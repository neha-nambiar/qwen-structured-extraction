"""RQ jobs: cpu_stage (decode, boxes, CLIP) and gpu_stage (fallback, extraction, flags)."""

import io
import logging
import time
from dataclasses import dataclass

import rq
from openai import OpenAI
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from eroll.extraction.qwen import classify_page, extract_pages
from eroll.extraction.validators import expected_serials, flag_reasons, serial_gaps
from eroll.pages.boxes import PAGE_TYPE_VOTER, crop_box
from eroll.pages.classify import (
    CLIP_THRESHOLD,
    OUTCOME_CLIP_VOTER,
    OUTCOME_QWEN_FALLBACK,
    Clip,
    analyse_page,
)
from eroll.pages.pdf import image_xrefs, open_pdf, page_images
from eroll.service.db import (
    JobPdfRow,
    JobRow,
    PageRow,
    RecordRow,
    add_problem,
    mark_failed,
    record_row,
    to_record,
)
from eroll.service.metrics import (
    CLIP_MARGIN,
    FLAGGED_RECORDS,
    JOBS_FINISHED,
    PAGE_DECISIONS,
    RECORD_CONFIDENCE,
    RECORDS_EXTRACTED,
    STAGE_SECONDS,
)

log = logging.getLogger(__name__)

JOB_TIMEOUT_SECONDS = 6 * 3600


@dataclass(frozen=True)
class Context:
    sessions: sessionmaker
    gpu_queue: rq.Queue | None  # CPU worker only
    clip: Clip | None  # CPU worker only
    llm: OpenAI | None  # GPU worker only
    vllm_model: str
    max_concurrent_requests: int
    confidence_threshold: float | None


# Bound once at worker start; jobs receive only the job id.
_context: Context | None = None


def bind(context: Context) -> None:
    global _context
    _context = context


def _bound() -> Context:
    if _context is None:
        raise RuntimeError("jobs.bind was not called before a job ran")
    return _context


def cpu_stage(job_id: str) -> None:
    context = _bound()
    if context.clip is None or context.gpu_queue is None:
        raise RuntimeError("cpu_stage needs a Context with clip and gpu_queue")

    start = time.perf_counter()
    page: int | None = None
    with context.sessions() as session:
        try:
            job = session.get(JobRow, job_id)
            if job is None:
                raise ValueError(f"no job {job_id}")
            job.status = "running"
            session.commit()

            pdf = session.get(JobPdfRow, job_id)
            if pdf is None:
                raise ValueError(f"job {job_id}: stored PDF is missing")
            data = pdf.data

            # Both run before any page is stored, so a malformed PDF fails with nothing stored.
            with open_pdf(data, job_id) as doc:
                xrefs = image_xrefs(doc, job_id)
                bad_voter_pages: list[tuple[int, list[str]]] = []
                for page_image in page_images(doc, xrefs, job_id):
                    page = page_image.number
                    clip_start = time.perf_counter()
                    analysis = analyse_page(context.clip, page_image.image, CLIP_THRESHOLD)
                    STAGE_SECONDS.labels("clip").observe(time.perf_counter() - clip_start)
                    PAGE_DECISIONS.labels(analysis.outcome).inc()
                    CLIP_MARGIN.observe(analysis.margin)
                    session.merge(
                        PageRow(
                            job_id=job_id,
                            page=page,
                            image_data=page_image.data,
                            width=page_image.image.size[0],
                            height=page_image.image.size[1],
                            boxes=[list(box) for box in analysis.boxes],
                            geometry_label=analysis.geometry_label,
                            p_voter=analysis.p_voter,
                            p_non_voter=analysis.p_non_voter,
                            margin=analysis.margin,
                            outcome=analysis.outcome,
                            final_label=analysis.final_label,
                            grid_problems=analysis.grid_problems,
                        )
                    )
                    session.commit()
                    log.info(
                        "job %s page %d: %d boxes, %s",
                        job_id,
                        page,
                        len(analysis.boxes),
                        analysis.outcome,
                    )
                    if analysis.outcome == OUTCOME_CLIP_VOTER and analysis.grid_problems:
                        bad_voter_pages.append((page, analysis.grid_problems))
            page = None

            job = session.get(JobRow, job_id)
            job.page_count = len(xrefs)
            session.commit()

            # Every bad page in one error, so the operator sees the whole list.
            if bad_voter_pages:
                details = "; ".join(
                    f"page {number}: {', '.join(problems)}" for number, problems in bad_voter_pages
                )
                raise ValueError(
                    f"{len(bad_voter_pages)} voter pages have grid problems: {details}"
                )

            context.gpu_queue.enqueue(gpu_stage, job_id, job_timeout=JOB_TIMEOUT_SECONDS)
            STAGE_SECONDS.labels("cpu_stage").observe(time.perf_counter() - start)
            log.info("job %s: %d pages classified, gpu_stage enqueued", job_id, len(xrefs))
        except Exception as exc:
            mark_failed(session, job_id, "cpu_stage", page, str(exc))
            JOBS_FINISHED.labels("failed").inc()
            raise


def _page_image(session: Session, job_id: str, page: int) -> Image.Image:
    data = session.scalar(
        select(PageRow.image_data).where(PageRow.job_id == job_id, PageRow.page == page)
    )
    if data is None:
        raise ValueError(f"job {job_id} page {page}: stored page image is missing")
    # Decoded exactly as pdf.page_images decoded it, so the stored boxes line up.
    return Image.open(io.BytesIO(data)).convert("RGB")


def gpu_stage(job_id: str) -> None:
    context = _bound()
    if context.llm is None:
        raise RuntimeError("gpu_stage needs a Context with llm")

    start = time.perf_counter()
    page: int | None = None
    with context.sessions() as session:
        try:
            # Step 1: Qwen labels for the pages the cascade could not decide.
            fallback_pages = session.execute(
                select(PageRow.page, PageRow.grid_problems)
                .where(
                    PageRow.job_id == job_id,
                    PageRow.outcome == OUTCOME_QWEN_FALLBACK,
                    PageRow.final_label.is_(None),
                )
                .order_by(PageRow.page)
            ).all()
            for page, problems in fallback_pages:
                image = _page_image(session, job_id, page)
                fallback_start = time.perf_counter()
                label = classify_page(context.llm, context.vllm_model, image, page)
                STAGE_SECONDS.labels("fallback").observe(time.perf_counter() - fallback_start)
                if label == PAGE_TYPE_VOTER and problems:
                    # Raised before the label is stored, so a retry asks again.
                    raise ValueError(
                        f"page {page}: fallback voter page has grid problems: {', '.join(problems)}"
                    )
                session.get(PageRow, (job_id, page)).final_label = label
                session.commit()
                log.info("job %s page %d: fallback label %s", job_id, page, label)
            page = None

            # Step 2: extract voter pages with no records yet, so a retry resumes.
            done = set(
                session.scalars(select(RecordRow.page).where(RecordRow.job_id == job_id).distinct())
            )
            todo = [
                (number, boxes)
                for number, boxes in session.execute(
                    select(PageRow.page, PageRow.boxes)
                    .where(PageRow.job_id == job_id, PageRow.final_label == PAGE_TYPE_VOTER)
                    .order_by(PageRow.page)
                )
                if number not in done
            ]

            def crops():
                # Lazy: one page is decoded at a time, overlapping requests in flight.
                for number, boxes in todo:
                    image = _page_image(session, job_id, number)
                    yield number, [crop_box(image, tuple(box)) for box in boxes]

            # Several pages are in flight, so `page` stays None; the error names page and box.
            page_start = time.perf_counter()
            for number, records in extract_pages(
                context.llm, context.vllm_model, crops(), context.max_concurrent_requests
            ):
                # One transaction per page: a failure later keeps this page.
                session.add_all(record_row(job_id, record) for record in records)
                session.commit()
                now = time.perf_counter()
                # Time between consecutive finished pages, the extraction throughput.
                STAGE_SECONDS.labels("extract_page").observe(now - page_start)
                page_start = now
                RECORDS_EXTRACTED.inc(len(records))
                for record in records:
                    RECORD_CONFIDENCE.observe(record.confidence)
                log.info("job %s page %d: %d records", job_id, number, len(records))

            # Step 3: flags over the whole job in reading order, since the serial base is
            # the job's first record.
            rows = list(
                session.scalars(
                    select(RecordRow)
                    .where(RecordRow.job_id == job_id)
                    .order_by(RecordRow.page, RecordRow.box)
                )
            )
            records = [to_record(row) for row in rows]
            flagged = 0
            for row, record, expected in zip(rows, records, expected_serials(records), strict=True):
                row.flags = flag_reasons(record, expected, context.confidence_threshold)
                flagged += bool(row.flags)
                for reason in row.flags:
                    FLAGGED_RECORDS.labels(reason).inc()

            by_page: dict[int, list] = {}
            for record in records:
                by_page.setdefault(record.page, []).append(record)
            gaps = serial_gaps(sorted(by_page.items()))
            for earlier, later in gaps:
                add_problem(
                    session,
                    job_id,
                    {
                        "kind": "serial_gap",
                        "stage": "gpu_stage",
                        "pages": [earlier, later],
                        "message": f"serials do not continue from page {earlier} to page {later}",
                    },
                )

            job = session.get(JobRow, job_id)
            # Wall time of this call; a failed call's time is not added.
            job.gpu_seconds += time.perf_counter() - start
            job.status = "completed"
            session.commit()
            STAGE_SECONDS.labels("gpu_stage").observe(time.perf_counter() - start)
            JOBS_FINISHED.labels("completed").inc()
            log.info(
                "job %s completed: %d records, %d flagged, %d serial gaps",
                job_id,
                len(records),
                flagged,
                len(gaps),
            )
        except Exception as exc:
            mark_failed(session, job_id, "gpu_stage", page, str(exc))
            JOBS_FINISHED.labels("failed").inc()
            raise

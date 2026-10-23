"""HTTP API: upload a PDF, poll its job, download the CSV, review flagged records."""

import hashlib
import io
import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import pymupdf
import redis
import rq
from fastapi import Depends, FastAPI, HTTPException, Request, Security, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.security import APIKeyHeader, HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest
from sqlalchemy import Select, create_engine, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from eroll.extraction.records import FIELD_ATTRS, FIELD_KEYS, apply_corrections, write_csv
from eroll.extraction.validators import LOW_CONFIDENCE
from eroll.pages.boxes import crop_box
from eroll.service.db import (
    CorrectionRow,
    JobPdfRow,
    JobRow,
    PageRow,
    RecordRow,
    check_head,
    latest_corrections,
    to_record,
)
from eroll.service.jobs import JOB_TIMEOUT_SECONDS, cpu_stage
from eroll.service.metrics import JOBS_SUBMITTED, QueueDepthCollector
from eroll.settings import load_settings, require

log = logging.getLogger(__name__)

API_KEY_MIN_LENGTH = 32
CPU_QUEUE = "cpu"
GPU_QUEUE = "gpu"
ALEMBIC_INI = Path("alembic.ini")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    require(settings, "database_url", "redis_url")
    api_key = settings.api_key.get_secret_value() if settings.api_key else ""
    if len(api_key) < API_KEY_MIN_LENGTH:
        log.error("API_KEY must be set and at least %d characters", API_KEY_MIN_LENGTH)
        raise SystemExit(2)

    engine = create_engine(settings.database_url.get_secret_value())
    check_head(engine, ALEMBIC_INI)

    connection = redis.Redis.from_url(settings.redis_url.get_secret_value())
    queue_depth = QueueDepthCollector(connection, (CPU_QUEUE, GPU_QUEUE))
    REGISTRY.register(queue_depth)
    app.state.sessions = sessionmaker(engine)
    app.state.cpu_queue = rq.Queue(CPU_QUEUE, connection=connection)
    app.state.api_key = api_key
    app.state.max_upload_bytes = settings.max_upload_bytes
    yield
    REGISTRY.unregister(queue_depth)
    connection.close()
    engine.dispose()


API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
HTTP_BASIC = HTTPBasic(auto_error=False)


def key_matches(provided: str | None, expected: str) -> bool:
    if not provided or not expected:
        return False
    return secrets.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def require_api_key(
    request: Request,
    header: Annotated[str | None, Security(API_KEY_HEADER)],
    basic: Annotated[HTTPBasicCredentials | None, Security(HTTP_BASIC)],
) -> None:
    expected = request.app.state.api_key
    if key_matches(header, expected) or (basic and key_matches(basic.password, expected)):
        return
    raise HTTPException(
        status_code=401, detail="invalid API key", headers={"WWW-Authenticate": "Basic"}
    )


# Every route, including /metrics and the review pages, needs the API key.
app = FastAPI(
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    dependencies=[Depends(require_api_key)],
)


@app.post("/jobs")
async def submit_job(request: Request, file: UploadFile) -> dict:
    state = request.app.state
    data = await file.read(state.max_upload_bytes + 1)
    if len(data) > state.max_upload_bytes:
        raise HTTPException(413, f"upload exceeds MAX_UPLOAD_BYTES ({state.max_upload_bytes})")
    job_id = hashlib.sha256(data).hexdigest()

    # Validated before anything touches Postgres or Redis.
    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            page_count = doc.page_count
    except (pymupdf.FileDataError, pymupdf.EmptyFileError) as exc:
        raise HTTPException(422, f"not a readable PDF: {exc}") from None
    if page_count < 1:
        raise HTTPException(422, "PDF has no pages")

    with state.sessions() as session:
        job = session.get(JobRow, job_id)
        if job is None:
            session.add(JobRow(id=job_id, status="queued"))
            try:
                # One commit, so a job never exists without its PDF. Flush first: with no
                # ORM relationship, insert order is not derived from the foreign key.
                session.flush()
                session.add(JobPdfRow(job_id=job_id, data=data))
                session.commit()
            except IntegrityError:
                # A concurrent identical upload inserted first; treat it as existing.
                session.rollback()
                job = session.get(JobRow, job_id)
            else:
                state.cpu_queue.enqueue(cpu_stage, job_id, job_timeout=JOB_TIMEOUT_SECONDS)
                JOBS_SUBMITTED.inc()
                log.info("job %s submitted, %d pages", job_id, page_count)
                return {"id": job_id, "status": "queued"}

        # Existing job: only a failed one is retried, keeping its records.
        if job.status == "failed":
            job.status = "queued"
            session.commit()
            state.cpu_queue.enqueue(cpu_stage, job_id, job_timeout=JOB_TIMEOUT_SECONDS)
            JOBS_SUBMITTED.inc()
            log.info("job %s requeued after failure", job_id)
        return {"id": job_id, "status": job.status}


def _job_or_404(session, job_id: str) -> JobRow:
    job = session.get(JobRow, job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return job


@app.get("/jobs/{job_id}")
def get_job(request: Request, job_id: str) -> dict:
    with request.app.state.sessions() as session:
        job = _job_or_404(session, job_id)
        pages = session.scalars(
            select(PageRow).where(PageRow.job_id == job_id).order_by(PageRow.page)
        )
        record_count = session.scalar(
            select(func.count()).select_from(RecordRow).where(RecordRow.job_id == job_id)
        )
        flagged_count = session.scalar(
            select(func.count())
            .select_from(RecordRow)
            .where(RecordRow.job_id == job_id, func.jsonb_array_length(RecordRow.flags) > 0)
        )
        return {
            "id": job.id,
            "status": job.status,
            "page_count": job.page_count,
            "pages": [
                {
                    "page": page.page,
                    "box_count": len(page.boxes),
                    "geometry_label": page.geometry_label,
                    "p_voter": page.p_voter,
                    "p_non_voter": page.p_non_voter,
                    "margin": page.margin,
                    "outcome": page.outcome,
                    "final_label": page.final_label,
                    "grid_problems": page.grid_problems,
                }
                for page in pages
            ],
            "problems": job.problems,
            "record_count": record_count,
            "flagged_count": flagged_count,
            "gpu_seconds": job.gpu_seconds,
        }


@app.get("/jobs/{job_id}/records.csv")
def get_records_csv(request: Request, job_id: str, raw: int = 0) -> Response:
    with request.app.state.sessions() as session:
        job = _job_or_404(session, job_id)
        if job.status != "completed":
            raise HTTPException(409, f"job is {job.status}, not completed")
        rows = list(session.scalars(select(RecordRow).where(RecordRow.job_id == job_id)))
        if raw == 1:
            records = [to_record(row) for row in rows]
        else:
            corrections = latest_corrections(session, [row.id for row in rows])
            records = [
                apply_corrections(to_record(row), corrections.get(row.id, [])) for row in rows
            ]

    buffer = io.StringIO(newline="")
    write_csv(records, buffer)
    return Response(
        buffer.getvalue().encode("utf-8"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{job_id}.csv"'},
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def changed_fields(prefilled: dict[str, str], submitted: dict[str, str]) -> dict[str, str]:
    """The submitted fields whose stripped value differs from the pre-filled one."""
    changed = {}
    for key, value in submitted.items():
        stripped = value.strip()
        if stripped != prefilled[key]:
            changed[key] = stripped
    return changed


def _unreviewed_flagged() -> Select:
    # Oldest job first, then page and box; job id breaks submission-time ties.
    return (
        select(
            RecordRow.id,
            RecordRow.job_id,
            RecordRow.page,
            RecordRow.box,
            RecordRow.flags,
            RecordRow.confidence,
        )
        .join(JobRow, JobRow.id == RecordRow.job_id)
        .where(RecordRow.reviewed.is_(False), func.jsonb_array_length(RecordRow.flags) > 0)
        .order_by(JobRow.submitted_at, RecordRow.job_id, RecordRow.page, RecordRow.box)
    )


def _flagged_or_404(session: Session, record_id: int) -> RecordRow:
    row = session.get(RecordRow, record_id)
    if row is None or not row.flags:
        raise HTTPException(404, "no such flagged record")
    return row


def _prefilled(session: Session, row: RecordRow) -> dict[str, str]:
    # The record's values with the latest correction per field applied.
    record = apply_corrections(
        to_record(row), latest_corrections(session, [row.id]).get(row.id, [])
    )
    values = record.model_dump(by_alias=True, include=FIELD_ATTRS)
    return {key: values[key] for key in FIELD_KEYS}


def _shown_confidence(flags: list[str], confidence: float | None) -> float | None:
    # Shown only where the confidence threshold caused the flag.
    return confidence if LOW_CONFIDENCE in flags else None


@app.get("/review", response_class=HTMLResponse)
def review_list(request: Request) -> HTMLResponse:
    with request.app.state.sessions() as session:
        rows = session.execute(_unreviewed_flagged()).all()
    items = [
        {
            "id": row.id,
            "job_id": row.job_id,
            "page": row.page,
            "box": row.box,
            "reasons": row.flags,
            "confidence": _shown_confidence(row.flags, row.confidence),
        }
        for row in rows
    ]
    return templates.TemplateResponse(request, "review_list.html", {"items": items})


@app.get("/review/{record_id}", response_class=HTMLResponse)
def review_record(request: Request, record_id: int) -> HTMLResponse:
    with request.app.state.sessions() as session:
        row = _flagged_or_404(session, record_id)
        context = {
            "record_id": row.id,
            "job_id": row.job_id,
            "page": row.page,
            "box": row.box,
            "reasons": row.flags,
            "confidence": _shown_confidence(row.flags, row.confidence),
            "fields": list(_prefilled(session, row).items()),
        }
    return templates.TemplateResponse(request, "review_record.html", context)


@app.get("/review/{record_id}/crop.png")
def review_crop(request: Request, record_id: int) -> Response:
    with request.app.state.sessions() as session:
        row = _flagged_or_404(session, record_id)
        boxes, data = session.execute(
            select(PageRow.boxes, PageRow.image_data).where(
                PageRow.job_id == row.job_id, PageRow.page == row.page
            )
        ).one()
        box = boxes[row.box - 1]
    # PNG so the crop is shown losslessly.
    image = Image.open(io.BytesIO(data)).convert("RGB")
    buffer = io.BytesIO()
    crop_box(image, tuple(box)).save(buffer, format="PNG")
    # The crop shows a voter's details, so the browser does not keep a copy.
    return Response(
        buffer.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"}
    )


@app.post("/review/{record_id}")
async def submit_review(request: Request, record_id: int) -> RedirectResponse:
    form = await request.form()
    with request.app.state.sessions() as session:
        row = _flagged_or_404(session, record_id)
        names = {key: f"f{index}" for index, key in enumerate(FIELD_KEYS)}
        missing = [key for key, name in names.items() if not isinstance(form.get(name), str)]
        if missing:
            raise HTTPException(422, f"missing fields: {', '.join(missing)}")
        submitted = {key: form[name] for key, name in names.items()}

        # One transaction: corrections and the reviewed mark land together.
        changes = changed_fields(_prefilled(session, row), submitted)
        session.add_all(
            CorrectionRow(record_id=row.id, field=key, value=value)
            for key, value in changes.items()
        )
        row.reviewed = True
        session.commit()
        log.info("record %d reviewed: %d fields corrected", record_id, len(changes))

        following = session.execute(_unreviewed_flagged().limit(1)).first()
    target = f"/review/{following.id}" if following else "/review"
    return RedirectResponse(target, status_code=303)

"""Run the deployed pipeline code in-process beside vLLM on the pod, into out/eval-<name>/."""

import argparse
import csv
import json
import logging
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import openai
from openai import OpenAI

from eroll.extraction.qwen import (
    MIN_PROBE_PROMPT_TOKENS,
    classify_page,
    extract_pages,
    probe,
    prompt_tokens,
)
from eroll.extraction.records import Record, write_csv
from eroll.pages.boxes import PAGE_TYPE_VOTER, crop_box
from eroll.pages.classify import CLIP_THRESHOLD, OUTCOME_QWEN_FALLBACK, analyse_page, load_clip
from eroll.pages.pdf import image_xrefs, open_pdf, page_images
from eroll.settings import load_settings

log = logging.getLogger(__name__)

RUN_NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")

VLLM_STARTUP_WAIT_SECONDS = 30
PROBE_RETRY_SECONDS = 2
PROBE_REQUEST_TIMEOUT_SECONDS = 5

REPO_ROOT = Path(__file__).resolve().parents[3]

PAGES_CSV_HEADER = (
    "pdf",
    "page",
    "width",
    "height",
    "box_count",
    "column_fill",
    "geometry_label",
    "p_voter",
    "p_non_voter",
    "margin",
    "outcome",
    "final_label",
    "qwen_label",
    "grid_problems",
)


@dataclass
class PageResult:
    pdf: str
    page: int
    width: int
    height: int
    boxes: list[tuple[int, int, int, int]]
    column_fill: float | None
    geometry_label: str
    p_voter: float
    p_non_voter: float
    margin: float
    outcome: str
    final_label: str | None
    qwen_label: str | None
    grid_problems: list[str]


def eval_dir(run_name: str, repo_root: Path) -> Path:
    """Resolved out/eval-<run_name>/; raise if it escapes out/ or already exists."""
    if not RUN_NAME_PATTERN.fullmatch(run_name):
        raise ValueError(f"run name {run_name!r} must be 1 to 64 ASCII letters, digits, '-' or '_'")
    root = repo_root.resolve()
    out_root = (root / "out").resolve()
    data_root = (root / "data").resolve()
    path = (out_root / f"eval-{run_name}").resolve()
    if path.is_relative_to(data_root):
        raise ValueError(f"output path {path} resolves inside {data_root}")
    if path.parent != out_root or path.name != f"eval-{run_name}":
        raise ValueError(f"output path {path} resolves outside {out_root}/eval-{run_name}")
    if path.exists():
        raise ValueError(f"output path {path} already exists")
    return path


def wait_for_probe(client: OpenAI, model: str, base_url: str) -> int:
    """Probe prompt token count, retrying connection failures for VLLM_STARTUP_WAIT_SECONDS."""
    short = client.with_options(timeout=PROBE_REQUEST_TIMEOUT_SECONDS)
    deadline = time.monotonic() + VLLM_STARTUP_WAIT_SECONDS
    while True:
        try:
            return probe(short, model)
        except openai.APIConnectionError:
            if time.monotonic() >= deadline:
                raise SystemExit(
                    f"vLLM server at {base_url} did not answer within {VLLM_STARTUP_WAIT_SECONDS} s"
                ) from None
            time.sleep(PROBE_RETRY_SECONDS)


def gpu_name() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().splitlines()[0]


def write_pages_csv(path: Path, results: list[PageResult]) -> None:
    with path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(PAGES_CSV_HEADER)
        for r in results:
            writer.writerow(
                (
                    r.pdf,
                    r.page,
                    r.width,
                    r.height,
                    len(r.boxes),
                    "" if r.column_fill is None else f"{r.column_fill:.4f}",
                    r.geometry_label,
                    f"{r.p_voter:.6f}",
                    f"{r.p_non_voter:.6f}",
                    f"{r.margin:.6f}",
                    r.outcome,
                    r.final_label or "",
                    r.qwen_label or "",
                    "; ".join(r.grid_problems),
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m eroll.evaluation.run_local")
    parser.add_argument("run_name")
    parser.add_argument("pdfs", nargs="+", type=Path)
    parser.add_argument(
        "--force-fallback",
        action="store_true",
        help="also send every page through the Qwen fallback; "
        "its answer goes to qwen_label, final labels stay the cascade's",
    )
    args = parser.parse_args()
    settings = load_settings()

    missing = [str(path) for path in args.pdfs if not path.is_file()]
    if missing:
        raise SystemExit(f"PDF not found: {', '.join(missing)}")
    stems = [path.stem for path in args.pdfs]
    if len(set(stems)) != len(stems):
        raise SystemExit("two PDFs share a filename stem; their <stem>.csv would collide")
    try:
        out_dir = eval_dir(args.run_name, REPO_ROOT)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    gpu = gpu_name()

    client = OpenAI(
        base_url=settings.vllm_base_url,
        api_key="unused",
        max_retries=0,
        timeout=settings.request_timeout_seconds,
    )
    model = settings.vllm_model
    probe_tokens = wait_for_probe(client, model, settings.vllm_base_url)
    if probe_tokens < MIN_PROBE_PROMPT_TOKENS:
        raise SystemExit(
            f"probe prompt tokens {probe_tokens} below MIN_PROBE_PROMPT_TOKENS "
            f"{MIN_PROBE_PROMPT_TOKENS}: min_pixels is not being applied"
        )
    log.info("probe: %d prompt tokens", probe_tokens)

    out_dir.mkdir(parents=True)
    clip = load_clip()

    results: list[PageResult] = []
    crop_tokens: dict[str, int] = {}
    for path in args.pdfs:
        with open_pdf(path.read_bytes(), path.name) as doc:
            xrefs = image_xrefs(doc, path.name)
            for page in page_images(doc, xrefs, path.name):
                a = analyse_page(clip, page.image, CLIP_THRESHOLD)
                width, height = page.image.size
                size = f"{width}x{height}"
                if a.geometry_label == PAGE_TYPE_VOTER and a.boxes and size not in crop_tokens:
                    crop = crop_box(page.image, a.boxes[0])
                    crop_tokens[size] = prompt_tokens(client, model, crop)
                    log.info("crop prompt tokens at %s: %d", size, crop_tokens[size])
                results.append(
                    PageResult(
                        pdf=path.stem,
                        page=page.number,
                        width=width,
                        height=height,
                        boxes=a.boxes,
                        column_fill=a.column_fill,
                        geometry_label=a.geometry_label,
                        p_voter=a.p_voter,
                        p_non_voter=a.p_non_voter,
                        margin=a.margin,
                        outcome=a.outcome,
                        final_label=a.final_label,
                        qwen_label=None,
                        grid_problems=a.grid_problems,
                    )
                )
        log.info("%s: analysed %d pages", path.name, len(xrefs))

    by_key = {(r.pdf, r.page): r for r in results}

    # Timed from here: Qwen fallback, extraction and CSV writing, not CLIP.
    start = time.perf_counter()
    for path in args.pdfs:
        wanted = {
            r.page
            for r in results
            if r.pdf == path.stem and (args.force_fallback or r.outcome == OUTCOME_QWEN_FALLBACK)
        }
        if not wanted:
            continue
        with open_pdf(path.read_bytes(), path.name) as doc:
            for page in page_images(doc, image_xrefs(doc, path.name), path.name):
                if page.number not in wanted:
                    continue
                r = by_key[(path.stem, page.number)]
                r.qwen_label = classify_page(client, model, page.image, page.number)
                if r.outcome == OUTCOME_QWEN_FALLBACK:
                    r.final_label = r.qwen_label

    # The service fails a job on a voter page with grid problems, so abort here too.
    bad = [r for r in results if r.final_label == PAGE_TYPE_VOTER and r.grid_problems]
    if bad:
        raise SystemExit(
            "voter pages with grid problems: "
            + "; ".join(f"{r.pdf} page {r.page}: {', '.join(r.grid_problems)}" for r in bad)
        )

    def voter_crops(path: Path, numbers: set[int]):
        with open_pdf(path.read_bytes(), path.name) as doc:
            for page in page_images(doc, image_xrefs(doc, path.name), path.name):
                if page.number in numbers:
                    r = by_key[(path.stem, page.number)]
                    yield page.number, [crop_box(page.image, box) for box in r.boxes]

    records: dict[str, list[Record]] = {path.stem: [] for path in args.pdfs}
    for path in args.pdfs:
        numbers = {
            r.page for r in results if r.pdf == path.stem and r.final_label == PAGE_TYPE_VOTER
        }
        if not numbers:
            continue
        for number, page_records in extract_pages(
            client, model, voter_crops(path, numbers), settings.max_concurrent_requests
        ):
            records[path.stem].extend(page_records)
            log.info("%s page %d: %d records", path.stem, number, len(page_records))

    for stem, pdf_records in records.items():
        with (out_dir / f"{stem}.csv").open("x", encoding="utf-8", newline="") as file:
            # extract_pages yields pages as they finish, not in page order.
            write_csv(sorted(pdf_records, key=lambda rec: (rec.page, rec.box)), file)
    seconds = time.perf_counter() - start

    write_pages_csv(out_dir / "pages.csv", results)
    record_count = sum(len(pdf_records) for pdf_records in records.values())
    timing = {
        "seconds": seconds,
        "records": record_count,
        "gpu_name": gpu,
        "probe_prompt_tokens": probe_tokens,
        "crop_prompt_tokens": crop_tokens,
        "max_concurrent_requests": settings.max_concurrent_requests,
        "force_fallback": args.force_fallback,
    }
    with (out_dir / "timing.json").open("x", encoding="utf-8") as file:
        json.dump(timing, file, indent=2)
    log.info(
        "%d records in %.1f s (%.3f s/record)",
        record_count,
        seconds,
        seconds / max(record_count, 1),
    )


if __name__ == "__main__":
    main()

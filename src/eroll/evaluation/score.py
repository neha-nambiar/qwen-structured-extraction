"""Score an evaluation run against the hand-corrected ground truth."""

import argparse
import csv
import json
import logging
import statistics
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from eroll.extraction.records import FIELD_KEYS, Record, normalise_house_number, read_csv
from eroll.pages.boxes import (
    EXPECTED_BOXES_PER_PAGE,
    MIN_COLUMN_FILL_RATIO,
    PAGE_TYPE_NON_VOTER,
    PAGE_TYPE_VOTER,
)
from eroll.pages.classify import CLIP_CALIBRATION_FALLBACK_RATE, OUTCOME_QWEN_FALLBACK
from eroll.pages.pdf import open_pdf
from eroll.settings import load_settings

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
TRUTH_CSV_DIR = REPO_ROOT / "data" / "extracted"
TRUTH_PDFS_DIR = REPO_ROOT / "data" / "pdfs"
BENCHMARKS_MD = REPO_ROOT / "BENCHMARKS.md"

GROUND_TRUTH_PDF_COUNT = 6  # fewer scored PDFs marks the report partial

RUN_FILES = {"pages.csv", "timing.json", "mistakes.csv"}
MISTAKES_HEADER = ("pdf", "page", "box", "field", "predicted", "truth")

# (pdf stem, page, box)
Key = tuple[str, int, int]


@dataclass(frozen=True)
class TruthPdf:
    """One ground-truth PDF: its page count and its records from the CSV."""

    page_count: int
    records: tuple[Record, ...]


def true_label(page: int, page_count: int) -> str:
    """True page label: pages 1, 2 and the last page are non-voter, the rest voter."""
    if not 1 <= page <= page_count:
        raise ValueError(f"page {page} is outside 1..{page_count}")
    if page in (1, 2, page_count):
        return PAGE_TYPE_NON_VOTER
    return PAGE_TYPE_VOTER


def check_truth(name: str, records: list[Record], page_count: int) -> None:
    """Raise on the first record with a bad page or box, a non-voter page, or a duplicate."""
    seen = set()
    for record in records:
        where = f"{name}: row with page {record.page}, box {record.box}"
        if not 1 <= record.page <= page_count:
            raise ValueError(f"{where}: page is outside 1..{page_count}")
        if not 1 <= record.box <= EXPECTED_BOXES_PER_PAGE:
            raise ValueError(f"{where}: box is outside 1..{EXPECTED_BOXES_PER_PAGE}")
        if true_label(record.page, page_count) == PAGE_TYPE_NON_VOTER:
            raise ValueError(f"{where}: page {record.page} is a non-voter page")
        key = (record.page, record.box)
        if key in seen:
            raise ValueError(f"{where}: duplicate (page, box)")
        seen.add(key)


def load_truth(extracted_dir: Path, pdfs_dir: Path) -> dict[str, TruthPdf]:
    """Read and check every ground-truth CSV with its PDF, keyed by stem; all or nothing."""
    csv_paths = sorted(extracted_dir.glob("*.csv"))
    if not csv_paths:
        raise ValueError(f"{extracted_dir}: contains no ground truth CSV")

    truth = {}
    for csv_path in csv_paths:
        pdf_path = pdfs_dir / f"{csv_path.stem}.pdf"
        if not pdf_path.is_file():
            raise ValueError(f"{csv_path.name}: no PDF of the same stem at {pdf_path}")

        with csv_path.open("r", encoding="utf-8", newline="") as file:
            records = read_csv(file, csv_path.name)
        with open_pdf(pdf_path.read_bytes(), pdf_path.name) as doc:
            page_count = doc.page_count

        check_truth(csv_path.name, records, page_count)
        truth[csv_path.stem] = TruthPdf(page_count=page_count, records=tuple(records))
    return truth


def fields_match(predicted: str, truth: str, field: str) -> bool:
    """Case-insensitive, whitespace-stripped match; a dash-only house number equals ""."""
    if field == "house number":
        predicted = normalise_house_number(predicted)
        truth = normalise_house_number(truth)
    return predicted.strip().casefold() == truth.strip().casefold()


@dataclass
class Scores:
    records: int = 0
    records_correct: int = 0
    fields: int = 0
    fields_correct: int = 0
    parse_failures: int = 0
    field_errors: dict[str, int] = field(default_factory=lambda: dict.fromkeys(FIELD_KEYS, 0))
    # (pdf, page, box, field, predicted, truth); written to mistakes.csv only, never logged.
    mistakes: list[tuple[str, int, int, str, str, str]] = field(default_factory=list)


def score_records(truth: dict[Key, Record], predicted: dict[Key, Record]) -> Scores:
    """Score predictions by field match; a missing or extra prediction raises."""
    missing = sorted(truth.keys() - predicted.keys())
    if missing:
        raise ValueError(
            f"{len(missing)} ground truth Records have no prediction, e.g. {missing[0]}"
        )
    extra = sorted(predicted.keys() - truth.keys())
    if extra:
        raise ValueError(f"{len(extra)} predictions have no ground truth Record, e.g. {extra[0]}")

    scores = Scores()
    for key in sorted(truth):
        want = truth[key].model_dump(by_alias=True)
        got_record = predicted[key]
        got = got_record.model_dump(by_alias=True)
        if got_record.error or all(got[k] == "" for k in FIELD_KEYS):
            scores.parse_failures += 1
        wrong = 0
        for k in FIELD_KEYS:
            if fields_match(got[k], want[k], k):
                continue
            wrong += 1
            scores.field_errors[k] += 1
            scores.mistakes.append((*key, k, got[k], want[k]))
        scores.records += 1
        scores.records_correct += wrong == 0
        scores.fields += len(FIELD_KEYS)
        scores.fields_correct += len(FIELD_KEYS) - wrong
    return scores


def read_pages(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"{path}: page decisions are missing")
    with path.open("r", encoding="utf-8", newline="") as file:
        return {(row["pdf"], int(row["page"])): row for row in csv.DictReader(file)}


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def verdict(passed: bool, evaluated: bool) -> str:
    if not evaluated:
        return "not evaluated"
    return "pass" if passed else "fail"


def accuracy_rows(label: str, s: Scores) -> list[str]:
    return [
        f"| {label} | {s.records} | {percent(s.fields_correct / s.fields)} "
        f"| {percent(s.records_correct / s.records)} | {s.parse_failures} | "
        + ", ".join(f"{k}: {n}" for k, n in s.field_errors.items() if n)
        + " |"
    ]


def write_mistakes(run_dir: Path, scores: Scores) -> None:
    with (run_dir / "mistakes.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(MISTAKES_HEADER)
        writer.writerows(scores.mistakes)


def log_accuracy(label: str, s: Scores) -> None:
    log.info(
        "%s: %d Records, field accuracy %s, Record accuracy %s, %d parse failures",
        label,
        s.records,
        percent(s.fields_correct / s.fields),
        percent(s.records_correct / s.records),
        s.parse_failures,
    )


def score_records_only(run_dir: Path, truth_pdfs: dict[str, TruthPdf]) -> None:
    """Score CSVs exported from the API: accuracy to the log and mistakes.csv only."""
    stems = sorted(p.stem for p in run_dir.glob("*.csv") if p.name not in RUN_FILES)
    if not stems:
        raise SystemExit(f"{run_dir}: contains no prediction CSV")
    unknown = [s for s in stems if s not in truth_pdfs]
    if unknown:
        raise SystemExit(f"{len(unknown)} prediction CSVs have no ground truth, e.g. {unknown[0]}")

    truth: dict[Key, Record] = {}
    predicted: dict[Key, Record] = {}
    size_of: dict[Key, str] = {}
    for stem in stems:
        pdf_path = TRUTH_PDFS_DIR / f"{stem}.pdf"
        with open_pdf(pdf_path.read_bytes(), pdf_path.name) as doc:
            for r in truth_pdfs[stem].records:
                width, height = doc[r.page - 1].get_images(full=True)[0][2:4]
                truth[(stem, r.page, r.box)] = r
                size_of[(stem, r.page, r.box)] = f"{width}x{height}"
        with (run_dir / f"{stem}.csv").open("r", encoding="utf-8", newline="") as file:
            for r in read_csv(file, f"{stem}.csv"):
                predicted[(stem, r.page, r.box)] = r

    try:
        overall = score_records(truth, predicted)
        by_size = {}
        for size in sorted(set(size_of.values())):
            keys = [k for k in truth if size_of[k] == size]
            by_size[size] = score_records(
                {k: truth[k] for k in keys}, {k: predicted[k] for k in keys}
            )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    write_mistakes(run_dir, overall)
    log.info("records-only score of %d PDFs in %s", len(stems), run_dir.name)
    log_accuracy("overall", overall)
    for size, s in by_size.items():
        log_accuracy(size, s)
    log.info("%d mistakes written to mistakes.csv", len(overall.mistakes))


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m eroll.evaluation.score")
    parser.add_argument("run_name")
    parser.add_argument(
        "--records-only",
        action="store_true",
        help="score Record CSVs exported from the API: accuracy only, to the log; "
        "no page decisions or timing needed, BENCHMARKS.md not written",
    )
    args = parser.parse_args()
    settings = load_settings()

    run_dir = (REPO_ROOT / "out" / f"eval-{args.run_name}").resolve()
    if run_dir.parent != (REPO_ROOT / "out").resolve() or not run_dir.is_dir():
        raise SystemExit(f"no evaluation run at {run_dir}")

    truth_pdfs = load_truth(TRUTH_CSV_DIR, TRUTH_PDFS_DIR)
    if args.records_only:
        score_records_only(run_dir, truth_pdfs)
        return
    pages = read_pages(run_dir / "pages.csv")
    timing_path = run_dir / "timing.json"
    if not timing_path.is_file():
        raise SystemExit(f"{timing_path}: extraction wall-clock time is missing")
    timing = json.loads(timing_path.read_text(encoding="utf-8"))

    stems = sorted(p.stem for p in run_dir.glob("*.csv") if p.name not in RUN_FILES)
    if not stems:
        raise SystemExit(f"{run_dir}: contains no prediction CSV")
    unknown = [s for s in stems if s not in truth_pdfs]
    if unknown:
        raise SystemExit(f"{len(unknown)} prediction CSVs have no ground truth, e.g. {unknown[0]}")

    truth: dict[Key, Record] = {}
    predicted: dict[Key, Record] = {}
    for stem in stems:
        for r in truth_pdfs[stem].records:
            truth[(stem, r.page, r.box)] = r
        with (run_dir / f"{stem}.csv").open("r", encoding="utf-8", newline="") as file:
            for r in read_csv(file, f"{stem}.csv"):
                predicted[(stem, r.page, r.box)] = r
        page_count = truth_pdfs[stem].page_count
        absent = [n for n in range(1, page_count + 1) if (stem, n) not in pages]
        if absent:
            raise SystemExit(
                f"pages.csv lacks {len(absent)} pages of {stem}, e.g. page {absent[0]}"
            )

    try:
        overall = score_records(truth, predicted)
        by_size: dict[str, Scores] = {}
        size_of = {k: f"{pages[k[:2]]['width']}x{pages[k[:2]]['height']}" for k in truth}
        for size in sorted(set(size_of.values())):
            keys = [k for k in truth if size_of[k] == size]
            by_size[size] = score_records(
                {k: truth[k] for k in keys}, {k: predicted[k] for k in keys}
            )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    # Page classification against true labels, over every page of the scored PDFs.
    truth_per_page: dict[tuple[str, int], int] = {}
    for stem, page, _ in truth:
        truth_per_page[(stem, page)] = truth_per_page.get((stem, page), 0) + 1
    written_per_page: dict[tuple[str, int], int] = {}
    for stem, page, _ in predicted:
        written_per_page[(stem, page)] = written_per_page.get((stem, page), 0) + 1

    class_pages = {PAGE_TYPE_VOTER: 0, PAGE_TYPE_NON_VOTER: 0}
    class_wrong = {PAGE_TYPE_VOTER: 0, PAGE_TYPE_NON_VOTER: 0}
    dropped = fallback = qwen_scored = qwen_wrong = 0
    detection_pages = detection_agree = 0
    voter_fills: list[float] = []
    non_voter_fills: list[float] = []
    for stem in stems:
        page_count = truth_pdfs[stem].page_count
        for n in range(1, page_count + 1):
            row = pages[(stem, n)]
            label = true_label(n, page_count)
            class_pages[label] += 1
            class_wrong[label] += row["final_label"] != label
            fallback += row["outcome"] == OUTCOME_QWEN_FALLBACK
            if row.get("qwen_label"):
                qwen_scored += 1
                qwen_wrong += row["qwen_label"] != label
            fill = float(row["column_fill"]) if row["column_fill"] else None
            if label == PAGE_TYPE_VOTER:
                if row["final_label"] != PAGE_TYPE_VOTER or not written_per_page.get((stem, n)):
                    dropped += 1
                    log.warning("%s page %d: voter page dropped", stem, n)
                detection_pages += 1
                if int(row["box_count"]) == truth_per_page.get((stem, n), 0):
                    detection_agree += 1
                else:
                    log.warning(
                        "%s page %d: detected %s boxes, ground truth has %d",
                        stem,
                        n,
                        row["box_count"],
                        truth_per_page.get((stem, n), 0),
                    )
                if fill is not None:
                    voter_fills.append(fill)
            elif fill is not None and int(row["box_count"]) >= 1:
                non_voter_fills.append(fill)
    total_pages = sum(class_pages.values())

    comparable = not timing.get("force_fallback", False)
    record_count = len(predicted)
    seconds_per_record = timing["seconds"] / record_count
    usd_per_1000 = seconds_per_record * 1000 / 3600 * settings.hourly_usd

    complete = len(stems) >= GROUND_TRUTH_PDF_COUNT
    timing_note = "" if comparable else " (not evaluated: forced fallback)"
    overall_field = overall.fields_correct / overall.fields

    lines = [
        "# Benchmarks",
        "",
        f"Run `{args.run_name}`, scored {date.today().isoformat()}"
        + ("" if complete else " (partial: fewer than six ground truth PDFs)")
        + ".",
        "",
        f"Scored {len(stems)} ground truth PDFs, {total_pages} pages and "
        f"{overall.records} Records. GPU: {timing['gpu_name']}, `MAX_CONCURRENT_REQUESTS` "
        f"{timing['max_concurrent_requests']}, `HOURLY_USD` {settings.hourly_usd}.",
        "",
        "## Summary",
        "",
        "| Measure | This run |",
        "|---|---|",
        f"| Field accuracy, overall | {percent(overall_field)} |",
        *(
            f"| Field accuracy, {size} | {percent(s.fields_correct / s.fields)} |"
            for size, s in by_size.items()
        ),
        f"| Voter pages dropped | {dropped} ({verdict(dropped == 0, complete)}) |",
        f"| Seconds per Record | {seconds_per_record:.3f}{timing_note} |",
        f"| $ per 1000 Records | {usd_per_1000:.3f}{timing_note} |",
        "",
        "## Extraction accuracy",
        "",
        "| Scope | Records | Field accuracy | Record accuracy | Parse failures | Field errors |",
        "|---|---|---|---|---|---|",
        *accuracy_rows("overall", overall),
        *(row for size, s in by_size.items() for row in accuracy_rows(size, s)),
        "",
        "## Page classification",
        "",
        "| True class | Pages | Final label wrong |",
        "|---|---|---|",
        *(f"| {c} | {class_pages[c]} | {class_wrong[c]} |" for c in class_pages),
        "",
        f"Voter pages dropped: {dropped}.",
        "",
        f"Qwen fallback rate: {fallback} of {total_pages} pages, "
        f"{percent(fallback / total_pages)}; calibration set rate "
        f"{percent(CLIP_CALIBRATION_FALLBACK_RATE)}.",
    ]
    if qwen_scored:
        lines += [
            "",
            "Fallback accuracy ("
            + (
                "every page forced through the fallback"
                if timing.get("force_fallback", False)
                else "pages the cascade sent to the fallback"
            )
            + f"): {qwen_wrong} wrong of "
            f"{qwen_scored} pages.",
        ]
    lines += [
        "",
        "## Detection",
        "",
        f"Voter pages whose detected box count equals the ground truth Record count: "
        f"{detection_agree} of {detection_pages}.",
        "",
        "Column fill (median box width / (page width / 3)), for the `MIN_COLUMN_FILL_RATIO` "
        f"({MIN_COLUMN_FILL_RATIO}) comment: voter median "
        + (
            f"{statistics.median(voter_fills):.3f}, lowest {min(voter_fills):.3f}"
            if voter_fills
            else "n/a"
        )
        + "; highest non-voter page with a detected rectangle "
        + (f"{max(non_voter_fills):.3f}" if non_voter_fills else "n/a")
        + ".",
        "",
        "## Speed",
        "",
        f"Wall clock {timing['seconds']:.1f} s for {record_count} Records. Probe prompt tokens: "
        f"{timing['probe_prompt_tokens']}; first-crop prompt tokens per page image size: "
        + ", ".join(f"{size}: {n}" for size, n in sorted(timing["crop_prompt_tokens"].items()))
        + ".",
        "",
    ]

    # Written after every check passed.
    write_mistakes(run_dir, overall)
    BENCHMARKS_MD.write_text("\n".join(lines), encoding="utf-8")
    log.info(
        "scored %d Records: field %.4f, record %.4f; %d mistakes",
        overall.records,
        overall_field,
        overall.records_correct / overall.records,
        len(overall.mistakes),
    )


if __name__ == "__main__":
    main()

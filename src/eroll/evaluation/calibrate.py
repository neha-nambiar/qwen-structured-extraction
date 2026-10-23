"""Fit the CLIP margin threshold on calibration PDFs disjoint from the ground truth."""

import datetime
import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from eroll.evaluation.score import true_label
from eroll.pages.boxes import PAGE_TYPE_NON_VOTER, PAGE_TYPE_VOTER
from eroll.pages.classify import (
    CLIP_MODEL_ID,
    NON_VOTER_PROMPTS,
    OUTCOME_QWEN_FALLBACK,
    VOTER_PROMPTS,
    analyse_page,
    decide,
    load_clip,
)
from eroll.pages.pdf import image_xrefs, open_pdf, page_images
from eroll.settings import load_settings

log = logging.getLogger(__name__)

MAX_CLASS_ERROR_RATE = 0.0

# Several constituencies so the threshold isn't fitted to one printer's scans; sixty
# PDFs for a large enough check set.
MIN_CALIBRATION_PDFS = 60
MIN_CONSTITUENCIES = 6

CONSTITUENCY_PATTERN = re.compile(r"-S\d+-(\d+)-")

# 0.00..1.00 built from integers so each candidate is the float nearest its 2-dp value.
CANDIDATE_THRESHOLDS = tuple(round(i / 100, 2) for i in range(101))

PAGE_TYPES = (PAGE_TYPE_VOTER, PAGE_TYPE_NON_VOTER)

REPO_ROOT = Path(__file__).resolve().parents[3]
CALIBRATION_PDFS_DIR = REPO_ROOT / "data" / "calibration" / "pdfs"
TRUTH_PDFS_DIR = REPO_ROOT / "data" / "pdfs"
CALIBRATION_MD = REPO_ROOT / "CALIBRATION.md"


@dataclass(frozen=True)
class CalibrationPage:
    true_label: str
    clip_label: str
    geometry_label: str
    margin: float


def class_error_counts(
    pages: list[CalibrationPage], threshold: float
) -> dict[str, tuple[int, int]]:
    """Per true class, (cascade errors, pages) at `threshold`."""
    # An error is a page decide() would trust whose CLIP label is wrong.
    counts = {label: [0, 0] for label in PAGE_TYPES}
    for page in pages:
        if page.true_label not in counts:
            raise ValueError(f"true label {page.true_label!r} is not a page type")
        counts[page.true_label][1] += 1
        trusted = page.margin >= threshold and page.clip_label == page.geometry_label
        if trusted and page.clip_label != page.true_label:
            counts[page.true_label][0] += 1
    return {label: (errors, total) for label, (errors, total) in counts.items()}


def class_error_rates(pages: list[CalibrationPage], threshold: float) -> dict[str, float]:
    """Per true class, cascade errors over pages; raises on a class with no pages."""
    rates = {}
    for label, (errors, total) in class_error_counts(pages, threshold).items():
        if total == 0:
            raise ValueError(f"no {label} pages, so its error rate is undefined")
        rates[label] = errors / total
    return rates


def select_threshold(pages: list[CalibrationPage], max_rate: float) -> float | None:
    """The smallest candidate where every class's error rate is at most max_rate, or None."""
    for threshold in CANDIDATE_THRESHOLDS:
        if all(rate <= max_rate for rate in class_error_rates(pages, threshold).values()):
            return threshold
    return None


def check_disjoint(calibration: list[tuple[str, str]], truth: list[tuple[str, str]]) -> None:
    """Raise if any calibration PDF matches a ground-truth PDF by SHA-256 (catches renames)."""
    truth_by_hash = {sha256: name for name, sha256 in truth}
    for name, sha256 in calibration:
        if sha256 in truth_by_hash:
            raise ValueError(
                f"Calibration set PDF {name} is identical (SHA-256 {sha256}) to "
                f"ground truth PDF {truth_by_hash[sha256]}; refusing to calibrate"
            )


def constituency(name: str) -> int:
    """The constituency number in a roll file name; raises if there is none."""
    match = CONSTITUENCY_PATTERN.search(name)
    if match is None:
        raise ValueError(
            f"{name}: no constituency number in the file name "
            f"(expected {CONSTITUENCY_PATTERN.pattern!r})"
        )
    # int, so "076" and "76" are one constituency.
    return int(match.group(1))


def check_counts(names: list[str]) -> None:
    """Raise if the calibration set has too few PDFs or constituencies."""
    constituencies = {constituency(name) for name in names}
    if len(names) < MIN_CALIBRATION_PDFS or len(constituencies) < MIN_CONSTITUENCIES:
        raise ValueError(
            f"Calibration set has {len(names)} PDFs from {len(constituencies)} "
            f"constituencies; at least {MIN_CALIBRATION_PDFS} PDFs from "
            f"{MIN_CONSTITUENCIES} constituencies are required"
        )


def hash_pdfs(directory: Path) -> list[tuple[str, str]]:
    """(file name, hex SHA-256) for every PDF in directory, sorted by name."""
    return [
        (path.name, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(directory.glob("*.pdf"))
    ]


def write_calibration_md(
    path: Path,
    fitted_on: datetime.date,
    pdf_count: int,
    constituency_count: int,
    threshold: float,
    error_counts: dict[str, tuple[int, int]],
    fallback_count: int,
    page_count: int,
) -> None:
    """Write CALIBRATION.md: aggregates only, no file names or per-page rows."""
    lines = [
        "# CLIP threshold calibration",
        "",
        "Written by `python -m eroll.evaluation.calibrate`. Copy the threshold and the",
        "`qwen_fallback` rate into `src/eroll/pages/classify.py` by hand.",
        "",
        f"- Date of fitting: {fitted_on.isoformat()}",
        f"- CLIP model: `{CLIP_MODEL_ID}`",
        f"- `MAX_CLASS_ERROR_RATE`: {MAX_CLASS_ERROR_RATE}",
        f"- Calibration set: {pdf_count} PDFs, {constituency_count} constituencies, "
        f"{page_count} pages",
        f"- Selected threshold: {threshold:.2f}",
        "",
        "## Text prompts",
        "",
        "Voter:",
        "",
        *[f"- {prompt}" for prompt in VOTER_PROMPTS],
        "",
        "Non-voter:",
        "",
        *[f"- {prompt}" for prompt in NON_VOTER_PROMPTS],
        "",
        "## Cascade errors at the selected threshold",
        "",
        "| Class | Pages | Errors | Rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, (errors, total) in error_counts.items():
        lines.append(f"| {label} | {total} | {errors} | {errors / total:.4f} |")
    lines += [
        "",
        "## Pages sent to the Qwen fallback at the selected threshold",
        "",
        "| Pages | `qwen_fallback` | Rate |",
        "| ---: | ---: | ---: |",
        f"| {page_count} | {fallback_count} | {fallback_count / page_count:.4f} |",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    load_settings()

    calibration = hash_pdfs(CALIBRATION_PDFS_DIR)
    truth = hash_pdfs(TRUTH_PDFS_DIR)
    log.info("hashed %d calibration set and %d ground truth PDFs", len(calibration), len(truth))
    check_disjoint(calibration, truth)
    names = [name for name, _ in calibration]
    check_counts(names)
    constituency_count = len({constituency(name) for name in names})

    clip = load_clip()

    pages: list[CalibrationPage] = []
    scores: list[tuple[float, float, str]] = []
    for name in names:
        with open_pdf((CALIBRATION_PDFS_DIR / name).read_bytes(), name) as doc:
            xrefs = image_xrefs(doc, name)
            decoded = page_images(doc, xrefs, name)
            for number in range(1, len(xrefs) + 1):
                try:
                    page = next(decoded)
                except OSError as exc:
                    raise ValueError(
                        f"{name} page {number}: cannot be decoded into a page image ({exc})"
                    ) from exc
                analysis = analyse_page(clip, page.image, threshold=1.0)
                clip_label = (
                    PAGE_TYPE_VOTER
                    if analysis.p_voter > analysis.p_non_voter
                    else PAGE_TYPE_NON_VOTER
                )
                pages.append(
                    CalibrationPage(
                        true_label=true_label(number, len(xrefs)),
                        clip_label=clip_label,
                        geometry_label=analysis.geometry_label,
                        margin=analysis.margin,
                    )
                )
                scores.append((analysis.p_voter, analysis.p_non_voter, analysis.geometry_label))
        log.info("%s: analysed %d pages", name, len(xrefs))

    threshold = select_threshold(pages, MAX_CLASS_ERROR_RATE)
    if threshold is None:
        lowest = {
            label: min(class_error_rates(pages, t)[label] for t in CANDIDATE_THRESHOLDS)
            for label in PAGE_TYPES
        }
        raise SystemExit(
            f"no threshold in 0.00..1.00 keeps both classes at or below "
            f"MAX_CLASS_ERROR_RATE {MAX_CLASS_ERROR_RATE}; lowest error rate reached: "
            + ", ".join(f"{label} {rate:.4f}" for label, rate in lowest.items())
            + "; CALIBRATION.md not written"
        )

    error_counts = class_error_counts(pages, threshold)
    fallback_count = sum(
        decide(p_voter, p_non_voter, geometry, threshold)[0] == OUTCOME_QWEN_FALLBACK
        for p_voter, p_non_voter, geometry in scores
    )

    log.info(
        "selected threshold %.2f over %d pages from %d PDFs in %d constituencies",
        threshold,
        len(pages),
        len(names),
        constituency_count,
    )
    for label, (errors, total) in error_counts.items():
        log.info("  %s: %d errors in %d pages", label, errors, total)
    log.info(
        "  qwen_fallback: %d of %d pages (%.4f)",
        fallback_count,
        len(pages),
        fallback_count / len(pages),
    )

    write_calibration_md(
        CALIBRATION_MD,
        datetime.date.today(),
        len(names),
        constituency_count,
        threshold,
        error_counts,
        fallback_count,
        len(pages),
    )
    log.info("written to %s", CALIBRATION_MD)


if __name__ == "__main__":
    main()

"""Check decoding, box detection and serials against the ground truth, with no model."""

import argparse
import csv
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path

from eroll.evaluation.score import TruthPdf, load_truth, true_label
from eroll.extraction.records import Record
from eroll.extraction.validators import serial_gaps
from eroll.pages.boxes import (
    PAGE_TYPE_NON_VOTER,
    PAGE_TYPE_VOTER,
    column_fill,
    crop_box,
    detect_boxes,
    geometry_label,
    grid_problems,
)
from eroll.pages.pdf import image_xrefs, open_pdf, page_images
from eroll.settings import load_settings

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_EXTRACTED_DIR = DATA_DIR / "extracted"
DEFAULT_PDFS_DIR = DATA_DIR / "pdfs"
DEFAULT_OUT_DIR = REPO_ROOT / "out" / "truth-check"

PAGES_CSV_HEADER = (
    "pdf",
    "page",
    "width",
    "height",
    "true_label",
    "geometry_label",
    "box_count",
    "truth_records",
    "column_fill",
    "grid_problems",
)


@dataclass(frozen=True)
class PageCheck:
    pdf: str
    page: int
    width: int
    height: int
    true_label: str
    geometry_label: str
    box_count: int
    truth_records: int
    fill: float | None
    grid_problems: tuple[str, ...]

    @property
    def size(self) -> str:
        return f"{self.width}x{self.height}"

    @property
    def where(self) -> str:
        return f"{self.pdf} p{self.page}"

    @property
    def label_mismatch(self) -> bool:
        return self.geometry_label != self.true_label

    @property
    def count_mismatch(self) -> bool:
        return self.true_label == PAGE_TYPE_VOTER and self.box_count != self.truth_records


def check_output_dir(out_dir: Path, forbidden: list[Path]) -> Path:
    """out_dir resolved (so `..` or symlinks cannot reach data/); raise if forbidden."""
    resolved = out_dir.resolve()
    for directory in forbidden:
        if resolved.is_relative_to(directory.resolve()):
            raise ValueError(f"output directory {resolved} is inside {directory}; refusing")
    return resolved


def records_by_page(records: tuple[Record, ...]) -> dict[int, list[Record]]:
    """Records grouped by page, each page sorted by box number."""
    pages: dict[int, list[Record]] = {}
    for record in sorted(records, key=lambda r: (r.page, r.box)):
        pages.setdefault(record.page, []).append(record)
    return pages


def check_pages(stem: str, truth: TruthPdf, pdf_path: Path) -> list[PageCheck]:
    truth_pages = records_by_page(truth.records)
    checks = []
    with open_pdf(pdf_path.read_bytes(), pdf_path.name) as doc:
        xrefs = image_xrefs(doc, pdf_path.name)
        for page in page_images(doc, xrefs, pdf_path.name):
            width, height = page.image.size
            boxes = detect_boxes(page.image)
            label = true_label(page.number, truth.page_count)
            problems = grid_problems(boxes) if label == PAGE_TYPE_VOTER else []
            checks.append(
                PageCheck(
                    pdf=stem,
                    page=page.number,
                    width=width,
                    height=height,
                    true_label=label,
                    geometry_label=geometry_label(boxes, width),
                    box_count=len(boxes),
                    truth_records=len(truth_pages.get(page.number, [])),
                    fill=column_fill(boxes, width),
                    grid_problems=tuple(problems),
                )
            )
    log.info("%s: checked %d pages", stem, len(checks))
    return checks


def pdf_serial_gaps(stem: str, truth: TruthPdf) -> list[str]:
    """Serial gaps between consecutive voter pages, as "stem pA-pB"."""
    # Voter pages with no records are skipped; the box count check still fails them.
    truth_pages = records_by_page(truth.records)
    voter_pages = [
        (page, truth_pages[page])
        for page in range(1, truth.page_count + 1)
        if true_label(page, truth.page_count) == PAGE_TYPE_VOTER and page in truth_pages
    ]
    return [f"{stem} p{earlier}-p{later}" for earlier, later in serial_gaps(voter_pages)]


def log_failures(what: str, places: list[str]) -> None:
    if places:
        log.info("  %s: %d (%s)", what, len(places), ", ".join(places))
    else:
        log.info("  %s: 0", what)


def log_summary(group: str, checks: list[PageCheck]) -> None:
    """Log one group's aggregates. No field values are logged."""
    voter = [c for c in checks if c.true_label == PAGE_TYPE_VOTER]
    non_voter = [c for c in checks if c.true_label == PAGE_TYPE_NON_VOTER]
    log.info(
        "%s: %d pages from %d PDFs (%d voter, %d non-voter), %d ground truth Records",
        group,
        len(checks),
        len({c.pdf for c in checks}),
        len(voter),
        len(non_voter),
        sum(c.truth_records for c in checks),
    )
    log_failures("geometry label mismatches", [c.where for c in checks if c.label_mismatch])
    log_failures("voter-page box count mismatches", [c.where for c in voter if c.count_mismatch])
    log_failures("voter pages with grid problems", [c.where for c in voter if c.grid_problems])

    voter_fills = [c.fill for c in voter if c.fill is not None]
    if voter_fills:
        log.info(
            "  voter-page fill: median %.3f, lowest %.3f over %d pages with rectangles",
            statistics.median(voter_fills),
            min(voter_fills),
            len(voter_fills),
        )
    else:
        log.info("  voter-page fill: no voter page has a rectangle")

    non_voter_fills = [c.fill for c in non_voter if c.fill is not None]
    if non_voter_fills:
        log.info(
            "  highest non-voter fill: %.3f, over %d of %d non-voter pages with rectangles",
            max(non_voter_fills),
            len(non_voter_fills),
            len(non_voter),
        )
    else:
        log.info("  highest non-voter fill: no non-voter page has a rectangle")


def write_pages_csv(checks: list[PageCheck], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(PAGES_CSV_HEADER)
        for c in checks:
            writer.writerow(
                [
                    c.pdf,
                    c.page,
                    c.width,
                    c.height,
                    c.true_label,
                    c.geometry_label,
                    c.box_count,
                    c.truth_records,
                    "" if c.fill is None else f"{c.fill:.4f}",
                    "; ".join(c.grid_problems),
                ]
            )


def run_check(truth: dict[str, TruthPdf], pdfs_dir: Path, out_dir: Path) -> bool:
    """Check every ground-truth PDF, log aggregates, write pages.csv; True if all pass."""
    checks = []
    gaps = []
    for stem, truth_pdf in truth.items():
        checks.extend(check_pages(stem, truth_pdf, pdfs_dir / f"{stem}.pdf"))
        gaps.extend(pdf_serial_gaps(stem, truth_pdf))

    log.info("checked %d ground truth PDFs", len(truth))
    for size in sorted({c.size for c in checks}):
        log_summary(size, [c for c in checks if c.size == size])
    log_summary("all sizes", checks)
    log_failures("serial gaps between voter pages", gaps)

    out_dir.mkdir(parents=True, exist_ok=True)
    pages_csv = out_dir / "pages.csv"
    write_pages_csv(checks, pages_csv)
    log.info("per-page rows written to %s", pages_csv)

    return not (
        gaps or any(c.label_mismatch or c.count_mismatch or c.grid_problems for c in checks)
    )


def write_crop(
    stem: str, truth: TruthPdf, pdf_path: Path, page_number: int, box: int, out_dir: Path
) -> None:
    if not 1 <= page_number <= truth.page_count:
        raise ValueError(f"{stem}: page {page_number} is outside 1..{truth.page_count}")

    with open_pdf(pdf_path.read_bytes(), pdf_path.name) as doc:
        xrefs = image_xrefs(doc, pdf_path.name)
        page = next(p for p in page_images(doc, xrefs, pdf_path.name) if p.number == page_number)

    boxes = detect_boxes(page.image)
    if not 1 <= box <= len(boxes):
        raise ValueError(
            f"{stem} page {page_number}: box {box} is outside 1..{len(boxes)} detected"
        )

    records = records_by_page(truth.records).get(page_number, [])
    if len(records) != len(boxes):
        log.warning(
            "%s page %d: %d boxes detected but %d ground truth Records; box numbers may not "
            "line up",
            stem,
            page_number,
            len(boxes),
            len(records),
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    crop_path = out_dir / f"crop-{page_number}-{box}.png"
    crop_box(page.image, boxes[box - 1]).save(crop_path)
    log.info("crop written to %s", crop_path)

    record = next((r for r in records if r.box == box), None)
    if record is None:
        log.warning("%s page %d: no ground truth Record at box %d", stem, page_number, box)
        return
    print(f"Ground truth name at {stem} page {page_number} box {box}: {record.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m eroll.evaluation.check_truth",
        description="Check page decoding, box detection and serials on the ground truth.",
    )
    parser.add_argument("--extracted-dir", type=Path, default=DEFAULT_EXTRACTED_DIR)
    parser.add_argument("--pdfs-dir", type=Path, default=DEFAULT_PDFS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--crop",
        nargs=3,
        metavar=("STEM", "PAGE", "BOX"),
        help="write one detected box's crop instead of running the check",
    )
    args = parser.parse_args()
    if args.crop is not None:
        stem, page, box = args.crop
        if not (page.isascii() and page.isdigit() and box.isascii() and box.isdigit()):
            parser.error(f"--crop PAGE and BOX must be whole numbers, got {page!r} {box!r}")
        args.crop = (stem, int(page), int(box))
    return args


def main() -> None:
    load_settings()
    args = parse_args()

    out_dir = check_output_dir(args.out_dir, [DATA_DIR, args.extracted_dir, args.pdfs_dir])
    truth = load_truth(args.extracted_dir, args.pdfs_dir)

    if args.crop is not None:
        stem, page, box = args.crop
        if stem not in truth:
            raise SystemExit(f"--crop: no ground truth PDF with stem {stem!r}")
        write_crop(stem, truth[stem], args.pdfs_dir / f"{stem}.pdf", page, box, out_dir)
        return

    if not run_check(truth, args.pdfs_dir, out_dir):
        log.error("ground-truth check failed; see the counts above and %s", out_dir)
        raise SystemExit(1)
    log.info("ground-truth check passed")


if __name__ == "__main__":
    main()

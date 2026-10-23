"""Voter-box detection, page geometry and crops for one page image (pure functions)."""

import statistics

import cv2
import numpy as np
from PIL.Image import Image

Box = tuple[int, int, int, int]  # (x, y, w, h) in page image pixels

# The electoral roll format: a voter page is a grid of up to 10 rows of 3 voter boxes.
EXPECTED_ROWS = 10
EXPECTED_COLUMNS = 3
EXPECTED_BOXES_PER_PAGE = EXPECTED_ROWS * EXPECTED_COLUMNS

# Detection thresholds are fractions of the page's width, height or area (or unitless
# ratios), so they hold at any resolution.
ADAPTIVE_BLOCK_FRACTION = 0.017  # adaptive-threshold block, fraction of width, forced odd, floor 3
ADAPTIVE_C = 11  # constant subtracted from the local mean (unitless)
H_KERNEL_WIDTH_FRACTION = 0.022  # horizontal opening kernel length, fraction of width
V_KERNEL_HEIGHT_FRACTION = 0.010  # vertical opening kernel length, fraction of height
CLOSE_KERNEL_FRACTION = 0.002  # square closing kernel side, fraction of width, forced odd, floor 3
MIN_WIDTH_FRACTION = 0.24  # rectangle width lower bound, fraction of width
MAX_WIDTH_FRACTION = 0.38  # rectangle width upper bound, fraction of width
MIN_HEIGHT_FRACTION = 0.06  # rectangle height lower bound, fraction of height
MAX_HEIGHT_FRACTION = 0.12  # rectangle height upper bound, fraction of height
MIN_ASPECT = 1.5  # width-to-height ratio lower bound
MAX_ASPECT = 4.5  # width-to-height ratio upper bound
MIN_AREA_FRACTION = 0.015  # rectangle area floor, fraction of page area
IOU_THRESHOLD = 0.70  # discard an overlapping rectangle at or above this IoU
# inward inset per edge, fraction of that rectangle's own width or height
CROP_INSET_FRACTION = 0.02
MIN_CROP_INSET_PX = 1  # floor on the inset, in pixels
ROW_TOLERANCE_FRACTION = 0.45  # row membership window, multiple of median rectangle height
COLUMN_TOLERANCE_FRACTION = 0.25  # column membership window, multiple of median rectangle width

# Voter boxes span ~0.945 of a column, table cells ~0.744 (measured by check_truth).
# Re-measure for a new roll format.
MIN_COLUMN_FILL_RATIO = 0.92

PAGE_TYPE_VOTER = "voter"
PAGE_TYPE_NON_VOTER = "non_voter"


def _make_odd(value: int) -> int:
    """Return the smallest odd integer >= value (odd kernel sizes only)."""
    return value if value % 2 == 1 else value + 1


def _iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union


def _vertical_centre(box: Box) -> float:
    return box[1] + box[3] / 2


def group_rows(boxes: list[Box]) -> list[list[Box]]:
    """Cluster boxes into rows by vertical centre, top to bottom, each left to right."""
    if not boxes:
        return []

    median_h = statistics.median(b[3] for b in boxes)
    tolerance = ROW_TOLERANCE_FRACTION * median_h

    rows: list[list[Box]] = []
    for box in sorted(boxes, key=_vertical_centre):
        centre = _vertical_centre(box)
        for row in rows:
            mean_centre = sum(_vertical_centre(b) for b in row) / len(row)
            if abs(centre - mean_centre) <= tolerance:
                row.append(box)
                break
        else:
            rows.append([box])

    rows.sort(key=lambda row: sum(_vertical_centre(b) for b in row) / len(row))
    for row in rows:
        row.sort(key=lambda b: b[0])
    return rows


def column_fill(boxes: list[Box], page_width: int) -> float | None:
    """Median box width as a fraction of one grid column; None when there are no boxes."""
    if not boxes:
        return None
    column_width = page_width / EXPECTED_COLUMNS
    return statistics.median(b[2] for b in boxes) / column_width


def geometry_label(boxes: list[Box], page_width: int) -> str:
    """Voter if the boxes are as wide as a voter box, else non-voter."""
    # Width, not count: a voter page with a missed box stays a voter page and fails loudly.
    fill = column_fill(boxes, page_width)
    if fill is None or fill < MIN_COLUMN_FILL_RATIO:
        return PAGE_TYPE_NON_VOTER
    return PAGE_TYPE_VOTER


def grid_problems(boxes: list[Box]) -> list[str]:
    """Every structural problem with a voter page's box grid; empty if consistent."""
    # Full rows then one short final row is a valid partial page.
    if not boxes:
        return [f"0 boxes detected, expected 1 to {EXPECTED_BOXES_PER_PAGE}"]

    problems = []
    if len(boxes) > EXPECTED_BOXES_PER_PAGE:
        problems.append(
            f"{len(boxes)} boxes exceeds the {EXPECTED_ROWS}x{EXPECTED_COLUMNS} grid "
            f"of {EXPECTED_BOXES_PER_PAGE}"
        )

    rows = group_rows(boxes)
    counts = [len(row) for row in rows]

    if len(rows) > EXPECTED_ROWS:
        problems.append(f"{len(rows)} rows exceeds the expected {EXPECTED_ROWS}")

    short_or_long = [n for n in counts[:-1] if n != EXPECTED_COLUMNS]
    if short_or_long:
        problems.append(
            f"non-final rows hold {short_or_long} boxes in per-row counts {counts}: every "
            f"row but the last must hold {EXPECTED_COLUMNS}, so a box is missing mid-page"
        )

    if counts[-1] > EXPECTED_COLUMNS:
        problems.append(f"final row holds {counts[-1]} boxes, expected 1 to {EXPECTED_COLUMNS}")

    median_width = statistics.median(b[2] for b in boxes)
    tolerance = COLUMN_TOLERANCE_FRACTION * median_width
    columns: list[list[int]] = []
    for left in sorted(b[0] for b in boxes):
        for column in columns:
            if abs(left - sum(column) / len(column)) <= tolerance:
                column.append(left)
                break
        else:
            columns.append([left])
    if len(columns) > EXPECTED_COLUMNS:
        problems.append(
            f"{len(columns)} distinct left-edge columns exceeds {EXPECTED_COLUMNS}; "
            "columns do not line up"
        )

    return problems


def detect_boxes(image: Image) -> list[Box]:
    """Voter-box rectangles on one page, in reading order; possibly []."""
    W, H = image.size
    page_area = W * H

    # Binarise with a local threshold so uneven scan brightness does not matter.
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    block = max(3, _make_odd(round(W * ADAPTIVE_BLOCK_FRACTION)))
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        ADAPTIVE_C,
    )

    # Keep only long horizontal and vertical strokes (box borders), then close small gaps.
    h_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(3, int(W * H_KERNEL_WIDTH_FRACTION)), 1)
    )
    v_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(3, int(H * V_KERNEL_HEIGHT_FRACTION)))
    )
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    mask = cv2.bitwise_or(horizontal, vertical)
    close_side = _make_odd(max(3, int(W * CLOSE_KERNEL_FRACTION)))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_side, close_side))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Keep rectangles with a voter box's size and shape.
    candidates: list[Box] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not (MIN_WIDTH_FRACTION * W <= w <= MAX_WIDTH_FRACTION * W):
            continue
        if not (MIN_HEIGHT_FRACTION * H <= h <= MAX_HEIGHT_FRACTION * H):
            continue
        aspect = w / h
        if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
            continue
        if w * h < MIN_AREA_FRACTION * page_area:
            continue
        candidates.append((x, y, w, h))

    if not candidates:
        return []

    # IoU dedup, largest first, so an outer contour wins over a nested inner one.
    candidates.sort(key=lambda b: b[2] * b[3], reverse=True)
    retained: list[Box] = []
    for box in candidates:
        if all(_iou(box, kept) < IOU_THRESHOLD for kept in retained):
            retained.append(box)

    return [box for row in group_rows(retained) for box in row]


def crop_box(image: Image, box: Box) -> Image:
    """Crop one box, inset slightly and clamped to the page."""
    # The inset drops the drawn border; the serial number and EPIC id stay inside.
    W, H = image.size
    x, y, w, h = box
    inset_x = max(MIN_CROP_INSET_PX, int(w * CROP_INSET_FRACTION))
    inset_y = max(MIN_CROP_INSET_PX, int(h * CROP_INSET_FRACTION))
    left = max(0, x + inset_x)
    top = max(0, y + inset_y)
    right = min(W, x + w - inset_x)
    bottom = min(H, y + h - inset_y)
    return image.crop((left, top, right, bottom))

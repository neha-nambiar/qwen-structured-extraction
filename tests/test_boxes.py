import cv2
import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from PIL import Image

from eroll.pages.boxes import (
    COLUMN_TOLERANCE_FRACTION,
    CROP_INSET_FRACTION,
    MIN_CROP_INSET_PX,
    ROW_TOLERANCE_FRACTION,
    Box,
    crop_box,
    detect_boxes,
    grid_problems,
)

PAGE_WIDTH = 1983  # the 240 DPI roll format
PAGE_HEIGHT = 2806
BOX_WIDTH = round(0.94 * PAGE_WIDTH / 3)  # 0.94 of a grid column
BOX_HEIGHT = round(0.08 * PAGE_HEIGHT)  # 0.08 of the page height
TOP_MARGIN_FRACTION = 0.05
ROW_PITCH_FRACTION = 0.09  # 0.08 box plus 0.01 gap; ten rows end at 0.94 of the height
BORDER_PX = 3  # ink drawn inward from each rectangle's outer edge
SCALES = (0.5, 1.0, 2.0)


# Max edge offset for a detected box to match the drawn one: border plus one resampled pixel.
def edge_tolerance(scale: float) -> float:
    return BORDER_PX + 1 / scale


def draw_grid(full_rows: int, final_row: int) -> tuple[Image.Image, list[Box]]:
    """A white page with full_rows rows of 3 rectangles then final_row more, left-aligned.

    Returns the page and the drawn rectangles as (x, y, w, h) in reading order.
    """
    page = np.full((PAGE_HEIGHT, PAGE_WIDTH, 3), 255, dtype=np.uint8)
    column_width = PAGE_WIDTH / 3
    drawn: list[Box] = []
    for row in range(full_rows + 1):
        for column in range(3 if row < full_rows else final_row):
            x = round(column * column_width + (column_width - BOX_WIDTH) / 2)
            y = round(PAGE_HEIGHT * (TOP_MARGIN_FRACTION + ROW_PITCH_FRACTION * row))
            right, bottom = x + BOX_WIDTH - 1, y + BOX_HEIGHT - 1
            # Filled black then white inside, so the outer edge is exact (a stroke
            # thickness would be centred on the corners).
            cv2.rectangle(page, (x, y), (right, bottom), (0, 0, 0), thickness=-1)
            cv2.rectangle(
                page,
                (x + BORDER_PX, y + BORDER_PX),
                (right - BORDER_PX, bottom - BORDER_PX),
                (255, 255, 255),
                thickness=-1,
            )
            drawn.append((x, y, BOX_WIDTH, BOX_HEIGHT))
    return Image.fromarray(page), drawn


def edges(box: Box, scale: float) -> tuple[float, float, float, float]:
    """Left, top, right and bottom of a box, converted back to page pixels."""
    x, y, w, h = box
    return x / scale, y / scale, (x + w) / scale, (y + h) / scale


GRIDS = [(full_rows, final_row) for full_rows in range(10) for final_row in range(1, 4)]


# Synthetic grids are detected exactly at any scale; all 30 grids x 3 scales.
@pytest.mark.parametrize(("full_rows", "final_row"), GRIDS)
def test_synthetic_grid_detected_exactly_at_every_scale(full_rows: int, final_row: int) -> None:
    page, drawn = draw_grid(full_rows, final_row)
    for scale in SCALES:
        size = (round(PAGE_WIDTH * scale), round(PAGE_HEIGHT * scale))
        scaled = page if scale == 1.0 else page.resize(size, Image.Resampling.BICUBIC)
        detected = detect_boxes(scaled)

        assert len(detected) == len(drawn), f"scale {scale}: {len(detected)} boxes detected"
        for i, (found, expected) in enumerate(zip(detected, drawn, strict=True)):
            offsets = [
                abs(a - b) for a, b in zip(edges(found, scale), edges(expected, 1.0), strict=True)
            ]
            assert max(offsets) <= edge_tolerance(scale), (
                f"scale {scale}, box {i}: detected {found}, drawn {expected}"
            )


def test_blank_page_returns_no_boxes() -> None:
    page = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")
    assert detect_boxes(page) == []


# Max jitter in page pixels: half the row/column tolerance, so two jittered members of
# one row or column still cluster together.
MAX_JITTER_X = int(COLUMN_TOLERANCE_FRACTION * BOX_WIDTH / 2)
MAX_JITTER_Y = int(ROW_TOLERANCE_FRACTION * BOX_HEIGHT / 2)
# Half a grid column puts a left edge midway between columns, forming a new column.
COLUMN_SHIFT = round(PAGE_WIDTH / 3 / 2)


def grid_box(row: int, column: int, dx: int = 0, dy: int = 0) -> Box:
    """The box at one grid position, placed as draw_grid places it, then offset by dx, dy."""
    column_width = PAGE_WIDTH / 3
    x = round(column * column_width + (column_width - BOX_WIDTH) / 2)
    y = round(PAGE_HEIGHT * (TOP_MARGIN_FRACTION + ROW_PITCH_FRACTION * row))
    return (x + dx, y + dy, BOX_WIDTH, BOX_HEIGHT)


ANY_FULL_ROWS = st.integers(0, 9)
ANY_FINAL_ROW = st.integers(1, 3)


@st.composite
def jittered_grids(
    draw: st.DrawFn,
    full_rows: st.SearchStrategy[int] = ANY_FULL_ROWS,
    final_row: st.SearchStrategy[int] = ANY_FINAL_ROW,
) -> list[list[Box]]:
    """A valid grid as rows of boxes: full rows of 3 then a final row, every box jittered."""
    rows_of_3, last = draw(full_rows), draw(final_row)
    jitter_x = st.integers(-MAX_JITTER_X, MAX_JITTER_X)
    jitter_y = st.integers(-MAX_JITTER_Y, MAX_JITTER_Y)
    return [
        [
            grid_box(row, column, draw(jitter_x), draw(jitter_y))
            for column in range(3 if row < rows_of_3 else last)
        ]
        for row in range(rows_of_3 + 1)
    ]


def assert_problems(problems: list[str], fragments: list[str]) -> None:
    """Each problem contains exactly one fragment (the observed count), one to one."""
    matches = [[fragment for fragment in fragments if fragment in p] for p in problems]
    assert len(problems) == len(fragments), problems
    assert all(len(m) == 1 for m in matches), problems
    assert sorted(m[0] for m in matches) == sorted(fragments), problems


# Grid validator accepts valid grids and names every violation, one test per violation.
@given(jittered_grids())
def test_grid_validator_accepts_valid_grids(rows: list[list[Box]]) -> None:
    assert grid_problems([box for row in rows for box in row]) == []


def test_grid_validator_names_zero_boxes() -> None:
    assert_problems(grid_problems([]), ["0 boxes"])


# An eleventh row: always "11 rows"; also a short row and/or too many boxes depending on
# how many the old final row held.
@given(jittered_grids(full_rows=st.just(9)), st.integers(1, 3))
def test_grid_validator_names_extra_row(rows: list[list[Box]], extra: int) -> None:
    boxes = [box for row in rows for box in row] + [grid_box(10, c) for c in range(extra)]
    old_final = len(rows[-1])
    expected = ["11 rows"]
    if len(boxes) > 30:
        expected.append(f"{len(boxes)} boxes")
    if old_final < 3:
        expected.append(str([old_final]))
    assert_problems(grid_problems(boxes), expected)


# Boxes removed from non-final rows: one problem listing every short row's count.
@given(st.data())
def test_grid_validator_names_short_rows(data: st.DataObject) -> None:
    rows = data.draw(jittered_grids(full_rows=st.integers(1, 9)))
    shortened = data.draw(st.lists(st.integers(0, len(rows) - 2), min_size=1, unique=True))
    for index in shortened:
        removed = data.draw(st.lists(st.integers(0, 2), min_size=1, max_size=2, unique=True))
        rows[index] = [box for column, box in enumerate(rows[index]) if column not in removed]
    short = [len(row) for row in rows[:-1] if len(row) != 3]
    assert_problems(grid_problems([box for row in rows for box in row]), [str(short)])


# One box shifted sideways between columns adds a fourth column. Two full rows ensure
# every column keeps another box, and row counts are unchanged.
@given(st.data())
def test_grid_validator_names_fourth_column(data: st.DataObject) -> None:
    rows = data.draw(jittered_grids(full_rows=st.integers(2, 9)))
    shifted_row = data.draw(st.integers(0, len(rows) - 1))
    column = data.draw(st.integers(0, len(rows[shifted_row]) - 1))
    x, y, w, h = rows[shifted_row][column]
    rows[shifted_row][column] = (x + (COLUMN_SHIFT if column == 0 else -COLUMN_SHIFT), y, w, h)
    assert_problems(grid_problems([box for row in rows for box in row]), ["4 distinct"])


def coordinate_page(width: int, height: int) -> np.ndarray:
    """Every pixel holds y * width + x, so a crop's pixels say exactly where it was cut."""
    return np.arange(width * height, dtype=np.int32).reshape(height, width)


# Crops follow the inset rule and stay inside the page. Box sizes exercise both the
# MIN_CROP_INSET_PX floor and the fraction; origins make boxes cross every page edge.
@given(st.data())
def test_crop_follows_inset_rule_and_stays_inside_page(data: st.DataObject) -> None:
    width, height = data.draw(st.integers(1, 120)), data.draw(st.integers(1, 120))
    w, h = data.draw(st.integers(3, 300)), data.draw(st.integers(3, 300))
    x, y = data.draw(st.integers(-w, width)), data.draw(st.integers(-h, height))

    inset_x = max(MIN_CROP_INSET_PX, int(CROP_INSET_FRACTION * w))
    inset_y = max(MIN_CROP_INSET_PX, int(CROP_INSET_FRACTION * h))
    columns = [c for c in range(x + inset_x, x + w - inset_x) if 0 <= c < width]
    rows = [r for r in range(y + inset_y, y + h - inset_y) if 0 <= r < height]
    # Skip boxes whose inset interior misses the page entirely.
    assume(columns and rows)

    page = coordinate_page(width, height)
    cropped = np.array(crop_box(Image.fromarray(page), (x, y, w, h)))
    np.testing.assert_array_equal(cropped, page[np.ix_(rows, columns)])


# Pins the constants: a 100 x 40 box has insets 2 across and 1 down (the floor), and
# boxes hanging off the page are clamped.
def test_crop_inset_values() -> None:
    page = coordinate_page(200, 100)
    inside = np.array(crop_box(Image.fromarray(page), (20, 10, 100, 40)))
    np.testing.assert_array_equal(inside, page[11:49, 22:118])
    top_left = np.array(crop_box(Image.fromarray(page), (-10, -10, 100, 40)))
    np.testing.assert_array_equal(top_left, page[0:29, 0:88])
    bottom_right = np.array(crop_box(Image.fromarray(page), (150, 70, 100, 40)))
    np.testing.assert_array_equal(bottom_right, page[71:100, 152:200])

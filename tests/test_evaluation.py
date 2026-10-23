import re
import tempfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

from eroll.evaluation.calibrate import (
    CONSTITUENCY_PATTERN,
    CalibrationPage,
    check_counts,
    check_disjoint,
    class_error_rates,
    select_threshold,
)
from eroll.evaluation.run_local import RUN_NAME_PATTERN, eval_dir
from eroll.evaluation.score import check_truth, fields_match, score_records, true_label
from eroll.extraction.records import FIELD_KEYS, Record
from eroll.pages.boxes import PAGE_TYPE_NON_VOTER, PAGE_TYPE_VOTER

PAGE_COUNT = 10


def record(page: int, box: int) -> Record:
    return Record.model_validate({"page": page, "box": box, **{key: "x" for key in FIELD_KEYS}})


def test_true_label_follows_page_type_rule():
    assert true_label(1, PAGE_COUNT) == PAGE_TYPE_NON_VOTER
    assert true_label(2, PAGE_COUNT) == PAGE_TYPE_NON_VOTER
    assert true_label(3, PAGE_COUNT) == PAGE_TYPE_VOTER
    assert true_label(PAGE_COUNT - 1, PAGE_COUNT) == PAGE_TYPE_VOTER
    assert true_label(PAGE_COUNT, PAGE_COUNT) == PAGE_TYPE_NON_VOTER


def test_check_truth_accepts_voter_pages_only():
    check_truth("t.csv", [record(3, 1), record(3, 30), record(PAGE_COUNT - 1, 1)], PAGE_COUNT)


def test_check_truth_rejects_duplicate_page_and_box():
    records = [record(3, 7), record(4, 7), record(3, 7)]
    with pytest.raises(ValueError, match=re.escape("t.csv: row with page 3, box 7: duplicate")):
        check_truth("t.csv", records, PAGE_COUNT)


def test_check_truth_rejects_box_31():
    with pytest.raises(ValueError, match=re.escape("t.csv: row with page 3, box 31: box")):
        check_truth("t.csv", [record(3, 31)], PAGE_COUNT)


def test_check_truth_rejects_page_past_last():
    with pytest.raises(ValueError, match=re.escape("t.csv: row with page 11, box 1: page")):
        check_truth("t.csv", [record(PAGE_COUNT + 1, 1)], PAGE_COUNT)


def test_check_truth_rejects_record_on_non_voter_page():
    with pytest.raises(ValueError, match=re.escape("t.csv: row with page 2, box 5: page 2 is")):
        check_truth("t.csv", [record(2, 5)], PAGE_COUNT)


def test_fields_match_ignores_case_and_surrounding_whitespace():
    assert fields_match("  ABC1234567 ", "abc1234567", "epic id")
    assert not fields_match("abc1234567", "abc1234568", "epic id")


def test_fields_match_treats_dash_only_house_number_as_empty():
    assert fields_match(" - ", "", "house number")
    assert fields_match("", "--", "house number")
    assert not fields_match("12-A", "12", "house number")
    # The dash rule is for house numbers only.
    assert not fields_match("-", "", "age")


def test_score_records_raises_naming_count_and_key_of_dropped_prediction():
    truth = {("roll", 3, b): record(3, b) for b in (1, 2, 3)}
    predicted = {("roll", 3, 1): record(3, 1)}
    with pytest.raises(
        ValueError,
        match=re.escape("2 ground truth Records have no prediction, e.g. ('roll', 3, 2)"),
    ):
        score_records(truth, predicted)


PAGE_TYPES = st.sampled_from([PAGE_TYPE_VOTER, PAGE_TYPE_NON_VOTER])
# Half the margins land exactly on a grid point, where margin >= t is decided by equality.
MARGINS = st.one_of(st.integers(0, 100).map(lambda i: i / 100), st.floats(0.0, 1.0))


def calibration_page(true: str):
    return st.builds(
        CalibrationPage,
        true_label=st.just(true),
        clip_label=PAGE_TYPES,
        geometry_label=PAGE_TYPES,
        margin=MARGINS,
    )


# One page of each true class always present: with none, a class's rate is 0/0 and
# class_error_rates raises by design.
CALIBRATION_PAGES = st.tuples(
    calibration_page(PAGE_TYPE_VOTER),
    calibration_page(PAGE_TYPE_NON_VOTER),
    st.lists(PAGE_TYPES.flatmap(calibration_page), max_size=30),
).map(lambda drawn: [drawn[0], drawn[1], *drawn[2]])


# Threshold selection picks the smallest admissible grid value.
@given(CALIBRATION_PAGES, st.one_of(st.just(0.0), st.floats(0.0, 1.0)))
def test_select_threshold_is_smallest_grid_value_within_max_rate(pages, max_rate):
    def counted_rate(label: str, t: float) -> float:
        of_class = [p for p in pages if p.true_label == label]
        errors = [
            p
            for p in of_class
            if p.margin >= t and p.clip_label == p.geometry_label and p.clip_label != label
        ]
        return len(errors) / len(of_class)

    admissible = []
    for t in [i / 100 for i in range(101)]:
        expected = {
            label: counted_rate(label, t) for label in (PAGE_TYPE_VOTER, PAGE_TYPE_NON_VOTER)
        }
        assert class_error_rates(pages, t) == expected
        if all(rate <= max_rate for rate in expected.values()):
            admissible.append(t)

    assert select_threshold(pages, max_rate) == (admissible[0] if admissible else None)


def roll_name(constituency: int, part: int) -> str:
    return f"2026-EROLLGEN-S13-{constituency}-SIR-DraftRoll-Revision1-ENG-{part}-WI.pdf"


def roll_names(pdfs: int, constituencies: int) -> list[str]:
    return [roll_name(170 + i % constituencies, i) for i in range(pdfs)]


def test_check_disjoint_raises_naming_both_files_on_shared_hash():
    calibration = [("cal-a.pdf", "aa"), ("cal-b.pdf", "bb")]
    truth = [("truth-x.pdf", "cc"), ("truth-y.pdf", "bb")]
    with pytest.raises(ValueError, match=r"cal-b\.pdf .* truth-y\.pdf"):
        check_disjoint(calibration, truth)


def test_check_disjoint_passes_on_disjoint_hashes():
    check_disjoint([("cal-a.pdf", "aa")], [("truth-x.pdf", "cc")])


def test_check_counts_passes_at_exactly_60_pdfs_from_6_constituencies():
    check_counts(roll_names(60, 6))


def test_check_counts_raises_on_59_pdfs_stating_both_counts():
    with pytest.raises(ValueError, match="has 59 PDFs from 6 constituencies"):
        check_counts(roll_names(59, 6))


def test_check_counts_raises_on_5_constituencies_stating_both_counts():
    with pytest.raises(ValueError, match="has 60 PDFs from 5 constituencies"):
        check_counts(roll_names(60, 5))


def test_check_counts_raises_on_name_without_constituency():
    with pytest.raises(ValueError, match=re.escape("roll.pdf: no constituency number")):
        check_counts([*roll_names(60, 6), "roll.pdf"])


def test_constituency_pattern_reads_ground_truth_style_names():
    assert CONSTITUENCY_PATTERN.search(roll_name(76, 1)).group(1) == "76"
    assert CONSTITUENCY_PATTERN.search(roll_name(79, 10)).group(1) == "79"


RUN_NAME_CHARS = "abcXYZ019_-./\\ :"


# Evaluation output stays inside its own directory.
@given(
    name=st.one_of(
        st.text(alphabet=RUN_NAME_CHARS, max_size=70),
        st.sampled_from(["..", "../data", "data", "../../data/x", "a/b", "x" * 64, "x" * 65]),
    ),
    precreate=st.booleans(),
)
# Windows temp dirs are slow enough to trip the default deadline.
@hypothesis_settings(deadline=None)
def test_eval_dir_stays_inside_its_own_directory(name: str, precreate: bool):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "data").mkdir()
        (root / "out").mkdir()
        valid = RUN_NAME_PATTERN.fullmatch(name) is not None
        if valid and precreate:
            (root / "out" / f"eval-{name}").mkdir()
        if not valid or precreate:
            with pytest.raises(ValueError):
                eval_dir(name, root)
            return
        path = eval_dir(name, root)
        assert path == (root / "out" / f"eval-{name}").resolve()
        assert not path.is_relative_to((root / "data").resolve())
        assert not path.exists()

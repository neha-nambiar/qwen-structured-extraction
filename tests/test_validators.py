import re

import pytest
from hypothesis import given
from hypothesis import strategies as st

from eroll.extraction.records import Record
from eroll.extraction.validators import (
    LOW_CONFIDENCE,
    VALIDATOR_NAMES,
    expected_serials,
    flag_reasons,
    serial_gaps,
)
from eroll.pages.boxes import EXPECTED_BOXES_PER_PAGE

# Lone surrogates are excluded: they are not text a page, a model or a CSV can carry.
any_text = st.text(st.characters(exclude_categories=["Cs"]), max_size=12)

# Each field mixes passing values, near misses and arbitrary text, including non-ASCII
# letters and digits that str.isalpha/isdigit accept.
epic_ids = st.one_of(
    st.from_regex(r"[A-Z]{3}[0-9]{7}", fullmatch=True),
    st.from_regex(r"[A-Za-z]{3}[0-9]{7}", fullmatch=True),
    st.from_regex(r"[A-Za-z]{2,4}[0-9]{6,8}", fullmatch=True),
    st.builds(
        lambda letters, digits: letters + digits,
        st.text(st.characters(categories=["L"]), min_size=3, max_size=3),
        st.text(st.characters(categories=["Nd"]), min_size=7, max_size=7),
    ),
    any_text,
)
ages = st.one_of(
    st.integers(min_value=0, max_value=200).map(str),
    st.sampled_from(["17", "18", "120", "121", "018", " 30", "3O", "²", "\u0967\u096e", ""]),
    any_text,
)
genders = st.one_of(st.sampled_from(["Male", "FEMALE", "female", "Other", " Male", ""]), any_text)
relation_types = st.one_of(
    st.sampled_from(["Husband", "father", "MOTHER", "Others", "Other", "Wife", ""]), any_text
)
errors = st.one_of(st.none(), st.just(""), any_text)
# Confidence and threshold are mean log probabilities, so both are at most 0.
log_probs = st.floats(max_value=0.0, allow_nan=False)


@st.composite
def flag_cases(draw):
    page = draw(st.integers(min_value=1, max_value=2000))
    box = draw(st.integers(min_value=1, max_value=EXPECTED_BOXES_PER_PAGE))
    expected = draw(st.integers(min_value=0, max_value=10_000))
    serial = draw(
        st.one_of(
            st.just(str(expected)),
            st.sampled_from([str(expected + 1), str(expected - 1), f"0{expected}", f" {expected}"]),
            st.integers(min_value=0, max_value=100_000).map(str),
            any_text,
        )
    )
    record = Record.model_validate(
        {
            "page": page,
            "box": box,
            "serial number": serial,
            "epic id": draw(epic_ids),
            "name": draw(any_text),
            "relation type": draw(relation_types),
            "relation name": draw(any_text),
            "house number": draw(any_text),
            "age": draw(ages),
            "gender": draw(genders),
            "confidence": draw(st.none() | log_probs),
            "error": draw(errors),
        }
    )
    threshold = draw(st.none() | log_probs)
    return record, expected, threshold


# Flag reasons match their definitions and come from a fixed set.
@given(flag_cases())
def test_flag_reasons_match_definitions(case):
    record, expected_serial, threshold = case

    # Independent oracle: explicit ASCII regexes rather than the helpers under test.
    epic = record.epic_id
    epic_ok = re.fullmatch(r"[A-Za-z]{3}[0-9]{7}", epic) is not None
    age_ok = re.fullmatch(r"[0-9]+", record.age) is not None and 18 <= int(record.age) <= 120
    expected = {
        "serial": record.serial_number != str(expected_serial),
        "epic_format": not epic_ok,
        "epic_lowercase": epic_ok and re.fullmatch(r"[A-Z]{3}", epic[:3]) is None,
        "age_range": not age_ok,
        "gender_vocab": record.gender.lower() not in {"male", "female"},
        "relation_vocab": record.relation_type.lower()
        not in {"husband", "father", "mother", "others"},
        "parse_error": record.error not in (None, ""),
        LOW_CONFIDENCE: threshold is not None
        and record.confidence is not None
        and record.confidence < threshold,
    }

    reasons = flag_reasons(record, expected_serial, threshold)

    assert set(reasons) <= {*VALIDATOR_NAMES, LOW_CONFIDENCE}
    # Sorted lists rather than sets, so a reason reported twice also fails.
    assert sorted(reasons) == sorted(name for name, failed in expected.items() if failed)


def serial_record(page: int, box: int, serial: str) -> Record:
    # Only page, box and serial number matter to serial_gaps; the rest is filler.
    return Record.model_validate(
        {
            "page": page,
            "box": box,
            "serial number": serial,
            "epic id": "ABC1234567",
            "name": "",
            "relation type": "Father",
            "relation name": "",
            "house number": "",
            "age": "30",
            "gender": "Male",
        }
    )


# How each page boundary is made; "none" is weighted double. "zero_padded" continues,
# since "0571" is a whole number.
boundary_kinds = st.sampled_from(
    ["none", "none", "off_by", "non_digit", "space_padded", "zero_padded"]
)
# Strings int() or str.isdigit would accept but the definition rejects, plus any text.
non_digit_serials = st.one_of(
    st.sampled_from(["", "57l", "S71", "-1", "+5", "5.0", "1_000", "²", "\u0967\u0968"]),
    any_text,
)


@st.composite
def voter_pages(draw):
    page = draw(st.integers(min_value=1, max_value=50))
    serial = draw(st.integers(min_value=0, max_value=5000))
    pages: list[tuple[int, list[str]]] = []
    for index in range(draw(st.integers(min_value=0, max_value=6))):
        count = draw(st.integers(min_value=1, max_value=4))
        kind = draw(boundary_kinds) if index else "none"
        if index:
            page += draw(st.integers(min_value=1, max_value=3))
        if kind == "off_by":
            serial += draw(st.integers(min_value=-5, max_value=5).filter(bool))
        texts = [str(serial + offset) for offset in range(count)]
        serial += count
        if kind not in ("none", "off_by"):
            # Corrupt the earlier page's last serial or this page's first.
            if draw(st.booleans()):
                side, position = pages[-1][1], -1
            else:
                side, position = texts, 0
            if kind == "non_digit":
                side[position] = draw(non_digit_serials)
            elif kind == "space_padded":
                side[position] = draw(st.sampled_from([" {}", "{} ", "\t{}", "{}\n"])).format(
                    side[position]
                )
            else:
                side[position] = "0" + side[position]
        pages.append((page, texts))
    return [
        (number, [serial_record(number, box, text) for box, text in enumerate(texts, 1)])
        for number, texts in pages
    ]


# Serial gaps are reported exactly.
@given(voter_pages())
def test_serial_gaps_reported_exactly(pages):
    # Independent oracle; fullmatch, unlike "$", rejects a trailing newline.
    def whole(text):
        return re.fullmatch(r"[0-9]+", text) is not None

    expected = []
    for (earlier, earlier_records), (later, later_records) in zip(pages, pages[1:], strict=False):
        last = earlier_records[-1].serial_number
        first = later_records[0].serial_number
        if not (whole(last) and whole(first) and int(first) == int(last) + 1):
            expected.append((earlier, later))

    assert serial_gaps(pages) == expected


def test_serial_gaps_rejects_voter_page_without_records():
    pages = [(3, [serial_record(3, 1, "811")]), (4, [])]
    with pytest.raises(ValueError, match="page 4"):
        serial_gaps(pages)


def test_expected_serials_run_on_across_a_partial_mid_roll_page():
    # A 2-box page followed by a full page: per-page arithmetic would expect 841 here.
    records = [serial_record(30, 1, "811"), serial_record(30, 2, "812")]
    records += [serial_record(31, 1, "813"), serial_record(31, 2, "814")]
    assert expected_serials(records) == [811, 812, 813, 814]
    assert expected_serials([serial_record(3, 1, "x"), serial_record(3, 2, "1")]) == [0, 1]

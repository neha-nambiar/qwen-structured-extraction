import io
import json
from datetime import datetime

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from eroll.extraction.records import (
    CSV_HEADER,
    ERROR_TEXT_LIMIT,
    FIELD_ATTRS,
    FIELD_KEYS,
    Record,
    apply_corrections,
    normalise_house_number,
    parse_record,
    read_csv,
    write_csv,
)

# Text weighted towards characters that break naive CSV handling (no lone surrogates,
# which UTF-8 cannot encode).
field_text = st.text(
    st.one_of(st.sampled_from(',"\r\n\x00 é\u0905'), st.characters(exclude_categories=["Cs"]))
)

records = st.builds(
    lambda page, box, fields, confidence, error: Record.model_validate(
        {
            "page": page,
            "box": box,
            **dict(zip(FIELD_KEYS, fields, strict=True)),
            "confidence": confidence,
            "error": error,
        }
    ),
    page=st.integers(min_value=1, max_value=2000),
    box=st.integers(min_value=1, max_value=30),
    fields=st.lists(field_text, min_size=len(FIELD_KEYS), max_size=len(FIELD_KEYS)),
    confidence=st.none() | st.floats(max_value=0.0, allow_nan=False),
    error=st.none() | field_text,
)


def compared(record: Record) -> dict:
    return record.model_dump(include={"page", "box", *FIELD_ATTRS})


# CSV round trip and format.
@given(st.lists(records, max_size=12))
def test_csv_round_trip_and_format(written):
    file = io.StringIO(newline="")
    write_csv(written, file)
    text = file.getvalue()

    # The header as the very first bytes also rules out a byte-order mark.
    assert text.encode("utf-8").startswith(",".join(CSV_HEADER).encode("utf-8") + b"\r\n")

    file.seek(0)
    read = read_csv(file, "roll.csv")
    expected = sorted(written, key=lambda r: (r.page, r.box))
    assert [compared(r) for r in read] == [compared(r) for r in expected]


def test_read_csv_wrong_header_names_file():
    file = io.StringIO("page,box,name\r\n1,1,x\r\n", newline="")
    with pytest.raises(ValueError, match="roll-7.csv"):
        read_csv(file, "roll-7.csv")


# Any JSON value, with object keys biased towards the real field keys so that parsed
# objects often carry fields of the wrong type (numbers, nulls, nested objects).
json_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | field_text,
    lambda children: (
        st.lists(children, max_size=4)
        | st.dictionaries(st.sampled_from(FIELD_KEYS) | field_text, children, max_size=8)
    ),
    max_leaves=20,
)

# Model-like replies: prose, JSON and stray braces glued together, often broken.
model_responses = st.lists(
    st.one_of(
        field_text,
        json_values.map(json.dumps),
        st.sampled_from(["{", "}", "[", "]", ":", ",", '"', "null", "```json\n", "\n```"]),
    ),
    max_size=6,
).map("".join)


def decodes_to_object(text: str) -> bool:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        return False
    try:
        return isinstance(json.loads(text[start : end + 1]), dict)
    except (ValueError, RecursionError):
        return False


# Parsing never raises, and sets error exactly on responses without a JSON object.
@given(st.one_of(field_text, model_responses))
@example("x" * (ERROR_TEXT_LIMIT + 1))
# json.loads raises ValueError, not JSONDecodeError, past the int-to-str digit limit.
@example('{"age": ' + "1" * 5000 + "}")
# Deep nesting makes json.loads raise RecursionError.
@example('{"name": ' + "[" * 100_000 + "]" * 100_000 + "}")
def test_parsing_is_total(text):
    record = parse_record(text, 3, 7)

    assert (record.page, record.box) == (3, 7)
    assert (record.error is None) == decodes_to_object(text)
    if record.error is not None:
        assert record.error == text[:ERROR_TEXT_LIMIT]
        assert len(record.error) <= ERROR_TEXT_LIMIT
        fields = record.model_dump(by_alias=True, include=FIELD_ATTRS)
        assert set(fields.values()) == {""}


# Wrappers a model puts around its JSON; none contains a brace.
WRAPPERS = [
    ("", ""),
    ("Here is the record:\n", ""),
    ("```json\n", "\n```"),
    ("Sure.\n```json\n", "\n```\nLet me know if anything is unclear."),
]

# A house number is sometimes dash-only, so normalisation is exercised, not just assumed.
house_numbers = field_text | st.sampled_from(["-", " \u2013 ", "\u2014-", "29a", ""])


# JSON round trip through any wrapper and key order.
@given(
    fields=st.fixed_dictionaries(
        {key: house_numbers if key == "house number" else field_text for key in FIELD_KEYS}
    ),
    key_order=st.permutations(FIELD_KEYS),
    ensure_ascii=st.booleans(),
    wrapper=st.sampled_from(WRAPPERS),
)
@example(
    fields={
        **dict.fromkeys(FIELD_KEYS, '{"x": "}"}\n\u0905\u0928\u093f\u0932 é'),
        "house number": "\u2013",
    },
    key_order=FIELD_KEYS,
    ensure_ascii=False,
    wrapper=WRAPPERS[2],
)
def test_json_round_trip(fields, key_order, ensure_ascii, wrapper):
    obj = {key: fields[key] for key in key_order}
    before, after = wrapper
    text = before + json.dumps(obj, ensure_ascii=ensure_ascii) + after

    record = parse_record(text, 3, 7)

    assert record.error is None
    expected = {**fields, "house number": normalise_house_number(fields["house number"])}
    assert record.model_dump(by_alias=True, include=FIELD_ATTRS) == expected


# Three timestamps, so that one field corrected several times often has ties.
correction_times = st.sampled_from(
    [datetime(2026, 1, 1, 9, 0), datetime(2026, 1, 1, 9, 5), datetime(2026, 1, 2, 8, 0)]
)
corrections = st.lists(
    st.tuples(st.sampled_from(FIELD_KEYS), field_text, correction_times), max_size=12
)


# The latest correction wins, and the stored record is untouched.
@given(records, corrections)
def test_latest_correction_wins(record, applied):
    before = record.model_copy(deep=True)

    corrected = apply_corrections(record, applied)

    # Latest timestamp per field; on a tie the later correction in input order wins.
    latest = {}
    for field, value, timestamp in applied:
        if field not in latest or timestamp >= latest[field][0]:
            latest[field] = (timestamp, value)
    expected = {
        **record.model_dump(by_alias=True),
        **{field: value for field, (_, value) in latest.items()},
    }
    assert corrected.model_dump(by_alias=True) == expected
    # The stored record, which the raw export reads, is left as it was.
    assert record == before
    if not applied:
        assert corrected is record


def test_apply_corrections_unknown_field_names_it():
    record = parse_record("{}", 1, 1)
    # The attribute name, not the field key, is still an unknown field.
    with pytest.raises(ValueError, match="'epic_id'"):
        apply_corrections(record, [("epic_id", "ABC1234567", datetime(2026, 1, 1))])

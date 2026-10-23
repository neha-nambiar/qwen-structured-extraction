"""Format checks that flag suspicious records for review, and the serial check across pages."""

from itertools import pairwise

from eroll.extraction.records import Record

# Every reason validate() can return; with LOW_CONFIDENCE these are the metric labels.
VALIDATOR_NAMES = (
    "serial",
    "epic_format",
    "epic_lowercase",
    "age_range",
    "gender_vocab",
    "relation_vocab",
    "parse_error",
)
LOW_CONFIDENCE = "low_confidence"

EPIC_LENGTH = 10
EPIC_LETTERS = 3
MIN_AGE = 18
MAX_AGE = 120
GENDER_VOCAB = ("male", "female")
RELATION_VOCAB = ("husband", "father", "mother", "others")


def is_whole_number(text: str) -> bool:
    # isdigit alone accepts "²", which int() rejects; ASCII-only is safe for int().
    return text.isascii() and text.isdigit()


def expected_serials(records: list[Record]) -> list[int]:
    """Serial each record should carry, counting on from the job's first record."""
    # Counting along records, not page and box, handles partial pages. An unreadable
    # first serial gives base 0, so the whole job is flagged rather than guessed.
    if not records:
        return []
    first = records[0].serial_number.strip()
    base = int(first) if is_whole_number(first) else 0
    return [base + index for index in range(len(records))]


def validate(record: Record, expected_serial: int) -> list[str]:
    """Every check the record fails; empty means not suspicious, not necessarily correct."""
    failures = []

    # Exact string comparison: "0571" or " 571" is not what the roll prints.
    if record.serial_number != str(expected_serial):
        failures.append("serial")

    epic = record.epic_id
    letters, digits = epic[:EPIC_LETTERS], epic[EPIC_LETTERS:]
    if (
        len(epic) != EPIC_LENGTH
        or not (letters.isascii() and letters.isalpha())
        or not is_whole_number(digits)
    ):
        failures.append("epic_format")
    elif not letters.isupper():
        # Valid format, but a lower-cased EPIC tends to come with other errors.
        failures.append("epic_lowercase")

    age = record.age
    if not is_whole_number(age) or not (MIN_AGE <= int(age) <= MAX_AGE):
        failures.append("age_range")

    if record.gender.lower() not in GENDER_VOCAB:
        failures.append("gender_vocab")

    if record.relation_type.lower() not in RELATION_VOCAB:
        failures.append("relation_vocab")

    # parse_record stores the raw text of an unparsable response here.
    if record.error:
        failures.append("parse_error")

    return failures


def flag_reasons(
    record: Record, expected_serial: int, confidence_threshold: float | None
) -> list[str]:
    """validate()'s reasons, plus LOW_CONFIDENCE below a set threshold."""
    reasons = validate(record, expected_serial)
    if (
        confidence_threshold is not None
        and record.confidence is not None
        and record.confidence < confidence_threshold
    ):
        reasons.append(LOW_CONFIDENCE)
    return reasons


def serial_gaps(pages: list[tuple[int, list[Record]]]) -> list[tuple[int, int]]:
    """Adjacent voter pages (earlier, later) whose serials do not continue by 1."""
    for page, records in pages:
        if not records:
            raise ValueError(f"voter page {page} has no Records to check serials against")

    gaps = []
    for (earlier, earlier_records), (later, later_records) in pairwise(pages):
        # Not stripped: " 571" is not what the roll prints, so it is not a whole number.
        last = earlier_records[-1].serial_number
        first = later_records[0].serial_number
        if not (is_whole_number(last) and is_whole_number(first)) or int(first) != int(last) + 1:
            gaps.append((earlier, later))
    return gaps

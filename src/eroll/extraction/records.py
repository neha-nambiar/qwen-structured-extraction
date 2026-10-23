"""Record model, extraction prompt and schema, JSON recovery and the CSV format."""

import csv
import json
from collections.abc import Iterable
from datetime import datetime
from typing import TextIO

from pydantic import BaseModel, ConfigDict, Field, ValidationError

# The eight extracted keys, verbatim in lower case with spaces, in a fixed order.
FIELD_KEYS = (
    "serial number",
    "epic id",
    "name",
    "relation type",
    "relation name",
    "house number",
    "age",
    "gender",
)

CSV_HEADER = ("page", "box", *FIELD_KEYS)

# Raw model text kept on an unparsable response, truncated to this many characters.
ERROR_TEXT_LIMIT = 2000

# Sent with every crop. Without the capitalisation rule the model lower-cases printed
# capitals; without the relation-type rule that field is often empty.
PROMPT = (
    "You are reading one voter's details from a cropped electoral roll box.\n"
    "Return a single JSON object and nothing else: no prose, no explanation, and "
    "no markdown code fence.\n"
    "The object must have exactly these eight keys, spelled verbatim in lower case "
    "with spaces, and no other keys:\n"
    '  "serial number", "epic id", "name", "relation type", "relation name", '
    '"house number", "age", "gender".\n'
    'Every value must be a string. Use an empty string "" for any field that is '
    "not readable in the crop.\n"
    "Copy each value exactly as printed, preserving capitalisation character for "
    "character. Do not change the case of anything. If a value is printed in "
    "capitals, such as ABHISHEK KARDILE or NIE7470107, return it in capitals.\n"
    '"relation type" is the relationship named by the label on the second name '
    'line, not a name: a line reading "Husband\'s Name" means "Husband", '
    '"Father\'s Name" means "Father", "Mother\'s Name" means "Mother", and '
    'anything else means "Others". Put the name printed on that line in '
    '"relation name".\n'
    '"gender" is usually one of Male, Female. If the crop shows something else, '
    "report exactly what you read."
)


# JSON schema for constrained decoding. Names and house numbers stay free text.
RECORD_SCHEMA = {
    "type": "object",
    "properties": {
        "serial number": {"type": "string", "pattern": r"^([0-9]{1,5})?$"},
        "epic id": {"type": "string", "pattern": r"^([A-Z]{3}[0-9]{7})?$"},
        "name": {"type": "string"},
        "relation type": {"enum": ["Husband", "Father", "Mother", "Others", ""]},
        "relation name": {"type": "string"},
        "house number": {"type": "string"},
        "age": {"type": "string", "pattern": r"^([0-9]{1,3})?$"},
        "gender": {"enum": ["Male", "Female", ""]},
    },
    "required": list(FIELD_KEYS),
    "additionalProperties": False,
}


class Record(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        populate_by_name=True,
        hide_input_in_errors=True,
        frozen=True,
    )

    page: int = Field(ge=1)
    box: int = Field(ge=1)
    serial_number: str = Field(alias="serial number")
    epic_id: str = Field(alias="epic id")
    name: str
    relation_type: str = Field(alias="relation type")
    relation_name: str = Field(alias="relation name")
    house_number: str = Field(alias="house number")
    age: str
    gender: str
    # Mean per-token log probability of the completion, so never above 0.
    confidence: float | None = Field(default=None, le=0.0)
    error: str | None = None


# Attribute names of the eight field keys.
FIELD_ATTRS = frozenset(
    {
        "serial_number",
        "epic_id",
        "name",
        "relation_type",
        "relation_name",
        "house_number",
        "age",
        "gender",
    }
)


# Characters a dash-only house number may be rendered with. A vision model may emit an
# en or em dash where the page prints a hyphen.
DASH_CHARACTERS = "-\u2013\u2014"


def normalise_house_number(value: str) -> str:
    """Collapse a dash-only house number to an empty string."""
    # "-" and empty mean the same, and at 115 DPI a thin dash is hard to tell from nothing.
    stripped = value.strip()
    if stripped and set(stripped) <= set(DASH_CHARACTERS):
        return ""
    return value


def parse_record(text: str, page: int, box: int) -> Record:
    """Turn one model response into a Record; unparsable text goes to `error`."""
    # First '{' to last '}' tolerates code fences and surrounding prose.
    parsed = None
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
        # Huge integers raise plain ValueError and deep nesting RecursionError.
        except (ValueError, RecursionError):
            parsed = None

    if not isinstance(parsed, dict):
        fields = {key: "" for key in FIELD_KEYS}
        return Record.model_validate(
            {"page": page, "box": box, **fields, "error": text[:ERROR_TEXT_LIMIT]}
        )

    # Missing keys and nulls become "", other values are kept for the validators to flag.
    fields = {}
    for key in FIELD_KEYS:
        value = parsed.get(key)
        if value is None:
            fields[key] = ""
        elif isinstance(value, str):
            fields[key] = value
        else:
            fields[key] = str(value)

    fields["house number"] = normalise_house_number(fields["house number"])
    return Record.model_validate({"page": page, "box": box, **fields})


def write_csv(records: Iterable[Record], file: TextIO) -> None:
    """Write records as CSV (page, box, 8 fields), sorted by (page, box)."""
    writer = csv.writer(file)
    writer.writerow(CSV_HEADER)
    for record in sorted(records, key=lambda r: (r.page, r.box)):
        row = record.model_dump(by_alias=True)
        writer.writerow([row[column] for column in CSV_HEADER])


def read_csv(file: TextIO, name: str) -> list[Record]:
    """Read a CSV written by write_csv; errors name the row's page and box, not values."""
    reader = csv.reader(file)
    header = next(reader, None)
    if header != list(CSV_HEADER):
        raise ValueError(f"{name}: header is {header}, expected {list(CSV_HEADER)}")

    records = []
    for row in reader:
        page_text = row[0] if row else ""
        box_text = row[1] if len(row) > 1 else ""
        where = f"{name}: row with page {page_text!r}, box {box_text!r}"
        if len(row) != len(CSV_HEADER):
            raise ValueError(f"{where}: has {len(row)} columns, expected {len(CSV_HEADER)}")
        try:
            page = int(page_text)
            box = int(box_text)
        except ValueError:
            raise ValueError(f"{where}: page and box must be integers") from None
        try:
            record = Record.model_validate(
                {"page": page, "box": box, **dict(zip(FIELD_KEYS, row[2:], strict=True))}
            )
        except ValidationError as error:
            raise ValueError(f"{where}: {error}") from error
        records.append(record)
    return records


def apply_corrections(record: Record, corrections: Iterable[tuple[str, str, datetime]]) -> Record:
    """Record with each field set to its latest (field, value, timestamp) correction."""
    latest = {}
    for field, value, _ in sorted(corrections, key=lambda correction: correction[2]):
        if field not in FIELD_KEYS:
            raise ValueError(f"correction names unknown field {field!r}")
        latest[field] = value
    if not latest:
        return record
    return Record.model_validate({**record.model_dump(by_alias=True), **latest})

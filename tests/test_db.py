from eroll.extraction.records import FIELD_KEYS, Record
from eroll.service.db import CorrectionRow, record_row, to_record


def test_record_row_round_trip():
    record = Record.model_validate(
        {"page": 3, "box": 7, **{key: f"v-{key}" for key in FIELD_KEYS}, "confidence": -0.5}
    )
    row = record_row("a" * 64, record)
    assert row.job_id == "a" * 64
    assert to_record(row) == record


def test_correction_field_check_lists_every_field_key():
    (check,) = [c for c in CorrectionRow.__table__.constraints if c.name == "corrections_field"]
    for key in FIELD_KEYS:
        assert repr(key) in str(check.sqltext)

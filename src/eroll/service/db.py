"""ORM models for the schema, plus writes shared by both stages (only mark_failed commits)."""

import logging
from datetime import datetime
from pathlib import Path

from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Engine,
    Float,
    ForeignKey,
    Index,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from eroll.extraction.records import FIELD_KEYS, Record

log = logging.getLogger(__name__)

JOB_STATUSES = ("queued", "running", "completed", "failed")


def _in_list(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(value) for value in values)})"


class Base(DeclarativeBase):
    pass


class JobRow(Base):
    __tablename__ = "jobs"
    __table_args__ = (CheckConstraint(_in_list("status", JOB_STATUSES), name="jobs_status"),)

    # Hex SHA-256 of the PDF: identical uploads are the same job.
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(Text)
    page_count: Mapped[int | None]
    problems: Mapped[list] = mapped_column(JSONB, default=list, server_default="[]")
    submitted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    gpu_seconds: Mapped[float] = mapped_column(Float, default=0.0, server_default="0")


class JobPdfRow(Base):
    __tablename__ = "job_pdfs"

    job_id: Mapped[str] = mapped_column(String(64), ForeignKey("jobs.id"), primary_key=True)
    data: Mapped[bytes] = mapped_column(LargeBinary)


class PageRow(Base):
    __tablename__ = "pages"

    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), primary_key=True)
    page: Mapped[int] = mapped_column(primary_key=True)
    image_data: Mapped[bytes] = mapped_column(LargeBinary, deferred=True)
    width: Mapped[int]
    height: Mapped[int]
    # Voter boxes in reading order as [x, y, w, h].
    boxes: Mapped[list] = mapped_column(JSONB)
    geometry_label: Mapped[str] = mapped_column(Text)
    p_voter: Mapped[float] = mapped_column(Float)
    p_non_voter: Mapped[float] = mapped_column(Float)
    margin: Mapped[float] = mapped_column(Float)
    outcome: Mapped[str] = mapped_column(Text)
    final_label: Mapped[str | None] = mapped_column(Text)
    grid_problems: Mapped[list] = mapped_column(JSONB)


class RecordRow(Base):
    __tablename__ = "records"
    # One record per voter box, which makes a resumed job idempotent.
    __table_args__ = (UniqueConstraint("job_id", "page", "box", name="records_job_page_box"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"))
    page: Mapped[int]
    box: Mapped[int]
    serial_number: Mapped[str] = mapped_column(Text)
    epic_id: Mapped[str] = mapped_column(Text)
    name: Mapped[str] = mapped_column(Text)
    relation_type: Mapped[str] = mapped_column(Text)
    relation_name: Mapped[str] = mapped_column(Text)
    house_number: Mapped[str] = mapped_column(Text)
    age: Mapped[str] = mapped_column(Text)
    gender: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(Text)
    flags: Mapped[list] = mapped_column(JSONB, default=list, server_default="[]")
    reviewed: Mapped[bool] = mapped_column(default=False, server_default="false")


class CorrectionRow(Base):
    __tablename__ = "corrections"
    __table_args__ = (
        CheckConstraint(_in_list("field", FIELD_KEYS), name="corrections_field"),
        Index("corrections_record_created", "record_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    record_id: Mapped[int] = mapped_column(ForeignKey("records.id"))
    # A field key as written in the CSV header, e.g. "epic id".
    field: Mapped[str] = mapped_column(Text)
    value: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


def check_head(engine: Engine, alembic_ini: Path) -> None:
    """Exit unless the database is at the Alembic head revision."""
    expected = ScriptDirectory.from_config(Config(str(alembic_ini))).get_current_head()
    with engine.connect() as connection:
        current = MigrationContext.configure(connection).get_current_revision()
    if current != expected:
        raise SystemExit(
            f"database is at revision {current}, expected head {expected}; "
            "run `alembic upgrade head`"
        )
    log.info("database at head revision %s", current)


def record_row(job_id: str, record: Record) -> RecordRow:
    return RecordRow(job_id=job_id, **record.model_dump())


def to_record(row: RecordRow) -> Record:
    return Record(
        page=row.page,
        box=row.box,
        serial_number=row.serial_number,
        epic_id=row.epic_id,
        name=row.name,
        relation_type=row.relation_type,
        relation_name=row.relation_name,
        house_number=row.house_number,
        age=row.age,
        gender=row.gender,
        confidence=row.confidence,
        error=row.error,
    )


def add_problem(session: Session, job_id: str, problem: dict) -> None:
    """Append a problem to the job. The caller commits."""
    job = session.get(JobRow, job_id)
    if job is None:
        raise ValueError(f"no job {job_id}")
    job.problems = [*job.problems, problem]


def mark_failed(session: Session, job_id: str, stage: str, page: int | None, message: str) -> None:
    """Roll back, mark the job failed with an error problem, and commit."""
    # Records from earlier pages are kept so a retry can resume.
    session.rollback()
    problem = {
        "kind": "error",
        "stage": stage,
        "pages": [] if page is None else [page],
        "message": message,
    }
    add_problem(session, job_id, problem)
    session.get(JobRow, job_id).status = "failed"
    session.commit()
    log.info("job %s failed in %s at page %s", job_id, stage, page)


def latest_corrections(
    session: Session, record_ids: list[int]
) -> dict[int, list[tuple[str, str, datetime]]]:
    """Every correction for the given records, oldest first, keyed by record id."""
    # id after created_at keeps insertion order for equal timestamps.
    if not record_ids:
        return {}
    rows = session.execute(
        select(
            CorrectionRow.record_id,
            CorrectionRow.field,
            CorrectionRow.value,
            CorrectionRow.created_at,
        )
        .where(CorrectionRow.record_id.in_(record_ids))
        .order_by(CorrectionRow.record_id, CorrectionRow.created_at, CorrectionRow.id)
    )
    corrections: dict[int, list[tuple[str, str, datetime]]] = {}
    for record_id, field, value, created_at in rows:
        corrections.setdefault(record_id, []).append((field, value, created_at))
    return corrections

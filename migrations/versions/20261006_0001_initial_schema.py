"""Initial schema: jobs, job_pdfs, pages, records, corrections."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column(
            "problems",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="[]",
            nullable=False,
        ),
        sa.Column(
            "submitted_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("gpu_seconds", sa.Float(), server_default="0", nullable=False),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'completed', 'failed')", name="jobs_status"
        ),
    )
    op.create_table(
        "job_pdfs",
        sa.Column("job_id", sa.String(64), sa.ForeignKey("jobs.id"), primary_key=True),
        sa.Column("data", sa.LargeBinary(), nullable=False),
    )
    op.create_table(
        "pages",
        sa.Column("job_id", sa.String(64), sa.ForeignKey("jobs.id"), primary_key=True),
        sa.Column("page", sa.Integer(), primary_key=True),
        sa.Column("image_data", sa.LargeBinary(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("boxes", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("geometry_label", sa.Text(), nullable=False),
        sa.Column("p_voter", sa.Float(), nullable=False),
        sa.Column("p_non_voter", sa.Float(), nullable=False),
        sa.Column("margin", sa.Float(), nullable=False),
        sa.Column("outcome", sa.Text(), nullable=False),
        sa.Column("final_label", sa.Text(), nullable=True),
        sa.Column("grid_problems", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    )
    op.create_table(
        "records",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("job_id", sa.String(64), sa.ForeignKey("jobs.id"), nullable=False),
        sa.Column("page", sa.Integer(), nullable=False),
        sa.Column("box", sa.Integer(), nullable=False),
        sa.Column("serial_number", sa.Text(), nullable=False),
        sa.Column("epic_id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("relation_type", sa.Text(), nullable=False),
        sa.Column("relation_name", sa.Text(), nullable=False),
        sa.Column("house_number", sa.Text(), nullable=False),
        sa.Column("age", sa.Text(), nullable=False),
        sa.Column("gender", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "flags",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="[]",
            nullable=False,
        ),
        sa.Column("reviewed", sa.Boolean(), server_default="false", nullable=False),
        sa.UniqueConstraint("job_id", "page", "box", name="records_job_page_box"),
    )
    op.create_table(
        "corrections",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("record_id", sa.Integer(), sa.ForeignKey("records.id"), nullable=False),
        sa.Column("field", sa.Text(), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "field IN ('serial number', 'epic id', 'name', 'relation type', "
            "'relation name', 'house number', 'age', 'gender')",
            name="corrections_field",
        ),
    )
    op.create_index("corrections_record_created", "corrections", ["record_id", "created_at"])


def downgrade() -> None:
    op.drop_index("corrections_record_created", table_name="corrections")
    op.drop_table("corrections")
    op.drop_table("records")
    op.drop_table("pages")
    op.drop_table("job_pdfs")
    op.drop_table("jobs")

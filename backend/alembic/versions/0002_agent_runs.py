from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("metrics", postgresql.JSONB(), nullable=True),
        sa.Column("trace", postgresql.JSONB(), nullable=True),
        sa.Column("submission_csv", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_agent_runs")),
    )
    op.create_index("ix_agent_runs_started_at", "agent_runs", ["started_at"])
    for table in ("pilot_results", "campaign_results"):
        op.create_table(
            table,
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("run_id", sa.Uuid(), nullable=False),
            sa.Column("sequence", sa.Integer(), nullable=False),
            sa.Column("payload", postgresql.JSONB(), nullable=False),
            sa.ForeignKeyConstraint(
                ["run_id"], ["agent_runs.id"],
                name=op.f(f"fk_{table}_run_id_agent_runs"), ondelete="CASCADE",
            ),
            sa.PrimaryKeyConstraint("id", name=op.f(f"pk_{table}")),
            sa.UniqueConstraint("run_id", "sequence", name=op.f(f"uq_{table}_run_id")),
        )
        op.create_index(f"ix_{table}_run_id", table, ["run_id"])


def downgrade() -> None:
    op.drop_table("campaign_results")
    op.drop_table("pilot_results")
    op.drop_table("agent_runs")

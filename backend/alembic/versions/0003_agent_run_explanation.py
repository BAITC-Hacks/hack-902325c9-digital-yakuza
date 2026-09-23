from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("agent_runs", sa.Column("explanation", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("agent_runs", "explanation")

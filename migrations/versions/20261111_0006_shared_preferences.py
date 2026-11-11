"""Add explicit shared preference choices."""

import sqlalchemy as sa
from alembic import op

revision = "20261111_0006"
down_revision = "20260826_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "preference_feedback",
        sa.Column("event_id", sa.String(36), primary_key=True),
        sa.Column("participant_hash", sa.String(64), nullable=False),
        sa.Column("request_digest", sa.String(64), nullable=False),
        sa.Column("model_version", sa.String(24), nullable=False),
        sa.Column("profile_version", sa.Integer(), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_preference_feedback_participant_hash",
        "preference_feedback",
        ["participant_hash"],
    )
    op.create_index(
        "ix_preference_feedback_created_at", "preference_feedback", ["created_at"]
    )


def downgrade() -> None:
    op.drop_table("preference_feedback")

"""Add dead_letter_events table for failed Kafka messages.

Revision ID: 002
Revises: 001
Create Date: 2026-06-22
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "dead_letter_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("worker_name", sa.Text(), nullable=False),
        sa.Column("source_topic", sa.Text(), nullable=False),
        sa.Column("kafka_partition", sa.Integer(), nullable=False),
        sa.Column("kafka_offset", sa.BigInteger(), nullable=False),
        sa.Column("event_id", sa.Text(), nullable=True),
        sa.Column("error_type", sa.Text(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=False),
        sa.Column("raw_payload", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_dead_letter_events_source_topic_created_at",
        "dead_letter_events",
        ["source_topic", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_dead_letter_events_error_type",
        "dead_letter_events",
        ["error_type"],
        unique=False,
    )
    op.create_index(
        "ix_dead_letter_events_worker_name",
        "dead_letter_events",
        ["worker_name"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_dead_letter_events_worker_name", table_name="dead_letter_events")
    op.drop_index("ix_dead_letter_events_error_type", table_name="dead_letter_events")
    op.drop_index(
        "ix_dead_letter_events_source_topic_created_at",
        table_name="dead_letter_events",
    )
    op.drop_table("dead_letter_events")

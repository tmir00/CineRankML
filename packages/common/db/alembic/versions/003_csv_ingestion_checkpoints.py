"""Add csv_ingestion_checkpoints table for Postgres-backed producer resume.

Revision ID: 003
Revises: 002
Create Date: 2026-06-22
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "csv_ingestion_checkpoints",
        sa.Column("source_file", sa.Text(), nullable=False),
        sa.Column("last_row_number", sa.Integer(), nullable=False),
        sa.Column("last_event_id", sa.Text(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("source_file"),
    )


def downgrade() -> None:
    op.drop_table("csv_ingestion_checkpoints")

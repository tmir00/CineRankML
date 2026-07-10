"""Add event_type to ratings_events and allow null ratings for deletes.

Revision ID: 010
Revises: 009
Create Date: 2026-07-05
"""

import sqlalchemy as sa

from alembic import op
from typing import Sequence, Union

revision: str = "010"
down_revision: Union[str, None] = "009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "ratings_events",
        sa.Column(
            "event_type",
            sa.Text(),
            nullable=False,
            server_default="rating_created",
        ),
    )
    op.alter_column("ratings_events", "rating", existing_type=sa.Float(), nullable=True)


def downgrade() -> None:
    op.alter_column("ratings_events", "rating", existing_type=sa.Float(), nullable=False)
    op.drop_column("ratings_events", "event_type")

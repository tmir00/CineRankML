"""Add snapshot export order index on ratings_events.

Revision ID: 006
Revises: 005
Create Date: 2026-06-29
"""

from typing import Sequence, Union

from alembic import op

revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_ratings_events_rating_timestamp_id",
        "ratings_events",
        ["rating_timestamp", "id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_ratings_events_rating_timestamp_id",
        table_name="ratings_events",
    )

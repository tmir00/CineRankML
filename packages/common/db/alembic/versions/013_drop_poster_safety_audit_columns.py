"""Drop poster safety audit columns from catalog_movies.

Revision ID: 013
Revises: 012
Create Date: 2026-07-11
"""

import sqlalchemy as sa

from alembic import op
from typing import Sequence, Union

revision: str = "013"
down_revision: Union[str, None] = "012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column("catalog_movies", "poster_safety_reason")
    op.drop_column("catalog_movies", "poster_safety_score")
    op.drop_column("catalog_movies", "poster_safety_provider")


def downgrade() -> None:
    op.add_column(
        "catalog_movies",
        sa.Column("poster_safety_provider", sa.Text(), nullable=True),
    )
    op.add_column(
        "catalog_movies",
        sa.Column("poster_safety_score", sa.Float(), nullable=True),
    )
    op.add_column(
        "catalog_movies",
        sa.Column("poster_safety_reason", sa.Text(), nullable=True),
    )

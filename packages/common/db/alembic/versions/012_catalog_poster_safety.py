"""Add poster safety and certification columns to catalog_movies.

Revision ID: 012
Revises: 011
Create Date: 2026-07-08
"""

import sqlalchemy as sa

from alembic import op
from typing import Sequence, Union

revision: str = "012"
down_revision: Union[str, None] = "011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "catalog_movies",
        sa.Column("adult", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "catalog_movies",
        sa.Column("certification_us", sa.Text(), nullable=True),
    )
    op.add_column(
        "catalog_movies",
        sa.Column("poster_checked", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "catalog_movies",
        sa.Column("poster_safe", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
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
    op.add_column(
        "catalog_movies",
        sa.Column("poster_checked_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("catalog_movies", "poster_checked_at")
    op.drop_column("catalog_movies", "poster_safety_reason")
    op.drop_column("catalog_movies", "poster_safety_score")
    op.drop_column("catalog_movies", "poster_safety_provider")
    op.drop_column("catalog_movies", "poster_safe")
    op.drop_column("catalog_movies", "poster_checked")
    op.drop_column("catalog_movies", "certification_us")
    op.drop_column("catalog_movies", "adult")

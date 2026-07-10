"""Add index for catalog enrichment pending queries.

Revision ID: 004
Revises: 003
Create Date: 2026-06-23
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_catalog_movies_enrichment_pending",
        "catalog_movies",
        ["movie_id"],
        unique=False,
        postgresql_where=sa.text(
            "tmdb_id IS NOT NULL AND (enrichment_status IS NULL OR enrichment_status = 'failed')"
        ),
    )
    op.create_index(
        "ix_catalog_movies_enrichment_status",
        "catalog_movies",
        ["enrichment_status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_catalog_movies_enrichment_status", table_name="catalog_movies")
    op.drop_index("ix_catalog_movies_enrichment_pending", table_name="catalog_movies")

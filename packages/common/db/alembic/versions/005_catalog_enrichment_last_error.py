"""Add enrichment_last_error to catalog_movies.

Revision ID: 005
Revises: 004
Create Date: 2026-06-24
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "catalog_movies",
        sa.Column("enrichment_last_error", sa.Text(), nullable=True),
    )
    op.create_index(
        "ix_catalog_movies_enrichment_last_error_failed",
        "catalog_movies",
        ["updated_at"],
        unique=False,
        postgresql_where=sa.text("enrichment_status = 'failed'"),
    )


def downgrade() -> None:
    op.drop_index(
        "ix_catalog_movies_enrichment_last_error_failed",
        table_name="catalog_movies",
    )
    op.drop_column("catalog_movies", "enrichment_last_error")

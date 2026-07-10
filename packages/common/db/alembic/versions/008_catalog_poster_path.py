"""Add poster_path column to catalog_movies.

Revision ID: 008
Revises: 007
Create Date: 2026-07-03
"""

import sqlalchemy as sa

from alembic import op
from typing import Sequence, Union

revision: str = "008"
down_revision: Union[str, None] = "007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("catalog_movies", sa.Column("poster_path", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("catalog_movies", "poster_path")

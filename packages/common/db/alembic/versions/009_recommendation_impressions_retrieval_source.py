"""Add retrieval_source column to recommendation_impressions.

Revision ID: 009
Revises: 008
Create Date: 2026-07-05
"""

import sqlalchemy as sa

from alembic import op
from typing import Sequence, Union

revision: str = "009"
down_revision: Union[str, None] = "008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "recommendation_impressions",
        sa.Column(
            "retrieval_source",
            sa.Text(),
            nullable=False,
            server_default="unknown",
        ),
    )


def downgrade() -> None:
    op.drop_column("recommendation_impressions", "retrieval_source")

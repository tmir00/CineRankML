"""Add recommendation_experiments table for online A/B split state.

Revision ID: 011
Revises: 010
Create Date: 2026-07-06
"""

import sqlalchemy as sa

from alembic import op
from typing import Sequence, Union

revision: str = "011"
down_revision: Union[str, None] = "010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "recommendation_experiments",
        sa.Column("experiment_id", sa.Text(), nullable=False),
        sa.Column("main_model_version", sa.Text(), nullable=False),
        sa.Column("candidate_model_version", sa.Text(), nullable=True),
        sa.Column(
            "main_split_fraction",
            sa.Float(),
            nullable=False,
            server_default="0.70",
        ),
        sa.Column(
            "candidate_split_fraction",
            sa.Float(),
            nullable=False,
            server_default="0.30",
        ),
        sa.Column(
            "status",
            sa.Text(),
            nullable=False,
            server_default="active",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("experiment_id"),
    )
    op.create_index(
        "ix_recommendation_experiments_status",
        "recommendation_experiments",
        ["status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_recommendation_experiments_status", table_name="recommendation_experiments")
    op.drop_table("recommendation_experiments")

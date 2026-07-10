"""Initial database schema for Phase 1.

Revision ID: 001
Revises:
Create Date: 2026-06-22
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "catalog_movies",
        sa.Column("movie_id", sa.Integer(), nullable=False),
        sa.Column("tmdb_id", sa.Integer(), nullable=True),
        sa.Column("imdb_id", sa.Text(), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("genres", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("original_language", sa.Text(), nullable=True),
        sa.Column("overview", sa.Text(), nullable=True),
        sa.Column("tagline", sa.Text(), nullable=True),
        sa.Column("tmdb_keywords", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("runtime", sa.Integer(), nullable=True),
        sa.Column("tmdb_popularity", sa.Float(), nullable=True),
        sa.Column("tmdb_vote_average", sa.Float(), nullable=True),
        sa.Column("tmdb_vote_count", sa.Integer(), nullable=True),
        sa.Column("enrichment_status", sa.Text(), nullable=True),
        sa.Column("enriched_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source", sa.Text(), nullable=True),
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
        sa.PrimaryKeyConstraint("movie_id"),
    )

    op.create_table(
        "ratings_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("event_id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("movie_id", sa.Integer(), nullable=False),
        sa.Column("rating", sa.Float(), nullable=False),
        sa.Column("rating_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("stream_pipeline_version", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("source", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("event_id"),
    )
    op.create_index(
        "ix_ratings_events_user_id_rating_timestamp",
        "ratings_events",
        ["user_id", "rating_timestamp"],
        unique=False,
    )
    op.create_index("ix_ratings_events_movie_id", "ratings_events", ["movie_id"], unique=False)

    op.create_table(
        "tag_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("event_id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("movie_id", sa.Integer(), nullable=False),
        sa.Column("tag", sa.Text(), nullable=False),
        sa.Column("tag_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("stream_pipeline_version", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("source", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("event_id"),
    )
    op.create_index("ix_tag_events_movie_id", "tag_events", ["movie_id"], unique=False)
    op.create_index("ix_tag_events_event_id", "tag_events", ["event_id"], unique=False)

    op.create_table(
        "movie_tag_counts",
        sa.Column("movie_id", sa.Integer(), nullable=False),
        sa.Column("tag", sa.Text(), nullable=False),
        sa.Column("count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("movie_id", "tag"),
    )
    op.create_index("ix_movie_tag_counts_movie_id", "movie_tag_counts", ["movie_id"], unique=False)

    op.create_table(
        "catalog_dirty_movies",
        sa.Column("movie_id", sa.Integer(), nullable=False),
        sa.Column("first_dirty_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_dirty_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("attempt_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["movie_id"], ["catalog_movies.movie_id"]),
        sa.PrimaryKeyConstraint("movie_id"),
    )
    op.create_index(
        "ix_catalog_dirty_movies_last_dirty_at",
        "catalog_dirty_movies",
        ["last_dirty_at"],
        unique=False,
    )

    op.create_table(
        "embedding_versions",
        sa.Column("embedding_version", sa.Text(), nullable=False),
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column("dimension", sa.Integer(), nullable=False),
        sa.Column("text_template_version", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("embedding_version"),
    )

    op.create_table(
        "movie_content_embeddings",
        sa.Column("movie_id", sa.Integer(), nullable=False),
        sa.Column("embedding_version", sa.Text(), nullable=False),
        sa.Column("embedding", postgresql.ARRAY(sa.Float()), nullable=False),
        sa.Column("embedding_text_hash", sa.Text(), nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["embedding_version"], ["embedding_versions.embedding_version"]),
        sa.ForeignKeyConstraint(["movie_id"], ["catalog_movies.movie_id"]),
        sa.PrimaryKeyConstraint("movie_id"),
    )
    op.create_index(
        "ix_movie_content_embeddings_embedding_version",
        "movie_content_embeddings",
        ["embedding_version"],
        unique=False,
    )

    op.create_table(
        "pipeline_runs",
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("job_name", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("records_processed", sa.Integer(), nullable=True),
        sa.Column("records_failed", sa.Integer(), nullable=True),
        sa.Column("code_version", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("run_id"),
    )

    op.create_table(
        "recommendation_impressions",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("request_id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("movie_id", sa.Integer(), nullable=False),
        sa.Column("rank_position", sa.Integer(), nullable=False),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column("model_role", sa.Text(), nullable=False),
        sa.Column("experiment_id", sa.Text(), nullable=False),
        sa.Column("predicted_score", sa.Float(), nullable=False),
        sa.Column("shown_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_recommendation_impressions_experiment_id_model_version",
        "recommendation_impressions",
        ["experiment_id", "model_version"],
        unique=False,
    )

    op.create_table(
        "recommendation_ratings",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("request_id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("movie_id", sa.Integer(), nullable=False),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column("model_role", sa.Text(), nullable=False),
        sa.Column("experiment_id", sa.Text(), nullable=False),
        sa.Column("rating", sa.Float(), nullable=False),
        sa.Column("rated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_recommendation_ratings_experiment_id_model_version",
        "recommendation_ratings",
        ["experiment_id", "model_version"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_table("recommendation_ratings")
    op.drop_table("recommendation_impressions")
    op.drop_table("pipeline_runs")
    op.drop_table("movie_content_embeddings")
    op.drop_table("embedding_versions")
    op.drop_table("catalog_dirty_movies")
    op.drop_table("movie_tag_counts")
    op.drop_table("tag_events")
    op.drop_table("ratings_events")
    op.drop_table("catalog_movies")

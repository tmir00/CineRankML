from datetime import datetime
from common.db.base import Base
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, Text, func


class CatalogMovie(Base):
    __tablename__ = "catalog_movies"

    movie_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tmdb_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    imdb_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    genres: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
    original_language: Mapped[str | None] = mapped_column(Text, nullable=True)
    overview: Mapped[str | None] = mapped_column(Text, nullable=True)
    tagline: Mapped[str | None] = mapped_column(Text, nullable=True)
    tmdb_keywords: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
    runtime: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tmdb_popularity: Mapped[float | None] = mapped_column(Float, nullable=True)
    tmdb_vote_average: Mapped[float | None] = mapped_column(Float, nullable=True)
    tmdb_vote_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    poster_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    # TMDB adult flag; adult movies never show posters in the UI.
    adult: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    # US certification from TMDB release_dates (G, PG, PG-13, R, NC-17, etc.).
    certification_us: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Offline poster safety check has run (see scripts/check_poster_safety.py).
    poster_checked: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    # When false, the frontend hides the poster image (precomputed offline, not request-time).
    poster_safe: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    poster_safety_provider: Mapped[str | None] = mapped_column(Text, nullable=True)
    poster_safety_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    poster_safety_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    poster_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    enrichment_status: Mapped[str | None] = mapped_column(Text, nullable=True)
    enrichment_last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    enriched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    source: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class CatalogDirtyMovie(Base):
    __tablename__ = "catalog_dirty_movies"

    movie_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("catalog_movies.movie_id"), primary_key=True
    )
    first_dirty_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_dirty_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)


class MovieTagCount(Base):
    __tablename__ = "movie_tag_counts"

    movie_id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    tag: Mapped[str] = mapped_column(Text, primary_key=True)
    count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

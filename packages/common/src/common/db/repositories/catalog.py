"""Catalog movie reads and writes."""

from __future__ import annotations

import logging

from dataclasses import dataclass
from datetime import UTC, datetime
from sqlalchemy.orm import Session
from sqlalchemy import func, or_, select
from common.tmdb.client import TmdbMovieDetails
from sqlalchemy.dialects.postgresql import insert
from common.db.models.catalog import CatalogDirtyMovie, CatalogMovie


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CatalogSeedRow:
    """One movie row ready for bulk upsert into catalog_movies."""

    movie_id: int
    title: str
    year: int | None
    genres: list[str]
    tmdb_id: int | None
    imdb_id: str | None


def bulk_upsert_catalog_movies(session: Session, rows: list[CatalogSeedRow]) -> int:
    """
    Insert or update catalog_movies rows from MovieLens seed data.

    Do this by:
    1. Building a bulk insert from the seed rows.
    2. Upserting on movie_id so re-runs are idempotent.
    3. Updating only seed columns on conflict so enrichment fields are preserved.

    ============================ Arguments ============================
    session: An open SQLAlchemy session inside a transaction.
    rows: Parsed MovieLens movie rows to upsert.

    ============================ Returns ============================
    The number of rows in this batch.
    """
    if not rows:
        return 0

    now = datetime.now(tz=UTC)
    
    # Build the a list of dictionaries per CSV rowfor the bulk insert.
    values = [
        {
            "movie_id": row.movie_id,
            "title": row.title,
            "year": row.year,
            "genres": row.genres or None,
            "tmdb_id": row.tmdb_id,
            "imdb_id": row.imdb_id,
            "source": "movielens",
            "created_at": now,
            "updated_at": now,
        }
        for row in rows
    ]

    # Build the SQLAlchemy insert statement.
    stmt = insert(CatalogMovie).values(values)

    # Get the excluded columns for the on conflict do update.
    excluded = stmt.excluded
    # Build the SQLAlchemy on conflict do update statement.
    stmt = stmt.on_conflict_do_update(
        index_elements=["movie_id"],
        set_={
            "title": excluded.title,
            "year": excluded.year,
            "genres": excluded.genres,
            "tmdb_id": excluded.tmdb_id,
            "imdb_id": excluded.imdb_id,
            "source": excluded.source,
            "updated_at": now,
        },
    )
    session.execute(stmt)
    return len(rows)


def count_pending_enrichment(session: Session) -> int:
    """
    Count movies that still need TMDB enrichment.

    ============================ Arguments ============================
    session: An open SQLAlchemy session.

    ============================ Returns ============================
    The number of movies with a tmdb_id and no successful enrichment yet.
    """
    # Build the SQLAlchemy select statement.
    result = session.execute(
        select(func.count())
        .select_from(CatalogMovie)
        .where(
            CatalogMovie.tmdb_id.is_not(None),
            or_(
                CatalogMovie.enrichment_status.is_(None),
                CatalogMovie.enrichment_status == "failed",
            ),
        )
    )
    return int(result.scalar_one())


def fetch_pending_enrichment(session: Session, batch_size: int, remaining_limit: int | None) -> list[CatalogMovie]:
    """
    Fetch the next batch of movies waiting for TMDB enrichment.

    Do this by:
    1. Selecting rows with a tmdb_id and enrichment_status NULL or failed.
    2. Ordering by movie_id for stable batching.
    3. Applying the smaller of batch_size and remaining_limit when a cap is set.

    ============================ Arguments ============================
    session: An open SQLAlchemy session.
    batch_size: Maximum rows to fetch in one batch.
    remaining_limit: Optional cap on how many movies this job run may still process.

    ============================ Returns ============================
    A list of CatalogMovie ORM rows to enrich.
    """
    # Calculate the limit based on the batch size and remaining limit.
    limit = batch_size
    if remaining_limit is not None:
        limit = min(batch_size, max(0, remaining_limit))

    # If the limit is less than or equal to 0, return an empty list.
    if limit <= 0:
        return []

    # Build the select statem,ent to fetch the next batch of movies to enrich.
    stmt = (
        select(CatalogMovie)
        .where(
            CatalogMovie.tmdb_id.is_not(None),
            or_(
                CatalogMovie.enrichment_status.is_(None),
                CatalogMovie.enrichment_status == "failed",
            ),
        )
        .order_by(CatalogMovie.movie_id)
        .limit(limit)
    )
    # Return the list of CatalogMovie ORM rows to enrich.
    return list(session.scalars(stmt).all())


def apply_enrichment(session: Session, movie_id: int, details: TmdbMovieDetails | None, \
                        status: str, last_error: str | None = None,) -> None:
    """
    Write TMDB enrichment results onto one catalog_movies row.
    If details is None, set enrichment_status to failed.
    Otherwise, set enrichment_status to enriched and update the other fields.

    Do this by:
    1. Updating enrichment_status and updated_at for every outcome.
    2. Writing TMDB detail columns when enrichment succeeded.
    3. Setting enrichment_last_error on failure and clearing it on success.

    ============================ Arguments ============================
    session: An open SQLAlchemy session inside a transaction.
    movie_id: The catalog movie to update.
    details: TMDB field values when enrichment succeeded; None otherwise.
    status: One of enriched, failed, or skipped.
    last_error: TMDB failure reason when status is failed (e.g. not_found, http_error).
        Cleared to NULL when status is enriched. Ignored for skipped.

    ============================ Returns ============================
    None
    """
    now = datetime.now(tz=UTC)

    # Build the dictionary of values to update.
    values: dict = {
        "enrichment_status": status,
        "updated_at": now,
    }

    # If enrichment failed, persist the last error reason on the catalog row.
    if status == "failed":
        values["enrichment_last_error"] = last_error or "http_error"

    # If the status is enriched and the details are not None, update the values 
    # dictionary with the TMDB movie details.
    if status == "enriched" and details is not None:
        values.update(
            {
                "overview": details.overview,
                "tagline": details.tagline,
                "original_language": details.original_language,
                "runtime": details.runtime,
                "tmdb_popularity": details.tmdb_popularity,
                "tmdb_vote_average": details.tmdb_vote_average,
                "tmdb_vote_count": details.tmdb_vote_count,
                "tmdb_keywords": details.tmdb_keywords or None,
                "enriched_at": now,
                "enrichment_last_error": None,
            }
        )

    # Build the SQLAlchemy update statement.
    session.execute(
        CatalogMovie.__table__.update()
        .where(CatalogMovie.movie_id == movie_id)
        .values(**values)
    )


def mark_catalog_movie_dirty(session: Session, movie_id: int) -> None:
    """
    Mark one catalog movie as needing OpenSearch sync after successful enrichment.

    Do this by:
    1. Inserting a catalog_dirty_movies row when the movie is not yet dirty.
    2. Updating last_dirty_at when the row already exists.

    ============================ Arguments ============================
    session: An open SQLAlchemy session inside a transaction.
    movie_id: The movie that was just enriched.
    """
    now = datetime.now(tz=UTC)

    # Build the SQLAlchemy insert statement.
    # Insert a catalog_dirty_movies row when the movie is not yet dirty.
    # Update last_dirty_at when the row already exists.
    stmt = (
        insert(CatalogDirtyMovie)
        .values(
            movie_id=movie_id,
            first_dirty_at=now,
            last_dirty_at=now,
            attempt_count=0,
            last_error=None,
        )
        .on_conflict_do_update(
            index_elements=["movie_id"],
            set_={
                "last_dirty_at": now,
            },
        )
    )
    session.execute(stmt)


def mark_movies_without_tmdb_skipped(session: Session) -> int:
    """
    Mark catalog movies with no tmdb_id as skipped so enrichment does not retry them.

    ============================ Arguments ============================
    session: An open SQLAlchemy session inside a transaction.

    ============================ Returns ============================
    The number of rows updated.
    """
    now = datetime.now(tz=UTC)

    # Build the SQLAlchemy update statement.
    # Update the enrichment_status to skipped and the updated_at to the current time.
    result = session.query(CatalogMovie).filter(
        CatalogMovie.tmdb_id.is_(None),
        CatalogMovie.enrichment_status.is_(None),
    ).update(
        {
            "enrichment_status": "skipped",
            "updated_at": now,
        }
    )
    return int(result)

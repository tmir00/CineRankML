"""Catalog movie reads and writes."""

from __future__ import annotations

import logging

from dataclasses import dataclass
from datetime import UTC, datetime
from sqlalchemy.orm import Session
from sqlalchemy import func, select
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
    The number of movies with a tmdb_id that have never been attempted
    (enrichment_status IS NULL).
    """
    # Build the SQLAlchemy select statement.
    result = session.execute(
        select(func.count())
        .select_from(CatalogMovie)
        .where(
            CatalogMovie.tmdb_id.is_not(None),
            CatalogMovie.enrichment_status.is_(None),
        )
    )
    return int(result.scalar_one())


def fetch_pending_enrichment(session: Session, batch_size: int, remaining_limit: int | None) -> list[CatalogMovie]:
    """
    Fetch the next batch of movies waiting for TMDB enrichment.

    Do this by:
    1. Selecting rows with a tmdb_id and never-attempted enrichment_status (NULL).
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
            CatalogMovie.enrichment_status.is_(None),
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


@dataclass(frozen=True)
class DirtyMovieRow:
    """One dirty catalog movie joined with catalog metadata."""

    movie_id: int
    title: str
    year: int | None
    genres: list[str] | None
    overview: str | None
    tagline: str | None
    original_language: str | None
    tmdb_keywords: list[str] | None
    runtime: int | None
    tmdb_popularity: float | None
    tmdb_vote_average: float | None
    tmdb_vote_count: int | None
    tmdb_id: int | None
    imdb_id: str | None
    attempt_count: int


def fetch_dirty_movie_batch(session: Session, limit: int) -> list[DirtyMovieRow]:
    """
    Fetch the next batch of movies waiting for OpenSearch sync.

    Do this by:
    1. Joining catalog_dirty_movies with catalog_movies.
    2. Ordering by last_dirty_at so the oldest work is processed first.
    3. Returning up to limit rows.

    ============================ Arguments ============================
    session: An open SQLAlchemy session.
    limit: Maximum dirty movies to fetch.

    ============================ Returns ============================
    Dirty movie rows with catalog metadata attached.
    """
    if limit <= 0:
        return []

    # Build the SQLAlchemy select statement.
    stmt = (
        select(CatalogMovie, CatalogDirtyMovie.attempt_count)
        .join(CatalogDirtyMovie, CatalogDirtyMovie.movie_id == CatalogMovie.movie_id)
        .order_by(CatalogDirtyMovie.last_dirty_at)
        .limit(limit)
    )

    # Execute the select statement and return a list of DirtyMovieRow objects.
    rows = session.execute(stmt).all()
    return [
        DirtyMovieRow(
            movie_id=movie.movie_id,
            title=movie.title,
            year=movie.year,
            genres=movie.genres,
            overview=movie.overview,
            tagline=movie.tagline,
            original_language=movie.original_language,
            tmdb_keywords=movie.tmdb_keywords,
            runtime=movie.runtime,
            tmdb_popularity=movie.tmdb_popularity,
            tmdb_vote_average=movie.tmdb_vote_average,
            tmdb_vote_count=movie.tmdb_vote_count,
            tmdb_id=movie.tmdb_id,
            imdb_id=movie.imdb_id,
            attempt_count=int(attempt_count),
        )
        for movie, attempt_count in rows
    ]


def clear_dirty_movie(session: Session, movie_id: int) -> None:
    """
    Remove one movie from the dirty queue after a successful OpenSearch sync.

    ============================ Arguments ============================
    session: An open SQLAlchemy session inside a transaction.
    movie_id: Movie that finished syncing successfully.
    """
    session.query(CatalogDirtyMovie).filter(CatalogDirtyMovie.movie_id == movie_id).delete()


def record_dirty_sync_failure(session: Session, movie_id: int, error: str) -> None:
    """
    Record a failed sync attempt for one dirty movie.

    ============================ Arguments ============================
    session: An open SQLAlchemy session inside a transaction.
    movie_id: Movie that failed to sync.
    error: Short error message to store on the dirty row.
    """
    # Get the current time.
    now = datetime.now(tz=UTC)

    # Record the failed sync attempt by updating the dirty movie row and incrementing the attempt count.
    session.execute(
        CatalogDirtyMovie.__table__.update()
        .where(CatalogDirtyMovie.movie_id == movie_id)
        .values(
            attempt_count=CatalogDirtyMovie.attempt_count + 1,
            last_error=error[:2000],
            last_dirty_at=now,
        )
    )


def mark_all_catalog_movies_dirty(session: Session) -> int:
    """
    Mark every catalog movie dirty for a full OpenSearch rebuild.

    Do this by:
    1. Selecting all catalog movie ids.
    2. Upserting catalog_dirty_movies rows for each id.

    ============================ Arguments ============================
    session: An open SQLAlchemy session inside a transaction.

    ============================ Returns ============================
    The number of movies marked dirty.
    """
    # Get the current time.
    now = datetime.now(tz=UTC)

    # Get all the movie ids.
    movie_ids = list(session.scalars(select(CatalogMovie.movie_id)).all())
    if not movie_ids:
        return 0

    # Build the list of dictionaries for the bulk insert.
    values = [
        {
            "movie_id": movie_id,
            "first_dirty_at": now,
            "last_dirty_at": now,
            "attempt_count": 0,
            "last_error": None,
        }
        for movie_id in movie_ids
    ]

    # Build the SQLAlchemy insert statement for the bulk insert into catalog_dirty_movies.
    stmt = insert(CatalogDirtyMovie).values(values)

    # Build the SQLAlchemy on conflict do update statement for movies that already exist in the table.
    stmt = stmt.on_conflict_do_update(
        index_elements=["movie_id"],
        set_={
            "last_dirty_at": now,
            "attempt_count": 0,
            "last_error": None,
        },
    )
    
    session.execute(stmt)
    # Return the number of movies marked dirty.
    return len(movie_ids)


def count_dirty_movies(session: Session) -> int:
    """
    Count movies still waiting for OpenSearch sync.

    ============================ Returns ============================
    Number of rows in catalog_dirty_movies.
    """
    # Build the SQLAlchemy select statement to count the number of rows in catalog_dirty_movies.
    result = session.execute(select(func.count()).select_from(CatalogDirtyMovie))
    return int(result.scalar_one())

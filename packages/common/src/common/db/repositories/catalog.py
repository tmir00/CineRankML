"""Catalog movie reads and writes."""

from __future__ import annotations

import logging

from dataclasses import dataclass
from datetime import UTC, datetime
from sqlalchemy.orm import Session
from sqlalchemy import func, or_, select, text
from common.tmdb.client import TmdbMovieDetails
from common.poster_safety.show_poster import compute_show_poster
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


def count_enrichable_movies(session: Session, *, enrich_all: bool) -> int:
    """
    Count movies eligible for TMDB enrichment in the current mode.

    ============================ Arguments ============================
    session: An open SQLAlchemy session.
    enrich_all: When True, count all movies with a tmdb_id. When False,
        count only never-attempted rows (enrichment_status IS NULL).

    ============================ Returns ============================
    Number of enrichable catalog movies.
    """
    filters = [CatalogMovie.tmdb_id.is_not(None)]
    if not enrich_all:
        filters.append(CatalogMovie.enrichment_status.is_(None))

    result = session.execute(
        select(func.count()).select_from(CatalogMovie).where(*filters)
    )
    return int(result.scalar_one())


def count_pending_enrichment(session: Session) -> int:
    """
    Count movies that still need TMDB enrichment.

    ============================ Arguments ============================
    session: An open SQLAlchemy session.

    ============================ Returns ============================
    The number of movies with a tmdb_id that have never been attempted
    (enrichment_status IS NULL).
    """
    return count_enrichable_movies(session, enrich_all=False)


def fetch_enrichment_batch(
    session: Session,
    batch_size: int,
    remaining_limit: int | None,
    *,
    enrich_all: bool = False,
    after_movie_id: int = 0,
) -> list[CatalogMovie]:
    """
    Fetch the next batch of movies for TMDB enrichment.

    Do this by:
    1. Selecting rows with a tmdb_id.
    2. In default mode, limiting to never-attempted enrichment_status (NULL).
    3. In enrich_all mode, paging forward with movie_id cursor.
    4. Applying the smaller of batch_size and remaining_limit when a cap is set.

    ============================ Arguments ============================
    session: An open SQLAlchemy session.
    batch_size: Maximum rows to fetch in one batch.
    remaining_limit: Optional cap on how many movies this job run may still process.
    enrich_all: When True, include all enrichment statuses and use after_movie_id.
    after_movie_id: Cursor for enrich_all mode; returns rows with movie_id greater than this.

    ============================ Returns ============================
    A list of CatalogMovie ORM rows to enrich.
    """
    limit = batch_size
    if remaining_limit is not None:
        limit = min(batch_size, max(0, remaining_limit))

    if limit <= 0:
        return []

    filters = [CatalogMovie.tmdb_id.is_not(None)]
    if enrich_all:
        filters.append(CatalogMovie.movie_id > after_movie_id)
    else:
        filters.append(CatalogMovie.enrichment_status.is_(None))

    stmt = (
        select(CatalogMovie)
        .where(*filters)
        .order_by(CatalogMovie.movie_id)
        .limit(limit)
    )
    return list(session.scalars(stmt).all())


def fetch_pending_enrichment(session: Session, batch_size: int, remaining_limit: int | None) -> list[CatalogMovie]:
    """
    Fetch the next batch of movies waiting for TMDB enrichment.

    ============================ Arguments ============================
    session: An open SQLAlchemy session.
    batch_size: Maximum rows to fetch in one batch.
    remaining_limit: Optional cap on how many movies this job run may still process.

    ============================ Returns ============================
    A list of CatalogMovie ORM rows to enrich.
    """
    return fetch_enrichment_batch(
        session,
        batch_size,
        remaining_limit,
        enrich_all=False,
    )


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
                "poster_path": details.poster_path,
                "adult": details.adult,
                "certification_us": details.certification_us,
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
    poster_path: str | None
    adult: bool
    certification_us: str | None
    poster_safe: bool
    poster_checked: bool
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
            poster_path=movie.poster_path,
            adult=bool(movie.adult),
            certification_us=movie.certification_us,
            poster_safe=bool(movie.poster_safe),
            poster_checked=bool(movie.poster_checked),
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
    1. Counting catalog movies (return early when the catalog is empty).
    2. Upserting dirty rows with INSERT…SELECT so Postgres stays under the
       bind-parameter limit (a per-id VALUES list blows past 65535 for large catalogs).

    ============================ Arguments ============================
    session: An open SQLAlchemy session inside a transaction.

    ============================ Returns ============================
    The number of movies marked dirty.
    """
    # Count catalog movies first so we can return without a no-op upsert.
    count = session.scalar(select(func.count()).select_from(CatalogMovie)) or 0
    if count == 0:
        return 0

    # Get the current time for first/last dirty timestamps.
    now = datetime.now(tz=UTC)

    # One statement copies every movie_id into catalog_dirty_movies (or refreshes existing rows).
    session.execute(
        text(
            """
            INSERT INTO catalog_dirty_movies
                (movie_id, first_dirty_at, last_dirty_at, attempt_count, last_error)
            SELECT movie_id, :now, :now, 0, NULL
            FROM catalog_movies
            ON CONFLICT (movie_id) DO UPDATE SET
                last_dirty_at = EXCLUDED.last_dirty_at,
                attempt_count = 0,
                last_error = NULL
            """
        ),
        {"now": now},
    )
    # Return how many catalog movies were marked dirty.
    return count


def count_dirty_movies(session: Session) -> int:
    """
    Count movies still waiting for OpenSearch sync.

    ============================ Returns ============================
    Number of rows in catalog_dirty_movies.
    """
    # Build the SQLAlchemy select statement to count the number of rows in catalog_dirty_movies.
    result = session.execute(select(func.count()).select_from(CatalogDirtyMovie))
    return int(result.scalar_one())


@dataclass(frozen=True)
class CatalogMovieRow:
    """Catalog metadata needed for hybrid ranker candidate features."""

    movie_id: int
    title: str
    year: int | None
    runtime: int | None
    tmdb_popularity: float | None
    tmdb_vote_average: float | None
    tmdb_vote_count: int | None


def get_catalog_movies_by_ids(session: Session, movie_ids: list[int]) -> dict[int, CatalogMovieRow]:
    """
    Load catalog metadata for a batch of movie ids.

    ============================ Arguments ============================
    session: An open SQLAlchemy session.
    movie_ids: Movies to look up in catalog_movies.

    ============================ Returns ============================
    Mapping of movie_id to catalog metadata rows.
    """
    if not movie_ids:
        return {}

    stmt = select(CatalogMovie).where(CatalogMovie.movie_id.in_(movie_ids))
    rows = session.scalars(stmt).all()
    return {
        row.movie_id: CatalogMovieRow(
            movie_id=row.movie_id,
            title=row.title,
            year=row.year,
            runtime=row.runtime,
            tmdb_popularity=row.tmdb_popularity,
            tmdb_vote_average=row.tmdb_vote_average,
            tmdb_vote_count=row.tmdb_vote_count,
        )
        for row in rows
    }


@dataclass(frozen=True)
class CatalogDisplayRow:
    """Catalog metadata needed to render a movie card in the UI."""

    movie_id: int
    title: str
    year: int | None
    genres: list[str]
    poster_path: str | None
    poster_safe: bool
    show_poster: bool
    certification_us: str | None


def get_catalog_display_by_ids(session: Session, movie_ids: list[int]) -> dict[int, CatalogDisplayRow]:
    """
    Load title, year, genres, and poster_path for a batch of catalog movies.

    ============================ Arguments ============================
    session: An open SQLAlchemy session.
    movie_ids: Movies to look up in catalog_movies.

    ============================ Returns ============================
    Mapping of movie_id to display metadata rows.
    """
    if not movie_ids:
        return {}

    stmt = select(CatalogMovie).where(CatalogMovie.movie_id.in_(movie_ids))
    rows = session.scalars(stmt).all()
    return {
        row.movie_id: CatalogDisplayRow(
            movie_id=row.movie_id,
            title=row.title,
            year=row.year,
            genres=list(row.genres or []),
            poster_path=row.poster_path,
            poster_safe=bool(row.poster_safe),
            show_poster=compute_show_poster(
                poster_path=row.poster_path,
                poster_safe=bool(row.poster_safe),
                poster_checked=bool(row.poster_checked),
                adult=bool(row.adult),
                certification_us=row.certification_us,
            ),
            certification_us=row.certification_us,
        )
        for row in rows
    }


def get_movie_genres_by_ids(session: Session, movie_ids: list[int]) -> dict[int, list[str]]:
    """
    Load genre lists for a batch of catalog movies.

    Do this by:
    1. Selecting movie_id and genres from catalog_movies for the given ids.
    2. Returning only rows that have at least one genre string.

    ============================ Arguments ============================
    session: An open SQLAlchemy session.
    movie_ids: Movies to look up in catalog_movies.

    ============================ Returns ============================
    Mapping of movie_id to a non-empty genre list.
    """
    if not movie_ids:
        return {}

    stmt = select(CatalogMovie.movie_id, CatalogMovie.genres).where(
        CatalogMovie.movie_id.in_(movie_ids)
    )
    rows = session.execute(stmt).all()

    result: dict[int, list[str]] = {}
    for movie_id, genres in rows:
        if genres:
            result[int(movie_id)] = list(genres)
    return result


def catalog_movie_exists(session: Session, movie_id: int) -> bool:
    """Return True when one movie_id exists in catalog_movies."""
    stmt = select(CatalogMovie.movie_id).where(CatalogMovie.movie_id == movie_id)
    return session.execute(stmt).scalar_one_or_none() is not None


@dataclass(frozen=True)
class PosterSafetyStats:
    """Counts used by the offline poster safety script preamble."""

    total_catalog: int
    total_with_poster: int
    total_questionable: int
    total_already_checked: int


@dataclass(frozen=True)
class PosterSafetyCandidate:
    """One catalog movie queued for offline poster safety checking."""

    movie_id: int
    title: str
    poster_path: str
    adult: bool
    certification_us: str | None


@dataclass(frozen=True)
class PosterSafetyUpdate:
    """Poster safety fields to write back to catalog_movies."""

    movie_id: int
    poster_checked: bool
    poster_safe: bool
    poster_checked_at: datetime


def _questionable_movie_filter():
    """SQL filter for movies that need offline poster safety checks."""
    return or_(
        CatalogMovie.adult.is_(True),
        CatalogMovie.certification_us.in_(("R", "NC-17")),
    )


def count_poster_safety_stats(session: Session) -> PosterSafetyStats:
    """
    Count catalog movies relevant to the offline poster safety script.

    ============================ Returns ============================
    Totals for catalog size, poster coverage, questionable movies, and checked rows.
    """
    total_catalog = int(
        session.execute(select(func.count()).select_from(CatalogMovie)).scalar_one()
    )
    total_with_poster = int(
        session.execute(
            select(func.count())
            .select_from(CatalogMovie)
            .where(CatalogMovie.poster_path.is_not(None))
        ).scalar_one()
    )
    questionable_filter = _questionable_movie_filter()
    total_questionable = int(
        session.execute(
            select(func.count())
            .select_from(CatalogMovie)
            .where(
                CatalogMovie.poster_path.is_not(None),
                questionable_filter,
            )
        ).scalar_one()
    )
    total_already_checked = int(
        session.execute(
            select(func.count())
            .select_from(CatalogMovie)
            .where(
                CatalogMovie.poster_path.is_not(None),
                questionable_filter,
                CatalogMovie.poster_checked.is_(True),
            )
        ).scalar_one()
    )
    return PosterSafetyStats(
        total_catalog=total_catalog,
        total_with_poster=total_with_poster,
        total_questionable=total_questionable,
        total_already_checked=total_already_checked,
    )


def fetch_questionable_poster_candidates(
    session: Session,
    *,
    limit: int | None,
    force: bool,
    only_certification: str | None,
) -> list[PosterSafetyCandidate]:
    """
    Fetch questionable catalog movies that still need poster safety checks.

    ============================ Arguments ============================
    session: An open SQLAlchemy session.
    limit: Optional maximum number of rows to return.
    force: When True, include movies even if poster_checked is already true.
    only_certification: Optional US certification filter (e.g. R).

    ============================ Returns ============================
    Candidate rows ordered by movie_id.
    """
    filters = [
        CatalogMovie.poster_path.is_not(None),
        _questionable_movie_filter(),
    ]
    if not force:
        filters.append(CatalogMovie.poster_checked.is_(False))
    if only_certification:
        filters.append(CatalogMovie.certification_us == only_certification)

    stmt = (
        select(CatalogMovie)
        .where(*filters)
        .order_by(CatalogMovie.movie_id)
    )
    if limit is not None and limit > 0:
        stmt = stmt.limit(limit)

    rows = session.scalars(stmt).all()
    return [
        PosterSafetyCandidate(
            movie_id=row.movie_id,
            title=row.title,
            poster_path=str(row.poster_path),
            adult=bool(row.adult),
            certification_us=row.certification_us,
        )
        for row in rows
        if row.poster_path
    ]


def update_poster_safety_batch(session: Session, updates: list[PosterSafetyUpdate]) -> int:
    """
    Write poster safety results for a batch of catalog movies.

    ============================ Arguments ============================
    session: An open SQLAlchemy session inside a transaction.
    updates: Poster safety rows to persist.

    ============================ Returns ============================
    Number of rows updated.
    """
    if not updates:
        return 0

    now = datetime.now(tz=UTC)
    for update in updates:
        session.execute(
            CatalogMovie.__table__.update()
            .where(CatalogMovie.movie_id == update.movie_id)
            .values(
                poster_checked=update.poster_checked,
                poster_safe=update.poster_safe,
                poster_checked_at=update.poster_checked_at,
                updated_at=now,
            )
        )
    return len(updates)


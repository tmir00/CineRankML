"""TMDB enrichment batch loop."""

from __future__ import annotations

import logging

from dataclasses import dataclass
from sqlalchemy.orm import Session

from common.config.settings import EnrichmentSettings
from common.db.repositories.catalog import (
    apply_enrichment,
    count_enrichable_movies,
    fetch_enrichment_batch,
    mark_catalog_movie_dirty,
    mark_movies_without_tmdb_skipped,
)
from common.tmdb.client import TmdbClient

logger = logging.getLogger(__name__)


@dataclass
class EnrichmentStats:
    """Mutable counters updated during the enrichment loop."""

    processed: int = 0
    failed: int = 0


def run_enrichment_loop(
    session: Session,
    client: TmdbClient,
    settings: EnrichmentSettings,
    stats: EnrichmentStats,
    *,
    enrich_all: bool = False,
) -> None:
    """
    Enrich catalog movies with TMDB metadata in batches.

    Do this by:
    1. Marking movies without a tmdb_id as skipped.
    2. Fetching rows in batches until none remain or the limit is reached.
    3. Calling TMDB for each movie, updating catalog_movies, and marking dirty on success.
    4. Persisting enrichment_last_error on catalog_movies when TMDB enrichment fails.

    ============================ Arguments ============================
    session: An open SQLAlchemy session (commits happen inside this function).
    client: TMDB HTTP client with rate limiting.
    settings: Batch size, optional limit, and log frequency.
    stats: Mutable counters updated as movies are enriched or fail.
    enrich_all: When True, re-fetch TMDB metadata for all movies with a tmdb_id.

    ============================ Returns ============================
    None. Progress counts are stored on stats.processed and stats.failed.
    """
    skipped_no_tmdb = mark_movies_without_tmdb_skipped(session)
    session.commit()

    if skipped_no_tmdb:
        logger.info(
            "Marked movies without tmdb_id as skipped",
            extra={"count": skipped_no_tmdb},
        )

    remaining_limit = settings.enrichment_limit
    after_movie_id = 0

    while True:
        batch = fetch_enrichment_batch(
            session,
            batch_size=settings.enrichment_batch_size,
            remaining_limit=remaining_limit,
            enrich_all=enrich_all,
            after_movie_id=after_movie_id,
        )
        if not batch:
            break

        for movie in batch:
            if movie.tmdb_id is None:
                continue

            details, error_type = client.fetch_movie(movie.tmdb_id)

            if details is not None:
                apply_enrichment(session, movie.movie_id, details, "enriched", last_error=None)
                mark_catalog_movie_dirty(session, movie.movie_id)
                stats.processed += 1
                logger.info(
                    "Enriched %s (poster_path=%s)",
                    movie.title,
                    details.poster_path,
                    extra={
                        "movie_id": movie.movie_id,
                        "title": movie.title,
                        "poster_path": details.poster_path,
                    },
                )
            else:
                apply_enrichment(
                    session,
                    movie.movie_id,
                    None,
                    "failed",
                    last_error=error_type or "http_error",
                )
                stats.failed += 1

            if remaining_limit is not None:
                remaining_limit -= 1
                if remaining_limit <= 0:
                    break

            total = stats.processed + stats.failed
            if total % settings.enrichment_log_every_n == 0:
                logger.info(
                    "Enrichment progress",
                    extra={
                        "processed": stats.processed,
                        "failed": stats.failed,
                        "enrich_all": enrich_all,
                        "remaining": count_enrichable_movies(session, enrich_all=enrich_all),
                    },
                )

        if enrich_all and batch:
            after_movie_id = batch[-1].movie_id

        session.commit()

        if remaining_limit is not None and remaining_limit <= 0:
            break

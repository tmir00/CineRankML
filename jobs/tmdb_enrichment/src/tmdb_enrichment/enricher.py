"""TMDB enrichment batch loop."""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from common.config.settings import EnrichmentSettings
from common.db.repositories.catalog import (
    apply_enrichment,
    count_pending_enrichment,
    fetch_pending_enrichment,
    mark_catalog_movie_dirty,
    mark_movies_without_tmdb_skipped,
)
from common.tmdb.client import TmdbClient

logger = logging.getLogger(__name__)


def run_enrichment_loop(session: Session, client: TmdbClient, settings: EnrichmentSettings) -> tuple[int, int]:
    """
    Enrich pending catalog movies with TMDB metadata in batches.

    Do this by:
    1. Marking movies without a tmdb_id as skipped.
    2. Fetching pending rows in batches until none remain or the limit is reached.
    3. Calling TMDB for each movie, updating catalog_movies, and marking dirty on success.
    4. Persisting enrichment_last_error on catalog_movies when TMDB enrichment fails.

    ============================ Arguments ============================
    session: An open SQLAlchemy session (commits happen inside this function).
    client: TMDB HTTP client with rate limiting.
    settings: Batch size, optional limit, and log frequency.

    ============================ Returns ============================
    A tuple of (records_processed, records_failed).
    """
    # Mark movies without a tmdb_id as skipped.
    skipped_no_tmdb = mark_movies_without_tmdb_skipped(session)
    session.commit()

    # If any movies were marked as skipped, log the count and record the metric.
    if skipped_no_tmdb:
        logger.info(
            "Marked movies without tmdb_id as skipped",
            extra={"count": skipped_no_tmdb},
        )

    processed = 0
    failed = 0
    remaining_limit = settings.enrichment_limit

    # Loop until no more pending movies or the limit is reached.
    while True:
        batch = fetch_pending_enrichment(
            session,
            batch_size=settings.enrichment_batch_size,
            remaining_limit=remaining_limit,
        )
        # If there are no more pending movies, break the loop.
        if not batch:
            break

        # Iterate over the batch of pending movies.
        for movie in batch:
            # If the movie has no tmdb_id, skip it.
            if movie.tmdb_id is None:
                continue

            details, error_type = client.fetch_movie(movie.tmdb_id)

            # If the fetched movie details are not None, update the movie and mark it as dirty.
            if details is not None:
                # Apply the enrichment to the movie and clear any prior failure reason.
                apply_enrichment(session, movie.movie_id, details, "enriched", last_error=None)
                # Mark the movie as dirty.
                mark_catalog_movie_dirty(session, movie.movie_id)
                processed += 1
            else:
                # Set the enrichment status to failed and persist the TMDB error reason.
                apply_enrichment(
                    session,
                    movie.movie_id,
                    None,
                    "failed",
                    last_error=error_type or "http_error",
                )
                failed += 1

            # If there is a remaining limit, decrement it and check if it has reached 0.
            if remaining_limit is not None:
                remaining_limit -= 1
                # If the remaining limit has reached 0, break the loop.
                if remaining_limit <= 0:
                    break

            # Calculate the total number of processed and failed movies.
            total = processed + failed
            # If the total number of processed and failed movies is a multiple of the log frequency, log the progress.
            if total % settings.enrichment_log_every_n == 0:
                logger.info(
                    "Enrichment progress",
                    extra={
                        "processed": processed,
                        "failed": failed,
                        "pending": count_pending_enrichment(session),
                    },
                )

        # Commit the session.
        session.commit()

        # If there is a remaining limit and it has reached 0, break the loop.
        if remaining_limit is not None and remaining_limit <= 0:
            break

    return processed, failed

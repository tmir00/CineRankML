"""Batch job entrypoint for TMDB catalog enrichment."""

from __future__ import annotations

import logging
import sys
import time
import uuid

from common.config.settings import get_enrichment_settings, get_tmdb_settings
from common.db.repositories.pipeline import finish_pipeline_run, start_pipeline_run
from common.db.session import get_session_factory
from common.tmdb.client import TmdbClient
from tmdb_enrichment.enricher import run_enrichment_loop

logger = logging.getLogger(__name__)

JOB_NAME = "tmdb_enrichment"


def configure_logging() -> None:
    """Set up basic structured logging for the TMDB enrichment job."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )


def main() -> None:
    """
    Run the tmdb_enrichment batch job.

    Do this by:
    1. Creating a pipeline_runs row.
    2. Enriching pending catalog movies via the TMDB API.
    3. Recording success or failure in pipeline_runs and logs.
    """
    configure_logging()

    # Get the TMDB settings.
    tmdb_settings = get_tmdb_settings()
    # If the TMDB API key is not set, raise an error.
    if not tmdb_settings.tmdb_api_key:
        raise RuntimeError("TMDB_API_KEY is required for tmdb_enrichment")

    enrichment_settings = get_enrichment_settings()

    # Generate a unique run ID for the job run.
    # This is used to identify the job run in the pipeline_runs table.
    run_id = str(uuid.uuid4())
    # Start the timer for the job run.
    start = time.perf_counter()
    records_processed = 0
    records_failed = 0
    error_message: str | None = None
    status = "success"

    # Create a new SQLAlchemy session.
    session_factory = get_session_factory()
    session = session_factory()

    # Insert a new pipeline run row to indicate the job has started.
    try:
        start_pipeline_run(session, run_id, JOB_NAME)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    # Try to run the TMDB enrichment job.
    try:
        # Create a new TMDB client.
        client = TmdbClient(tmdb_settings)
        # Create a new SQLAlchemy session.
        session = session_factory()
        # Try to run the enrichment loop.
        try:
            records_processed, records_failed = run_enrichment_loop(
                session,
                client,
                enrichment_settings,
            )
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    except Exception as exc:
        status = "failure"
        error_message = str(exc)
        logger.exception("TMDB enrichment job failed")
        raise
    
    # If the TMDB job is done, finish the pipeline run by updating the pipeline run row with the job run results.
    finally:
        # Calculate the elapsed time for the job run.
        elapsed = time.perf_counter() - start

        session = session_factory()
        try:
            finish_pipeline_run(
                session,
                run_id,
                status=status,
                records_processed=records_processed,
                records_failed=records_failed,
                error_message=error_message,
            )
            session.commit()
        except Exception:
            session.rollback()
            logger.exception("Failed to update pipeline run")
        finally:
            session.close()

        logger.info(
            "TMDB enrichment job finished",
            extra={
                "run_id": run_id,
                "status": status,
                "records_processed": records_processed,
                "records_failed": records_failed,
                "duration_seconds": elapsed,
            },
        )


if __name__ == "__main__":
    main()

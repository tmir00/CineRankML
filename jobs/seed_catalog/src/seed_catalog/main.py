"""Batch job entrypoint for seeding catalog_movies from MovieLens CSVs."""

from __future__ import annotations

import sys
import time
import uuid
import logging

from seed_catalog.loader import iter_seed_rows
from common.db.session import get_session_factory
from common.config.settings import get_catalog_seed_settings
from common.db.repositories.catalog import bulk_upsert_catalog_movies
from common.db.repositories.pipeline import finish_pipeline_run, start_pipeline_run

logger = logging.getLogger(__name__)

JOB_NAME = "seed_catalog"


def configure_logging() -> None:
    """Set up basic structured logging for the seed catalog job."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )


def run_seed_catalog() -> int:
    """
    Load MovieLens CSVs and upsert all movies into catalog_movies.

    Do this by:
    1. Reading joined rows from movies.csv and links.csv.
    2. Upserting into catalog_movies in configurable batches.
    3. Returning the total number of rows processed.

    ============================ Returns ============================
    The total number of catalog rows upserted.
    """
    settings = get_catalog_seed_settings()
    session_factory = get_session_factory()
    total_processed = 0
    # Initialize an empty list to store the batch of catalog seed rows.
    batch: list = []

    # Iterate over the rows from the MovieLens CSVs.
    for row in iter_seed_rows(settings):
        # Add the row to the batch.
        batch.append(row)
        
        # If the batch size is reached, upsert the batch into catalog_movies.
        if len(batch) >= settings.seed_batch_size:
            # Create a new session.
            session = session_factory()
            try:
                # Upsert the batch into catalog_movies.
                bulk_upsert_catalog_movies(session, batch)
                session.commit()
                total_processed += len(batch)
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

            # Clear the batch.
            batch = []

    # If there are any remaining rows in the batch, upsert the batch into catalog_movies.
    if batch:
        session = session_factory()
        # Try to upsert the batch into catalog_movies.
        try:
            bulk_upsert_catalog_movies(session, batch)
            session.commit()
            total_processed += len(batch)

        # If the upsert fails, rollback the session and raise the exception.
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    logger.info("Seed catalog complete", extra={"records_processed": total_processed})
    return total_processed


def main() -> None:
    """
    Run the seed_catalog batch job.

    Do this by:
    1. Creating a pipeline_runs row.
    2. Loading and upserting MovieLens catalog rows.
    3. Recording success or failure in pipeline_runs and logs.
    """
    # Set up basic structured logging for the seed catalog job.
    configure_logging()

    # Generate a unique run ID for the job run.
    run_id = str(uuid.uuid4())
    # Start the timer for the job run.
    start = time.perf_counter()
    records_processed = 0
    error_message: str | None = None
    status = "success"

    # Create a new SQLAlchemy session.
    session_factory = get_session_factory()
    session = session_factory()
    # Insert a new pipeline run row to indicate the job has started.
    try:
        start_pipeline_run(session, run_id, JOB_NAME)
        session.commit()

    # If the pipeline run fails, rollback the session and raise the exception.
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    # Try to run the seed catalog job.
    try:
        records_processed = run_seed_catalog()
    except Exception as exc:
        status = "failure"
        error_message = str(exc)
        logger.exception("Seed catalog job failed")
        raise

    # Calculate the elapsed time for the job run.
    finally:
        elapsed = time.perf_counter() - start

        session = session_factory()
        # Finish the pipeline run by updating the pipeline run row with the job run results.
        try:
            finish_pipeline_run(
                session,
                run_id,
                status=status,
                records_processed=records_processed,
                records_failed=0 if status == "success" else None,
                error_message=error_message,
            )
            session.commit()

        # If the pipeline run fails, rollback the session and log the exception.
        except Exception:
            session.rollback()
            logger.exception("Failed to update pipeline run")
        finally:
            session.close()

        # Log the job run results.
        logger.info(
            "Seed catalog job finished",
            extra={
                "run_id": run_id,
                "status": status,
                "records_processed": records_processed,
                "duration_seconds": elapsed,
            },
        )


if __name__ == "__main__":
    main()

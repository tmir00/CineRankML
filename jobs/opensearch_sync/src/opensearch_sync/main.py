"""Batch job entrypoint for OpenSearch catalog sync."""

from __future__ import annotations

import logging
import sys
import time
import uuid

from common.config.settings import (
    get_embedder_settings,
    get_opensearch_settings,
    get_sync_settings,
)

from opensearch_sync.syncer import run_sync_loop
from common.db.session import get_session_factory
from common.embeddings.client import EmbedderClient
from common.opensearch.client import create_opensearch_client
from common.db.repositories.pipeline import finish_pipeline_run, start_pipeline_run


logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Set up basic structured logging for the OpenSearch sync job."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )


def main() -> None:
    """
    Run the opensearch_sync batch job.

    Do this by:
    1. Creating a pipeline_runs row.
    2. Syncing dirty catalog movies to OpenSearch via embedder-api.
    3. Recording success or failure in pipeline_runs and logs.
    """
    configure_logging()

    # Get all the settings from the config.
    sync_settings = get_sync_settings()
    opensearch_settings = get_opensearch_settings()
    embedder_settings = get_embedder_settings()

    # Generate a unique run id.
    run_id = str(uuid.uuid4())
    # Start the timer.
    start = time.perf_counter()
    stats_processed = 0
    stats_failed = 0
    error_message: str | None = None
    status = "success"

    # Create a session factory to get a new session for each batch.
    session_factory = get_session_factory()
    session = session_factory()
    # Start the pipeline run by marking the start time in the pipeline_runs table.
    try:
        start_pipeline_run(session, run_id, sync_settings.job_name)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    # Create the OpenSearch client and embedder client.
    try:
        # Create the OpenSearch client.
        client = create_opensearch_client(opensearch_settings)
        # Create the Embedder client.
        with EmbedderClient(embedder_settings) as embedder:
            # Run the sync loop.
            sync_stats = run_sync_loop(
                session_factory,
                client,
                embedder,
                sync_settings,
                opensearch_settings,
                embedder_settings,
            )

            # Update the stats.
            stats_processed = sync_stats.processed
            stats_failed = sync_stats.failed
            # If there were failed movies and no movies were processed, set the status to failure.
            if stats_failed > 0 and stats_processed == 0:
                status = "failure"
                error_message = f"{stats_failed} movies failed to sync"
    
    except KeyboardInterrupt:
        status = "cancelled"
        error_message = "Interrupted by user (Ctrl+C)"
        logger.warning(
            "OpenSearch sync interrupted",
            extra={
                "records_processed": stats_processed,
                "records_failed": stats_failed,
            },
        )
        raise
    
    except Exception as exc:
        status = "failure"
        error_message = str(exc)
        logger.exception("OpenSearch sync job failed")
        raise

    # On completion, finish the pipeline run by updating the pipeline_runs table.
    finally:
        # Stop the timer.
        elapsed = time.perf_counter() - start

        # Create a session to finish the pipeline run.
        session = session_factory()
        try:
            finish_pipeline_run(
                session,
                run_id,
                status=status,
                records_processed=stats_processed,
                records_failed=stats_failed,
                error_message=error_message,
            )
            session.commit()
        except Exception:
            session.rollback()
            logger.exception("Failed to update pipeline run")
        finally:
            session.close()

        # Log the completion of the sync job.
        logger.info(
            "OpenSearch sync job finished",
            extra={
                "run_id": run_id,
                "status": status,
                "records_processed": stats_processed,
                "records_failed": stats_failed,
                "duration_seconds": elapsed,
            },
        )


if __name__ == "__main__":
    main()

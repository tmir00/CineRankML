"""Batch job entrypoint for CF dataset preparation."""

from __future__ import annotations

import sys
import time
import uuid
import logging

from common.storage.s3 import create_s3_client
from common.db.session import get_session_factory
from common.config.settings import get_cf_dataset_settings
from common.metrics.cf_dataset import push_cf_dataset_metrics
from prepare_cf_dataset.prep import resolve_cf_dataset_version, run_cf_dataset_prep
from common.db.repositories.pipeline import finish_pipeline_run, start_pipeline_run


logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Set up basic structured logging for the CF dataset prep job."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )


def main() -> None:
    """
    Run the prepare_cf_dataset batch job.

    Do this by:
    1. Creating a pipeline_runs row.
    2. Building a versioned CF dataset from the latest complete snapshot.
    3. Writing manifest.json last and pushing metrics to Pushgateway.
    """
    # Configure logging and load settings.
    configure_logging()
    settings = get_cf_dataset_settings()
    run_id = str(uuid.uuid4())
    start = time.perf_counter()

    # Initialize counters and error message.
    stats_processed = 0
    stats_failed = 0
    error_message: str | None = None
    status = "success"
    cf_dataset_version = resolve_cf_dataset_version(settings)
    train_row_count = 0
    holdout_row_count = 0

    # Insert a new row into pipeline_runs table to mark the start of the job.
    session_factory = get_session_factory()
    session = session_factory()
    try:
        start_pipeline_run(session, run_id, settings.job_name)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    # Run the CF dataset prep.
    try:
        # Create an S3 client so that we can read/write to MinIO/S3.
        client = create_s3_client(settings)
        # Run the CF dataset prep.
        prep_stats = run_cf_dataset_prep(
            client,
            settings,
            pipeline_run_id=run_id,
        )

        # Update the counters and error message.
        cf_dataset_version = prep_stats.cf_dataset_version
        train_row_count = prep_stats.train_row_count
        holdout_row_count = prep_stats.holdout_row_count
        stats_processed = train_row_count + holdout_row_count

    except KeyboardInterrupt:
        status = "cancelled"
        error_message = "Interrupted by user (Ctrl+C)"
        logger.warning("CF dataset prep interrupted")
        raise

    except Exception as exc:
        status = "failure"
        error_message = str(exc)
        logger.exception("CF dataset prep job failed")
        raise
    
    # When the job is done, update the pipeline_runs table and push metrics to Pushgateway.
    finally:
        # Calculate the elapsed time.
        elapsed = time.perf_counter() - start
        success = status == "success"

        # Open a session to the database and update the pipeline_runs table.
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

        # Push metrics to Pushgateway.
        push_cf_dataset_metrics(
            settings.pushgateway_url,
            settings.metrics_job_name,
            cf_dataset_version,
            success=success,
            duration_seconds=elapsed,
            train_row_count=train_row_count,
            holdout_row_count=holdout_row_count,
        )

        logger.info(
            "CF dataset prep job finished",
            extra={
                "run_id": run_id,
                "cf_dataset_version": cf_dataset_version,
                "status": status,
                "records_processed": stats_processed,
                "records_failed": stats_failed,
                "train_row_count": train_row_count,
                "holdout_row_count": holdout_row_count,
                "duration_seconds": elapsed,
            },
        )


if __name__ == "__main__":
    main()

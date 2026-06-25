"""Batch job entrypoint for Postgres-to-MinIO snapshots."""

from __future__ import annotations

import sys
import time
import uuid
import logging

from common.storage.s3 import create_s3_client
from common.db.session import get_session_factory
from common.config.settings import get_snapshot_settings
from common.metrics.snapshot import push_snapshot_metrics
from snapshot_to_s3.exporter import resolve_snapshot_id, run_snapshot_export
from common.db.repositories.pipeline import finish_pipeline_run, start_pipeline_run


logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Set up basic structured logging for the snapshot job."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )


def main() -> None:
    """
    Run the snapshot_to_s3 batch job.

    Do this by:
    1. Creating a pipeline_runs row.
    2. Exporting Postgres tables to MinIO as partitioned Parquet parts.
    3. Writing manifest.json last and pushing metrics to Pushgateway.
    """
    # Configure the logging.
    configure_logging()

    # Get the settings.
    settings = get_snapshot_settings()
    # Generate a unique run id.
    run_id = str(uuid.uuid4())
    # Get the start time.
    start = time.perf_counter()

    # Initialize the statistics.
    stats_processed = 0
    stats_failed = 0
    error_message: str | None = None
    status = "success"
    snapshot_id = resolve_snapshot_id(settings)
    table_row_counts: dict[str, int] = {}
    upload_failures = 0

    # Get the session factory to start a new pipeline run.
    session_factory = get_session_factory()
    session = session_factory()
    
    # Start a new pipeline run by creating a pipeline_runs row in the database.
    try:
        start_pipeline_run(session, run_id, settings.job_name)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    # Create an S3 client and run the snapshot export.
    try:
        # Create an S3 client.
        client = create_s3_client(settings)
        # Run the snapshot export.
        export_stats = run_snapshot_export(
            session_factory,
            client,
            settings,
            pipeline_run_id=run_id,
        )
        snapshot_id = export_stats.snapshot_id
        table_row_counts = export_stats.table_row_counts
        upload_failures = export_stats.upload_failures
        stats_processed = sum(table_row_counts.values())
        stats_failed = upload_failures

    # Handle keyboard interrupt.
    except KeyboardInterrupt:
        status = "cancelled"
        error_message = "Interrupted by user (Ctrl+C)"
        logger.warning("Snapshot export interrupted")
        raise

    # Handle other exceptions.
    except Exception as exc:
        status = "failure"
        error_message = str(exc)
        logger.exception("Snapshot export job failed")
        raise

    # Finish the pipeline run and push metrics.
    finally:
        # Get the elapsed time.
        elapsed = time.perf_counter() - start
        success = status == "success"

        session = session_factory()
        try:
            # Finish the pipeline run by updating the pipeline_runs row in the database.
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

        # Push the snapshot metrics to the Pushgateway.
        if snapshot_id:
            # Push the snapshot metrics.
            push_snapshot_metrics(
                settings.pushgateway_url,
                settings.metrics_job_name,
                snapshot_id,
                success=success,
                duration_seconds=elapsed,
                table_row_counts=table_row_counts,
                upload_failures=upload_failures,
            )

        # Log the final status.
        logger.info(
            "Snapshot export job finished",
            extra={
                "run_id": run_id,
                "snapshot_id": snapshot_id,
                "status": status,
                "records_processed": stats_processed,
                "records_failed": stats_failed,
                "table_row_counts": table_row_counts,
                "duration_seconds": elapsed,
            },
        )


if __name__ == "__main__":
    main()

"""Batch job entrypoint for CF PyTorch training."""

from __future__ import annotations

import logging
import sys
import time
import uuid

from train_cf.train import run_cf_training
from common.storage.s3 import create_s3_client
from train_cf.version import resolve_cf_version
from common.db.session import get_session_factory
from common.config.settings import get_cf_training_settings
from common.metrics.cf_training import push_cf_training_metrics
from common.db.repositories.pipeline import finish_pipeline_run, start_pipeline_run


logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Set up basic structured logging for the CF training job."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )


def main() -> None:
    """
    Run the train_cf batch job.

    Do this by:
    1. Creating a pipeline_runs row.
    2. Training the CF model on the latest complete CF dataset.
    3. Writing manifest.json last and pushing metrics to Pushgateway.
    """
    # Configure logging.
    configure_logging()

    settings = get_cf_training_settings()
    run_id = str(uuid.uuid4())
    start = time.perf_counter()

    stats_processed = 0
    stats_failed = 0
    error_message: str | None = None
    status = "success"
    cf_version = resolve_cf_version(settings)
    best_validation_rmse: float | None = None

    session_factory = get_session_factory()
    session = session_factory()

    # Create a pipeline_runs row to indicate the start of the CF training pipeline run.
    try:
        start_pipeline_run(session, run_id, settings.job_name)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    # Create an S3 client to access the MinIO bucket.
    try:
        client = create_s3_client(settings)
        # Train the CF model.
        training_stats = run_cf_training(
            client,
            settings,
            pipeline_run_id=run_id,
        )
        # Update the CF version and best validation RMSE.
        cf_version = training_stats.cf_version
        best_validation_rmse = training_stats.best_validation_rmse
        stats_processed = 1

    except KeyboardInterrupt:
        status = "cancelled"
        error_message = "Interrupted by user (Ctrl+C)"
        logger.warning("CF training interrupted")
        raise

    except Exception as exc:
        status = "failure"
        error_message = str(exc)
        logger.exception("CF training job failed")
        raise

    # Update the pipeline_runs row to indicate the end of the CF training pipeline run.
    finally:
        elapsed = time.perf_counter() - start
        success = status == "success"

        session = session_factory()
        
        # Try to update the pipeline_runs row to indicate the end of the CF training pipeline run.
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

        # Push the CF training metrics to the Pushgateway.
        push_cf_training_metrics(
            settings.pushgateway_url,
            settings.metrics_job_name,
            cf_version,
            success=success,
            duration_seconds=elapsed,
            best_validation_rmse=best_validation_rmse,
        )

        # Log the end of the CF training job.
        logger.info(
            "CF training job finished",
            extra={
                "run_id": run_id,
                "cf_version": cf_version,
                "status": status,
                "records_processed": stats_processed,
                "records_failed": stats_failed,
                "best_validation_rmse": best_validation_rmse,
                "duration_seconds": elapsed,
            },
        )


if __name__ == "__main__":
    main()

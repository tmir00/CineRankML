"""Batch job entrypoint for hybrid ranker PyTorch training."""

from __future__ import annotations

import sys
import time
import uuid
import logging

from common.storage.s3 import create_s3_client
from common.db.session import get_session_factory
from train_hybrid_ranker.train import run_hybrid_training
from train_hybrid_ranker.version import resolve_model_version
from common.config.settings import get_hybrid_training_settings
from common.metrics.hybrid_training import push_hybrid_training_metrics
from common.db.repositories.pipeline import finish_pipeline_run, start_pipeline_run


logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Set up basic structured logging for the hybrid training job."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )


def main() -> None:
    """
    Run the train_hybrid_ranker batch job.

    Do this by:
    1. Creating a pipeline_runs row.
    2. Training the hybrid ranker on the latest complete feature dataset.
    3. Uploading model artifacts and pushing metrics to Pushgateway.
    """
    configure_logging()

    # Initialize the settings and run ID and start timer.
    settings = get_hybrid_training_settings()
    run_id = str(uuid.uuid4())
    start = time.perf_counter()

    logger.info(
        "Hybrid training job starting run_id=%s job_name=%s model_version=%s",
        run_id,
        settings.job_name,
        resolve_model_version(settings),
    )

    stats_processed = 0
    stats_failed = 0
    error_message: str | None = None
    status = "success"
    model_version = resolve_model_version(settings)
    best_validation_rmse: float | None = None

    session_factory = get_session_factory()
    session = session_factory()

    # Add a row to pipeline_runs to indicate the start of the training.
    try:
        start_pipeline_run(session, run_id, settings.job_name)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    # Try to train the hybrid ranker.
    try:
        client = create_s3_client(settings)
        # Train the hybrid ranker.
        training_stats = run_hybrid_training(
            client,
            settings,
            pipeline_run_id=run_id,
        )
        model_version = training_stats.model_version
        best_validation_rmse = training_stats.best_validation_rmse
        stats_processed = 1

    except KeyboardInterrupt:
        status = "cancelled"
        error_message = "Interrupted by user (Ctrl+C)"
        logger.warning("Hybrid training interrupted")
        raise

    except Exception as exc:
        status = "failure"
        error_message = str(exc)
        logger.exception("Hybrid training job failed")
        raise

    # Finish the pipeline run and push metrics to the Pushgateway once the training is complete.
    finally:
        elapsed = time.perf_counter() - start
        success = status == "success"

        session = session_factory()
        try:
            # Finish the pipeline run by updating the pipeline_runs row.
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

        # Push the hybrid training metrics to the Pushgateway.
        push_hybrid_training_metrics(
            settings.pushgateway_url,
            settings.metrics_job_name,
            model_version,
            success=success,
            duration_seconds=elapsed,
            best_validation_rmse=best_validation_rmse,
        )

        logger.info(
            "Hybrid training job finished run_id=%s model_version=%s status=%s "
            "best_validation_rmse=%s duration_seconds=%.1f",
            run_id,
            model_version,
            status,
            best_validation_rmse,
            elapsed,
        )


if __name__ == "__main__":
    main()

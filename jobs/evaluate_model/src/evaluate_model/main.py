"""Batch job entrypoint for hybrid ranker evaluation."""

from __future__ import annotations

import sys
import time
import uuid
import logging

from common.storage.s3 import create_s3_client
from common.db.session import get_session_factory
from evaluate_model.evaluate import run_hybrid_evaluation
from common.config.settings import get_evaluate_model_settings
from common.metrics.hybrid_evaluation import push_hybrid_evaluation_metrics
from common.db.repositories.pipeline import finish_pipeline_run, start_pipeline_run


logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Set up basic structured logging for the evaluation job."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )


def main() -> None:
    """
    Run the evaluate_model batch job.

    Do this by:
    1. Creating a pipeline_runs row.
    2. Loading a trained hybrid ranker and evaluating on the test split.
    3. Writing manifest.json last and pushing metrics to Pushgateway.
    """
    configure_logging()

    # Fetch the evaluate model settings and initialize the run id and start time.
    settings = get_evaluate_model_settings()
    run_id = str(uuid.uuid4())
    start = time.perf_counter()

    stats_processed = 0
    stats_failed = 0
    error_message: str | None = None
    status = "success"
    model_version = settings.model_version or "unknown"
    test_rmse: float | None = None

    session_factory = get_session_factory()
    session = session_factory()

    # Insert a pipeline_runs row to indicate the start of the evaluation job.
    try:
        start_pipeline_run(session, run_id, settings.job_name)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    # Try to run the hybrid model evaluation.
    try:
        client = create_s3_client(settings)
        evaluation_stats = run_hybrid_evaluation(
            client,
            settings,
            pipeline_run_id=run_id,
        )
        model_version = evaluation_stats.model_version
        test_rmse = evaluation_stats.test_rmse
        stats_processed = 1

    except KeyboardInterrupt:
        status = "cancelled"
        error_message = "Interrupted by user (Ctrl+C)"
        logger.warning("Hybrid evaluation interrupted")
        raise

    except Exception as exc:
        status = "failure"
        error_message = str(exc)
        logger.exception("Hybrid evaluation job failed")
        raise

    # Once the evaluation is complete, mark the job as finished and push the metrics to the Pushgateway.
    finally:
        elapsed = time.perf_counter() - start
        success = status == "success"

        session = session_factory()
        try:
            # Update the pipeline_runs row to indicate the end of the evaluation job.
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

        # Push the hybrid evaluation metrics to the Pushgateway.
        push_hybrid_evaluation_metrics(
            settings.pushgateway_url,
            settings.metrics_job_name,
            model_version,
            success=success,
            duration_seconds=elapsed,
            test_rmse=test_rmse,
        )

        logger.info(
            "Hybrid evaluation job finished",
            extra={
                "run_id": run_id,
                "model_version": model_version,
                "status": status,
                "records_processed": stats_processed,
                "records_failed": stats_failed,
                "test_rmse": test_rmse,
                "duration_seconds": elapsed,
            },
        )


if __name__ == "__main__":
    main()

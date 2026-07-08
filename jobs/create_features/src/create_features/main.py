"""Batch job entrypoint for hybrid ranker feature generation."""

from __future__ import annotations

import logging
import sys
import time
import uuid

from common.config.settings import get_create_features_settings
from common.db.repositories.pipeline import finish_pipeline_run, start_pipeline_run
from common.db.session import get_session_factory
from common.metrics.hybrid_features import push_hybrid_features_metrics
from common.storage.s3 import create_s3_client
from create_features.prep import run_hybrid_feature_generation
from create_features.version import resolve_dataset_version

logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    """Set up basic structured logging for the create_features job."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    for noisy_logger in ("botocore", "boto3", "urllib3"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def main() -> None:
    """
    Run the create_features batch job.

    Do this by:
    1. Creating a pipeline_runs row.
    2. Building versioned hybrid feature datasets from frozen snapshot and CF artifacts.
    3. Writing manifest.json last and pushing metrics to Pushgateway.
    """
    # Configure logging and load settings.
    settings = get_create_features_settings()
    configure_logging(settings.log_level)
    run_id = str(uuid.uuid4())
    start = time.perf_counter()

    # Initialize counters and error message.
    stats_processed = 0
    stats_failed = 0
    error_message: str | None = None
    status = "success"
    dataset_version = resolve_dataset_version(settings)
    train_row_count = 0
    validation_row_count = 0
    test_row_count = 0
    quality_stats = None

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

    logger.info(
        "create_features job started run_id=%s dataset_version=%s",
        run_id,
        dataset_version,
    )

    # Run hybrid feature generation.
    try:
        client = create_s3_client(settings)
        prep_stats = run_hybrid_feature_generation(
            client,
            settings,
            pipeline_run_id=run_id,
        )

        dataset_version = prep_stats.dataset_version
        train_row_count = prep_stats.train_row_count
        validation_row_count = prep_stats.validation_row_count
        test_row_count = prep_stats.test_row_count
        quality_stats = prep_stats.quality
        stats_processed = train_row_count + validation_row_count + test_row_count

    except KeyboardInterrupt:
        status = "cancelled"
        error_message = "Interrupted by user (Ctrl+C)"
        logger.warning("Hybrid feature generation interrupted")
        raise

    except Exception as exc:
        status = "failure"
        error_message = str(exc)
        logger.exception("create_features job failed")
        raise

    finally:
        elapsed = time.perf_counter() - start
        success = status == "success"

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

        push_hybrid_features_metrics(
            settings.pushgateway_url,
            settings.metrics_job_name,
            dataset_version,
            success=success,
            duration_seconds=elapsed,
            train_row_count=train_row_count,
            validation_row_count=validation_row_count,
            test_row_count=test_row_count,
            quality=quality_stats,
        )

        logger.info(
            "create_features job finished run_id=%s dataset_version=%s status=%s "
            "records_processed=%s train_row_count=%s validation_row_count=%s "
            "test_row_count=%s duration_seconds=%.1f",
            run_id,
            dataset_version,
            status,
            stats_processed,
            train_row_count,
            validation_row_count,
            test_row_count,
            elapsed,
        )


if __name__ == "__main__":
    main()

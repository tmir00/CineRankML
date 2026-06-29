"""Prometheus metrics for the prepare_cf_dataset batch job."""

from __future__ import annotations

import logging

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

logger = logging.getLogger(__name__)


def push_cf_dataset_metrics(gateway_url: str, metrics_job_name: str, cf_dataset_version: str,
                                *, success: bool, duration_seconds: float, train_row_count: int,
                                    holdout_row_count: int) -> None:
    """
    Push final CF dataset prep metrics to Pushgateway.

    Do this by:
    1. Creating a fresh CollectorRegistry for this job run.
    2. Setting gauges for success, duration, train/holdout row counts, and status.
    3. Pushing to Pushgateway grouped by cf_dataset_version.

    Metrics Pushed:
    1. cf_dataset_job_success: Whether the CF dataset prep job completed successfully (1) or failed (0).
    2. cf_dataset_job_duration_seconds: Wall-clock duration of the CF dataset prep job in seconds.
    3. cf_dataset_train_rows_total: Total rows written to the CF train split.
    4. cf_dataset_holdout_rows_total: Total rows written to the CF holdout split.
    5. cf_dataset_status: CF dataset completeness (1=complete manifest, 0=failed or incomplete).

    ============================ Arguments ============================
    gateway_url: Pushgateway base URL.
    metrics_job_name: Value for the job_name label.
    cf_dataset_version: Dataset version identifier for grouping and labels.
    success: Whether prep completed with a complete manifest.
    duration_seconds: Wall-clock job duration.
    train_row_count: Number of rows written to train/.
    holdout_row_count: Number of rows written to holdout/.
    """
    registry = CollectorRegistry()
    labels = ["job_name", "cf_dataset_version"]
    success_value = 1 if success else 0

    job_success = Gauge(
        "cf_dataset_job_success",
        "Whether the CF dataset prep job completed successfully (1) or failed (0)",
        labels,
        registry=registry,
    )
    job_duration = Gauge(
        "cf_dataset_job_duration_seconds",
        "Wall-clock duration of the CF dataset prep job in seconds",
        labels,
        registry=registry,
    )
    train_rows = Gauge(
        "cf_dataset_train_rows_total",
        "Total rows written to the CF train split",
        labels,
        registry=registry,
    )
    holdout_rows = Gauge(
        "cf_dataset_holdout_rows_total",
        "Total rows written to the CF holdout split",
        labels,
        registry=registry,
    )
    dataset_status = Gauge(
        "cf_dataset_status",
        "CF dataset completeness (1=complete manifest, 0=failed or incomplete)",
        labels,
        registry=registry,
    )

    job_success.labels(
        job_name=metrics_job_name,
        cf_dataset_version=cf_dataset_version,
    ).set(success_value)
    job_duration.labels(
        job_name=metrics_job_name,
        cf_dataset_version=cf_dataset_version,
    ).set(duration_seconds)
    train_rows.labels(
        job_name=metrics_job_name,
        cf_dataset_version=cf_dataset_version,
    ).set(train_row_count)
    holdout_rows.labels(
        job_name=metrics_job_name,
        cf_dataset_version=cf_dataset_version,
    ).set(holdout_row_count)
    dataset_status.labels(
        job_name=metrics_job_name,
        cf_dataset_version=cf_dataset_version,
    ).set(success_value)

    try:
        push_to_gateway(
            gateway_url,
            job=metrics_job_name,
            registry=registry,
            grouping_key={"cf_dataset_version": cf_dataset_version},
        )
    except Exception:
        logger.warning(
            "Failed to push CF dataset metrics to Pushgateway",
            extra={
                "gateway_url": gateway_url,
                "cf_dataset_version": cf_dataset_version,
            },
            exc_info=True,
        )

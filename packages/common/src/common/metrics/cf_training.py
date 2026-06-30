"""Prometheus metrics for the train_cf batch job."""

from __future__ import annotations

import logging

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

logger = logging.getLogger(__name__)


def push_cf_training_metrics(gateway_url: str, metrics_job_name: str, cf_version: str, *, success: bool,
                                duration_seconds: float, best_validation_rmse: float | None = None) -> None:
    """
    Push final CF training metrics to Pushgateway.

    Do this by:
    1. Creating a fresh CollectorRegistry for this job run.
    2. Setting gauges for success, duration, best validation RMSE, and status.
    3. Pushing to Pushgateway grouped by cf_version.

    ============================ Arguments ============================
    gateway_url: Pushgateway base URL.
    metrics_job_name: Value for the job_name label.
    cf_version: CF artifact version identifier for grouping and labels.
    success: Whether training completed with a complete manifest.
    duration_seconds: Wall-clock job duration.
    best_validation_rmse: Best validation RMSE across epochs, when available.
    """
    registry = CollectorRegistry()
    labels = ["job_name", "cf_version"]
    success_value = 1 if success else 0

    job_success = Gauge(
        "cf_training_job_success",
        "Whether the CF training job completed successfully (1) or failed (0)",
        labels,
        registry=registry,
    )
    job_duration = Gauge(
        "cf_training_job_duration_seconds",
        "Wall-clock duration of the CF training job in seconds",
        labels,
        registry=registry,
    )
    best_rmse = Gauge(
        "cf_training_best_validation_rmse",
        "Best validation RMSE achieved during CF training",
        labels,
        registry=registry,
    )
    training_status = Gauge(
        "cf_training_status",
        "CF artifact completeness (1=complete manifest, 0=failed or incomplete)",
        labels,
        registry=registry,
    )

    label_values = {"job_name": metrics_job_name, "cf_version": cf_version}
    job_success.labels(**label_values).set(success_value)
    job_duration.labels(**label_values).set(duration_seconds)
    if best_validation_rmse is not None:
        best_rmse.labels(**label_values).set(best_validation_rmse)
    training_status.labels(**label_values).set(success_value)

    try:
        push_to_gateway(
            gateway_url,
            job=metrics_job_name,
            registry=registry,
            grouping_key={"cf_version": cf_version},
        )
    except Exception:
        logger.warning(
            "Failed to push CF training metrics to Pushgateway",
            extra={
                "gateway_url": gateway_url,
                "cf_version": cf_version,
            },
            exc_info=True,
        )

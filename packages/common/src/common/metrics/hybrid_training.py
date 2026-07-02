"""Prometheus metrics for the train_hybrid_ranker batch job."""

from __future__ import annotations

import logging

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

logger = logging.getLogger(__name__)


def push_hybrid_training_metrics(gateway_url: str, metrics_job_name: str, model_version: str, *, \
                                    success: bool, duration_seconds: float, best_validation_rmse: float | None = None) -> None:
    """
    Push final hybrid training metrics to Pushgateway.

    Do this by:
    1. Creating a fresh CollectorRegistry for this job run.
    2. Setting gauges for success, duration, best validation RMSE, and status.
    3. Pushing to Pushgateway grouped by model_version.

    ============================ Arguments ============================
    gateway_url: Pushgateway base URL.
    metrics_job_name: Value for the job_name label.
    model_version: Model version identifier for grouping and labels.
    success: Whether training completed with uploaded artifacts.
    duration_seconds: Wall-clock job duration.
    best_validation_rmse: Best validation RMSE across epochs, when available.
    """
    registry = CollectorRegistry()
    labels = ["job_name", "model_version"]
    success_value = 1 if success else 0

    job_success = Gauge(
        "hybrid_training_job_success",
        "Whether the hybrid training job completed successfully (1) or failed (0)",
        labels,
        registry=registry,
    )
    job_duration = Gauge(
        "hybrid_training_job_duration_seconds",
        "Wall-clock duration of the hybrid training job in seconds",
        labels,
        registry=registry,
    )
    best_rmse = Gauge(
        "hybrid_training_best_validation_rmse",
        "Best validation RMSE achieved during hybrid training",
        labels,
        registry=registry,
    )
    training_status = Gauge(
        "hybrid_training_status",
        "Hybrid model artifact completeness (1=uploaded, 0=failed or incomplete)",
        labels,
        registry=registry,
    )

    label_values = {"job_name": metrics_job_name, "model_version": model_version}
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
            grouping_key={"model_version": model_version},
        )
    except Exception:
        logger.warning(
            "Failed to push hybrid training metrics to Pushgateway",
            extra={
                "gateway_url": gateway_url,
                "model_version": model_version,
            },
            exc_info=True,
        )

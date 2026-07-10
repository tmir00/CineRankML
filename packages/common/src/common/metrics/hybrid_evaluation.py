"""Prometheus metrics for the evaluate_model batch job."""

from __future__ import annotations

import logging

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

logger = logging.getLogger(__name__)


def push_hybrid_evaluation_metrics(gateway_url: str, metrics_job_name: str, model_version: str, *, \
                                    success: bool, duration_seconds: float, test_rmse: float | None = None) -> None:
    """
    Push final hybrid evaluation metrics to Pushgateway.

    Do this by:
    1. Creating a fresh CollectorRegistry for this job run.
    2. Setting gauges for success, duration, test RMSE, and status.
    3. Pushing to Pushgateway grouped by model_version.

    ============================ Arguments ============================
    gateway_url: Pushgateway base URL.
    metrics_job_name: Value for the job_name label.
    model_version: Model version identifier for grouping and labels.
    success: Whether evaluation completed with a complete manifest.
    duration_seconds: Wall-clock job duration.
    test_rmse: Test RMSE from the evaluation run, when available.
    """
    # Create a fresh CollectorRegistry for this job run.
    # This is the set of metrics I want to push to the Pushgateway for this job run.
    registry = CollectorRegistry()
    labels = ["job_name", "model_version"]
    success_value = 1 if success else 0

    job_success = Gauge(
        "hybrid_evaluation_job_success",
        "Whether the hybrid evaluation job completed successfully (1) or failed (0)",
        labels,
        registry=registry,
    )
    job_duration = Gauge(
        "hybrid_evaluation_job_duration_seconds",
        "Wall-clock duration of the hybrid evaluation job in seconds",
        labels,
        registry=registry,
    )
    test_rmse_gauge = Gauge(
        "hybrid_evaluation_test_rmse",
        "Test RMSE from the hybrid evaluation job",
        labels,
        registry=registry,
    )
    evaluation_status = Gauge(
        "hybrid_evaluation_status",
        "Hybrid model evaluation completeness (1=complete manifest, 0=failed)",
        labels,
        registry=registry,
    )

    label_values = {"job_name": metrics_job_name, "model_version": model_version}
    job_success.labels(**label_values).set(success_value)
    job_duration.labels(**label_values).set(duration_seconds)
    if test_rmse is not None:
        test_rmse_gauge.labels(**label_values).set(test_rmse)
    evaluation_status.labels(**label_values).set(success_value)

    try:
        push_to_gateway(
            gateway_url,
            job=metrics_job_name,
            registry=registry,
            grouping_key={"model_version": model_version},
        )
    except Exception:
        logger.warning(
            "Failed to push hybrid evaluation metrics to Pushgateway",
            extra={
                "gateway_url": gateway_url,
                "model_version": model_version,
            },
            exc_info=True,
        )

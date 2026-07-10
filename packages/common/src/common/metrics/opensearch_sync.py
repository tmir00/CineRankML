"""Prometheus metrics for the opensearch_sync batch job."""

from __future__ import annotations

import logging

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

logger = logging.getLogger(__name__)


def push_opensearch_sync_metrics(
    gateway_url: str,
    metrics_job_name: str,
    *,
    success: bool,
    duration_seconds: float,
    records_processed: int,
    records_failed: int,
) -> None:
    """
    Push final OpenSearch sync job metrics to Pushgateway.

    Do this by:
    1. Creating a fresh CollectorRegistry for this one-off push.
    2. Setting gauges for success, duration, processed rows, and failed rows.
    3. Calling push_to_gateway with the configured job name label.

    ============================ Arguments ============================
    gateway_url: Pushgateway base URL.
    metrics_job_name: Value for the job_name label (e.g. opensearch_sync).
    success: Whether the sync completed without fatal errors.
    duration_seconds: Wall-clock job duration.
    records_processed: Movies successfully synced.
    records_failed: Movies that failed to sync.
    """
    registry = CollectorRegistry()
    labels = ["job_name"]

    job_success = Gauge(
        "opensearch_sync_job_success",
        "Whether the OpenSearch sync job completed successfully (1) or failed (0)",
        labels,
        registry=registry,
    )
    job_duration = Gauge(
        "opensearch_sync_job_duration_seconds",
        "Wall-clock duration of the OpenSearch sync job in seconds",
        labels,
        registry=registry,
    )
    records_processed_total = Gauge(
        "opensearch_sync_records_processed_total",
        "Movies successfully synced to OpenSearch",
        labels,
        registry=registry,
    )
    records_failed_total = Gauge(
        "opensearch_sync_records_failed_total",
        "Movies that failed to sync to OpenSearch",
        labels,
        registry=registry,
    )

    success_value = 1 if success else 0
    job_success.labels(job_name=metrics_job_name).set(success_value)
    job_duration.labels(job_name=metrics_job_name).set(duration_seconds)
    records_processed_total.labels(job_name=metrics_job_name).set(records_processed)
    records_failed_total.labels(job_name=metrics_job_name).set(records_failed)

    try:
        push_to_gateway(gateway_url, job=metrics_job_name, registry=registry)
    except Exception:
        logger.exception("Failed to push OpenSearch sync metrics to Pushgateway")

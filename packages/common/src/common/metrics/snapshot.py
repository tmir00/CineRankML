"""Prometheus metrics for the snapshot_to_s3 batch job."""

from __future__ import annotations

import logging

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

logger = logging.getLogger(__name__)


def push_snapshot_metrics(gateway_url: str, metrics_job_name: str, snapshot_id: str, *, success: bool, duration_seconds: float,
                            table_row_counts: dict[str, int], upload_failures: int) -> None:
    """
    Push final snapshot job metrics to Pushgateway.

    Do this by:
    1. Creating a fresh CollectorRegistry (an empty container for metrics).
    2. Setting gauges for success, duration, per-table row counts, upload failures, and status.
    3. Calling push_to_gateway with grouping_key snapshot_id (grouping metrics by snapshot_id).

    ============================ Arguments ============================
    gateway_url: Pushgateway base URL.
    metrics_job_name: Value for the job_name label (e.g. s3_snapshot).
    snapshot_id: Snapshot identifier for metric grouping and labels.
    success: Whether the export completed with a complete manifest.
    duration_seconds: Wall-clock job duration.
    table_row_counts: Final exported row count per table name.
    upload_failures: Number of failed S3 uploads during the run.
    """

    registry = CollectorRegistry()

    # Define the labels for the metrics.
    labels = ["job_name", "snapshot_id"]

    # Define the gauges for the metrics.
    job_success = Gauge(
        "snapshot_job_success",
        "Whether the snapshot job completed successfully (1) or failed (0)",
        labels,
        registry=registry,
    )

    # Define the gauge for the job duration.
    job_duration = Gauge(
        "snapshot_job_duration_seconds",
        "Wall-clock duration of the snapshot job in seconds",
        labels,
        registry=registry,
    )

    # Define the gauge for the rows exported.
    rows_exported = Gauge(
        "snapshot_rows_exported_total",
        "Total rows exported per table in the snapshot",
        ["job_name", "table", "snapshot_id"],
        registry=registry,
    )

    # Define the gauge for the upload failures.
    upload_failures_total = Gauge(
        "snapshot_upload_failures_total",
        "Total S3 upload failures during the snapshot job",
        labels,
        registry=registry,
    )

    # Define the gauge for the snapshot status.
    snapshot_status = Gauge(
        "snapshot_status",
        "Snapshot completeness (1=complete manifest, 0=failed or incomplete)",
        labels,
        registry=registry,
    )

    # Set the values for the gauges.
    success_value = 1 if success else 0
    job_success.labels(job_name=metrics_job_name, snapshot_id=snapshot_id).set(success_value)
    job_duration.labels(job_name=metrics_job_name, snapshot_id=snapshot_id).set(duration_seconds)
    upload_failures_total.labels(job_name=metrics_job_name, snapshot_id=snapshot_id).set(
        upload_failures
    )
    snapshot_status.labels(job_name=metrics_job_name, snapshot_id=snapshot_id).set(success_value)

    # Set the values for the rows exported gauge.
    for table_name, row_count in table_row_counts.items():
        rows_exported.labels(
            job_name=metrics_job_name,
            table=table_name,
            snapshot_id=snapshot_id,
        ).set(row_count)

    # Push the metrics to the Pushgateway.
    try:
        push_to_gateway(
            gateway_url,
            job=metrics_job_name,
            registry=registry,
            grouping_key={"snapshot_id": snapshot_id},
        )
    except Exception:
        logger.warning(
            "Failed to push snapshot metrics to Pushgateway",
            extra={"gateway_url": gateway_url, "snapshot_id": snapshot_id},
            exc_info=True,
        )

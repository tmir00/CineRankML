"""Prometheus metrics for the create_features batch job."""

from __future__ import annotations

import logging

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from common.metrics.hybrid_feature_stats import HybridFeatureQualityStats

logger = logging.getLogger(__name__)


def push_hybrid_features_metrics(
    gateway_url: str,
    metrics_job_name: str,
    dataset_version: str,
    *,
    success: bool,
    duration_seconds: float,
    train_row_count: int,
    validation_row_count: int,
    test_row_count: int,
    quality: HybridFeatureQualityStats | None = None,
) -> None:
    """
    Push final hybrid feature generation metrics to the Pushgateway.

    Do this by:
    1. Creating a fresh CollectorRegistry for this job run.
    2. Setting gauges for success, duration, split row counts, and status.
    3. Setting data-quality gauges when the job succeeded and quality stats are available.
    4. Pushing to Pushgateway grouped by dataset_version.

    ============================ Arguments ============================
    gateway_url: Pushgateway base URL.
    metrics_job_name: Value for the job_name label.
    dataset_version: Dataset version identifier for grouping and labels.
    success: Whether feature generation completed with a complete manifest.
    duration_seconds: Wall-clock job duration.
    train_row_count: Number of rows written to train/.
    validation_row_count: Number of rows written to validation/.
    test_row_count: Number of rows written to test/.
    quality: Optional data-quality counters from a successful prep run.
    """
    registry = CollectorRegistry()
    labels = ["job_name", "dataset_version"]
    success_value = 1 if success else 0

    job_success = Gauge(
        "hybrid_features_job_success",
        "Whether the create_features job completed successfully (1) or failed (0)",
        labels,
        registry=registry,
    )
    job_duration = Gauge(
        "hybrid_features_job_duration_seconds",
        "Wall-clock duration of the create_features job in seconds",
        labels,
        registry=registry,
    )
    train_rows = Gauge(
        "hybrid_features_train_rows_total",
        "Total rows written to the hybrid train split",
        labels,
        registry=registry,
    )
    validation_rows = Gauge(
        "hybrid_features_validation_rows_total",
        "Total rows written to the hybrid validation split",
        labels,
        registry=registry,
    )
    test_rows = Gauge(
        "hybrid_features_test_rows_total",
        "Total rows written to the hybrid test split",
        labels,
        registry=registry,
    )
    dataset_status = Gauge(
        "hybrid_features_dataset_status",
        "Hybrid dataset completeness (1=complete manifest, 0=failed or incomplete)",
        labels,
        registry=registry,
    )

    label_values = {"job_name": metrics_job_name, "dataset_version": dataset_version}
    job_success.labels(**label_values).set(success_value)
    job_duration.labels(**label_values).set(duration_seconds)
    train_rows.labels(**label_values).set(train_row_count)
    validation_rows.labels(**label_values).set(validation_row_count)
    test_rows.labels(**label_values).set(test_row_count)
    dataset_status.labels(**label_values).set(success_value)

    # Push data-quality gauges only for successful runs with collected stats.
    if success and quality is not None:
        total_rows = train_row_count + validation_row_count + test_row_count

        total_rows_gauge = Gauge(
            "hybrid_features_total_rows_total",
            "Total rows written across hybrid train, validation, and test splits",
            labels,
            registry=registry,
        )
        row_count_delta = Gauge(
            "hybrid_features_row_count_delta_vs_cf",
            "Absolute difference between hybrid output rows and upstream CF dataset rows",
            labels,
            registry=registry,
        )
        cold_start_rows = Gauge(
            "hybrid_features_cold_start_rows_total",
            "Hybrid rows where the user had no prior rating history",
            labels,
            registry=registry,
        )
        cold_start_fraction = Gauge(
            "hybrid_features_cold_start_fraction",
            "Fraction of hybrid rows with no prior user rating history",
            labels,
            registry=registry,
        )
        users_processed = Gauge(
            "hybrid_features_users_processed_total",
            "Distinct users processed during hybrid feature generation",
            labels,
            registry=registry,
        )
        train_parts = Gauge(
            "hybrid_features_train_parts_total",
            "Number of Parquet parts written to hybrid train/",
            labels,
            registry=registry,
        )
        validation_parts = Gauge(
            "hybrid_features_validation_parts_total",
            "Number of Parquet parts written to hybrid validation/",
            labels,
            registry=registry,
        )
        test_parts = Gauge(
            "hybrid_features_test_parts_total",
            "Number of Parquet parts written to hybrid test/",
            labels,
            registry=registry,
        )
        missing_content = Gauge(
            "hybrid_features_missing_content_embedding_rows_total",
            "Hybrid candidate rows with no content embedding lookup",
            labels,
            registry=registry,
        )
        missing_cf = Gauge(
            "hybrid_features_missing_cf_embedding_rows_total",
            "Hybrid candidate rows with no CF embedding lookup",
            labels,
            registry=registry,
        )
        join_dropped = Gauge(
            "hybrid_features_join_dropped_candidates_total",
            "CF candidate rows that failed to join back to ratings_events",
            labels,
            registry=registry,
        )

        total_rows_gauge.labels(**label_values).set(total_rows)
        row_count_delta.labels(**label_values).set(quality.row_count_delta_vs_cf)
        cold_start_rows.labels(**label_values).set(quality.cold_start_rows)
        cold_start_fraction.labels(**label_values).set(quality.cold_start_fraction)
        users_processed.labels(**label_values).set(quality.users_processed)
        train_parts.labels(**label_values).set(quality.train_parts)
        validation_parts.labels(**label_values).set(quality.validation_parts)
        test_parts.labels(**label_values).set(quality.test_parts)
        missing_content.labels(**label_values).set(quality.missing_content_embedding_rows)
        missing_cf.labels(**label_values).set(quality.missing_cf_embedding_rows)
        join_dropped.labels(**label_values).set(quality.join_dropped_candidates)

    try:
        push_to_gateway(
            gateway_url,
            job=metrics_job_name,
            registry=registry,
            grouping_key={"dataset_version": dataset_version},
        )
    except Exception:
        logger.warning(
            "Failed to push hybrid features metrics to Pushgateway",
            extra={
                "gateway_url": gateway_url,
                "dataset_version": dataset_version,
            },
            exc_info=True,
        )

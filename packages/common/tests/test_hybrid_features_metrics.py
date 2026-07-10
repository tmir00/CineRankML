"""Tests for hybrid features Prometheus push helpers."""

from __future__ import annotations

from common.metrics.hybrid_feature_stats import HybridFeatureQualityStats, compute_row_count_delta
from common.metrics.hybrid_features import push_hybrid_features_metrics


def _sample_quality() -> HybridFeatureQualityStats:
    return HybridFeatureQualityStats(
        cf_expected_row_count=100,
        row_count_delta_vs_cf=0,
        cold_start_rows=5,
        cold_start_fraction=0.05,
        users_processed=20,
        train_parts=1,
        validation_parts=1,
        test_parts=1,
        missing_content_embedding_rows=2,
        missing_cf_embedding_rows=1,
        join_dropped_candidates=0,
    )


def _collect_metric_values(registry) -> dict[str, float]:
    """Collect sample values from a Prometheus registry."""
    values: dict[str, float] = {}
    for metric in registry.collect():
        for sample in metric.samples:
            values[sample.name] = sample.value
    return values


def test_compute_row_count_delta_returns_absolute_difference() -> None:
    """Row parity delta should be the absolute difference between totals."""
    assert compute_row_count_delta(100, 100) == 0
    assert compute_row_count_delta(95, 100) == 5
    assert compute_row_count_delta(105, 100) == 5


def test_push_hybrid_features_metrics_includes_quality_gauges_on_success(
    monkeypatch,
) -> None:
    """Successful runs with quality stats should push Tier 1 and Tier 2 gauges."""
    captured: dict[str, object] = {}

    def _fake_push_to_gateway(_gateway_url, *, job, registry, grouping_key=None) -> None:
        captured["job"] = job
        captured["grouping_key"] = grouping_key
        captured["metrics"] = _collect_metric_values(registry)

    monkeypatch.setattr(
        "common.metrics.hybrid_features.push_to_gateway",
        _fake_push_to_gateway,
    )

    push_hybrid_features_metrics(
        "http://pushgateway:9091",
        "create_features",
        "2026-06-25T121000Z",
        success=True,
        duration_seconds=12.5,
        train_row_count=80,
        validation_row_count=10,
        test_row_count=10,
        quality=_sample_quality(),
    )

    metrics = captured["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["hybrid_features_job_success"] == 1.0
    assert metrics["hybrid_features_total_rows_total"] == 100.0
    assert metrics["hybrid_features_row_count_delta_vs_cf"] == 0.0
    assert metrics["hybrid_features_cold_start_fraction"] == 0.05
    assert metrics["hybrid_features_users_processed_total"] == 20.0
    assert metrics["hybrid_features_train_parts_total"] == 1.0
    assert metrics["hybrid_features_missing_content_embedding_rows_total"] == 2.0
    assert metrics["hybrid_features_missing_cf_embedding_rows_total"] == 1.0
    assert metrics["hybrid_features_join_dropped_candidates_total"] == 0.0


def test_push_hybrid_features_metrics_skips_quality_gauges_on_failure(
    monkeypatch,
) -> None:
    """Failed runs should keep base gauges but omit data-quality gauges."""
    captured: dict[str, object] = {}

    def _fake_push_to_gateway(_gateway_url, *, job, registry, grouping_key=None) -> None:
        captured["metrics"] = _collect_metric_values(registry)

    monkeypatch.setattr(
        "common.metrics.hybrid_features.push_to_gateway",
        _fake_push_to_gateway,
    )

    push_hybrid_features_metrics(
        "http://pushgateway:9091",
        "create_features",
        "2026-06-25T121000Z",
        success=False,
        duration_seconds=3.0,
        train_row_count=0,
        validation_row_count=0,
        test_row_count=0,
        quality=_sample_quality(),
    )

    metrics = captured["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["hybrid_features_job_success"] == 0.0
    assert "hybrid_features_total_rows_total" not in metrics
    assert "hybrid_features_cold_start_fraction" not in metrics
    assert "hybrid_features_join_dropped_candidates_total" not in metrics

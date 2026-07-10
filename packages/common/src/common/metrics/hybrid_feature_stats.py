"""Quality counters collected during hybrid feature generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HybridFeatureQualityStats:
    """Data-quality counters for one successful create_features run."""

    cf_expected_row_count: int
    row_count_delta_vs_cf: int
    cold_start_rows: int
    cold_start_fraction: float
    users_processed: int
    train_parts: int
    validation_parts: int
    test_parts: int
    missing_content_embedding_rows: int
    missing_cf_embedding_rows: int
    join_dropped_candidates: int


def compute_row_count_delta(hybrid_total: int, cf_total: int) -> int:
    """
    Return the absolute difference between hybrid and CF dataset row counts.

    ============================ Arguments ============================
    hybrid_total: Total rows written to hybrid train/validation/test.
    cf_total: Total rows in the upstream CF dataset manifest.

    ============================ Returns ============================
    Non-negative row-count delta.
    """
    return abs(hybrid_total - cf_total)

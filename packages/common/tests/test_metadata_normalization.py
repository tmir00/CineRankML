"""Tests for metadata normalization helpers."""

from __future__ import annotations

from common.features.normalization import MetadataNormalizationStats, min_max_norm, normalize_candidate_metadata


def test_min_max_norm_scales_into_unit_interval() -> None:
    """Values between min and max should map into [0, 1]."""
    assert min_max_norm(5.0, 0.0, 10.0) == 0.5
    assert min_max_norm(None, 0.0, 10.0) == 0.0
    assert min_max_norm(5.0, 5.0, 5.0) == 0.0


def test_normalize_candidate_metadata_uses_train_fit_stats() -> None:
    """Normalized metadata should use the manifest train-fit bounds."""
    stats = MetadataNormalizationStats(
        year_min=1990.0,
        year_max=2000.0,
        runtime_min=80.0,
        runtime_max=180.0,
        tmdb_popularity_log_min=0.0,
        tmdb_popularity_log_max=2.0,
        tmdb_vote_average_min=0.0,
        tmdb_vote_average_max=10.0,
        tmdb_vote_count_log_min=0.0,
        tmdb_vote_count_log_max=10.0,
    )

    normalized = normalize_candidate_metadata(
        year=1995.0,
        runtime=130.0,
        tmdb_popularity=1.0,
        tmdb_vote_average=8.0,
        tmdb_vote_count=100.0,
        stats=stats,
    )

    assert normalized[0] == 0.5
    assert normalized[1] == 0.5
    assert len(normalized) == 5

"""Tests for hybrid ranker dataset manifest schema."""

from __future__ import annotations

from datetime import UTC, datetime

from common.features.normalization import MetadataNormalizationStats
from common.schemas.hybrid_ranker_dataset_manifest import HybridRankerDatasetManifest, HybridRankerPartEntry


def test_hybrid_ranker_manifest_accepts_required_fields() -> None:
    """A complete manifest should validate with all required lineage fields."""
    manifest = HybridRankerDatasetManifest(
        dataset_version="2026-06-25T121000Z",
        snapshot_id="2026-06-25T120000Z",
        cf_dataset_version="2026-06-25T121500Z",
        cf_version="cf-v1-2026-06-25T122000Z",
        content_embedding_version="content-v1",
        feature_schema_version="hybrid-v1",
        input_dim=1356,
        status="complete",
        train_row_count=80,
        validation_row_count=10,
        test_row_count=10,
        metadata_normalization=MetadataNormalizationStats(
            year_min=1900.0,
            year_max=2025.0,
            runtime_min=60.0,
            runtime_max=240.0,
            tmdb_popularity_log_min=0.0,
            tmdb_popularity_log_max=10.0,
            tmdb_vote_average_min=0.0,
            tmdb_vote_average_max=10.0,
            tmdb_vote_count_log_min=0.0,
            tmdb_vote_count_log_max=12.0,
        ),
        created_at=datetime(2026, 6, 25, 12, 10, tzinfo=UTC),
        finished_at=datetime(2026, 6, 25, 12, 11, tzinfo=UTC),
        pipeline_run_id="run-123",
        train_parts=[
            HybridRankerPartEntry(
                object_key="features/hybrid_ranker/dataset_version=2026-06-25T121000Z/train/part-00000.parquet",
                row_count=80,
            )
        ],
        validation_parts=[
            HybridRankerPartEntry(
                object_key="features/hybrid_ranker/dataset_version=2026-06-25T121000Z/validation/part-00000.parquet",
                row_count=10,
            )
        ],
        test_parts=[
            HybridRankerPartEntry(
                object_key="features/hybrid_ranker/dataset_version=2026-06-25T121000Z/test/part-00000.parquet",
                row_count=10,
            )
        ],
    )

    assert manifest.input_dim == 1356
    assert manifest.feature_schema_version == "hybrid-v1"
    assert manifest.metadata_normalization.year_max == 2025.0

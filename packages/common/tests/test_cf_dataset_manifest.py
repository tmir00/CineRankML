"""Tests for CF dataset manifest schema."""

from __future__ import annotations

from datetime import UTC, datetime

from common.schemas.cf_dataset_manifest import CfDatasetManifest, CfDatasetPartEntry


def test_cf_dataset_manifest_accepts_required_fields() -> None:
    """A complete manifest should validate with all required lineage fields."""
    manifest = CfDatasetManifest(
        snapshot_id="2026-06-25T120000Z",
        cf_dataset_version="2026-06-25T121500Z",
        status="complete",
        train_row_count=80,
        validation_row_count=10,
        test_row_count=10,
        num_users=10,
        num_movies=50,
        train_fraction=0.8,
        validation_fraction=0.1,
        test_fraction=0.1,
        shuffle_seed=42,
        created_at=datetime(2026, 6, 25, 12, 15, tzinfo=UTC),
        finished_at=datetime(2026, 6, 25, 12, 16, tzinfo=UTC),
        pipeline_run_id="run-123",
        user_id_map_key="features/cf_dataset/cf_dataset_version=2026-06-25T121500Z/user_id_map.parquet",
        movie_id_map_key="features/cf_dataset/cf_dataset_version=2026-06-25T121500Z/movie_id_map.parquet",
        train_parts=[
            CfDatasetPartEntry(
                object_key="features/cf_dataset/cf_dataset_version=2026-06-25T121500Z/train/part-00000.parquet",
                row_count=80,
            )
        ],
        validation_parts=[
            CfDatasetPartEntry(
                object_key="features/cf_dataset/cf_dataset_version=2026-06-25T121500Z/validation/part-00000.parquet",
                row_count=10,
            )
        ],
        test_parts=[
            CfDatasetPartEntry(
                object_key="features/cf_dataset/cf_dataset_version=2026-06-25T121500Z/test/part-00000.parquet",
                row_count=10,
            )
        ],
    )

    assert manifest.shuffle_seed == 42
    assert manifest.snapshot_id == "2026-06-25T120000Z"
    assert manifest.train_parts[0].row_count == 80
    assert manifest.test_row_count == 10

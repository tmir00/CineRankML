"""Tests for CF artifact manifest schema."""

from __future__ import annotations

from datetime import UTC, datetime

from common.schemas.cf_artifact_manifest import CfArtifactManifest, CfTrainingMetrics


def test_cf_artifact_manifest_accepts_required_fields() -> None:
    """A complete CF artifact manifest should validate with lineage fields."""
    metrics = CfTrainingMetrics(
        best_epoch=3,
        best_validation_rmse=0.91,
        best_validation_mae=0.72,
        num_train_rows=80,
        num_validation_rows=20,
        num_users=10,
        num_movies=50,
        movie_embedding_coverage=0.5,
        default_embedding_count=2,
        nan_embedding_count=0,
        embedding_norm_mean=0.4,
        embedding_norm_std=0.1,
    )
    manifest = CfArtifactManifest(
        cf_version="cf-v1-2026-06-25T122000Z",
        cf_dataset_version="2026-06-25T121500Z",
        snapshot_id="2026-06-25T120000Z",
        embedding_dim=64,
        status="complete",
        movie_cf_embeddings_path="artifacts/collaborative_filtering/cf_version=cf-v1-2026-06-25T122000Z/movie_cf_embeddings.parquet",
        cf_model_path="artifacts/collaborative_filtering/cf_version=cf-v1-2026-06-25T122000Z/cf_model.pt",
        cf_config_path="artifacts/collaborative_filtering/cf_version=cf-v1-2026-06-25T122000Z/cf_config.json",
        cf_metrics_path="artifacts/collaborative_filtering/cf_version=cf-v1-2026-06-25T122000Z/cf_metrics.json",
        training_curve_path="artifacts/collaborative_filtering/cf_version=cf-v1-2026-06-25T122000Z/training_curve.png",
        metrics=metrics,
        created_at=datetime(2026, 6, 25, 12, 20, tzinfo=UTC),
        finished_at=datetime(2026, 6, 25, 12, 25, tzinfo=UTC),
        pipeline_run_id="run-456",
    )

    assert manifest.metrics.best_validation_rmse == 0.91
    assert manifest.embedding_dim == 64

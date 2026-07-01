"""Tests for CF artifact reader helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from common.schemas.cf_artifact_manifest import CfArtifactManifest, CfTrainingMetrics
from common.storage.cf_artifact_reader import (
    list_complete_cf_artifact_versions,
    resolve_cf_version,
)


def _sample_manifest(*, status: str = "complete", cf_version: str = "cf-v1") -> CfArtifactManifest:
    return CfArtifactManifest(
        cf_version=cf_version,
        cf_dataset_version="2026-06-25T121500Z",
        snapshot_id="2026-06-25T120000Z",
        embedding_dim=64,
        status=status,  # type: ignore[arg-type]
        movie_cf_embeddings_path="artifacts/collaborative_filtering/cf_version=cf-v1/movie_cf_embeddings.parquet",
        cf_model_path="artifacts/collaborative_filtering/cf_version=cf-v1/cf_model.pt",
        cf_config_path="artifacts/collaborative_filtering/cf_version=cf-v1/cf_config.json",
        cf_metrics_path="artifacts/collaborative_filtering/cf_version=cf-v1/cf_metrics.json",
        training_curve_path="artifacts/collaborative_filtering/cf_version=cf-v1/training_curve.png",
        metrics=CfTrainingMetrics(
            best_epoch=5,
            best_validation_rmse=0.9,
            best_validation_mae=0.7,
            num_train_rows=80,
            num_validation_rows=10,
            num_users=10,
            num_movies=50,
            movie_embedding_coverage=1.0,
            default_embedding_count=0,
            nan_embedding_count=0,
            embedding_norm_mean=1.0,
            embedding_norm_std=0.1,
        ),
        created_at=datetime(2026, 6, 25, 12, 20, tzinfo=UTC),
        finished_at=datetime(2026, 6, 25, 12, 21, tzinfo=UTC),
        pipeline_run_id="run-456",
    )


def test_resolve_cf_version_uses_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit CF_VERSION should be validated and returned."""
    client = MagicMock()
    monkeypatch.setattr(
        "common.storage.cf_artifact_reader.load_cf_artifact_manifest",
        lambda *_args, **_kwargs: _sample_manifest(),
    )

    version = resolve_cf_version(client, "cinerankml", "cf-v1")

    assert version == "cf-v1"


def test_resolve_cf_version_picks_latest_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    """When no override is set, the latest complete CF artifact version is returned."""
    client = MagicMock()
    monkeypatch.setattr(
        "common.storage.cf_artifact_reader.list_complete_cf_artifact_versions",
        lambda *_args, **_kwargs: ["cf-v0", "cf-v1"],
    )

    version = resolve_cf_version(client, "cinerankml", None)

    assert version == "cf-v1"


def test_list_complete_cf_artifact_versions_skips_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only manifests with status=complete should be returned."""
    client = MagicMock()
    monkeypatch.setattr(
        "common.storage.cf_artifact_reader.list_common_prefixes",
        lambda *_args, **_kwargs: [
            "artifacts/collaborative_filtering/cf_version=cf-v0/",
            "artifacts/collaborative_filtering/cf_version=cf-v1/",
        ],
    )

    def _load_manifest(_client, _bucket, version: str) -> CfArtifactManifest:
        status = "complete" if version == "cf-v1" else "failed"
        return _sample_manifest(status=status, cf_version=version)

    monkeypatch.setattr(
        "common.storage.cf_artifact_reader.load_cf_artifact_manifest",
        _load_manifest,
    )

    versions = list_complete_cf_artifact_versions(client, "cinerankml")

    assert versions == ["cf-v1"]

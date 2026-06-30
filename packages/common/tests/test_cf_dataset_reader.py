"""Tests for CF dataset reader helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from common.schemas.cf_dataset_manifest import CfDatasetManifest, CfDatasetPartEntry
from common.storage.cf_dataset_reader import (
    list_complete_cf_dataset_versions,
    resolve_cf_dataset_version,
)


def _sample_manifest(*, status: str = "complete") -> CfDatasetManifest:
    return CfDatasetManifest(
        snapshot_id="2026-06-25T120000Z",
        cf_dataset_version="2026-06-25T121500Z",
        status=status,  # type: ignore[arg-type]
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


def test_resolve_cf_dataset_version_uses_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit CF_DATASET_VERSION should be validated and returned."""
    client = MagicMock()
    monkeypatch.setattr(
        "common.storage.cf_dataset_reader.load_cf_dataset_manifest",
        lambda *_args, **_kwargs: _sample_manifest(),
    )

    version = resolve_cf_dataset_version(client, "cinerankml", "2026-06-25T121500Z")

    assert version == "2026-06-25T121500Z"


def test_resolve_cf_dataset_version_picks_latest_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no override is set, the latest complete CF dataset version is returned."""
    client = MagicMock()
    monkeypatch.setattr(
        "common.storage.cf_dataset_reader.list_complete_cf_dataset_versions",
        lambda *_args, **_kwargs: ["2026-06-25T120000Z", "2026-06-25T121500Z"],
    )

    version = resolve_cf_dataset_version(client, "cinerankml", None)

    assert version == "2026-06-25T121500Z"


def test_list_complete_cf_dataset_versions_skips_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only manifests with status=complete should be returned."""
    client = MagicMock()
    monkeypatch.setattr(
        "common.storage.cf_dataset_reader.list_common_prefixes",
        lambda *_args, **_kwargs: [
            "features/cf_dataset/cf_dataset_version=2026-06-25T120000Z/",
            "features/cf_dataset/cf_dataset_version=2026-06-25T121500Z/",
        ],
    )

    def _load_manifest(_client, _bucket, version: str) -> CfDatasetManifest:
        status = "complete" if version.endswith("121500Z") else "failed"
        manifest = _sample_manifest(status=status)
        manifest.cf_dataset_version = version
        return manifest

    monkeypatch.setattr(
        "common.storage.cf_dataset_reader.load_cf_dataset_manifest",
        _load_manifest,
    )

    versions = list_complete_cf_dataset_versions(client, "cinerankml")

    assert versions == ["2026-06-25T121500Z"]

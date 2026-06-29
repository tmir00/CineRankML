"""Tests for snapshot discovery helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from common.schemas.snapshot_manifest import SnapshotManifest, SnapshotTableEntry
from common.storage.snapshot_reader import (
    list_complete_snapshot_ids,
    resolve_snapshot_id,
    snapshot_table_glob_uri,
)


def _complete_manifest(snapshot_id: str) -> dict:
    """Build a minimal complete snapshot manifest payload."""
    now = datetime(2026, 6, 25, 12, 0, tzinfo=UTC)
    manifest = SnapshotManifest(
        snapshot_id=snapshot_id,
        status="complete",
        created_at=now,
        finished_at=now,
        pipeline_run_id="run-1",
        tables={
            "ratings_events": SnapshotTableEntry(
                prefix=f"snapshots/snapshot_id={snapshot_id}/ratings_events/",
                row_count=1,
                part_count=1,
                parts=[],
            )
        },
    )
    return manifest.model_dump(mode="json")


def test_snapshot_table_glob_uri() -> None:
    """Glob URIs should target all Parquet parts for one snapshot table."""
    uri = snapshot_table_glob_uri("cinerankml", "2026-06-25T120000Z", "ratings_events")
    assert uri == "s3://cinerankml/snapshots/snapshot_id=2026-06-25T120000Z/ratings_events/*.parquet"


def test_list_complete_snapshot_ids_filters_incomplete_manifests() -> None:
    """Only snapshots with status=complete should be returned."""
    client = MagicMock()
    client.list_objects_v2.return_value = {
        "CommonPrefixes": [
            {"Prefix": "snapshots/snapshot_id=2026-06-25T110000Z/"},
            {"Prefix": "snapshots/snapshot_id=2026-06-25T120000Z/"},
        ],
        "IsTruncated": False,
    }

    def get_object(Bucket: str, Key: str) -> dict:
        if Key.endswith("2026-06-25T110000Z/manifest.json"):
            body = MagicMock()
            body.read.return_value = (
                '{"snapshot_id":"2026-06-25T110000Z","status":"failed","created_at":"2026-06-25T11:00:00Z",'
                '"finished_at":"2026-06-25T11:00:00Z","pipeline_run_id":"run-1","tables":{}}'
            ).encode("utf-8")
            return {"Body": body}
        body = MagicMock()
        body.read.return_value = __import__("json").dumps(
            _complete_manifest("2026-06-25T120000Z")
        ).encode("utf-8")
        return {"Body": body}

    client.get_object.side_effect = get_object

    complete_ids = list_complete_snapshot_ids(client, "cinerankml")
    assert complete_ids == ["2026-06-25T120000Z"]


def test_resolve_snapshot_id_returns_latest_complete_snapshot() -> None:
    """When SNAPSHOT_ID is unset, the latest complete snapshot id should be used."""
    client = MagicMock()
    client.list_objects_v2.return_value = {
        "CommonPrefixes": [
            {"Prefix": "snapshots/snapshot_id=2026-06-25T110000Z/"},
            {"Prefix": "snapshots/snapshot_id=2026-06-25T120000Z/"},
        ],
        "IsTruncated": False,
    }

    def get_object(Bucket: str, Key: str) -> dict:
        snapshot_id = Key.split("snapshot_id=")[1].split("/")[0]
        body = MagicMock()
        body.read.return_value = __import__("json").dumps(
            _complete_manifest(snapshot_id)
        ).encode("utf-8")
        return {"Body": body}

    client.get_object.side_effect = get_object

    resolved = resolve_snapshot_id(client, "cinerankml", None)
    assert resolved == "2026-06-25T120000Z"


def test_resolve_snapshot_id_rejects_incomplete_override() -> None:
    """An explicit SNAPSHOT_ID must point at a complete manifest."""
    client = MagicMock()
    body = MagicMock()
    body.read.return_value = (
        '{"snapshot_id":"2026-06-25T120000Z","status":"failed","created_at":"2026-06-25T12:00:00Z",'
        '"finished_at":"2026-06-25T12:00:00Z","pipeline_run_id":"run-1","tables":{}}'
    ).encode("utf-8")
    client.get_object.return_value = {"Body": body}

    with pytest.raises(ValueError, match="not complete"):
        resolve_snapshot_id(client, "cinerankml", "2026-06-25T120000Z")


def test_list_complete_snapshot_ids_skips_missing_manifests() -> None:
    """Snapshots without readable manifests should be ignored."""
    client = MagicMock()
    client.list_objects_v2.return_value = {
        "CommonPrefixes": [{"Prefix": "snapshots/snapshot_id=2026-06-25T120000Z/"}],
        "IsTruncated": False,
    }
    client.get_object.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "not found"}},
        "GetObject",
    )

    assert list_complete_snapshot_ids(client, "cinerankml") == []

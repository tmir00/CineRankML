"""Resolve complete snapshots from MinIO for downstream batch jobs."""

from __future__ import annotations

from botocore.client import BaseClient
from botocore.exceptions import ClientError
from pydantic import ValidationError

from common.schemas.snapshot_manifest import SnapshotManifest
from common.storage.s3 import get_json, list_common_prefixes, manifest_object_key


def load_snapshot_manifest(client: BaseClient, bucket: str, snapshot_id: str) -> SnapshotManifest:
    """
    Download and validate one snapshot manifest from MinIO.
    
    Do this by:
    1. Reading manifest.json for the requested snapshot id.
    2. Parsing the payload into a SnapshotManifest model.    

    ============================ Arguments ============================
    client: The boto3 S3 client.
    bucket: Source bucket name.
    snapshot_id: Snapshot identifier under snapshots/snapshot_id=.../.

    ============================ Returns ============================
    A validated SnapshotManifest.
    """
    # Download and parse the manifest JSON.
    payload = get_json(client, bucket, manifest_object_key(snapshot_id))
    return SnapshotManifest.model_validate(payload)


def list_complete_snapshot_ids(client: BaseClient, bucket: str) -> list[str]:
    """
    List snapshot ids that have a complete manifest in MinIO.

    Do this by:
    1. Listing snapshot prefixes under snapshots/snapshot_id=.
    2. Loading each manifest and keeping only status=complete snapshots.

    ============================ Arguments ============================
    client: The boto3 S3 client.
    bucket: Source bucket name.

    ============================ Returns ============================
    Snapshot ids sorted ascending (ISO timestamps sort lexicographically).
    """
    # List every folder name prefix in the bucket under snapshots/.
    snapshot_prefixes = list_common_prefixes(client, bucket, "snapshots/")
    complete_ids: list[str] = []

    for prefix in snapshot_prefixes:
        # Extract the snapshot id from prefixes like snapshots/snapshot_id=2026-06-25T120000Z/.
        if not prefix.startswith("snapshots/snapshot_id="):
            continue

        # Extract the snapshot id from the folder name prefix.
        snapshot_id = prefix.removeprefix("snapshots/snapshot_id=").rstrip("/")
        if not snapshot_id:
            continue

        # Load the manifest for the snapshot id.
        try:
            manifest = load_snapshot_manifest(client, bucket, snapshot_id)
        except (ValidationError, ClientError, OSError):
            continue

        # Only completed snapshots are safe inputs for dataset prep.
        if manifest.status == "complete":
            complete_ids.append(snapshot_id)

    return sorted(complete_ids)


def resolve_snapshot_id(client: BaseClient, bucket: str, snapshot_id_override: str | None) -> str:
    """
    Return the snapshot id to use for dataset prep.

    Do this by:
    1. Returning the snapshot we are overriding with when SNAPSHOT_ID is set.
    2. Otherwise picking the latest complete snapshot from MinIO.

    ============================ Arguments ============================
    client: The boto3 S3 client.
    bucket: Source bucket name.
    snapshot_id_override: Optional explicit snapshot id from settings.

    ============================ Returns ============================
    The resolved snapshot id string.
    """
    # If we are overriding the snapshot id, load the manifest and check if it is complete.
    if snapshot_id_override:
        manifest = load_snapshot_manifest(client, bucket, snapshot_id_override)
        if manifest.status != "complete":
            raise ValueError(f"Snapshot {snapshot_id_override} is not complete (status={manifest.status})")
        return snapshot_id_override

    # Otherwise, list all complete snapshots and return the latest one.
    complete_ids = list_complete_snapshot_ids(client, bucket)
    if not complete_ids:
        raise ValueError("No complete snapshots found in MinIO")

    return complete_ids[-1]


def snapshot_table_glob_uri(bucket: str, snapshot_id: str, table_name: str) -> str:
    """
    Build an s3:// glob URI for all Parquet parts of one snapshot table.

    ============================ Arguments ============================
    bucket: Source bucket name.
    snapshot_id: Snapshot identifier.
    table_name: Snapshot table folder name (e.g. ratings_events).

    ============================ Returns ============================
    URI like s3://bucket/snapshots/snapshot_id=.../ratings_events/*.parquet.
    """
    return (
        f"s3://{bucket}/snapshots/snapshot_id={snapshot_id}/{table_name}/*.parquet"
    )

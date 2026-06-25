"""Thin S3 client helpers for MinIO uploads."""

from __future__ import annotations

import json
import boto3

from typing import Any
from pathlib import Path
from botocore.client import BaseClient
from botocore.exceptions import ClientError
from common.config.settings import SnapshotSettings


def ensure_bucket_exists(client: BaseClient, bucket: str) -> None:
    """
    Create the snapshot bucket when it does not already exist.

    ============================ Arguments ============================
    client: The boto3 S3 client.
    bucket: Target bucket name.
    """
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError:
        client.create_bucket(Bucket=bucket)


def create_s3_client(settings: SnapshotSettings) -> BaseClient:
    """
    Create a boto3 S3 client pointed at MinIO.

    ============================ Arguments ============================
    settings: Snapshot job configuration with endpoint and credentials.

    ============================ Returns ============================
    A boto3 S3 client.
    """
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name="us-east-1",
    )


def snapshot_prefix(snapshot_id: str) -> str:
    """Return the object prefix for one snapshot run."""
    return f"snapshots/snapshot_id={snapshot_id}/"


def table_prefix(snapshot_id: str, table_name: str) -> str:
    """Return the object prefix for one table inside a snapshot."""
    return f"{snapshot_prefix(snapshot_id)}{table_name}/"


def part_object_key(snapshot_id: str, table_name: str, part_index: int) -> str:
    """
    Build the S3 object key for one Parquet part file.

    ============================ Arguments ============================
    snapshot_id: UTC snapshot identifier.
    table_name: Postgres table name being exported.
    part_index: Zero-based part index within the table.

    ============================ Returns ============================
    Object key like snapshots/snapshot_id=.../ratings_events/part-00000.parquet.
    """
    return f"{table_prefix(snapshot_id, table_name)}part-{part_index:05d}.parquet"


def manifest_object_key(snapshot_id: str) -> str:
    """Return the manifest.json object key for one snapshot."""
    return f"{snapshot_prefix(snapshot_id)}manifest.json"


def upload_file(client: BaseClient, bucket: str, key: str, local_path: Path) -> None:
    """
    Upload one local file to S3.

    ============================ Arguments ============================
    client: The boto3 S3 client.
    bucket: Target bucket name.
    key: Object key inside the bucket.
    local_path: Path to the local file to upload.
    """
    client.upload_file(str(local_path), bucket, key)


def put_json(client: BaseClient, bucket: str, key: str, payload: Any) -> None:
    """
    Upload a JSON-serializable object to S3.

    ============================ Arguments ============================
    client: The boto3 S3 client.
    bucket: Target bucket name.
    key: Object key inside the bucket.
    payload: JSON-serializable manifest or metadata object.
    """
    body = json.dumps(payload, default=str, indent=2).encode("utf-8")
    client.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")

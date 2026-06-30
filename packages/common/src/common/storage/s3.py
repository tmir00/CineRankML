"""Thin S3 client helpers for MinIO uploads."""

from __future__ import annotations

import json
import boto3

from pathlib import Path
from typing import Any, Protocol
from botocore.client import BaseClient
from botocore.exceptions import ClientError


class S3ConnectionSettings(Protocol):
    """Minimal settings shape required to build a boto3 S3 client."""

    s3_endpoint_url: str
    s3_access_key: str
    s3_secret_key: str


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


def create_s3_client(settings: S3ConnectionSettings) -> BaseClient:
    """
    Create a boto3 S3 client pointed at MinIO.

    ============================ Arguments ============================
    settings: Job configuration with endpoint and credentials.

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


def cf_dataset_prefix(cf_dataset_version: str) -> str:
    """Return the object prefix for one CF dataset version."""
    return f"features/cf_dataset/cf_dataset_version={cf_dataset_version}/"


def cf_dataset_train_prefix(cf_dataset_version: str) -> str:
    """Return the object prefix for shuffled train parts."""
    return f"{cf_dataset_prefix(cf_dataset_version)}train/"


def cf_dataset_validation_prefix(cf_dataset_version: str) -> str:
    """Return the object prefix for time-ordered validation parts."""
    return f"{cf_dataset_prefix(cf_dataset_version)}validation/"


def cf_dataset_test_prefix(cf_dataset_version: str) -> str:
    """Return the object prefix for locked test parts."""
    return f"{cf_dataset_prefix(cf_dataset_version)}test/"


def cf_dataset_user_map_object_key(cf_dataset_version: str) -> str:
    """Return the user_id_map.parquet object key for one CF dataset version."""
    return f"{cf_dataset_prefix(cf_dataset_version)}user_id_map.parquet"


def cf_dataset_movie_map_object_key(cf_dataset_version: str) -> str:
    """Return the movie_id_map.parquet object key for one CF dataset version."""
    return f"{cf_dataset_prefix(cf_dataset_version)}movie_id_map.parquet"


def cf_dataset_train_part_object_key(cf_dataset_version: str, part_index: int) -> str:
    """Return one train part object key for a CF dataset version."""
    return f"{cf_dataset_train_prefix(cf_dataset_version)}part-{part_index:05d}.parquet"


def cf_dataset_validation_part_object_key(cf_dataset_version: str, part_index: int) -> str:
    """Return one validation part object key for a CF dataset version."""
    return f"{cf_dataset_validation_prefix(cf_dataset_version)}part-{part_index:05d}.parquet"


def cf_dataset_test_part_object_key(cf_dataset_version: str, part_index: int) -> str:
    """Return one test part object key for a CF dataset version."""
    return f"{cf_dataset_test_prefix(cf_dataset_version)}part-{part_index:05d}.parquet"


def cf_dataset_manifest_object_key(cf_dataset_version: str) -> str:
    """Return the manifest.json object key for one CF dataset version."""
    return f"{cf_dataset_prefix(cf_dataset_version)}manifest.json"


def cf_artifact_prefix(cf_version: str) -> str:
    """Return the object prefix for one CF training artifact version."""
    return f"artifacts/collaborative_filtering/cf_version={cf_version}/"


def cf_movie_embeddings_object_key(cf_version: str) -> str:
    """Return the movie_cf_embeddings.parquet object key for one CF version."""
    return f"{cf_artifact_prefix(cf_version)}movie_cf_embeddings.parquet"


def cf_model_object_key(cf_version: str) -> str:
    """Return the cf_model.pt object key for one CF version."""
    return f"{cf_artifact_prefix(cf_version)}cf_model.pt"


def cf_config_object_key(cf_version: str) -> str:
    """Return the cf_config.json object key for one CF version."""
    return f"{cf_artifact_prefix(cf_version)}cf_config.json"


def cf_metrics_object_key(cf_version: str) -> str:
    """Return the cf_metrics.json object key for one CF version."""
    return f"{cf_artifact_prefix(cf_version)}cf_metrics.json"


def cf_training_curve_object_key(cf_version: str) -> str:
    """Return the training_curve.png object key for one CF version."""
    return f"{cf_artifact_prefix(cf_version)}training_curve.png"


def cf_artifact_manifest_object_key(cf_version: str) -> str:
    """Return the manifest.json object key for one CF artifact version."""
    return f"{cf_artifact_prefix(cf_version)}manifest.json"


def list_common_prefixes(client: BaseClient, bucket: str, prefix: str) -> list[str]:
    """
    List the immediate child folder name prefixes under one S3 prefix.
    E.g: Under snapshots, we list snapshot_id=2026-06-25T120000Z/, snapshot_id=2026-06-26T120000Z/, etc.

    Do this by:
    1. Paginating list_objects_v2 with delimiter=/.
    2. Collecting CommonPrefixes entries.

    ============================ Arguments ============================
    client: The boto3 S3 client.
    bucket: Source bucket name.
    prefix: Parent prefix to list (should end with / when listing children).

    ============================ Returns ============================
    Sorted list of child prefix strings.
    """
    # Normalize the prefix to end with a trailing slash.
    normalized_prefix = prefix if prefix.endswith("/") else f"{prefix}/"
    # Initialize an empty list to store the child prefix strings.
    prefixes: list[str] = []
    # Initialize a continuation token to None.
    continuation_token: str | None = None

    # Loop until we have no more pages to fetch.
    while True:
        kwargs: dict[str, Any] = {
            "Bucket": bucket,
            "Prefix": normalized_prefix,
            "Delimiter": "/",
        }
        # If we have a continuation token, add it to the kwargs.
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token

        # List the objects in the bucket under the normalized prefix.
        response = client.list_objects_v2(**kwargs)
        for entry in response.get("CommonPrefixes", []):
            prefixes.append(entry["Prefix"])

        # If we have no more pages to fetch, break out of the loop.
        if not response.get("IsTruncated"):
            break
        # Set the continuation token to the next token from the response.
        continuation_token = response.get("NextContinuationToken")

    return sorted(prefixes)


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


def download_file(client: BaseClient, bucket: str, key: str, local_path: Path) -> None:
    """
    Download one S3 object to a local file path.

    ============================ Arguments ============================
    client: The boto3 S3 client.
    bucket: Source bucket name.
    key: Object key inside the bucket.
    local_path: Destination path on the local filesystem.
    """
    local_path.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(local_path))


def get_json(client: BaseClient, bucket: str, key: str) -> Any:
    """
    Download and parse one JSON object from S3.
    This is used to download and read the manifest.json file from S3.

    ============================ Arguments ============================
    client: The boto3 S3 client.
    bucket: Source bucket name.
    key: Object key inside the bucket.

    ============================ Returns ============================
    The parsed JSON payload (usually a dict).
    """
    # Download the object from S3.
    response = client.get_object(Bucket=bucket, Key=key)
    # Read the body of the object.
    body = response["Body"].read()
    # Parse the body as JSON and return the result.
    return json.loads(body)

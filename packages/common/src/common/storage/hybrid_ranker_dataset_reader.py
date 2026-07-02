"""Resolve complete hybrid ranker datasets from MinIO for training jobs."""

from __future__ import annotations

from pydantic import ValidationError
from botocore.client import BaseClient
from botocore.exceptions import ClientError
from common.schemas.hybrid_ranker_dataset_manifest import HybridRankerDatasetManifest
from common.storage.s3 import get_json, hybrid_ranker_manifest_object_key, list_common_prefixes


def load_hybrid_ranker_dataset_manifest(client: BaseClient, bucket: str, dataset_version: str) -> HybridRankerDatasetManifest:
    """
    Download and validate one hybrid ranker dataset manifest from MinIO.

    Do this by:
    1. Reading manifest.json for the requested dataset version.
    2. Parsing the payload into a HybridRankerDatasetManifest model.

    ============================ Arguments ============================
    client: The boto3 S3 client.
    bucket: Source bucket name.
    dataset_version: Version id under features/hybrid_ranker/dataset_version=.../.

    ============================ Returns ============================
    A validated HybridRankerDatasetManifest.
    """
    payload = get_json(client, bucket, hybrid_ranker_manifest_object_key(dataset_version))
    return HybridRankerDatasetManifest.model_validate(payload)


def list_complete_hybrid_ranker_dataset_versions(client: BaseClient, bucket: str) -> list[str]:
    """
    List hybrid ranker dataset versions that have a complete manifest in MinIO.

    Do this by:
    1. Listing version prefixes under features/hybrid_ranker/.
    2. Loading each manifest and keeping only status=complete datasets.

    ============================ Arguments ============================
    client: The boto3 S3 client.
    bucket: Source bucket name.

    ============================ Returns ============================
    Dataset version ids sorted ascending.
    """
    # List all the versions of the hybrid ranker dataset in the bucket.
    version_prefixes = list_common_prefixes(client, bucket, "features/hybrid_ranker/")
    complete_versions: list[str] = []

    # Iterate over all the versions of the hybrid ranker dataset.
    for prefix in version_prefixes:
        # Skip if the file name is not a dataset version.
        if not prefix.startswith("features/hybrid_ranker/dataset_version="):
            continue

        # Extract the dataset version from the prefix.
        dataset_version = prefix.removeprefix("features/hybrid_ranker/dataset_version=").rstrip("/")
        # Skip if the dataset version is not valid.
        if not dataset_version:
            continue

        # Load the manifest for the dataset version.
        try:
            manifest = load_hybrid_ranker_dataset_manifest(client, bucket, dataset_version)
        # Skip if the manifest is not valid.
        except (ValidationError, ClientError, OSError):
            continue

        # Add the dataset version to the list of complete versions if the status is complete.
        if manifest.status == "complete":
            complete_versions.append(dataset_version)

    # Return the list of complete dataset versions sorted ascending.
    return sorted(complete_versions)


def resolve_hybrid_ranker_dataset_version(client: BaseClient, bucket: str, dataset_version_override: str | None) -> str:
    """
    Return the hybrid ranker dataset version to use for training.

    Do this by:
    1. Returning the override version when DATASET_VERSION is set.
    2. Otherwise picking the latest complete hybrid dataset from MinIO.

    ============================ Arguments ============================
    client: The boto3 S3 client.
    bucket: Source bucket name.
    dataset_version_override: Optional explicit version from settings.

    ============================ Returns ============================
    The resolved dataset version string.
    """
    # Return the override version when DATASET_VERSION is set.
    if dataset_version_override:
        # Load the manifest for the override version to check if it is complete.
        manifest = load_hybrid_ranker_dataset_manifest(client, bucket, dataset_version_override)
        # Raise an error if the manifest is not complete.
        if manifest.status != "complete":
            raise ValueError(
                f"Hybrid ranker dataset {dataset_version_override} is not complete "
                f"(status={manifest.status})"
            )
        return dataset_version_override

    # List all the complete dataset versions in the bucket.
    complete_versions = list_complete_hybrid_ranker_dataset_versions(client, bucket)
    # Raise an error if no complete dataset versions are found.
    if not complete_versions:
        raise ValueError("No complete hybrid ranker datasets found in MinIO")

    # Return the latest complete dataset version.
    return complete_versions[-1]

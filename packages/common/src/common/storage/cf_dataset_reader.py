"""Resolve complete CF datasets from MinIO for downstream training jobs."""

from __future__ import annotations

from botocore.client import BaseClient
from botocore.exceptions import ClientError
from pydantic import ValidationError

from common.schemas.cf_dataset_manifest import CfDatasetManifest
from common.storage.s3 import cf_dataset_manifest_object_key, get_json, list_common_prefixes


def load_cf_dataset_manifest(client: BaseClient, bucket: str, cf_dataset_version: str) -> CfDatasetManifest:
    """
    Download and validate one CF dataset manifest from MinIO.

    Do this by:
    1. Reading manifest.json for the requested CF dataset version.
    2. Parsing the payload into a CfDatasetManifest model.

    ============================ Arguments ============================
    client: The boto3 S3 client.
    bucket: Source bucket name.
    cf_dataset_version: Version id under features/cf_dataset/cf_dataset_version=.../.

    ============================ Returns ============================
    A validated CfDatasetManifest.
    """
    payload = get_json(client, bucket, cf_dataset_manifest_object_key(cf_dataset_version))
    return CfDatasetManifest.model_validate(payload)


def list_complete_cf_dataset_versions(client: BaseClient, bucket: str) -> list[str]:
    """
    List CF dataset versions that have a complete manifest in MinIO.

    Do this by:
    1. Listing version prefixes under features/cf_dataset/.
    2. Loading each manifest and keeping only status=complete datasets.

    ============================ Arguments ============================
    client: The boto3 S3 client.
    bucket: Source bucket name.

    ============================ Returns ============================
    CF dataset version ids sorted ascending.
    """
    # list all the folder names under features/cf_dataset/, these are the version prefixes.
    version_prefixes = list_common_prefixes(client, bucket, "features/cf_dataset/")
    complete_versions: list[str] = []

    # for each version prefix, load the manifest and check if it is complete.
    for prefix in version_prefixes:
        # skip if the prefix does not start with features/cf_dataset/cf_dataset_version=
        if not prefix.startswith("features/cf_dataset/cf_dataset_version="):
            continue

        # remove the prefix features/cf_dataset/cf_dataset_version= and strip the trailing /
        cf_dataset_version = prefix.removeprefix("features/cf_dataset/cf_dataset_version=").rstrip("/")
        if not cf_dataset_version:
            continue

        # load the manifest and check if it is complete.
        try:
            manifest = load_cf_dataset_manifest(client, bucket, cf_dataset_version)
        except (ValidationError, ClientError, OSError):
            continue

        # if the manifest is complete, add the version to the list.
        if manifest.status == "complete":
            complete_versions.append(cf_dataset_version)

    # return the list of complete versions sorted ascending.
    return sorted(complete_versions)


def resolve_cf_dataset_version(client: BaseClient, bucket: str, cf_dataset_version_override: str | None) -> str:
    """
    Return the CF dataset version to use for training.

    Do this by:
    1. Returning the 'override' version when CF_DATASET_VERSION is set.
    2. Otherwise picking the latest complete CF dataset from MinIO.

    ============================ Arguments ============================
    client: The boto3 S3 client.
    bucket: Source bucket name.
    cf_dataset_version_override: Optional explicit version from settings.

    ============================ Returns ============================
    The resolved CF dataset version string.
    """
    # if the override version is set, load the manifest and check if it is complete.
    if cf_dataset_version_override:
        manifest = load_cf_dataset_manifest(client, bucket, cf_dataset_version_override)
        
        # if the manifest is not complete, raise an error.
        if manifest.status != "complete":
            raise ValueError(
                f"CF dataset {cf_dataset_version_override} is not complete "
                f"(status={manifest.status})"
            )
        return cf_dataset_version_override

    # if the override version is not set, list all the complete versions and return the latest one.
    complete_versions = list_complete_cf_dataset_versions(client, bucket)
    # if there are no complete versions, raise an error.
    if not complete_versions:
        raise ValueError("No complete CF datasets found in MinIO")

    # return the latest complete version.
    return complete_versions[-1]

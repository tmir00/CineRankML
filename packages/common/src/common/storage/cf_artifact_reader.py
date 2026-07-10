"""Resolve complete CF training artifacts from MinIO for downstream feature jobs."""

from __future__ import annotations

from pydantic import ValidationError
from botocore.client import BaseClient
from botocore.exceptions import ClientError
from common.schemas.cf_artifact_manifest import CfArtifactManifest
from common.storage.s3 import cf_artifact_manifest_object_key, get_json, list_common_prefixes


def load_cf_artifact_manifest(client: BaseClient, bucket: str, cf_version: str) -> CfArtifactManifest:
    """
    Download and validate one CF artifact manifest from MinIO.

    Do this by:
    1. Reading manifest.json for the requested CF version.
    2. Parsing the payload into a CfArtifactManifest model.

    ============================ Arguments ============================
    client: The boto3 S3 client.
    bucket: Source bucket name.
    cf_version: Version id under artifacts/collaborative_filtering/cf_version=.../.

    ============================ Returns ============================
    A validated CfArtifactManifest.
    """
    payload = get_json(client, bucket, cf_artifact_manifest_object_key(cf_version))
    return CfArtifactManifest.model_validate(payload)


def list_complete_cf_artifact_versions(client: BaseClient, bucket: str) -> list[str]:
    """
    List CF artifact versions that have a complete manifest in MinIO.

    Do this by:
    1. Listing version prefixes under artifacts/collaborative_filtering/.
    2. Loading each manifest and keeping only status=complete artifacts.

    ============================ Arguments ============================
    client: The boto3 S3 client.
    bucket: Source bucket name.

    ============================ Returns ============================
    CF version ids sorted ascending.
    """
    # List the version prefixes (filenames) under artifacts/collaborative_filtering/.
    version_prefixes = list_common_prefixes(client, bucket, "artifacts/collaborative_filtering/")
    complete_versions: list[str] = []

    # Iterate over the version prefixes.
    for prefix in version_prefixes:
        # Skip prefixes that don't start with artifacts/collaborative_filtering/cf_version=.
        if not prefix.startswith("artifacts/collaborative_filtering/cf_version="):
            continue

        # Extract the CF version from the prefix.
        cf_version = prefix.removeprefix("artifacts/collaborative_filtering/cf_version=").rstrip("/")
        if not cf_version:
            continue

        # Load the manifest for the CF version.
        try:
            manifest = load_cf_artifact_manifest(client, bucket, cf_version)
        # Skip versions that are not complete.
        except (ValidationError, ClientError, OSError):
            continue
        
        # Keep only complete versions.
        if manifest.status == "complete":
            complete_versions.append(cf_version)

    # Return the complete versions sorted ascending.
    return sorted(complete_versions)


def resolve_cf_version(client: BaseClient, bucket: str, cf_version_override: str | None) -> str:
    """
    Return the CF artifact version to use for hybrid feature generation.

    Do this by:
    1. Returning the override version when CF_VERSION is set.
    2. Otherwise picking the latest complete CF artifact from MinIO.

    ============================ Arguments ============================
    client: The boto3 S3 client.
    bucket: Source bucket name.
    cf_version_override: Optional explicit version from settings.

    ============================ Returns ============================
    The resolved CF version string.
    """
    # Return the override version when CF_VERSION is set.
    if cf_version_override:
        manifest = load_cf_artifact_manifest(client, bucket, cf_version_override)
        if manifest.status != "complete":
            raise ValueError(
                f"CF artifact {cf_version_override} is not complete (status={manifest.status})"
            )
        return cf_version_override

    # List the complete CF artifact versions and return the latest one.
    complete_versions = list_complete_cf_artifact_versions(client, bucket)
    if not complete_versions:
        raise ValueError("No complete CF artifacts found in MinIO")

    return complete_versions[-1]

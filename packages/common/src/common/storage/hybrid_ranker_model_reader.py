"""Load hybrid ranker model artifacts from MinIO for evaluation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from pydantic import ValidationError
from botocore.client import BaseClient
from botocore.exceptions import ClientError

from common.storage.s3 import (
    download_file,
    get_json,
    hybrid_ranker_model_config_object_key,
    hybrid_ranker_model_manifest_object_key,
    hybrid_ranker_model_object_key,
    hybrid_ranker_training_metrics_object_key,
    list_common_prefixes,
)

from common.schemas.hybrid_ranker_artifact_manifest import (
    HybridModelConfig,
    HybridRankerArtifactManifest,
    HybridTrainingMetrics,
)



def load_hybrid_model_config(client: BaseClient, bucket: str, model_version: str) -> HybridModelConfig:
    """
    Download and validate model_config.json for one hybrid ranker model version.

    ============================ Arguments ============================
    client: The boto3 S3 client.
    bucket: Source bucket name.
    model_version: Model version id under models/hybrid_ranker/model_version=.../.

    ============================ Returns ============================
    A validated HybridModelConfig.
    """
    payload = get_json(client, bucket, hybrid_ranker_model_config_object_key(model_version))
    return HybridModelConfig.model_validate(payload)


def load_hybrid_training_metrics(client: BaseClient, bucket: str, model_version: str) -> HybridTrainingMetrics:
    """
    Download and validate training_metrics.json for one hybrid ranker model version.

    ============================ Arguments ============================
    client: The boto3 S3 client.
    bucket: Source bucket name.
    model_version: Model version id.

    ============================ Returns ============================
    A validated HybridTrainingMetrics.
    """
    payload = get_json(client, bucket, hybrid_ranker_training_metrics_object_key(model_version))
    return HybridTrainingMetrics.model_validate(payload)


def load_hybrid_ranker_artifact_manifest(client: BaseClient, bucket: str, model_version: str) -> HybridRankerArtifactManifest:
    """
    Download and validate manifest.json for one complete hybrid ranker model bundle.

    ============================ Arguments ============================
    client: The boto3 S3 client.
    bucket: Source bucket name.
    model_version: Model version id.

    ============================ Returns ============================
    A validated HybridRankerArtifactManifest.
    """
    payload = get_json(client, bucket, hybrid_ranker_model_manifest_object_key(model_version))
    return HybridRankerArtifactManifest.model_validate(payload)


def download_hybrid_ranker_model_checkpoint(client: BaseClient, bucket: str, model_version: str, \
                                                local_path: Path) -> Path:
    """
    Download hybrid_ranker_model.pt to a local path for evaluation.

    ============================ Arguments ============================
    client: The boto3 S3 client.
    bucket: Source bucket name.
    model_version: Model version id.
    local_path: Destination file path.

    ============================ Returns ============================
    The local path where the checkpoint was saved.
    """
    download_file(client, bucket, hybrid_ranker_model_object_key(model_version), local_path)
    return local_path


def list_complete_hybrid_ranker_model_versions(client: BaseClient, bucket: str) -> list[str]:
    """
    List hybrid ranker model versions that have a complete manifest in MinIO.

    ============================ Arguments ============================
    client: The boto3 S3 client.
    bucket: Source bucket name.

    ============================ Returns ============================
    Model version ids sorted ascending.
    """
    # List all the versions of the hybrid ranker model in the bucket.
    version_prefixes = list_common_prefixes(client, bucket, "models/hybrid_ranker/")
    # Initialize an empty list to store the complete model versions.
    complete_versions: list[str] = []

    # Iterate over all the versions of the hybrid ranker model.
    for prefix in version_prefixes:
        # Skip if the file name is not a model version.
        if not prefix.startswith("models/hybrid_ranker/model_version="):
            continue

        # Extract the model version from the prefix.
        model_version = prefix.removeprefix("models/hybrid_ranker/model_version=").rstrip("/")
        # Skip if the model version is not valid.
        if not model_version:
            continue

        # Load the manifest for the model version.
        try:
            # Skip if the manifest is not valid.
            manifest = load_hybrid_ranker_artifact_manifest(client, bucket, model_version)
        except (ValidationError, ClientError, OSError):
            continue

        # Add the model version to the list of complete versions if the status is complete.
        if manifest.status == "complete":
            complete_versions.append(model_version)

    return sorted(complete_versions)


def resolve_hybrid_ranker_model_version(client: BaseClient, bucket: str, model_version_override: str | None) -> str:
    """
    Return the hybrid ranker model version to evaluate.

    Do this by:
    1. Returning the override when MODEL_VERSION is set.
    2. Otherwise picking the latest model that has training artifacts uploaded.

    ============================ Arguments ============================
    client: The boto3 S3 client.
    bucket: Source bucket name.
    model_version_override: Optional explicit version from settings.

    ============================ Returns ============================
    The resolved model version string.
    """
    # Return the override version when MODEL_VERSION is set.
    if model_version_override:
        # Load the model config for the override version to check if it is complete.
        load_hybrid_model_config(client, bucket, model_version_override)
        return model_version_override

    # List all the versions of the hybrid ranker model in the bucket.
    version_prefixes = list_common_prefixes(client, bucket, "models/hybrid_ranker/")
    # Initialize an empty list to store the candidate model versions.
    candidates: list[str] = []

    # Iterate over all the versions of the hybrid ranker model.
    for prefix in version_prefixes:
        # Skip if the file name is not a model version.
        if not prefix.startswith("models/hybrid_ranker/model_version="):
            continue

        # Extract the model version from the prefix.
        model_version = prefix.removeprefix("models/hybrid_ranker/model_version=").rstrip("/")
        # Skip if the model version is not valid.
        if not model_version:
            continue

        try:
            # Load the model config for the model version.
            load_hybrid_model_config(client, bucket, model_version)
        # Skip if the model config is not valid.
        except (ValidationError, ClientError, OSError):
            continue

        # Add the model version to the list of candidate versions.
        candidates.append(model_version)

    if not candidates:
        raise ValueError("No hybrid ranker models with model_config.json found in MinIO")

    # Return the latest candidate model version.
    return sorted(candidates)[-1]


def load_hybrid_model_checkpoint_dict(client: BaseClient, bucket: str, model_version: str) -> dict:
    """
    Download hybrid_ranker_model.pt and return the torch checkpoint dict.

    ============================ Arguments ============================
    client: The boto3 S3 client.
    bucket: Source bucket name.
    model_version: Model version id.

    ============================ Returns ============================
    The parsed checkpoint dictionary.
    """
    import torch

    # Download the model checkpoint to a temporary directory.
    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = Path(temp_dir) / "hybrid_ranker_model.pt"
        download_hybrid_ranker_model_checkpoint(client, bucket, model_version, local_path)
        return torch.load(local_path, map_location="cpu", weights_only=False)

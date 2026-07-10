"""MLflow Model Registry helpers for hybrid ranker main/candidate aliases."""

from __future__ import annotations

import logging

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

logger = logging.getLogger(__name__)

MODEL_VERSION_TAG = "model_version"


def _get_client(tracking_uri: str) -> MlflowClient:
    """Create an MLflow client pointed at the configured tracking URI."""
    return MlflowClient(tracking_uri=tracking_uri)


def ensure_registered_model(client: MlflowClient, registered_model_name: str) -> None:
    """
    Create the registered model entry if it does not exist yet.

    ============================ Arguments ============================
    client: MLflow client connected to the tracking server.
    registered_model_name: Registry name for the hybrid ranker (e.g. hybrid_ranker).
    """
    try:
        client.create_registered_model(registered_model_name)
    except MlflowException as exc:
        if "RESOURCE_ALREADY_EXISTS" not in str(exc):
            raise


def register_hybrid_model_version(*, tracking_uri: str, registered_model_name: str, mlflow_run_id: str, model_version: str) -> int:
    """
    Register one hybrid ranker training run as a new model version in MLflow.

    Do this by:
    1. Ensuring the registered model exists.
    2. Creating a model version from the training run artifact URI.
    3. Tagging the version with the MinIO model_version string for later lookup.

    ============================ Arguments ============================
    tracking_uri: MLflow tracking server URI.
    registered_model_name: Registry name for the hybrid ranker.
    mlflow_run_id: MLflow run id from model_config.json.
    model_version: MinIO artifact version id (e.g. hybrid-v1-2026-06-25T123000Z).

    ============================ Returns ============================
    The numeric MLflow model version assigned by the registry.
    """
    # 1. Ensure the registered model exists.
    client = _get_client(tracking_uri)
    ensure_registered_model(client, registered_model_name)

    # 2. Create a model version from the training run artifact URI.
    source_uri = f"runs:/{mlflow_run_id}"
    created = client.create_model_version(
        name=registered_model_name,
        source=source_uri,
        run_id=mlflow_run_id,
    )

    # 3. Tag the version with the MinIO model_version string for later lookup.
    registry_version = int(created.version)

    # 4. Tag the version with the MinIO model_version string for later lookup.
    client.set_model_version_tag(
        registered_model_name,
        str(registry_version),
        MODEL_VERSION_TAG,
        model_version,
    )

    return registry_version


def set_model_alias(*, tracking_uri: str, registered_model_name: str, alias: str, registry_version: int) -> None:
    """
    Point a registry alias (main or candidate) at one model version.

    ============================ Arguments ============================
    tracking_uri: MLflow tracking server URI.
    registered_model_name: Registry name for the hybrid ranker.
    alias: Alias name (main or candidate).
    registry_version: Numeric MLflow model version to assign.
    """
    client = _get_client(tracking_uri)
    client.set_registered_model_alias(registered_model_name, alias, str(registry_version))


def resolve_model_version_by_alias(*, tracking_uri: str, registered_model_name: str, alias: str) -> str | None:
    """
    Resolve a registry alias to the MinIO model_version string.

    Do this by:
    1. Looking up the alias in the Model Registry.
    2. Reading the model_version tag stored at registration time.

    ============================ Arguments ============================
    tracking_uri: MLflow tracking server URI.
    registered_model_name: Registry name for the hybrid ranker.
    alias: Alias name (main or candidate).

    ============================ Returns ============================
    The MinIO model_version string, or None when the alias is not set.
    """
    client = _get_client(tracking_uri)
    
    # Find the MLFlow version tied to 'main' or 'candidate'.
    try:
        # For this registered MLflow model, what MinIO model_version does alias X currently point to?
        model_version_info = client.get_model_version_by_alias(registered_model_name, alias)
    except MlflowException:
        return None

    # Get the tags associated with this MLFlow version.
    tags = model_version_info.tags or {}
    tagged_version = tags.get(MODEL_VERSION_TAG)
    if tagged_version:
        return tagged_version

    # If no model_version tag is found, look up the run_id and get the model_version tag from the run.
    run_id = model_version_info.run_id
    if run_id:
        run = client.get_run(run_id)
        run_tag = run.data.tags.get("model_version")
        if run_tag:
            return run_tag

    return None


def assign_main_or_candidate_alias(*, tracking_uri: str, registered_model_name: str, mlflow_run_id: str, model_version: str) -> str:
    """
    Register a evaluated model and assign main or candidate alias.

    Do this by:
    1. Registering the model version from the MLflow run.
    2. Setting main when no main alias exists yet.
    3. Otherwise setting candidate when the version differs from current main.

    ============================ Arguments ============================
    tracking_uri: MLflow tracking server URI.
    registered_model_name: Registry name for the hybrid ranker.
    mlflow_run_id: MLflow run id from model_config.json.
    model_version: MinIO artifact version id for this model.

    ============================ Returns ============================
    The alias assigned: main or candidate.
    """
    # 1. Register the model version from the MLflow run.
    registry_version = register_hybrid_model_version(
        tracking_uri=tracking_uri,
        registered_model_name=registered_model_name,
        mlflow_run_id=mlflow_run_id,
        model_version=model_version,
    )

    # 2. Check if the current main alias is set.
    current_main = resolve_model_version_by_alias(
        tracking_uri=tracking_uri,
        registered_model_name=registered_model_name,
        alias="main",
    )

    # 3. If no main alias is set, set it to the new model version.
    if current_main is None:
        set_model_alias(
            tracking_uri=tracking_uri,
            registered_model_name=registered_model_name,
            alias="main",
            registry_version=registry_version,
        )
        logger.info("Set MLflow alias main", extra={"model_version": model_version})
        return "main"

    # 4. If the current main alias is set to the new model version, return main.
    if current_main == model_version:
        logger.info("Evaluated model already main", extra={"model_version": model_version})
        return "main"

    # 5. If the current main alias is set to a different model version, set the candidate alias to the new model version.
    set_model_alias(
        tracking_uri=tracking_uri,
        registered_model_name=registered_model_name,
        alias="candidate",
        registry_version=registry_version,
    )
    logger.info("Set MLflow alias candidate", extra={"model_version": model_version})
    return "candidate"


def promote_candidate_to_main(*, tracking_uri: str, registered_model_name: str) -> str | None:
    """
    Promote the candidate alias to main and clear the candidate alias.

    Do this by:
    1. Reading the candidate model version from the registry.
    2. Moving the candidate registry version to the main alias.
    3. Deleting the candidate alias so a new challenger can be registered later.

    ============================ Arguments ============================
    tracking_uri: MLflow tracking server URI.
    registered_model_name: Registry name for the hybrid ranker.

    ============================ Returns ============================
    The promoted MinIO model_version string, or None when no candidate exists.
    """
    client = _get_client(tracking_uri)

    # 1. Read the candidate model version from the registry.
    candidate_version_str = resolve_model_version_by_alias(
        tracking_uri=tracking_uri,
        registered_model_name=registered_model_name,
        alias="candidate",
    )

    # 2. If no candidate alias is set, return None.
    if candidate_version_str is None:
        return None

    try:
        candidate_info = client.get_model_version_by_alias(registered_model_name, "candidate")
    except MlflowException:
        return None

    registry_version = int(candidate_info.version)

    try:
        old_main_info = client.get_model_version_by_alias(registered_model_name, "main")
        client.set_model_version_tag(
            registered_model_name,
            old_main_info.version,
            "status",
            "retired",
        )
    except MlflowException:
        pass

    set_model_alias(
        tracking_uri=tracking_uri,
        registered_model_name=registered_model_name,
        alias="main",
        registry_version=registry_version,
    )
    client.delete_registered_model_alias(registered_model_name, "candidate")
    logger.info("Promoted candidate to main", extra={"model_version": candidate_version_str})
    return candidate_version_str

"""Load hybrid ranker models and experiment state for recommender-api."""

from __future__ import annotations

import logging

import torch
from botocore.client import BaseClient

from common.features.schema import FEATURE_SCHEMA_VERSION, INPUT_DIM
from common.mlflow.registry import resolve_model_version_by_alias
from common.storage.cf_embedding_cache import CfEmbeddingCache

from train_hybrid_ranker.model import HybridRankerMLP
from recommender_api.runtime import LoadedHybridModel
from recommender_api.settings import RecommenderApiSettings
from common.db.repositories.experiments import get_or_create_active_experiment

from common.storage.hybrid_ranker_model_reader import (
    load_hybrid_model_checkpoint_dict,
    load_hybrid_model_config,
    resolve_hybrid_ranker_model_version,
)

logger = logging.getLogger(__name__)


def resolve_main_model_version(settings: RecommenderApiSettings, s3_client: BaseClient) -> str:
    """
    Resolve which MinIO hybrid model version to load as main.

    Do this by:
    1. Using MODEL_VERSION when explicitly set (local debugging override).
    2. Reading the MLflow main alias when available.
    3. Falling back to the latest complete MinIO manifest.

    ============================ Arguments ============================
    settings: Recommender API settings.
    s3_client: Boto3 S3 client for MinIO.

    ============================ Returns ============================
    The MinIO model_version string for the main model.
    """
    # If MODEL_VERSION is set, use it as an explicit override.
    # This is useful for local debugging or testing one specific trained model.
    if settings.model_version:
        return resolve_hybrid_ranker_model_version(
            s3_client,
            settings.s3_bucket,
            settings.model_version,
        )

    # If MODEL_VERSION is not set, use the MLflow main alias to resolve the model version.
    # This is the standard way to use the MLflow model registry to track and version models.
    alias_version = resolve_model_version_by_alias(
        tracking_uri=settings.mlflow_tracking_uri,
        registered_model_name=settings.mlflow_registered_model_name,
        alias="main",
    )
    if alias_version:
        return alias_version

    return resolve_hybrid_ranker_model_version(s3_client, settings.s3_bucket, None)


def resolve_candidate_model_version(settings: RecommenderApiSettings, main_version: str) -> str | None:
    """
    Resolve the MinIO model version for the candidate alias, if any.

    ============================ Arguments ============================
    settings: Recommender API settings.
    main_version: Already-resolved main model version (candidate must differ).

    ============================ Returns ============================
    Candidate model_version string, or None when no candidate alias is set.
    """
    # Look for the MLflow "candidate" alias.
    # This represents a challenger model we may want to A/B test against main.
    alias_version = resolve_model_version_by_alias(
        tracking_uri=settings.mlflow_tracking_uri,
        registered_model_name=settings.mlflow_registered_model_name,
        alias="candidate",
    )
    if alias_version is None or alias_version == main_version:
        return None
    return alias_version


def load_hybrid_model_bundle(*, s3_client: BaseClient, settings: RecommenderApiSettings, model_version: str, \
                                device: torch.device, cf_version_override: str | None = None) -> LoadedHybridModel:
    """
    Load one hybrid ranker model, config, and matching CF embedding cache.

    ============================ Arguments ============================
    s3_client: Boto3 S3 client for MinIO.
    settings: Recommender API settings.
    model_version: MinIO model version to load.
    device: Torch device for inference.
    cf_version_override: Optional CF version override for the main model only.

    ============================ Returns ============================
    LoadedHybridModel with eval-mode network and CF cache.
    """
    # Load the model's config/manifest first.
    # This tells us the expected input dimension, hidden layers, dropout, CF version, etc.
    model_config = load_hybrid_model_config(s3_client, settings.s3_bucket, model_version)

    # Make sure the model was trained with the same feature vector size
    # that the recommender API is about to build at inference time.
    if model_config.input_dim != INPUT_DIM:
        raise ValueError(
            f"Model input_dim {model_config.input_dim} does not match expected {INPUT_DIM}"
        )

    # Make sure the feature schema matches.
    # This protects us from serving a model trained on an older/different feature layout.
    if model_config.feature_schema_version != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            "Feature schema mismatch: "
            f"model={model_config.feature_schema_version}, expected={FEATURE_SCHEMA_VERSION}"
        )

    # Recreate the same MLP architecture used during training.
    # The architecture comes from the saved model config.
    checkpoint = load_hybrid_model_checkpoint_dict(s3_client, settings.s3_bucket, model_version)
    model = HybridRankerMLP(
        input_dim=model_config.input_dim,
        hidden_dims=model_config.hidden_dims,
        dropout=model_config.dropout,
    )

    # Load the trained weights into the model.
    model.load_state_dict(checkpoint["model_state_dict"])

    # Put the model in inference mode.
    # This disables training-only behavior like dropout.
    model.eval()
    model.to(device)

    # Load the CF movie embedding cache that matches this hybrid model.
    # The hybrid ranker expects CF features from the same cf_version used during training.
    cf_cache = CfEmbeddingCache.load(
        s3_client,
        settings.s3_bucket,
        cf_version_override=cf_version_override,
        expected_cf_version=model_config.cf_version,
    )

    # Return everything the recommender runtime needs to score candidates:
    # the neural ranker, its config/metadata, model version, and CF embeddings.
    return LoadedHybridModel(
        model=model,
        model_config=model_config,
        model_version=model_version,
        cf_cache=cf_cache,
    )


def bootstrap_experiment_state(
    session_factory,
    settings: RecommenderApiSettings,
    *,
    main_version: str,
    candidate_version: str | None,
):
    """
    Load or create the active experiment row for online split tracking.

    ============================ Arguments ============================
    session_factory: SQLAlchemy session factory.
    settings: Recommender API settings.
    main_version: Loaded main model version.
    candidate_version: Loaded candidate model version, if any.

    ============================ Returns ============================
    ExperimentState from Postgres.
    """
    session = session_factory()
    try:
        state = get_or_create_active_experiment(
            session,
            experiment_id=settings.experiment_id,
            main_model_version=main_version,
            candidate_model_version=candidate_version,
            initial_main_split=settings.initial_main_split,
            initial_candidate_split=settings.initial_candidate_split,
        )
        session.commit()
        return state
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

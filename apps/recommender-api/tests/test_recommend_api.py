"""Tests for /v1/recommend validation behavior."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import torch
from fastapi.testclient import TestClient
from train_hybrid_ranker.model import HybridRankerMLP

from common.db.repositories.experiments import ExperimentState
from common.features.normalization import MetadataNormalizationStats
from common.features.schema import INPUT_DIM
from common.metrics.recommender import RecommenderMetrics
from common.opensearch.retrieval import RetrievalSettings
from common.recommendation.split_policy import SplitPolicySettings
from common.schemas.hybrid_ranker_artifact_manifest import HybridModelConfig
from common.storage.cf_embedding_cache import CfEmbeddingCache
from recommender_api.app import create_app
from recommender_api.runtime import InferenceRuntime, LoadedHybridModel
from recommender_api.settings import RecommenderApiSettings

# One metrics instance for this module so Prometheus collectors are not re-registered.
_TEST_METRICS = RecommenderMetrics(service_name="recommender-api-tests")


def _build_test_runtime() -> InferenceRuntime:
    model = HybridRankerMLP()
    model.eval()
    now = datetime.now(tz=UTC)
    stats = MetadataNormalizationStats(
        year_min=1900,
        year_max=2025,
        runtime_min=0,
        runtime_max=300,
        tmdb_popularity_log_min=0,
        tmdb_popularity_log_max=10,
        tmdb_vote_average_min=0,
        tmdb_vote_average_max=10,
        tmdb_vote_count_log_min=0,
        tmdb_vote_count_log_max=10,
    )
    model_config = HybridModelConfig(
        model_version="test-model",
        dataset_version="test-dataset",
        snapshot_id="test-snapshot",
        cf_dataset_version="test-cf-dataset",
        cf_version="test-cf",
        content_embedding_version="content-v1",
        feature_schema_version="hybrid-v1",
        input_dim=INPUT_DIM,
        dropout=0.1,
        num_epochs=1,
        batch_size=1,
        learning_rate=0.001,
        early_stopping_patience=1,
        shuffle_seed=1,
        device="cpu",
        pipeline_run_id="test-run",
        metadata_normalization=stats,
        created_at=now,
    )
    main = LoadedHybridModel(
        model=model,
        model_config=model_config,
        model_version="test-model",
        cf_cache=CfEmbeddingCache(embeddings={}, cf_version="test-cf"),
    )
    return InferenceRuntime(
        main=main,
        candidate=None,
        experiment=ExperimentState(
            experiment_id="test-experiment",
            main_model_version="test-model",
            candidate_model_version=None,
            main_split_fraction=1.0,
            candidate_split_fraction=0.0,
            status="active",
        ),
        split_policy=SplitPolicySettings(),
        experiment_id="test-experiment",
        mlflow_tracking_uri="http://localhost:5000",
        mlflow_registered_model_name="hybrid-ranker",
        session_factory=MagicMock(),
        opensearch_client=MagicMock(),
        kafka_producer=MagicMock(),
        metrics=_TEST_METRICS,
        opensearch_index_alias="movies",
        retrieval=RetrievalSettings(knn_size=10, max_candidates=20),
        min_ratings_for_recommend=5,
        default_top_k=20,
        device=torch.device("cpu"),
        initial_main_split=1.0,
        initial_candidate_split=0.0,
    )


def test_recommend_requires_authentication() -> None:
    settings = RecommenderApiSettings()
    client = TestClient(create_app(_build_test_runtime(), settings))
    response = client.post("/v1/recommend", json={"ratings": [], "top_k": 5})
    assert response.status_code == 401


def test_recommend_accepts_exclude_movie_ids_before_auth_check() -> None:
    """Schema must accept exclude_movie_ids; auth still fails without a session."""
    settings = RecommenderApiSettings()
    client = TestClient(create_app(_build_test_runtime(), settings))
    response = client.post(
        "/v1/recommend",
        json={"ratings": [], "top_k": 5, "exclude_movie_ids": [101, 202, 303]},
    )
    assert response.status_code == 401

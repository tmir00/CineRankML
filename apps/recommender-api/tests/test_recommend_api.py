"""Tests for /v1/recommend validation behavior."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch
from fastapi.testclient import TestClient
from train_hybrid_ranker.model import HybridRankerMLP

from common.features.schema import INPUT_DIM
from common.metrics.recommender import RecommenderMetrics
from common.schemas.hybrid_ranker_artifact_manifest import HybridModelConfig
from common.features.normalization import MetadataNormalizationStats
from common.storage.cf_embedding_cache import CfEmbeddingCache
from recommender_api.app import create_app
from recommender_api.runtime import InferenceRuntime
from recommender_api.settings import RecommenderApiSettings
from datetime import UTC, datetime


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
    return InferenceRuntime(
        model=model,
        model_config=model_config,
        model_version="test-model",
        cf_cache=CfEmbeddingCache(embeddings={}, cf_version="test-cf"),
        session_factory=MagicMock(),
        opensearch_client=MagicMock(),
        kafka_producer=MagicMock(),
        metrics=RecommenderMetrics(),
        opensearch_index_alias="movies",
        candidate_pool_size=10,
        min_ratings_for_recommend=5,
        default_top_k=20,
        device=torch.device("cpu"),
    )


def test_recommend_requires_authentication() -> None:
    settings = RecommenderApiSettings()
    client = TestClient(create_app(_build_test_runtime(), settings))
    response = client.post("/v1/recommend", json={"ratings": [], "top_k": 5})
    assert response.status_code == 401

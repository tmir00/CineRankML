"""Tests for /v1/ratings list and delete routes."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import torch
from fastapi.testclient import TestClient
from train_hybrid_ranker.model import HybridRankerMLP

from common.features.normalization import MetadataNormalizationStats
from common.features.schema import INPUT_DIM
from common.opensearch.retrieval import RetrievalSettings
from common.schemas.hybrid_ranker_artifact_manifest import HybridModelConfig
from common.storage.cf_embedding_cache import CfEmbeddingCache
from recommender_api.app import create_app
from recommender_api.runtime import InferenceRuntime
from recommender_api.settings import RecommenderApiSettings


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
        metrics=MagicMock(),
        opensearch_index_alias="movies",
        retrieval=RetrievalSettings(knn_size=10, max_candidates=20),
        min_ratings_for_recommend=5,
        default_top_k=20,
        device=torch.device("cpu"),
    )


def test_list_ratings_requires_authentication() -> None:
    settings = RecommenderApiSettings()
    client = TestClient(create_app(_build_test_runtime(), settings))
    response = client.get("/v1/ratings")
    assert response.status_code == 401


@patch("recommender_api.routes.ratings.get_current_user")
@patch("recommender_api.routes.ratings.catalog_movie_exists", return_value=True)
@patch("recommender_api.routes.ratings.user_has_active_rating", return_value=False)
def test_delete_rating_returns_422_when_not_actively_rated(
    _mock_active: MagicMock,
    _mock_catalog: MagicMock,
    mock_user: MagicMock,
) -> None:
    mock_user.return_value = MagicMock(user_id=1)
    settings = RecommenderApiSettings()
    client = TestClient(create_app(_build_test_runtime(), settings))
    response = client.delete("/v1/ratings/50")
    assert response.status_code == 422


@patch("recommender_api.routes.ratings.get_current_user")
@patch("recommender_api.routes.ratings.catalog_movie_exists", return_value=True)
@patch("recommender_api.routes.ratings.user_has_active_rating", return_value=True)
@patch("recommender_api.routes.ratings.publish_rating_event")
def test_delete_rating_publishes_event_when_actively_rated(
    mock_publish: MagicMock,
    _mock_active: MagicMock,
    _mock_catalog: MagicMock,
    mock_user: MagicMock,
) -> None:
    mock_user.return_value = MagicMock(user_id=1)
    runtime = _build_test_runtime()
    settings = RecommenderApiSettings()
    client = TestClient(create_app(runtime, settings))
    response = client.delete("/v1/ratings/50")
    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    mock_publish.assert_called_once()
    runtime.kafka_producer.flush.assert_called_once()

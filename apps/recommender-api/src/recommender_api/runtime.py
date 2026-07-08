"""Loaded model and client dependencies for recommender-api."""

from __future__ import annotations

import torch

from dataclasses import dataclass
from opensearchpy import OpenSearch
from sqlalchemy.orm import sessionmaker
from common.kafka.producer import KafkaEventProducer
from train_hybrid_ranker.model import HybridRankerMLP
from common.metrics.recommender import RecommenderMetrics
from common.opensearch.retrieval import RetrievalSettings
from common.storage.cf_embedding_cache import CfEmbeddingCache
from common.recommendation.split_policy import SplitPolicySettings
from common.db.repositories.experiments import ExperimentState
from common.schemas.hybrid_ranker_artifact_manifest import HybridModelConfig


@dataclass
class LoadedHybridModel:
    """One hybrid ranker loaded from MinIO at startup."""

    model: HybridRankerMLP
    model_config: HybridModelConfig
    model_version: str
    cf_cache: CfEmbeddingCache


@dataclass
class InferenceRuntime:
    """Process-level objects loaded once at recommender-api startup."""

    main: LoadedHybridModel
    candidate: LoadedHybridModel | None
    experiment: ExperimentState
    split_policy: SplitPolicySettings
    experiment_id: str
    mlflow_tracking_uri: str
    mlflow_registered_model_name: str
    session_factory: sessionmaker
    opensearch_client: OpenSearch
    kafka_producer: KafkaEventProducer
    metrics: RecommenderMetrics
    opensearch_index_alias: str
    retrieval: RetrievalSettings
    min_ratings_for_recommend: int
    default_top_k: int
    device: torch.device
    initial_main_split: float
    initial_candidate_split: float

    @property
    def model(self) -> HybridRankerMLP:
        """Backward-compatible accessor for the main hybrid model."""
        return self.main.model

    @property
    def model_config(self) -> HybridModelConfig:
        """Backward-compatible accessor for the main model config."""
        return self.main.model_config

    @property
    def model_version(self) -> str:
        """Backward-compatible accessor for the main model version."""
        return self.main.model_version

    @property
    def cf_cache(self) -> CfEmbeddingCache:
        """Backward-compatible accessor for the main model CF cache."""
        return self.main.cf_cache

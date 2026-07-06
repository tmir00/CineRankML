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
from common.schemas.hybrid_ranker_artifact_manifest import HybridModelConfig


@dataclass
class InferenceRuntime:
    """Process-level objects loaded once at recommender-api startup."""

    model: HybridRankerMLP
    model_config: HybridModelConfig
    model_version: str
    cf_cache: CfEmbeddingCache
    session_factory: sessionmaker
    opensearch_client: OpenSearch
    kafka_producer: KafkaEventProducer
    metrics: RecommenderMetrics
    opensearch_index_alias: str
    retrieval: RetrievalSettings
    min_ratings_for_recommend: int
    default_top_k: int
    device: torch.device

"""Entrypoint for the recommender-api service."""

from __future__ import annotations

import sys
import torch
import uvicorn
import logging

from recommender_api.app import create_app
from common.storage.s3 import create_s3_client
from common.db.session import get_session_factory
from recommender_api.runtime import InferenceRuntime
from common.kafka.producer import KafkaEventProducer
from train_hybrid_ranker.model import HybridRankerMLP
from common.metrics.recommender import RecommenderMetrics
from recommender_api.settings import RecommenderApiSettings
from common.opensearch.client import create_opensearch_client
from common.storage.cf_embedding_cache import CfEmbeddingCache
from common.features.schema import FEATURE_SCHEMA_VERSION, INPUT_DIM
from common.config.settings import get_kafka_settings, get_opensearch_settings


from common.storage.hybrid_ranker_model_reader import (
    load_hybrid_model_checkpoint_dict,
    load_hybrid_model_config,
    resolve_hybrid_ranker_model_version,
)

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Set up basic structured logging for recommender-api."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )


def load_runtime(settings: RecommenderApiSettings) -> InferenceRuntime:
    """
    Load all process-level inference dependencies at startup.

    Do this by:
    1. Loading the main hybrid model and validating artifact lineage fields.
    2. Loading the matching CF embedding cache from MinIO.
    3. Creating Postgres, OpenSearch, and Kafka clients.

    ============================ Arguments ============================
    settings: Recommender API runtime settings.

    ============================ Returns ============================
    Fully initialized InferenceRuntime.
    """
    # Create the S3 client.
    s3_client = create_s3_client(settings)

    # Resolve the hybrid ranker model version.
    model_version = resolve_hybrid_ranker_model_version(
        s3_client,
        settings.s3_bucket,
        settings.model_version,
    )
    # Load the hybrid ranker model config.
    model_config = load_hybrid_model_config(s3_client, settings.s3_bucket, model_version)

    # Validate the model input_dim and feature schema version.
    if model_config.input_dim != INPUT_DIM:
        raise ValueError(
            f"Model input_dim {model_config.input_dim} does not match expected {INPUT_DIM}"
        )
    if model_config.feature_schema_version != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            "Feature schema mismatch: "
            f"model={model_config.feature_schema_version}, expected={FEATURE_SCHEMA_VERSION}"
        )

    # Download the trained MLP weights from MinIO and put them in the PyTorch model.
    checkpoint = load_hybrid_model_checkpoint_dict(s3_client, settings.s3_bucket, model_version)

    # Create the hybrid ranker model.
    model = HybridRankerMLP(
        input_dim=model_config.input_dim,
        hidden_dims=model_config.hidden_dims,
        dropout=model_config.dropout,
    )
    # Load the model state dict.
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load the CF embedding cache.
    cf_cache = CfEmbeddingCache.load(
        s3_client,
        settings.s3_bucket,
        cf_version_override=settings.cf_version,
        expected_cf_version=model_config.cf_version,
    )

    # Use the best available device (GPU or CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Get the Kafka and OpenSearch settings.
    kafka_settings = get_kafka_settings()
    opensearch_settings = get_opensearch_settings()
    # Create the recommender metrics.
    metrics = RecommenderMetrics()
    # Start the metrics server.
    metrics.start_server(settings.metrics_port)

    # Create the inference runtime.
    runtime = InferenceRuntime(
        model=model,
        model_config=model_config,
        model_version=model_version,
        cf_cache=cf_cache,
        session_factory=get_session_factory(),
        opensearch_client=create_opensearch_client(opensearch_settings),
        kafka_producer=KafkaEventProducer(kafka_settings.kafka_bootstrap_servers),
        metrics=metrics,
        opensearch_index_alias=opensearch_settings.opensearch_index_alias,
        candidate_pool_size=settings.candidate_pool_size,
        min_ratings_for_recommend=settings.min_ratings_for_recommend,
        default_top_k=settings.default_top_k,
        device=device,
    )

    # Log the runtime details.
    logger.info(
        "recommender-api runtime loaded",
        extra={
            "model_version": model_version,
            "cf_version": cf_cache.cf_version,
            "device": str(device),
            "cf_embeddings_loaded": len(cf_cache),
        },
    )
    return runtime


def main() -> None:
    """
    Start the recommender-api HTTP service.

    Do this by:
    1. Loading runtime dependencies and failing fast when artifacts are missing.
    2. Creating the FastAPI app with injected dependencies.
    3. Running uvicorn on the configured host and port.
    """
    configure_logging()
    settings = RecommenderApiSettings()
    runtime = load_runtime(settings)
    app = create_app(runtime, settings)
    uvicorn.run(app, host=settings.recommender_host, port=settings.recommender_port)


if __name__ == "__main__":
    main()

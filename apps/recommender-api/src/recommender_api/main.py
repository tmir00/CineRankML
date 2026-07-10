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
from common.metrics.recommender import RecommenderMetrics
from recommender_api.settings import RecommenderApiSettings
from common.opensearch.client import create_opensearch_client
from common.opensearch.retrieval import RetrievalSettings
from common.config.settings import get_kafka_settings, get_opensearch_settings

from recommender_api.services.model_loader import (
    bootstrap_experiment_state,
    load_hybrid_model_bundle,
    resolve_candidate_model_version,
    resolve_main_model_version,
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
    1. Resolving main and candidate model versions from MLflow aliases or MinIO.
    2. Loading hybrid models and CF embedding caches for each role.
    3. Bootstrapping the active online experiment row in Postgres.
    4. Creating Postgres, OpenSearch, and Kafka clients.

    ============================ Arguments ============================
    settings: Recommender API runtime settings.

    ============================ Returns ============================
    Fully initialized InferenceRuntime.
    """
    s3_client = create_s3_client(settings)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main_version = resolve_main_model_version(settings, s3_client)
    candidate_version = resolve_candidate_model_version(settings, main_version)

    main_bundle = load_hybrid_model_bundle(
        s3_client,
        settings,
        main_version,
        device,
        cf_version_override=settings.cf_version,
    )

    candidate_bundle = None
    if candidate_version is not None:
        candidate_bundle = load_hybrid_model_bundle(
            s3_client,
            settings,
            candidate_version,
            device,
        )

    session_factory = get_session_factory()
    experiment = bootstrap_experiment_state(
        session_factory,
        settings,
        main_version=main_version,
        candidate_version=candidate_version,
    )

    kafka_settings = get_kafka_settings()
    opensearch_settings = get_opensearch_settings()
    metrics = RecommenderMetrics()
    metrics.start_server(settings.metrics_port)
    metrics.set_experiment_split_fraction("main", experiment.main_split_fraction)
    metrics.set_experiment_split_fraction("candidate", experiment.candidate_split_fraction)

    retrieval = RetrievalSettings(
        knn_size=settings.retrieval_knn_size,
        popular_size=settings.retrieval_popular_size,
        random_genre_size=settings.retrieval_random_genre_size,
        random_knn_size=settings.retrieval_random_knn_size,
        knn_pool_size=settings.retrieval_knn_pool_size,
        random_knn_skip_top=settings.retrieval_random_knn_skip_top,
        max_candidates=settings.retrieval_max_candidates,
        liked_genre_count=settings.retrieval_liked_genre_count,
        min_vote_count=settings.retrieval_min_vote_count,
        min_vote_average=settings.retrieval_min_vote_average,
    )

    runtime = InferenceRuntime(
        main=main_bundle,
        candidate=candidate_bundle,
        experiment=experiment,
        split_policy=settings.split_policy,
        experiment_id=settings.experiment_id,
        mlflow_tracking_uri=settings.mlflow_tracking_uri,
        mlflow_registered_model_name=settings.mlflow_registered_model_name,
        session_factory=session_factory,
        opensearch_client=create_opensearch_client(opensearch_settings),
        kafka_producer=KafkaEventProducer(kafka_settings.kafka_bootstrap_servers),
        metrics=metrics,
        opensearch_index_alias=opensearch_settings.opensearch_index_alias,
        retrieval=retrieval,
        min_ratings_for_recommend=settings.min_ratings_for_recommend,
        default_top_k=settings.default_top_k,
        device=device,
        initial_main_split=settings.initial_main_split,
        initial_candidate_split=settings.initial_candidate_split,
    )

    logger.info(
        "recommender-api runtime loaded",
        extra={
            "main_model_version": main_version,
            "candidate_model_version": candidate_version,
            "experiment_id": settings.experiment_id,
            "main_split": experiment.main_split_fraction,
            "candidate_split": experiment.candidate_split_fraction,
            "device": str(device),
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

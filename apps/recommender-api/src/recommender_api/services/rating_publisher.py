"""Publish API rating events to Kafka."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from common.config.settings import get_kafka_settings
from common.kafka.producer import KafkaEventProducer
from common.schemas.events import RatingCreatedEvent, RatingDeletedEvent


def build_api_rating_event(*, user_id: int, movie_id: int, rating: float, request_id: str | None = None, \
                            model_version: str | None = None, experiment_id: str | None = None) -> RatingCreatedEvent:
    """
    Build one rating_created event for API-originated ratings.

    Do this by:
    1. Stamping the event with source=api and the current UTC time.
    2. Attaching optional recommendation lineage fields when present.

    ============================ Arguments ============================
    user_id: Authenticated app user id.
    movie_id: Rated catalog movie id.
    rating: MovieLens-style rating between 0.5 and 5.0.
    request_id: Optional recommendation request id.
    model_version: Optional model version that served the recommendation.
    experiment_id: Optional experiment id for online tests.

    ============================ Returns ============================
    A validated RatingCreatedEvent ready for Kafka.
    """
    now = datetime.now(tz=UTC)
    return RatingCreatedEvent(
        event_id=uuid4(),
        event_type="rating_created",
        stream_pipeline_version="ratings-v1",
        source="api",
        occurred_at=now,
        produced_at=now,
        user_id=user_id,
        movie_id=movie_id,
        rating=rating,
        rating_timestamp=now,
        request_id=request_id,
        model_version=model_version,
        experiment_id=experiment_id,
    )


def build_api_rating_deleted_event(
    *,
    user_id: int,
    movie_id: int,
    request_id: str | None = None,
    model_version: str | None = None,
    experiment_id: str | None = None,
) -> RatingDeletedEvent:
    """
    Build one rating_deleted event for API-originated rating removals.

    Do this by:
    1. Stamping the event with source=api and the current UTC time.
    2. Attaching optional recommendation lineage fields when present.

    ============================ Arguments ============================
    user_id: Authenticated app user id.
    movie_id: Catalog movie id whose active rating should be removed.
    request_id: Optional recommendation request id.
    model_version: Optional model version that served the recommendation.
    experiment_id: Optional experiment id for online tests.

    ============================ Returns ============================
    A validated RatingDeletedEvent ready for Kafka.
    """
    now = datetime.now(tz=UTC)
    return RatingDeletedEvent(
        event_id=uuid4(),
        event_type="rating_deleted",
        stream_pipeline_version="ratings-v1",
        source="api",
        occurred_at=now,
        produced_at=now,
        user_id=user_id,
        movie_id=movie_id,
        rating_timestamp=now,
        request_id=request_id,
        model_version=model_version,
        experiment_id=experiment_id,
    )


def publish_rating_event(producer: KafkaEventProducer, event: RatingCreatedEvent | RatingDeletedEvent) -> None:
    """
    Publish one ratings-topic event to Kafka.

    ============================ Arguments ============================
    producer: Shared Kafka producer created at startup.
    event: Validated rating_created or rating_deleted event to publish.
    """
    kafka_settings = get_kafka_settings()
    producer.produce(kafka_settings.ratings_topic, event, key=str(event.event_id))

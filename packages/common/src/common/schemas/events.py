"""Pydantic event schemas shared by Kafka producers and consumers."""

from __future__ import annotations

import uuid

from uuid import UUID
from datetime import UTC, datetime
from typing import Literal, TypeVar
from pydantic import BaseModel, ValidationError

# Fixed namespace for UUID5 event_ids so the same logical event always gets the same id.
# Example: "rating:1:1:2021-01-01T00:00:00:3.5" -> "61a8a3d4-ce59-56be-86c3-9c12c7367834"
# The same CSV row will always get the same event_id. This is useful for deduplication.
EVENT_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "cinerankml.io/kafka-events")


class BaseKafkaEvent(BaseModel):
    """Shared fields on every Kafka event."""

    event_id: UUID
    event_type: str
    stream_pipeline_version: str
    source: str
    occurred_at: datetime
    produced_at: datetime


class RatingCreatedEvent(BaseKafkaEvent):
    """A user rating event published to the ratings topic."""

    event_type: Literal["rating_created"]
    user_id: int
    movie_id: int
    rating: float
    rating_timestamp: datetime
    request_id: str | None = None
    model_version: str | None = None
    experiment_id: str | None = None


class RatingDeletedEvent(BaseKafkaEvent):
    """A user rating removal event published to the ratings topic."""

    event_type: Literal["rating_deleted"]
    user_id: int
    movie_id: int
    rating_timestamp: datetime
    request_id: str | None = None
    model_version: str | None = None
    experiment_id: str | None = None


RatingStreamEvent = RatingCreatedEvent | RatingDeletedEvent


class TagCreatedEvent(BaseKafkaEvent):
    """A user tag event published to the tags topic."""

    event_type: Literal["tag_created"]
    user_id: int
    movie_id: int
    tag: str
    tag_timestamp: datetime


TEvent = TypeVar("TEvent", bound=BaseKafkaEvent)


def unix_ts_to_utc(ts: int) -> datetime:
    """Convert a MovieLens unix timestamp into a UTC datetime."""
    return datetime.fromtimestamp(ts, tz=UTC)


def _deterministic_uuid(key: str) -> UUID:
    """
    Build a stable UUID5 from a string key so the same CSV row always gets the same event_id.

    =============================================== Arguments ===============================================
    key: The string key to hash. E.g. "rating:1:1:2021-01-01T00:00:00:3.5"

    =============================================== Returns ===============================================
    A stable UUID for the key.
    """
    return uuid.uuid5(EVENT_ID_NAMESPACE, key)


def rating_event_id(user_id: int, movie_id: int, rating_timestamp: datetime, rating: float) -> UUID:
    """
    Create a stable id so re-publishing the same CSV row does not create a new event_id.
    
    =============================================== Arguments ===============================================
    user_id: The user ID.
    movie_id: The movie ID.
    rating_timestamp: The timestamp of the rating.
    rating: The rating value.

    =============================================== Returns ===============================================
    A stable UUID for the rating event.
    """
    key = f"rating:{user_id}:{movie_id}:{rating_timestamp.isoformat()}:{rating}"
    return _deterministic_uuid(key)


def tag_event_id(user_id: int, movie_id: int, tag: str, tag_timestamp: datetime) -> UUID:
    """Stable id so re-publishing the same CSV row does not create a new event_id."""
    key = f"tag:{user_id}:{movie_id}:{tag}:{tag_timestamp.isoformat()}"
    return _deterministic_uuid(key)


def validate_event(raw: dict, model: type[TEvent]) -> TEvent:
    """
    Parse and validate a raw dict into a typed Kafka event.

    ============================ Arguments ============================
    raw: The decoded JSON payload from Kafka.
    model: The Pydantic model class to validate against.

    ============================ Returns ============================
    A validated event instance.
    """
    return model.model_validate(raw)


def try_validate_event(raw: dict, model: type[TEvent]) -> tuple[TEvent | None, str | None]:
    """
    Try to validate a raw dict and return either the event or an error message.

    Do this by:
    1. Running Pydantic validation on the payload.
    2. Returning the event on success, or a readable error string on failure.

    ============================ Arguments ============================
    raw: The decoded JSON payload from Kafka.
    model: The Pydantic model class to validate against.

    ============================ Returns ============================
    A tuple of (event, error_message). Exactly one side is non-None.
    """
    try:
        return validate_event(raw, model), None
    except ValidationError as exc:
        return None, str(exc)


def is_validation_error(exc: BaseException) -> bool:
    """Return True when an exception came from Pydantic validation."""
    return isinstance(exc, ValidationError)


def parse_rating_stream_event(raw: dict) -> RatingStreamEvent:
    """
    Validate one ratings-topic payload into the correct event model.

    Do this by:
    1. Reading event_type from the decoded JSON dict.
    2. Validating with RatingCreatedEvent or RatingDeletedEvent.
    3. Raising ValidationError when event_type is missing or unknown.

    ============================ Arguments ============================
    raw: The decoded JSON payload from the ratings Kafka topic.

    ============================ Returns ============================
    A validated rating_created or rating_deleted event.
    """
    event_type = raw.get("event_type")
    if event_type == "rating_created":
        return RatingCreatedEvent.model_validate(raw)
    if event_type == "rating_deleted":
        return RatingDeletedEvent.model_validate(raw)
    raise ValueError(f"Unknown ratings event_type: {event_type!r}")


def rating_row_to_event(row: dict[str, str], produced_at: datetime | None = None) -> RatingCreatedEvent:
    """
    Build a rating_created event from one MovieLens ratings.csv row.

    Do this by:
    1. Reading userId, movieId, rating, and timestamp from the CSV row.
    2. Building a stable event_id from those fields.
    3. Filling the shared fields for the movielens ratings pipeline.

    ============================ Arguments ============================
    row: One CSV row as a dict (keys: userId, movieId, rating, timestamp).
    produced_at: When the producer sent the event. Defaults to now (UTC).

    ============================ Returns ============================
    A validated RatingCreatedEvent ready to publish.
    """
    # Parse the CSV row into the event fields.
    user_id = int(row["userId"])
    movie_id = int(row["movieId"])
    rating = float(row["rating"])
    rating_timestamp = unix_ts_to_utc(int(row["timestamp"]))
    now = produced_at or datetime.now(tz=UTC)

    # Build the event and return it.
    return RatingCreatedEvent(
        event_id=rating_event_id(user_id, movie_id, rating_timestamp, rating),
        event_type="rating_created",
        stream_pipeline_version="ratings-v1",
        source="movielens",
        occurred_at=rating_timestamp,
        produced_at=now,
        user_id=user_id,
        movie_id=movie_id,
        rating=rating,
        rating_timestamp=rating_timestamp,
    )


def tag_row_to_event(row: dict[str, str], produced_at: datetime | None = None) -> TagCreatedEvent:
    """
    Build a tag_created event from one MovieLens tags.csv row.

    Do this by:
    1. Reading userId, movieId, tag, and timestamp from the CSV row.
    2. Building a stable event_id from those fields.
    3. Filling the shared envelope fields for the movielens tags pipeline.

    ============================ Arguments ============================
    row: One CSV row as a dict (keys: userId, movieId, tag, timestamp).
    produced_at: When the producer sent the event. Defaults to now (UTC).

    ============================ Returns ============================
    A validated TagCreatedEvent ready to publish.
    """
    # Parse the CSV row into the event fields.
    user_id = int(row["userId"])
    movie_id = int(row["movieId"])
    tag = row["tag"]
    tag_timestamp = unix_ts_to_utc(int(row["timestamp"]))
    now = produced_at or datetime.now(tz=UTC)

    # Build the event and return it.
    return TagCreatedEvent(
        event_id=tag_event_id(user_id, movie_id, tag, tag_timestamp),
        event_type="tag_created",
        stream_pipeline_version="tags-v1",
        source="movielens",
        occurred_at=tag_timestamp,
        produced_at=now,
        user_id=user_id,
        movie_id=movie_id,
        tag=tag,
        tag_timestamp=tag_timestamp,
    )

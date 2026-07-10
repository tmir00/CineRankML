"""Insert idempotent rating and tag events into Postgres."""

from __future__ import annotations

from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert
from common.db.models.events import RatingsEvent, TagEvent
from common.schemas.events import RatingCreatedEvent, RatingDeletedEvent, TagCreatedEvent


def insert_rating_event(session: Session, event: RatingCreatedEvent) -> bool:
    """
    Insert one rating event in postgres when that event_id is not already stored.

    Do this by:
    1. Building a row from the validated Kafka event.
    2. Using ON CONFLICT DO NOTHING on event_id so that the same event 
    is not inserted multiple times.

    ============================ Arguments ============================
    session: An open SQLAlchemy session inside a transaction.
    event: A validated rating_created event.

    ============================ Returns ============================
    True when a new row was inserted, False when event_id already existed.
    """
    # Build the insert statement.
    stmt = (
        insert(RatingsEvent)
        .values(
            event_id=str(event.event_id),
            user_id=event.user_id,
            movie_id=event.movie_id,
            event_type="rating_created",
            rating=event.rating,
            rating_timestamp=event.rating_timestamp,
            stream_pipeline_version=event.stream_pipeline_version,
            source=event.source,
        )
        .on_conflict_do_nothing(index_elements=["event_id"])
        .returning(RatingsEvent.id)
    )
    # Execute the insert statement and return True if a new row was inserted, False if the event_id already existed.
    result = session.execute(stmt).scalar_one_or_none()
    return result is not None


def insert_rating_deleted_event(session: Session, event: RatingDeletedEvent) -> bool:
    """
    Insert one rating_deleted event when that event_id is not already stored.

    Do this by:
    1. Building a row from the validated Kafka event with a null rating value.
    2. Using ON CONFLICT DO NOTHING on event_id so the same event is not inserted twice.

    ============================ Arguments ============================
    session: An open SQLAlchemy session inside a transaction.
    event: A validated rating_deleted event.

    ============================ Returns ============================
    True when a new row was inserted, False when event_id already existed.
    """
    stmt = (
        insert(RatingsEvent)
        .values(
            event_id=str(event.event_id),
            user_id=event.user_id,
            movie_id=event.movie_id,
            event_type="rating_deleted",
            rating=None,
            rating_timestamp=event.rating_timestamp,
            stream_pipeline_version=event.stream_pipeline_version,
            source=event.source,
        )
        .on_conflict_do_nothing(index_elements=["event_id"])
        .returning(RatingsEvent.id)
    )
    result = session.execute(stmt).scalar_one_or_none()
    return result is not None


def insert_tag_event(session: Session, event: TagCreatedEvent) -> bool:
    """
    Insert one tag event when that event_id is not already stored.

    Do this by:
    1. Building a row from the validated Kafka event.
    2. Using ON CONFLICT DO NOTHING on event_id so that the same event 
    is not inserted multiple times.

    ============================ Arguments ============================
    session: An open SQLAlchemy session inside a transaction.
    event: A validated tag_created event.

    ============================ Returns ============================
    True when a new row was inserted, False when event_id already existed.
    """
    # Build the insert statement.
    stmt = (
        insert(TagEvent)
        .values(
            event_id=str(event.event_id),
            user_id=event.user_id,
            movie_id=event.movie_id,
            tag=event.tag,
            tag_timestamp=event.tag_timestamp,
            stream_pipeline_version=event.stream_pipeline_version,
            source=event.source,
        )
        .on_conflict_do_nothing(index_elements=["event_id"])
        .returning(TagEvent.id)
    )
    # Execute the insert statement and return True if a new row was inserted, False if the event_id already existed.
    result = session.execute(stmt).scalar_one_or_none()
    return result is not None

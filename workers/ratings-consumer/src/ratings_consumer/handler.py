"""Postgres write handler for ratings-topic events."""

from __future__ import annotations

from pydantic import BaseModel
from sqlalchemy.orm import Session
from common.db.session import get_session_factory
from common.schemas.events import RatingCreatedEvent, RatingDeletedEvent
from common.db.repositories.events import insert_rating_deleted_event, insert_rating_event


def process_rating_stream_event(event: BaseModel) -> str:
    """
    Insert one validated ratings-topic event into ratings_events.

    Do this by:
    1. Opening a database session and starting a transaction.
    2. Routing rating_created and rating_deleted events to the matching insert helper.
    3. Committing on success or rolling back on failure.

    ============================ Arguments ============================
    event: A validated rating_created or rating_deleted Kafka event.

    ============================ Returns ============================
    "success" when a new row was inserted, "duplicate" when event_id already existed.
    """
    session_factory = get_session_factory()
    session: Session = session_factory()

    try:
        if isinstance(event, RatingCreatedEvent):
            inserted = insert_rating_event(session, event)
        elif isinstance(event, RatingDeletedEvent):
            inserted = insert_rating_deleted_event(session, event)
        else:
            raise TypeError(f"Unsupported ratings event type: {type(event)!r}")

        session.commit()
        return "success" if inserted else "duplicate"

    except Exception:
        session.rollback()
        raise

    finally:
        session.close()

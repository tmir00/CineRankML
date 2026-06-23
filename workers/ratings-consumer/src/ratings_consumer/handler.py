"""Postgres write handler for rating_created events."""

from __future__ import annotations

from sqlalchemy.orm import Session
from common.db.session import get_session_factory
from common.schemas.events import RatingCreatedEvent
from common.db.repositories.events import insert_rating_event


def process_rating_event(event: RatingCreatedEvent) -> str:
    """
    Insert one validated rating event into ratings_events.

    Do this by:
    1. Opening a database session and starting a transaction.
    2. Inserting the row with idempotent event_id handling.
    3. Committing on success or rolling back on failure.

    ============================ Arguments ============================
    event: A validated rating_created Kafka event.

    ============================ Returns ============================
    "success" when a new row was inserted, "duplicate" when event_id already existed.
    """
    # Get the session factory.
    session_factory = get_session_factory()
    # Create a new session.
    session: Session = session_factory()

    # Try to insert the event.
    try:
        # Insert the event.
        inserted = insert_rating_event(session, event)
        # Commit the session.
        session.commit()
        return "success" if inserted else "duplicate"
    
    except Exception:
        # In the case of an error, rollback the session.
        session.rollback()
        # Raise the exception.
        raise
    
    finally:
        session.close()

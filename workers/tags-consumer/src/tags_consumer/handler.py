"""Postgres write handler for tag_created events."""

from __future__ import annotations

from sqlalchemy.orm import Session

from common.db.repositories.events import insert_tag_event
from common.db.repositories.tags import mark_movie_dirty_if_catalog_exists, upsert_movie_tag_count
from common.db.session import get_session_factory
from common.schemas.events import TagCreatedEvent


def process_tag_event(event: TagCreatedEvent) -> str:
    """
    Insert one tag event and apply tag side effects in a single transaction.

    Do this by:
    1. Inserting the tag event idempotently.
    2. Upserting movie_tag_counts for the movie and tag.
    3. Marking catalog_dirty_movies when the movie exists in the catalog.

    ============================ Arguments ============================
    event: A validated tag_created Kafka event.

    ============================ Returns ============================
    "success" when a new tag event row was inserted, "duplicate" when event_id already existed.
    """
    session_factory = get_session_factory()
    session: Session = session_factory()

    try:
        inserted = insert_tag_event(session, event)

        # Only apply side effects when this is a new tag event row.
        if inserted:
            upsert_movie_tag_count(session, event.movie_id, event.tag)
            mark_movie_dirty_if_catalog_exists(session, event.movie_id)

        session.commit()
        return "success" if inserted else "duplicate"
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

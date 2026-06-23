"""Tag side-effect writes: counts and catalog dirty marking."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from common.db.models.catalog import CatalogDirtyMovie, CatalogMovie, MovieTagCount

logger = logging.getLogger(__name__)


def upsert_movie_tag_count(session: Session, movie_id: int, tag: str) -> None:
    """
    Increase the tag count for one movie by one.

    Do this by:
    1. Inserting a (movie_id, tag) row with count=1 when it does not exist.
    2. Bumping count and updated_at when the row already exists.

    ============================ Arguments ============================
    session: An open SQLAlchemy session inside a transaction.
    movie_id: The movie that received the tag.
    tag: The tag text.
    """
    # Get the current timestamp.
    now = datetime.now(tz=UTC)
    
    # Build the insert statement.
    stmt = (
        insert(MovieTagCount)
        .values(movie_id=movie_id, tag=tag, count=1, updated_at=now)
        .on_conflict_do_update(
            index_elements=["movie_id", "tag"],
            set_={
                "count": MovieTagCount.count + 1,
                "updated_at": now,
            },
        )
    )
    session.execute(stmt)


def mark_movie_dirty_if_catalog_exists(session: Session, movie_id: int) -> bool:
    """
    Mark a movie as needing OpenSearch sync when it exists in the catalog.

    Do this by:
    1. Checking whether the movie_id is present in catalog_movies.
    2. Inserting or updating catalog_dirty_movies when the movie is known.
    3. Skipping dirty marking when the catalog has not been seeded yet.

    ============================ Arguments ============================
    session: An open SQLAlchemy session inside a transaction.
    movie_id: The movie that changed because of a new tag.

    ============================ Returns ============================
    True when the movie was marked dirty, False when the movie is not in catalog yet.
    """
    # Check if the movie_id is present in catalog_movies.
    exists = session.execute(
        select(CatalogMovie.movie_id).where(CatalogMovie.movie_id == movie_id)
    ).scalar_one_or_none()

    # If the movie_id is not present in catalog_movies, log the skip and return False.
    if exists is None:
        logger.debug(
            "Skipping dirty mark because movie is not in catalog yet",
            extra={"movie_id": movie_id},
        )
        return False

    # Get the current timestamp.
    now = datetime.now(tz=UTC)
    
    # Build the insert statement.
    stmt = (
        insert(CatalogDirtyMovie)
        .values(
            movie_id=movie_id,
            first_dirty_at=now,
            last_dirty_at=now,
            attempt_count=0,
            last_error=None,
        )
        .on_conflict_do_update(
            index_elements=["movie_id"],
            set_={
                "last_dirty_at": now,
            },
        )
    )
    session.execute(stmt)
    return True

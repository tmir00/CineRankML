"""Reads PostgreSQL tables in chunks, so that we can export them in batches."""

from __future__ import annotations

from datetime import UTC, datetime

from typing import Any, Callable
from sqlalchemy.orm import Session
from collections.abc import Iterator
from sqlalchemy import and_, or_, select
from common.db.models.catalog import CatalogMovie
from common.db.models.embeddings import MovieContentEmbedding
from common.db.models.events import RatingsEvent, TagEvent


def _model_to_dict(row: Any) -> dict[str, Any]:
    """Convert one SQLAlchemy ORM model instance to a plain dict of column values."""
    return {col.name: getattr(row, col.name) for col in row.__table__.columns}


def _iter_keyset_batches(session: Session, batch_size: int, fetch_batch: Callable[[Session, int, int], list[Any]],
                            get_cursor: Callable[[Any], int]) -> Iterator[list[dict[str, Any]]]:
    """
    Yield model rows as dict batches by using a cursor to track the last row processed.

    ========================================== Arguments ==========================================
    session: An open SQLAlchemy session.
    batch_size: Maximum rows per batch.
    fetch_batch: Callable that loads the next batch given (session, last_cursor, batch_size).
    get_cursor: Callable that extracts the pagination cursor from the last row.

    ========================================== Returns ==========================================
    An iterator of lists of dicts, each representing a batch of rows.
    """
    # Initialize the last cursor to 0.
    last_cursor = 0

    # Loop until we break out of the loop.
    while True:
        # Fetch the next batch of rows.
        rows = fetch_batch(session, last_cursor, batch_size)
        
        # If there are no more rows, break out of the loop.
        if not rows:
            break
        
        # Yield the rows as dicts.
        yield [_model_to_dict(row) for row in rows]
        
        # Update the last cursor to the cursor of the last row in the batch.
        last_cursor = get_cursor(rows[-1])


def iter_ratings_events(session: Session, batch_size: int) -> Iterator[list[dict[str, Any]]]:
    """
    Yield ratings_events rows in time-ordered keyset-paginated batches.

    Do this by:
    1. Defining a fetch function that loads the next batch after (rating_timestamp, id).
    2. Yielding each batch as a list of row dicts until no rows remain.

    ========================================== Arguments ==========================================
    session: An open SQLAlchemy session.
    batch_size: Maximum rows per batch.

    ========================================== Returns ==========================================
    An iterator of lists of dicts, each representing a batch of rows.
    """
    # Start before the first possible row so the first page includes the earliest rating.
    last_timestamp = datetime.min.replace(tzinfo=UTC)
    last_id = 0

    # While we have rows to yield, yield the rows in time-ordered keyset-paginated batches.
    while True:
        # Build the SQL statement to fetch the next batch of rows.
        stmt = (
            # Select the next batch of rows.
            select(RatingsEvent)
            .where( # Where
                or_( # Either
                    RatingsEvent.rating_timestamp > last_timestamp, # The rating timestamp is greater than the last timestamp. Or..
                    and_( 
                        RatingsEvent.rating_timestamp == last_timestamp, # The rating timestamp is equal to the last timestamp. And..
                        RatingsEvent.id > last_id, # The id is greater than the last id.
                    ),
                )
            )
            .order_by(RatingsEvent.rating_timestamp, RatingsEvent.id)
            .limit(batch_size)
        )
        rows = list(session.scalars(stmt).all())
        if not rows:
            break

        yield [_model_to_dict(row) for row in rows]
        last_timestamp = rows[-1].rating_timestamp
        last_id = rows[-1].id


def iter_tag_events(session: Session, batch_size: int) -> Iterator[list[dict[str, Any]]]:
    """
    Yield tag_events rows in keyset-paginated batches.

    Do this by:
    1. Defining a fetch function that loads the next batch of rows given (session, last_cursor, batch_size).
    2. Calling _iter_keyset_batches so that it yields the rows as lists of dicts.
    
    ========================================== Arguments ==========================================
    session: An open SQLAlchemy session.

    ========================================== Returns ==========================================
    An iterator of lists of dicts, each representing a batch of rows.
    """

    def fetch(session: Session, last_id: int, limit: int) -> list[TagEvent]:
        stmt = (
            select(TagEvent)
            .where(TagEvent.id > last_id)
            .order_by(TagEvent.id)
            .limit(limit)
        )
        return list(session.scalars(stmt).all())

    yield from _iter_keyset_batches(session, batch_size, fetch, lambda row: row.id)


def iter_catalog_movies(session: Session, batch_size: int) -> Iterator[list[dict[str, Any]]]:
    """
    Yield catalog_movies rows in keyset-paginated batches.

    Do this by:
    1. Defining a fetch function that loads the next batch of rows given (session, last_cursor, batch_size).
    2. Calling _iter_keyset_batches so that it yields the rows as lists of dicts.
    
    ========================================== Arguments ==========================================
    session: An open SQLAlchemy session.
    batch_size: Maximum rows per batch.
    
    ========================================== Returns ==========================================
    An iterator of lists of dicts, each representing a batch of rows.
    """

    def fetch(session: Session, last_movie_id: int, limit: int) -> list[CatalogMovie]:
        stmt = (
            select(CatalogMovie)
            .where(CatalogMovie.movie_id > last_movie_id)
            .order_by(CatalogMovie.movie_id)
            .limit(limit)
        )
        return list(session.scalars(stmt).all())

    yield from _iter_keyset_batches(session, batch_size, fetch, lambda row: row.movie_id)


def iter_movie_content_embeddings(session: Session, batch_size: int) -> Iterator[list[dict[str, Any]]]:
    """
    Yield movie_content_embeddings rows in keyset-paginated batches.

    Do this by:
    1. Defining a fetch function that loads the next batch of rows given (session, last_cursor, batch_size).
    2. Calling _iter_keyset_batches so that it yields the rows as lists of dicts.
    
    ========================================== Arguments ==========================================
    session: An open SQLAlchemy session.
    batch_size: Maximum rows per batch.
    
    ========================================== Returns ==========================================
    An iterator of lists of dicts, each representing a batch of rows.
    """

    def fetch(session: Session, last_movie_id: int, limit: int) -> list[MovieContentEmbedding]:
        stmt = (
            select(MovieContentEmbedding)
            .where(MovieContentEmbedding.movie_id > last_movie_id)
            .order_by(MovieContentEmbedding.movie_id)
            .limit(limit)
        )
        return list(session.scalars(stmt).all())

    yield from _iter_keyset_batches(session, batch_size, fetch, lambda row: row.movie_id)


TABLE_EXPORT_ITERATORS: dict[str, Callable[[Session, int], Iterator[list[dict[str, Any]]]]] = {
    "catalog_movies": iter_catalog_movies,
    "movie_content_embeddings": iter_movie_content_embeddings,
    "tag_events": iter_tag_events,
    "ratings_events": iter_ratings_events,
}

EXPORT_TABLE_ORDER: list[str] = [
    "catalog_movies",
    "movie_content_embeddings",
    "tag_events",
    "ratings_events",
]

"""Read user rating history from ratings_events."""

from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sqlalchemy import func, select
from common.db.models.events import RatingsEvent
from common.db.repositories.catalog import CatalogDisplayRow, get_catalog_display_by_ids


@dataclass(frozen=True)
class UserRatingRow:
    """One active rating row from a user's history."""

    movie_id: int
    rating: float
    rating_timestamp: datetime


@dataclass(frozen=True)
class UserRatingDisplayRow:
    """One active user rating joined with catalog display metadata."""

    movie_id: int
    title: str
    year: int | None
    genres: list[str]
    poster_path: str | None
    rating: float
    rated_at: datetime


def _latest_rating_events_subquery(user_id: int):
    """
    Build a subquery with the newest ratings_events row per movie for one user.

    ============================ Arguments ============================
    user_id: App or MovieLens user id.

    ============================ Returns ============================
    SQLAlchemy subquery with movie_id, rating, rating_timestamp, and event_type.
    """
    return (
        select(
            RatingsEvent.movie_id,
            RatingsEvent.rating,
            RatingsEvent.rating_timestamp,
            RatingsEvent.event_type,
        )
        .where(RatingsEvent.user_id == user_id)
        .distinct(RatingsEvent.movie_id)
        .order_by(
            RatingsEvent.movie_id,
            RatingsEvent.rating_timestamp.desc(),
            RatingsEvent.id.desc(),
        )
        .subquery("latest_per_movie")
    )


def _active_rating_filter(latest_per_movie):
    """Return SQL filters that keep only movies whose latest event is a rating."""
    return (
        latest_per_movie.c.event_type == "rating_created",
        latest_per_movie.c.rating.is_not(None),
    )


def fetch_user_ratings(session: Session, user_id: int) -> list[UserRatingRow]:
    """
    Load active ratings for one user ordered by rating time ascending.

    Do this by:
    1. Selecting the newest ratings_events row per movie_id with DISTINCT ON.
    2. Keeping only rows whose latest event is rating_created.
    3. Sorting the final list by rating_timestamp ascending.

    ============================ Arguments ============================
    session: An open SQLAlchemy session.
    user_id: App or MovieLens user id.

    ============================ Returns ============================
    Active de-duplicated rating history for the user.
    """
    latest_per_movie = _latest_rating_events_subquery(user_id)
    stmt = (
        select(latest_per_movie)
        .where(*_active_rating_filter(latest_per_movie))
        .order_by(latest_per_movie.c.rating_timestamp.asc())
    )
    rows = session.execute(stmt).all()

    return [
        UserRatingRow(
            movie_id=row.movie_id,
            rating=float(row.rating),
            rating_timestamp=row.rating_timestamp,
        )
        for row in rows
    ]


def count_user_ratings(session: Session, user_id: int) -> int:
    """
    Count how many distinct movies a user actively rates right now.

    Do this by:
    1. Selecting the newest ratings_events row per movie_id.
    2. Counting only rows whose latest event is rating_created.

    ============================ Arguments ============================
    session: An open SQLAlchemy session.
    user_id: App or MovieLens user id.

    ============================ Returns ============================
    Number of unique actively rated movies.
    """
    latest_per_movie = _latest_rating_events_subquery(user_id)
    active = (
        select(latest_per_movie.c.movie_id)
        .where(*_active_rating_filter(latest_per_movie))
        .subquery("active_ratings")
    )
    stmt = select(func.count()).select_from(active)
    return int(session.execute(stmt).scalar_one())


def user_has_active_rating(session: Session, user_id: int, movie_id: int) -> bool:
    """
    Return True when the user's latest event for one movie is an active rating.

    ============================ Arguments ============================
    session: An open SQLAlchemy session.
    user_id: App user id.
    movie_id: Catalog movie id to check.

    ============================ Returns ============================
    True when the movie is actively rated by the user.
    """
    for row in fetch_user_ratings(session, user_id):
        if row.movie_id == movie_id:
            return True
    return False


def fetch_user_ratings_with_catalog(session: Session, user_id: int) -> list[UserRatingDisplayRow]:
    """
    Load active user ratings enriched with catalog title and poster metadata.

    Do this by:
    1. Loading active ratings from ratings_events.
    2. Looking up catalog_movies display fields for those movie ids.
    3. Sorting by rated_at descending so the newest ratings appear first.

    ============================ Arguments ============================
    session: An open SQLAlchemy session.
    user_id: Authenticated app user id.

    ============================ Returns ============================
    Display-ready active ratings for the user.
    """
    rating_rows = fetch_user_ratings(session, user_id)
    if not rating_rows:
        return []

    catalog_by_id = get_catalog_display_by_ids(session, [row.movie_id for row in rating_rows])
    display_rows: list[UserRatingDisplayRow] = []

    for row in rating_rows:
        catalog = catalog_by_id.get(row.movie_id)
        if catalog is None:
            continue
        display_rows.append(
            UserRatingDisplayRow(
                movie_id=row.movie_id,
                title=catalog.title,
                year=catalog.year,
                genres=catalog.genres,
                poster_path=catalog.poster_path,
                rating=row.rating,
                rated_at=row.rating_timestamp,
            )
        )

    display_rows.sort(key=lambda item: item.rated_at, reverse=True)
    return display_rows

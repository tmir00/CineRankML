"""Read user rating history from ratings_events."""

from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sqlalchemy import func, select
from common.db.models.events import RatingsEvent


@dataclass(frozen=True)
class UserRatingRow:
    """ One rating row from a user's history. """

    movie_id: int
    rating: float
    rating_timestamp: datetime


def fetch_user_ratings(session: Session, user_id: int) -> list[UserRatingRow]:
    """
    Load the latest rating per movie for one user ordered by rating time.

    Do this by:
    1. Selecting the newest ratings_events row per movie_id with DISTINCT ON.
    2. Breaking ties on rating_timestamp with the highest id.
    3. Sorting the final list by rating_timestamp ascending.

    ============================ Arguments ============================
    session: An open SQLAlchemy session.
    user_id: App or MovieLens user id.

    ============================ Returns ============================
    De-duplicated rating history for the user.
    """
    latest_per_movie = (
        select(
            RatingsEvent.movie_id,
            RatingsEvent.rating,
            RatingsEvent.rating_timestamp,
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

    stmt = select(latest_per_movie).order_by(latest_per_movie.c.rating_timestamp.asc())
    rows = session.execute(stmt).all()

    return [
        UserRatingRow(
            movie_id=row.movie_id,
            rating=row.rating,
            rating_timestamp=row.rating_timestamp,
        )
        for row in rows
    ]


def count_user_ratings(session: Session, user_id: int) -> int:
    """
    Count how many distinct movies a user has rated.

    ============================ Arguments ============================
    session: An open SQLAlchemy session.
    user_id: App or MovieLens user id.

    ============================ Returns ============================
    Number of unique rated movies.
    """
    stmt = select(func.count(func.distinct(RatingsEvent.movie_id))).where(
        RatingsEvent.user_id == user_id
    )
    return int(session.execute(stmt).scalar_one())

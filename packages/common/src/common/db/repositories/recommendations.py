"""Write recommendation impression rows for online serving."""

from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
from sqlalchemy.orm import Session
from common.db.models.recommendations import RecommendationImpression, RecommendationRating


@dataclass(frozen=True)
class ImpressionRow:
    """One recommendation impression ready for Postgres insert."""

    request_id: str
    user_id: int
    movie_id: int
    rank_position: int
    model_version: str
    model_role: str
    experiment_id: str
    predicted_score: float
    shown_at: datetime
    retrieval_source: str


@dataclass(frozen=True)
class RecommendationRatingRow:
    """One user rating tied to a prior recommendation impression."""

    request_id: str
    user_id: int
    movie_id: int
    model_version: str
    model_role: str
    experiment_id: str
    rating: float
    rated_at: datetime


def insert_recommendation_rating(session: Session, row: RecommendationRatingRow) -> None:
    """
    Insert one recommendation feedback rating with full experiment lineage.

    ============================ Arguments ============================
    session: An open SQLAlchemy session inside a transaction.
    row: Rating row including request_id, model_role, and experiment_id.
    """
    session.add(
        RecommendationRating(
            request_id=row.request_id,
            user_id=row.user_id,
            movie_id=row.movie_id,
            model_version=row.model_version,
            model_role=row.model_role,
            experiment_id=row.experiment_id,
            rating=row.rating,
            rated_at=row.rated_at,
        )
    )
    session.flush()


def insert_impressions(session: Session, rows: list[ImpressionRow]) -> int:
    """
    Bulk insert recommendation impression rows after ranking completes.

    ============================ Arguments ============================
    session: An open SQLAlchemy session inside a transaction.
    rows: One impression row per movie shown to the user.

    ============================ Returns ============================
    Number of rows inserted.
    """
    if not rows:
        return 0

    for row in rows:
        session.add(
            RecommendationImpression(
                request_id=row.request_id,
                user_id=row.user_id,
                movie_id=row.movie_id,
                rank_position=row.rank_position,
                model_version=row.model_version,
                model_role=row.model_role,
                experiment_id=row.experiment_id,
                predicted_score=row.predicted_score,
                shown_at=row.shown_at,
                retrieval_source=row.retrieval_source,
            )
        )
    session.flush()
    return len(rows)

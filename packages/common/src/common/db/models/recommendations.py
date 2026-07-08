from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Float, Index, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from common.db.base import Base


class RecommendationImpression(Base):
    __tablename__ = "recommendation_impressions"
    __table_args__ = (
        Index(
            "ix_recommendation_impressions_experiment_id_model_version",
            "experiment_id",
            "model_version",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(Text, nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    movie_id: Mapped[int] = mapped_column(Integer, nullable=False)
    rank_position: Mapped[int] = mapped_column(Integer, nullable=False)
    model_version: Mapped[str] = mapped_column(Text, nullable=False)
    model_role: Mapped[str] = mapped_column(Text, nullable=False)
    experiment_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    predicted_score: Mapped[float] = mapped_column(Float, nullable=False)
    shown_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    retrieval_source: Mapped[str] = mapped_column(Text, nullable=False, server_default="unknown")


class RecommendationRating(Base):
    __tablename__ = "recommendation_ratings"
    __table_args__ = (
        Index(
            "ix_recommendation_ratings_experiment_id_model_version",
            "experiment_id",
            "model_version",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(Text, nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    movie_id: Mapped[int] = mapped_column(Integer, nullable=False)
    model_version: Mapped[str] = mapped_column(Text, nullable=False)
    model_role: Mapped[str] = mapped_column(Text, nullable=False)
    experiment_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    rating: Mapped[float] = mapped_column(Float, nullable=False)
    rated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class RecommendationExperiment(Base):
    __tablename__ = "recommendation_experiments"
    __table_args__ = (
        Index("ix_recommendation_experiments_status", "status"),
    )

    experiment_id: Mapped[str] = mapped_column(Text, primary_key=True)
    main_model_version: Mapped[str] = mapped_column(Text, nullable=False)
    candidate_model_version: Mapped[str | None] = mapped_column(Text, nullable=True)
    main_split_fraction: Mapped[float] = mapped_column(Float, nullable=False, default=0.70)
    candidate_split_fraction: Mapped[float] = mapped_column(Float, nullable=False, default=0.30)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

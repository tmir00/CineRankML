from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Float, Index, Integer, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from common.db.base import Base


class RatingsEvent(Base):
    __tablename__ = "ratings_events"
    __table_args__ = (
        Index("ix_ratings_events_user_id_rating_timestamp", "user_id", "rating_timestamp"),
        Index("ix_ratings_events_rating_timestamp_id", "rating_timestamp", "id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    movie_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(Text, nullable=False, server_default="rating_created")
    rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    rating_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    stream_pipeline_version: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    source: Mapped[str | None] = mapped_column(Text, nullable=True)


class TagEvent(Base):
    __tablename__ = "tag_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(Text, unique=True, nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    movie_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    tag: Mapped[str] = mapped_column(Text, nullable=False)
    tag_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    stream_pipeline_version: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    source: Mapped[str | None] = mapped_column(Text, nullable=True)

"""ORM (Object-Relational Mapping) model for events that failed validation or database writes."""

from datetime import datetime
from common.db.base import Base
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import BigInteger, DateTime, Index, Integer, Text, func


class DeadLetterEvent(Base):
    """A Kafka message that could not be processed and was saved for later review."""

    __tablename__ = "dead_letter_events"
    __table_args__ = (
        Index("ix_dead_letter_events_source_topic_created_at", "source_topic", "created_at"),
        Index("ix_dead_letter_events_error_type", "error_type"),
        Index("ix_dead_letter_events_worker_name", "worker_name"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    worker_name: Mapped[str] = mapped_column(Text, nullable=False)
    source_topic: Mapped[str] = mapped_column(Text, nullable=False)
    kafka_partition: Mapped[int] = mapped_column(Integer, nullable=False)
    kafka_offset: Mapped[int] = mapped_column(BigInteger, nullable=False)
    event_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_type: Mapped[str] = mapped_column(Text, nullable=False)
    error_message: Mapped[str] = mapped_column(Text, nullable=False)
    raw_payload: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

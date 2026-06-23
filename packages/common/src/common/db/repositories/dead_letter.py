"""Save failed Kafka messages into dead_letter_events."""

from __future__ import annotations

from sqlalchemy.orm import Session
from common.db.models.dead_letter import DeadLetterEvent


def insert_dead_letter_event(session: Session, *, worker_name: str, source_topic: str, \
                                kafka_partition: int, kafka_offset: int, error_type: str, \
                                error_message: str, raw_payload: str, event_id: str | None = None) -> None:
    """
    Store one failed Kafka message in dead_letter_events for later review.

    Do this by:
    1. Building a row with the Kafka coordinates and error details.
    2. Inserting it in the same database session as the caller.

    ============================ Arguments ============================
    session: An open SQLAlchemy session (usually its own short transaction).
    worker_name: The consumer worker name (e.g. ratings-consumer).
    source_topic: Kafka topic the message came from.
    kafka_partition: Kafka partition number.
    kafka_offset: Kafka offset of the failed message.
    error_type: Short error category (e.g. validation_error, db_write_error).
    error_message: Human-readable error detail.
    raw_payload: Original message body as text.
    event_id: Parsed event_id when available.
    """
    # Build the row to insert in the database.
    row = DeadLetterEvent(
        worker_name=worker_name,
        source_topic=source_topic,
        kafka_partition=kafka_partition,
        kafka_offset=kafka_offset,
        event_id=event_id,
        error_type=error_type,
        error_message=error_message,
        raw_payload=raw_payload,
    )
    session.add(row)

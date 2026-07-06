"""Confluent Kafka consumer loop with validation, Postgres writes, and DLQ handling."""

from __future__ import annotations

import logging
import signal
import time

from pydantic import BaseModel
from typing import Any, TypeVar
from collections.abc import Callable
from common.kafka.serde import decode_json
from common.metrics.worker import WorkerMetrics
from common.schemas.events import try_validate_event
from confluent_kafka import Consumer, KafkaError, KafkaException, Message


logger = logging.getLogger(__name__)

TEvent = TypeVar("TEvent", bound=BaseModel)
ProcessEventFn = Callable[[BaseModel], str]
ParseEventFn = Callable[[dict], BaseModel]
SaveDeadLetterFn = Callable[..., None]


class GracefulShutdown:
    """
    Track whether a shutdown signal has been received.
    """

    def __init__(self) -> None:
        self._stop = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        logger.info("Shutdown signal received", extra={"signal": signum})
        self._stop = True

    @property
    def requested(self) -> bool:
        """True when the process should stop consuming."""
        return self._stop


class KafkaEventConsumer:
    """
    Subscribe to Kafka topics and poll messages with manual offset commits.

    Do this by:
    1. Creating a confluent Consumer with auto-commit disabled.
    2. Subscribing to the configured topic list.
    3. Exposing poll, commit, and lag helpers for the shared consumer loop.
    """

    def __init__(self, bootstrap_servers: str, group_id: str, topics: list[str]) -> None:
        """
        Create and subscribe a Kafka consumer to the given topics.

        ============================ Arguments ============================
        bootstrap_servers: Comma-separated Kafka broker addresses.
        group_id: Consumer group id for offset tracking.
        topics: Topic names to subscribe to.
        """
        # Create a confluent Consumer with auto-commit disabled.
        self._consumer = Consumer(
            {
                "bootstrap.servers": bootstrap_servers,
                "group.id": group_id,
                "enable.auto.commit": False,
                "auto.offset.reset": "earliest",
                "session.timeout.ms": 10000,
            }
        )
        # Subscribe to the given topics.
        self._consumer.subscribe(topics)
        # Store the consumer group id and topics.
        self._group_id = group_id
        self._topics = topics

    @property
    def group_id(self) -> str:
        """ Get the consumer group id. """
        return self._group_id

    @property
    def topics(self) -> list[str]:
        """ Get the subscribed topic names. """
        return self._topics

    def poll(self, timeout: float = 1.0) -> Message | None:
        """
        Wait for the next Kafka message.

        ============================ Arguments ============================
        timeout: Seconds to wait when no message is available.

        ============================ Returns ============================
        The next message, or None when nothing arrived in time.
        """
        # Get the next message from the consumer, waiting for up to timeout seconds.
        msg = self._consumer.poll(timeout)
        # If there is no message, return None.
        if msg is None:
            return None

        # If there is an error, check if it is a partition end error.
        if msg.error():
            error = msg.error()
            # If the error is a partition end error, return None.
            if error.code() == KafkaError._PARTITION_EOF:
                return None
            # Otherwise, raise the Kafka exception.
            raise KafkaException(error)

        return msg

    def commit(self, msg: Message) -> None:
        """
        Commit the offset for one successfully processed message.

        ============================ Arguments ============================
        msg: The Kafka message whose offset should be committed.
        """
        self._consumer.commit(message=msg, asynchronous=False)

    def close(self) -> None:
        """Leave the group and release broker resources."""
        self._consumer.close()

    def estimate_lag(self) -> int:
        """
        Estimate how far behind the consumer is from processing all messages 
        for all assigned partitions.

        Do this by:
        1. Getting the partitions that are assigned to this consumer.
        2. Iterating over the assigned partitions and calculating the lag.
        3. Returning the total lag.

        ============================ Returns ============================
        Sum of (high watermark - current position) for assigned partitions.
        """

        total_lag = 0
        # Get the partitions that are assigned to this consumer.
        assignments = self._consumer.assignment()
        # If there are no assignments, return 0.
        if not assignments:
            return 0

        # Iterate over the assigned partitions and calculate the lag.
        for tp in assignments:
            try:
                # Get the low and high watermark offsets for the partition.
                _low, high = self._consumer.get_watermark_offsets(tp, timeout=1.0)
                # Get the current position for the partition.
                positions = self._consumer.position([tp])
                # If there are no positions, continue.
                if not positions:
                    continue

                # Get the current position for the partition.
                current = positions[0].offset
                # If the current position is greater than or equal to 0 and the high watermark is greater than or equal to 0, add the lag to the total lag.
                if current >= 0 and high >= 0:
                    total_lag += max(0, high - current)
            
            except KafkaException:
                # If there is a Kafka exception, continue.
                continue

        return total_lag


def _message_context(msg: Message) -> dict[str, Any]:
    """ Return logging information from a Kafka message. """
    return {
        "topic": msg.topic(),
        "partition": msg.partition(),
        "offset": msg.offset(),
    }


def _raw_payload_text(msg: Message) -> str:
    """ Return the message body as text for dead_letter_events storage. """
    # Get the message value.
    value = msg.value()
    # If the value is None, return an empty string.
    if value is None:
        return ""
    # If the value is a bytes object, decode it to a string.
    if isinstance(value, bytes):
        # Decode the bytes to a string, replacing any invalid characters.
        return value.decode("utf-8", errors="replace")
    # Otherwise, convert the value to a string.
    return str(value)


def _log_received_message(msg: Message) -> None:
    """Emit a DEBUG line when a Kafka message is polled for processing."""
    logger.debug(
        "received message topic=%s partition=%s offset=%s",
        msg.topic(),
        msg.partition(),
        msg.offset(),
    )


def _log_db_write_finished(msg: Message, db_ms: float, *, status: str) -> None:
    """Emit a DEBUG line after a Postgres or DLQ write completes."""
    logger.debug(
        "db write finished partition=%s offset=%s db_ms=%.2f status=%s",
        msg.partition(),
        msg.offset(),
        db_ms,
        status,
    )


def _commit_with_logging(consumer: KafkaEventConsumer, msg: Message) -> None:
    """Commit one Kafka offset and emit a DEBUG line with commit latency."""
    commit_start = time.perf_counter()
    try:
        consumer.commit(msg)
    except KafkaException:
        logger.exception(
            "commit failed partition=%s offset=%s",
            msg.partition(),
            msg.offset(),
        )
        raise

    commit_ms = (time.perf_counter() - commit_start) * 1000
    logger.debug(
        "committed partition=%s offset=%s commit_ms=%.2f",
        msg.partition(),
        msg.offset(),
        commit_ms,
    )


def _log_consumer_progress_if_due(
    *,
    handled_count: int,
    progress_log_every_n: int,
    topic: str,
    worker_name: str,
    lag_resolver: Callable[[], int],
    ctx: dict[str, object],
    status: str | None = None,
) -> None:
    """Emit a periodic INFO progress line after every N handled Kafka messages."""
    if progress_log_every_n <= 0 or handled_count % progress_log_every_n != 0:
        return

    lag = lag_resolver()

    extra: dict[str, object] = {
        "topic": topic,
        "worker_name": worker_name,
        "handled_count": handled_count,
        "consumer_lag": lag,
        **ctx,
    }
    if status is not None:
        extra["status"] = status

    logger.info(
        "Consumer progress: handled %s messages (lag=%s, partition=%s, offset=%s)",
        handled_count,
        lag,
        ctx.get("partition"),
        ctx.get("offset"),
        extra=extra,
    )


def run_consumer_loop(consumer: KafkaEventConsumer, topic: str, worker_name: str, save_dead_letter: SaveDeadLetterFn, \
                        metrics: WorkerMetrics, process_event: ProcessEventFn, event_model: type[TEvent] | None = None, \
                        parse_event: ParseEventFn | None = None, shutdown: GracefulShutdown | None = None, \
                        lag_poll_interval_seconds: float = 15.0, progress_log_every_n: int = 10000) -> None:
    """
    Poll Kafka, validate events, write to Postgres, and commit offsets.

    Do this by:
    1. Polling for the next message until shutdown is requested.
    2. Decoding and validating the payload with Pydantic.
    3. Running the worker-specific database handler inside a transaction.
    4. Saving failed messages to dead_letter_events when validation or DB write fails.
    5. Committing the Kafka offset only after a successful path or DLQ save.

    ============================ Arguments ============================
    consumer: The subscribed KafkaEventConsumer instance.
    topic: Primary topic name (used in logs and metrics).
    worker_name: Worker name stored on dead_letter_events rows.
    save_dead_letter: Callable that persists one row to dead_letter_events.
    metrics: Prometheus metrics helper for this worker.
    process_event: Callable that writes one validated event to Postgres.
    event_model: Pydantic model class used when parse_event is not provided.
    parse_event: Optional callable that validates one raw dict into a typed event.
    shutdown: Optional GracefulShutdown tracker; created when omitted.
    lag_poll_interval_seconds: How often to refresh consumer_lag gauge.
    progress_log_every_n: How often to log handled-message progress at INFO.
    """
    if event_model is None and parse_event is None:
        raise ValueError("run_consumer_loop requires event_model or parse_event")
    # Create a GracefulShutdown tracker if not provided.
    shutdown = shutdown or GracefulShutdown()
    handled_count = 0
    # Log the consumer started with topic/group_id.
    logger.info(
        "Consumer started with topic/group_id",
        extra={"topic": topic, "group_id": consumer.group_id, "worker_name": worker_name},
    )

    # Track the last time the consumer lag was updated.
    last_lag_update = 0.0

    # Poll for messages until the shutdown signal is received.
    while not shutdown.requested:
        # Get the current time.
        now = time.monotonic()
        # If the current time is greater than the last lag update time plus the lag poll interval, update the consumer lag.
        if now - last_lag_update >= lag_poll_interval_seconds:
            metrics.set_consumer_lag(topic, consumer.estimate_lag())
            last_lag_update = now

        # Poll for the next message from the consumer.
        try:
            msg = consumer.poll(timeout=1.0)
        
        # If there is a Kafka exception, log the error.
        except KafkaException as exc:
            logger.error("Kafka poll failed", extra={"error": str(exc)})
            continue

        # If there is no message, continue.
        if msg is None:
            continue

        ctx = _message_context(msg)
        _log_received_message(msg)

        # Get the raw payload text from the message.
        raw_text = _raw_payload_text(msg)
        # Track the event id string.
        event_id_str: str | None = None

        # Decode JSON from the message body.
        try:
            raw_dict = decode_json(msg.value())
        
        # If there is a ValueError, log the error.
        except ValueError as exc:
            # Get the error type.
            error_type = str(exc)
            
            # Log the error.
            logger.warning(
                "Bad event schema / Pydantic error",
                extra={**ctx, "error_type": error_type, "error": "Could not decode JSON"},
            )
            
            # Save the bad event to the dead-letter table.
            db_start = time.perf_counter()
            save_dead_letter(
                worker_name=worker_name,
                source_topic=msg.topic() or topic,
                kafka_partition=msg.partition(),
                kafka_offset=msg.offset(),
                error_type=error_type,
                error_message="Could not decode JSON payload",
                raw_payload=raw_text,
                event_id=None,
            )
            _log_db_write_finished(msg, (time.perf_counter() - db_start) * 1000, status="validation_error")
            # Record the consumed event.
            metrics.record_consumed(topic, "validation_error")
            # Record the dead-letter event.
            metrics.record_dlq(error_type)
            logger.warning("Bad event saved to dead-letter table", extra=ctx)

            try:
                _commit_with_logging(consumer, msg)
            except KafkaException:
                pass
            
            handled_count += 1
            _log_consumer_progress_if_due(
                handled_count=handled_count,
                progress_log_every_n=progress_log_every_n,
                topic=topic,
                worker_name=worker_name,
                lag_resolver=consumer.estimate_lag,
                ctx=ctx,
                status="validation_error",
            )
            # Continue to the next message.
            continue

        # Validate the payload against the expected Pydantic model.
        if parse_event is not None:
            try:
                event = parse_event(raw_dict)
                validation_error = None
            except Exception as exc:
                event = None
                validation_error = str(exc)
        else:
            event, validation_error = try_validate_event(raw_dict, event_model)  # type: ignore[arg-type]

        # If there is a validation error or the event is None, log the error.
        if validation_error is not None or event is None:
            # Log the error.
            logger.warning(
                "Bad event schema / Pydantic error",
                extra={**ctx, "error": validation_error},
            )

            # Save the bad event to the dead-letter table.
            db_start = time.perf_counter()
            save_dead_letter(
                worker_name=worker_name,
                source_topic=msg.topic() or topic,
                kafka_partition=msg.partition(),
                kafka_offset=msg.offset(),
                error_type="validation_error",
                error_message=validation_error or "Unknown validation error",
                raw_payload=raw_text,
                event_id=str(raw_dict.get("event_id")) if raw_dict.get("event_id") else None,
            )
            _log_db_write_finished(msg, (time.perf_counter() - db_start) * 1000, status="validation_error")
            # Record the consumed event.
            metrics.record_consumed(topic, "validation_error")
            # Record the dead-letter event.
            metrics.record_dlq("validation_error")
            logger.warning("Bad event saved to dead-letter table", extra=ctx)

            try:
                _commit_with_logging(consumer, msg)
            except KafkaException:
                pass

            handled_count += 1
            _log_consumer_progress_if_due(
                handled_count=handled_count,
                progress_log_every_n=progress_log_every_n,
                topic=topic,
                worker_name=worker_name,
                lag_resolver=consumer.estimate_lag,
                ctx=ctx,
                status="validation_error",
            )
            continue

        # Get the event id string.
        event_id_str = str(event.event_id)
        # Get the context with the event id.
        ctx_with_event = {**ctx, "event_id": event_id_str}

        # Run the worker-specific Postgres handler.
        try:
            db_start = time.perf_counter()
            with metrics.time_db_write():
                status = process_event(event)
            _log_db_write_finished(msg, (time.perf_counter() - db_start) * 1000, status=status)

        # If there is an exception, log the error.
        except Exception as exc:
            logger.exception(
                "failed processing partition=%s offset=%s",
                msg.partition(),
                msg.offset(),
            )
            # Record the database failure.
            metrics.record_db_failure("db_write_error")

            # Save the bad event to the dead-letter table.
            db_start = time.perf_counter()
            save_dead_letter(
                worker_name=worker_name,
                source_topic=msg.topic() or topic,
                kafka_partition=msg.partition(),
                kafka_offset=msg.offset(),
                error_type="db_write_error",
                error_message=str(exc),
                raw_payload=raw_text,
                event_id=event_id_str,
            )
            _log_db_write_finished(msg, (time.perf_counter() - db_start) * 1000, status="dlq")
            # Record the consumed event.
            metrics.record_consumed(topic, "db_error")
            # Record the dead-letter event.
            metrics.record_dlq("db_write_error")
            # Log the bad event saved to the dead-letter table.
            logger.warning("Bad event saved to dead-letter table", extra=ctx_with_event)

            try:
                _commit_with_logging(consumer, msg)
            except KafkaException:
                pass

            handled_count += 1
            _log_consumer_progress_if_due(
                handled_count=handled_count,
                progress_log_every_n=progress_log_every_n,
                topic=topic,
                worker_name=worker_name,
                lag_resolver=consumer.estimate_lag,
                ctx=ctx_with_event,
                status="db_write_error",
            )
            continue

        # Record the consumed event.
        metrics.record_consumed(topic, status)

        try:
            _commit_with_logging(consumer, msg)
        except KafkaException:
            pass

        handled_count += 1
        _log_consumer_progress_if_due(
            handled_count=handled_count,
            progress_log_every_n=progress_log_every_n,
            topic=topic,
            worker_name=worker_name,
            lag_resolver=consumer.estimate_lag,
            ctx=ctx_with_event,
            status=status,
        )

    consumer.close()
    logger.info(
        "Consumer stopped cleanly",
        extra={"topic": topic, "worker_name": worker_name, "handled_count": handled_count},
    )

"""Confluent Kafka producer wrapper for JSON events."""

from __future__ import annotations

import logging
import threading
from typing import Any
from pydantic import BaseModel
from common.kafka.serde import encode_event
from confluent_kafka import KafkaException, Producer


logger = logging.getLogger(__name__)


class KafkaEventProducer:
    """
    Publish validated JSON events to a Kafka topic using confluent-kafka.

    Do this by:
    1. Creating a confluent Producer with the bootstrap servers.
    2. Encoding each Pydantic event as JSON bytes.
    3. Tracking delivery results in a callback for logging and metrics.
    """

    def __init__(self, bootstrap_servers: str, log_every_n: int = 1000) -> None:
        """
        Create a Kafka producer client.

        ============================ Arguments ============================
        bootstrap_servers: Comma-separated Kafka broker addresses.
        log_every_n: Log a batch INFO message after this many successful sends.
        """
        # Set the log every n successful sends.
        self._log_every_n = max(1, log_every_n)
        # Set the success count to 0.
        self._success_count = 0
        # Set the success lock to a threading lock.
        self._success_lock = threading.Lock()
        # Set the pending failures to 0.
        self._pending_failures = 0
        # Set the producer to a confluent Producer with the bootstrap servers.
        self._producer = Producer(
            {
                "bootstrap.servers": bootstrap_servers,
                "linger.ms": 10, # Wait up to 10ms to batch messages for efficiency.
                "acks": "all", # Producer consider success only when all Kafka replicas have acknowledged the message.
            }
        )
        # Log the bootstrap servers.
        logger.info("Kafka producer initialized", extra={"bootstrap_servers": bootstrap_servers})

    def _delivery_callback(self, err: Any, msg: Any) -> None:
        """
        Handle produce() delivery reports for logging.

        Do this by:
        1. If there is an error, increment the pending failures and log the error.
        2. If there is no error, increment the success count and log the success.

        ============================ Arguments ============================
        err: The error message from the delivery report.
        msg: The message from the delivery report.
        """
        # If there is an error, increment the pending failures and log the error.
        if err is not None:
            self._pending_failures += 1
            logger.error(
                "Event failed to send",
                extra={
                    "topic": msg.topic() if msg else None,
                    "error": str(err),
                },
            )
            return

        # If there is no error. Increment the success count and log the success.
        with self._success_lock:
            self._success_count += 1
            count = self._success_count

        # If the success count is a multiple of the log every n successful sends, log the success.
        if count % self._log_every_n == 0:
            logger.info(
                "Event sent to topic",
                extra={
                    "topic": msg.topic(),
                    "count": count,
                    "partition": msg.partition(),
                    "offset": msg.offset(),
                },
            )

    def produce(self, topic: str, event: BaseModel, key: str | None = None) -> None:
        """
        Send one event to Kafka without blocking for broker acknowledgement.

        Do this by:
        1. Encoding the event as JSON bytes.
        2. Calling produce() with an optional message key.
        3. Polling the producer so delivery callbacks can run.

        ============================ Arguments ============================
        topic: The Kafka topic name.
        event: A validated Pydantic event.
        key: Optional Kafka message key (often event_id as string).
        """
        # Encode the event as JSON bytes.
        payload = encode_event(event)
        # Encode the key as bytes if it is not None.
        key_bytes = key.encode("utf-8") if key is not None else None
        
        # Try to produce the message to the Kafka topic.
        try:
            self._producer.produce(
                topic=topic,
                value=payload,
                key=key_bytes,
                callback=self._delivery_callback,
            )
        # If there is a Kafka exception, log the error and raise it.
        except KafkaException as exc:
            logger.error(
                "Event failed to send",
                extra={"topic": topic, "error": str(exc)},
            )
            raise

        # Let the producer run delivery callbacks for completed sends.
        self._producer.poll(0)

    def flush(self, timeout: float = 30.0) -> None:
        """
        Wait until all in-flight messages are delivered or the timeout expires.

        ============================ Arguments ============================
        timeout: Maximum seconds to wait for pending messages.
        """
        #
        remaining = self._producer.flush(timeout)
        if remaining > 0:
            logger.error(
                "Kafka producer flush timed out with pending messages",
                extra={"remaining_messages": remaining},
            )

    @property
    def pending_failures(self) -> int:
        """Number of delivery failures reported since startup."""
        return self._pending_failures

    @property
    def success_count(self) -> int:
        """Number of successful delivery reports since startup."""
        with self._success_lock:
            return self._success_count

"""Tags Kafka consumer entrypoint."""

from __future__ import annotations

import logging

from common.config.settings import get_kafka_settings, get_worker_metrics_settings
from common.db.repositories.dead_letter import insert_dead_letter_event
from common.db.session import get_session_factory
from common.kafka.consumer import GracefulShutdown, KafkaEventConsumer, run_consumer_loop
from common.metrics.worker import WorkerMetrics
from common.schemas.events import TagCreatedEvent
from common.logging_config import configure_worker_logging

from tags_consumer.handler import process_tag_event

logger = logging.getLogger(__name__)


def save_dead_letter(**kwargs: object) -> None:
    """
    Persist one failed message to dead_letter_events in its own transaction.

    ============================ Arguments ============================
    kwargs: Fields passed through to insert_dead_letter_event.
    """
    # Get the session factory.
    session_factory = get_session_factory()
    # Create a new session.
    session = session_factory()
    
    # Try to insert the event.
    try:
        # Insert the event.
        insert_dead_letter_event(session, **kwargs)  # type: ignore[arg-type]
        session.commit()
    except Exception:
        # In the case of an error, rollback the session.
        session.rollback()
        # Raise the exception.
        raise
    finally:
        session.close()


def main() -> None:
    """
    Run the tags consumer worker.

    Do this by:
    1. Loading settings and starting the metrics HTTP server.
    2. Subscribing to the tags topic.
    3. Running the shared consumer loop with the tag write handler.
    """
    metrics_settings = get_worker_metrics_settings()
    configure_worker_logging(metrics_settings.log_level)

    kafka_settings = get_kafka_settings()
    # Create the WorkerMetrics instance.
    metrics = WorkerMetrics(metrics_settings.worker_name)
    # Start the metrics HTTP server.
    metrics.start_server(metrics_settings.metrics_port)
    
    # Create the Kafka consumer.
    consumer = KafkaEventConsumer(
        bootstrap_servers=kafka_settings.kafka_bootstrap_servers,
        group_id=kafka_settings.tags_consumer_group,
        topics=[kafka_settings.tags_topic],
    )
    shutdown = GracefulShutdown()
    
    # Run the consumer loop.
    run_consumer_loop(
        consumer=consumer,
        topic=kafka_settings.tags_topic,
        worker_name=metrics_settings.worker_name,
        event_model=TagCreatedEvent,
        save_dead_letter=save_dead_letter,
        metrics=metrics,
        process_event=process_tag_event,
        shutdown=shutdown,
        progress_log_every_n=metrics_settings.progress_log_every_n,
    )


if __name__ == "__main__":
    main()

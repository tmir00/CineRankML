"""Stream MovieLens tags.csv rows to the tags Kafka topic."""

from __future__ import annotations

import csv
import logging
import sys
import time

from common.config.settings import (
    get_kafka_settings,
    get_producer_settings,
    get_worker_metrics_settings,
    producer_row_delay_seconds,
)
from common.kafka.consumer import GracefulShutdown
from common.kafka.csv_checkpoint import read_csv_checkpoint, save_csv_checkpoint
from common.kafka.producer import KafkaEventProducer
from common.metrics.worker import WorkerMetrics
from common.schemas.events import TagCreatedEvent, tag_row_to_event

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Set up basic structured logging for the producer process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )


def stream_tags_csv(
    producer: KafkaEventProducer,
    topic: str,
    csv_path: str,
    source_file: str,
    start_row: int,
    row_delay_seconds: float,
    row_limit: int | None,
    checkpoint_every_n: int,
    metrics: WorkerMetrics,
    shutdown: GracefulShutdown,
) -> None:
    """
    Read tags.csv and publish tag_created events to Kafka.

    Do this by:
    1. Loading the saved Postgres checkpoint for source_file.
    2. Converting each new CSV row into a validated TagCreatedEvent.
    3. Waiting row_delay_seconds between each publish so ingestion is easy to follow locally.
    4. Saving the checkpoint to Postgres every checkpoint_every_n rows and once at the end.

    ============================ Arguments ============================
    producer: The Kafka producer client.
    topic: Kafka topic to publish to.
    csv_path: Path to tags.csv on disk.
    source_file: Stable logical file name used as the Postgres checkpoint key.
    start_row: Default starting row when no checkpoint exists.
    row_delay_seconds: Seconds to sleep after each published row.
    row_limit: Optional cap on rows for smoke tests.
    checkpoint_every_n: How often to persist progress to Postgres.
    metrics: Prometheus metrics helper.
    shutdown: Graceful shutdown tracker.
    """
    # Load saved progress from Postgres, or fall back to start_row.
    saved = read_csv_checkpoint(source_file, default_row=start_row)
    resume_row = max(start_row, saved.last_row_number)

    published = 0
    row_index = 0
    last_event_id: str | None = None
    last_saved_row = resume_row

    logger.info(
        "Starting tags CSV stream",
        extra={
            "csv_path": csv_path,
            "source_file": source_file,
            "resume_row": resume_row,
            "last_event_id": saved.last_event_id,
            "topic": topic,
        },
    )

    try:
        with open(csv_path, newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)

            for row in reader:
                if shutdown.requested:
                    break

                # Skip rows that were already published before the last checkpoint.
                if row_index < resume_row:
                    row_index += 1
                    continue

                if row_limit is not None and published >= row_limit:
                    break

                event: TagCreatedEvent = tag_row_to_event(row)
                producer.produce(topic, event, key=str(event.event_id))
                metrics.record_produced(topic)
                published += 1
                row_index += 1
                last_event_id = str(event.event_id)

                # Persist progress in batches so we do not write to Postgres on every row.
                if published % checkpoint_every_n == 0:
                    save_csv_checkpoint(source_file, row_index, last_event_id)
                    last_saved_row = row_index

                time.sleep(row_delay_seconds)
    finally:
        # Flush any remaining progress when we stop early or finish the final partial batch.
        if last_event_id is not None and row_index > last_saved_row:
            save_csv_checkpoint(source_file, row_index, last_event_id)

    producer.flush()
    logger.info(
        "Finished tags CSV stream",
        extra={"published": published, "last_row_index": row_index, "source_file": source_file},
    )


def main() -> None:
    """
    Run the tags producer worker.

    Do this by:
    1. Loading settings and starting the metrics HTTP server.
    2. Creating the Kafka producer.
    3. Streaming tags.csv until row_limit or shutdown.
    """
    configure_logging()

    kafka_settings = get_kafka_settings()
    producer_settings = get_producer_settings()
    metrics_settings = get_worker_metrics_settings()

    metrics = WorkerMetrics(metrics_settings.worker_name)
    metrics.start_server(metrics_settings.metrics_port)

    shutdown = GracefulShutdown()
    producer = KafkaEventProducer(
        bootstrap_servers=kafka_settings.kafka_bootstrap_servers,
        log_every_n=producer_settings.producer_log_every_n,
    )

    try:
        stream_tags_csv(
            producer=producer,
            topic=kafka_settings.tags_topic,
            csv_path=producer_settings.csv_path,
            source_file=producer_settings.source_file or "tags.csv",
            start_row=producer_settings.start_row,
            row_delay_seconds=producer_row_delay_seconds(producer_settings),
            row_limit=producer_settings.row_limit,
            checkpoint_every_n=producer_settings.checkpoint_every_n,
            metrics=metrics,
            shutdown=shutdown,
        )
    finally:
        producer.flush()


if __name__ == "__main__":
    main()

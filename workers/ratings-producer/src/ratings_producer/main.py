"""Stream MovieLens ratings.csv rows to the ratings Kafka topic."""

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
from common.schemas.events import RatingCreatedEvent, rating_row_to_event

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Set up basic structured logging for the producer process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )


def stream_ratings_csv(producer: KafkaEventProducer, topic: str, csv_path: str, source_file: str, start_row: int, \
                        row_delay_seconds: float, row_limit: int | None, checkpoint_every_n: int, metrics: WorkerMetrics, shutdown: GracefulShutdown) -> None:
    """
    Read ratings.csv and publish rating_created events to Kafka.

    Do this by:
    1. Loading the saved Postgres checkpoint for source_file.
    2. Converting each new CSV row into a validated RatingCreatedEvent.
    3. Waiting row_delay_seconds between each publish so ingestion is easy to follow locally.
    4. Saving the checkpoint to Postgres every checkpoint_every_n rows and once at the end.

    ============================ Arguments ============================
    producer: The Kafka producer client.
    topic: Kafka topic to publish to.
    csv_path: Path to ratings.csv on disk.
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
    # Get the resume row.
    resume_row = max(start_row, saved.last_row_number)

    published = 0
    row_index = 0
    last_event_id: str | None = None
    last_saved_row = resume_row

    logger.info(
        "Starting ratings CSV stream",
        extra={
            "csv_path": csv_path,
            "source_file": source_file,
            "resume_row": resume_row,
            "last_event_id": saved.last_event_id,
            "topic": topic,
        },
    )

    # Try to stream the ratings CSV.
    try:
        # Open the CSV file.
        with open(csv_path, newline="", encoding="utf-8") as csv_file:
            # Create a reader for the CSV file.
            reader = csv.DictReader(csv_file)

            # Iterate over the rows in the CSV file.
            for row in reader:
                # If the shutdown is requested, break the loop.
                if shutdown.requested:
                    break

                # Skip rows that were already published before the last checkpoint.
                if row_index < resume_row:
                    row_index += 1
                    continue

                # If the row limit is set and the published rows are greater than the row limit, break the loop.
                if row_limit is not None and published >= row_limit:
                    break

                # Convert the row to a RatingCreatedEvent.
                event: RatingCreatedEvent = rating_row_to_event(row)
                # Produce the event to the Kafka topic.
                producer.produce(topic, event, key=str(event.event_id))
                # Record the produced event.
                metrics.record_produced(topic)
                # Increment the published rows.
                published += 1
                # Increment the row index.
                row_index += 1
                # Set the last event ID.
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
        "Finished ratings CSV stream",
        extra={"published": published, "last_row_index": row_index, "source_file": source_file},
    )


def main() -> None:
    """
    Run the ratings producer worker.

    Do this by:
    1. Loading settings and starting the metrics HTTP server.
    2. Creating the Kafka producer.
    3. Streaming ratings.csv until row_limit or shutdown.
    """
    configure_logging()

    # Get the Kafka settings.
    kafka_settings = get_kafka_settings()
    # Get the producer settings.
    producer_settings = get_producer_settings()
    # Get the metrics settings.
    metrics_settings = get_worker_metrics_settings()

    # Create the WorkerMetrics instance.
    metrics = WorkerMetrics(metrics_settings.worker_name)
    # Start the metrics HTTP server.
    metrics.start_server(metrics_settings.metrics_port)

    # Create the GracefulShutdown instance.
    shutdown = GracefulShutdown()
    # Create the Kafka producer.
    producer = KafkaEventProducer(
        bootstrap_servers=kafka_settings.kafka_bootstrap_servers,
        log_every_n=producer_settings.producer_log_every_n,
    )

    try:
        # Stream the ratings CSV.
        stream_ratings_csv(
            producer=producer,
            topic=kafka_settings.ratings_topic,
            csv_path=producer_settings.csv_path,
            source_file=producer_settings.source_file or "ratings.csv",
            start_row=producer_settings.start_row,
            row_delay_seconds=producer_row_delay_seconds(producer_settings),
            row_limit=producer_settings.row_limit,
            checkpoint_every_n=producer_settings.checkpoint_every_n,
            metrics=metrics,
            shutdown=shutdown,
        )
    finally:
        # Flush the producer.
        producer.flush()


if __name__ == "__main__":
    main()

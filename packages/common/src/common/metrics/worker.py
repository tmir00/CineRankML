"""Prometheus metrics for Kafka workers."""

from __future__ import annotations

import time
from typing import Generator
from contextlib import contextmanager
from prometheus_client import Counter, Gauge, Histogram, start_http_server


class WorkerMetrics:
    """
    Expose standard ingestion metrics on an HTTP /metrics endpoint.

    Do this by:
    1. Registering counters, gauges, and histograms with low-cardinality labels.
    2. Offering small helper methods workers call at each processing step.
    """

    def __init__(self, worker_name: str) -> None:
        """
        Create metric collectors for one worker process.

        ============================ Arguments ============================
        worker_name: Stable worker identifier used as a metric label.
        """
        # Name of the worker. E.g. "ratings-consumer".
        self.worker_name = worker_name

        # Counter for the total number of events consumed.
        self.events_consumed_total = Counter(
            "events_consumed_total",
            "Kafka events processed by outcome",
            ["topic", "worker", "status"],
        )

        # Histogram for the latency of writing events to Postgres.
        self.db_write_latency_seconds = Histogram(
            "db_write_latency_seconds",
            "Time spent writing events to Postgres",
            ["worker"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
        )

        # Counter for the total number of failed writes to Postgres.
        self.db_write_failures_total = Counter(
            "db_write_failures_total",
            "Postgres write failures",
            ["worker", "error_type"],
        )

        # Counter for the total number of events saved to dead_letter_events.
        self.dlq_events_total = Counter(
            "dlq_events_total",
            "Messages saved to dead_letter_events",
            ["worker", "error_type"],
        )

        # Gauge for the approximate consumer lag for assigned partitions.
        self.consumer_lag = Gauge(
            "consumer_lag",
            "Approximate consumer lag for assigned partitions",
            ["topic", "worker"],
        )

        # Counter for the total number of events produced to Kafka.
        self.events_produced_total = Counter(
            "events_produced_total",
            "Kafka events produced successfully",
            ["topic", "worker"],
        )

    def start_server(self, port: int) -> None:
        """
        Start the Prometheus metrics HTTP server in a background thread.

        ============================ Arguments ============================
        port: TCP port to listen on (e.g. 9101).
        """
        # Start the Prometheus metrics HTTP server in a background thread.
        start_http_server(port)

    def record_consumed(self, topic: str, status: str) -> None:
        """Increment events_consumed_total for one processing outcome."""
        self.events_consumed_total.labels(topic=topic, worker=self.worker_name, status=status).inc()

    def record_db_failure(self, error_type: str) -> None:
        """Increment db_write_failures_total."""
        self.db_write_failures_total.labels(worker=self.worker_name, error_type=error_type).inc()

    def record_dlq(self, error_type: str) -> None:
        """Increment dlq_events_total."""
        self.dlq_events_total.labels(worker=self.worker_name, error_type=error_type).inc()

    def record_produced(self, topic: str, count: int = 1) -> None:
        """Increment events_produced_total."""
        self.events_produced_total.labels(topic=topic, worker=self.worker_name).inc(count)

    def set_consumer_lag(self, topic: str, lag: int) -> None:
        """Update consumer_lag gauge."""
        self.consumer_lag.labels(topic=topic, worker=self.worker_name).set(lag)

    @contextmanager
    def time_db_write(self) -> Generator[None, None, None]:
        """ 
        A context manager that starts a timer, then yields control to the caller.
        Once the caller is done, the context manager finally records the elapsed time.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.db_write_latency_seconds.labels(worker=self.worker_name).observe(elapsed)

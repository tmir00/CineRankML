"""Prometheus metrics for the embedder-api service."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

from prometheus_client import Counter, Histogram, start_http_server


class EmbedderMetrics:
    """
    Expose embedder-api request metrics on an HTTP /metrics endpoint.

    Do this by:
    1. Registering counters and histograms with low-cardinality labels.
    2. Offering helpers route handlers call per request.
    """

    def __init__(self, service_name: str = "embedder-api") -> None:
        """
        Create metric collectors for the embedder service.

        ============================ Arguments ============================
        service_name: Stable service identifier used as a metric label.
        """
        self.service_name = service_name

        self.embed_requests_total = Counter(
            "embed_requests_total",
            "Embedding requests handled by embedder-api",
            ["service", "status"],
        )
        self.embed_latency_seconds = Histogram(
            "embed_latency_seconds",
            "Latency of embedding requests",
            ["service"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
        )
        self.embed_batch_size = Histogram(
            "embed_batch_size",
            "Number of texts encoded per embedding request",
            ["service"],
            buckets=(1, 2, 5, 10, 20, 50, 100, 200),
        )

    def start_server(self, port: int) -> None:
        """
        Start the Prometheus metrics HTTP server in a background thread.

        ============================ Arguments ============================
        port: TCP port to listen on (e.g. 9110).
        """
        start_http_server(port)

    def record_request(self, status: str) -> None:
        """Increment embed_requests_total for one outcome."""
        self.embed_requests_total.labels(service=self.service_name, status=status).inc()

    def observe_batch_size(self, size: int) -> None:
        """Record how many texts were encoded in one request."""
        self.embed_batch_size.labels(service=self.service_name).observe(size)

    @contextmanager
    def time_request(self) -> Generator[None, None, None]:
        """Observe elapsed time for one embedding request."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.embed_latency_seconds.labels(service=self.service_name).observe(elapsed)

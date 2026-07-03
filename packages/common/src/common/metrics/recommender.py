"""Prometheus metrics for the recommender-api service."""

from __future__ import annotations

import time

from typing import Generator
from contextlib import contextmanager
from prometheus_client import Counter, Histogram, start_http_server


class RecommenderMetrics:
    """
    Expose recommender-api request and pipeline metrics on /metrics.

    Do this by:
    1. Registering counters and histograms with low-cardinality labels.
    2. Offering helpers route handlers and services call per request.
    """

    def __init__(self, service_name: str = "recommender-api") -> None:
        """
        Create metric collectors for the recommender service.

        ============================ Arguments ============================
        service_name: Stable service identifier used as a metric label.
        """
        self.service_name = service_name

        self.recommend_requests_total = Counter(
            "recommend_requests_total",
            "Recommendation API requests handled",
            ["endpoint", "status"],
        )
        self.recommend_errors_total = Counter(
            "recommend_errors_total",
            "Recommendation API errors",
            ["endpoint", "error_type"],
        )
        self.recommend_latency_ms = Histogram(
            "recommend_latency_ms",
            "End-to-end recommend request latency in milliseconds",
            ["endpoint"],
            buckets=(10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000),
        )
        self.postgres_query_latency_ms = Histogram(
            "postgres_query_latency_ms",
            "Postgres query latency in milliseconds",
            ["query"],
            buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000),
        )
        self.opensearch_query_latency_ms = Histogram(
            "opensearch_query_latency_ms",
            "OpenSearch query latency in milliseconds",
            [],
            buckets=(5, 10, 25, 50, 100, 250, 500, 1000, 2500),
        )
        self.feature_build_latency_ms = Histogram(
            "feature_build_latency_ms",
            "Online feature matrix build latency in milliseconds",
            [],
            buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000),
        )
        self.model_inference_latency_ms = Histogram(
            "model_inference_latency_ms",
            "Hybrid ranker inference latency in milliseconds",
            ["model_role"],
            buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000),
        )
        self.candidates_retrieved_count = Histogram(
            "candidates_retrieved_count",
            "Number of OpenSearch candidates retrieved per recommend request",
            [],
            buckets=(0, 1, 5, 10, 25, 50, 100, 200, 500),
        )
        self.recommendations_returned_count = Histogram(
            "recommendations_returned_count",
            "Number of recommendations returned per request",
            [],
            buckets=(0, 1, 5, 10, 20, 50),
        )

    def start_server(self, port: int) -> None:
        """
        Start the Prometheus metrics HTTP server in a background thread.

        ============================ Arguments ============================
        port: TCP port to listen on (e.g. 9111).
        """
        start_http_server(port)

    def record_request(self, endpoint: str, status: str) -> None:
        """Increment recommend_requests_total for one outcome."""
        self.recommend_requests_total.labels(endpoint=endpoint, status=status).inc()

    def record_error(self, endpoint: str, error_type: str) -> None:
        """Increment recommend_errors_total for one failure category."""
        self.recommend_errors_total.labels(endpoint=endpoint, error_type=error_type).inc()

    @contextmanager
    def time_recommend(self, endpoint: str) -> Generator[None, None, None]:
        """Observe elapsed milliseconds for one endpoint request."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.recommend_latency_ms.labels(endpoint=endpoint).observe(elapsed_ms)

    @contextmanager
    def time_postgres(self, query: str) -> Generator[None, None, None]:
        """Observe elapsed milliseconds for one Postgres query category."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.postgres_query_latency_ms.labels(query=query).observe(elapsed_ms)

    @contextmanager
    def time_opensearch(self) -> Generator[None, None, None]:
        """Observe elapsed milliseconds for one OpenSearch query."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.opensearch_query_latency_ms.observe(elapsed_ms)

    @contextmanager
    def time_feature_build(self) -> Generator[None, None, None]:
        """Observe elapsed milliseconds for feature matrix construction."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.feature_build_latency_ms.observe(elapsed_ms)

    @contextmanager
    def time_model_inference(self, model_role: str = "main") -> Generator[None, None, None]:
        """Observe elapsed milliseconds for hybrid model scoring."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.model_inference_latency_ms.labels(model_role=model_role).observe(elapsed_ms)

    def observe_candidates_retrieved(self, count: int) -> None:
        """Record how many candidates OpenSearch returned."""
        self.candidates_retrieved_count.observe(count)

    def observe_recommendations_returned(self, count: int) -> None:
        """Record how many recommendations were returned to the client."""
        self.recommendations_returned_count.observe(count)

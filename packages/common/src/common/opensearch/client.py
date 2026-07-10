"""OpenSearch client factory."""

from __future__ import annotations

from opensearchpy import OpenSearch
from common.config.settings import OpenSearchSettings


def create_opensearch_client(settings: OpenSearchSettings) -> OpenSearch:
    """
    Build an OpenSearch client from application settings.

    ============================ Arguments ============================
    settings: OpenSearch host, port, and timeout configuration.

    ============================ Returns ============================
    A connected OpenSearch client instance.
    """
    return OpenSearch(
        hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
        use_ssl=settings.opensearch_use_ssl,
        verify_certs=settings.opensearch_verify_certs,
        timeout=settings.opensearch_timeout_seconds,
    )

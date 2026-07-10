"""HTTP client for the embedder-api service."""

from __future__ import annotations

import httpx

from common.config.settings import EmbedderSettings


class EmbedderClient:
    """
    Call embedder-api over HTTP to turn text batches into vectors.

    Do this by:
    1. POSTing texts to /v1/embed.
    2. Returning the embedding matrix from the JSON response.
    """

    def __init__(self, settings: EmbedderSettings) -> None:
        """
        Create an HTTP client for embedder-api.

        ============================ Arguments ============================
        settings: Base URL and timeout configuration.
        """
        self._settings = settings
        self._client = httpx.Client(
            base_url=settings.embedder_base_url.rstrip("/"),
            timeout=settings.embedder_timeout_seconds,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Request embeddings for a batch of texts.

        ============================ Arguments ============================
        texts: Canonical embedding texts to encode.

        ============================ Returns ============================
        One embedding vector per input text, in the same order.
        """
        if not texts:
            return []

        response = self._client.post("/v1/embed", json={"texts": texts})
        response.raise_for_status()
        payload = response.json()
        return payload["embeddings"]

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> EmbedderClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

"""Entrypoint for the embedder-api service."""

from __future__ import annotations

import sys
import uvicorn
import logging

from embedder_api.app import create_app
from common.metrics.embedder import EmbedderMetrics
from pydantic_settings import BaseSettings, SettingsConfigDict
from embedder_api.model import DEFAULT_MODEL_NAME, EmbeddingModel

logger = logging.getLogger(__name__)


class EmbedderApiRuntimeSettings(BaseSettings):
    """Runtime settings for the embedder-api HTTP server."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    embedder_host: str = "0.0.0.0"
    embedder_port: int = 8080
    metrics_port: int = 9110
    embedding_model_name: str = DEFAULT_MODEL_NAME


def configure_logging() -> None:
    """Set up basic structured logging for embedder-api."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )


def main() -> None:
    """
    Start the embedder-api HTTP service.

    Do this by:
    1. Loading the sentence-transformer model.
    2. Starting the Prometheus metrics server.
    3. Serving FastAPI with uvicorn.
    """
    # Configure logging.
    configure_logging()
    # Load the runtime settings.
    settings = EmbedderApiRuntimeSettings()
    
    # Log the model name.
    logger.info(
        "Loading embedding model",
        extra={"model_name": settings.embedding_model_name},
    )
    # Load the embedding model.
    model = EmbeddingModel(settings.embedding_model_name)
    
    # Start the Prometheus metrics server.
    metrics = EmbedderMetrics()
    metrics.start_server(settings.metrics_port)

    # Create the FastAPI app.
    app = create_app(model, metrics)
    # Start the uvicorn server.
    uvicorn.run(
        app,
        host=settings.embedder_host,
        port=settings.embedder_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()

"""FastAPI application for embedder-api."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request

from common.metrics.embedder import EmbedderMetrics
from embedder_api.model import EmbeddingModel
from embedder_api.schemas import EmbedRequest, EmbedResponse


def create_app(model: EmbeddingModel, metrics: EmbedderMetrics) -> FastAPI:
    """
    Create the embedder-api FastAPI application.

    Do this by:
    1. Registering health and embedding routes.
    2. Wiring the loaded model and Prometheus metrics into handlers.

    ============================ Arguments ============================
    model: Loaded sentence-transformer model.
    metrics: Prometheus metrics collector for this service.

    ============================ Returns ============================
    Configured FastAPI app ready for uvicorn.
    """
    app = FastAPI(title="embedder-api", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str | int]:
        """
        ! Health check function.

        Returns ready when the embedding model is loaded in memory.
        """
        return {
            "status": "ok",
            "model_name": model.model_name,
            "dimension": model.dimension,
        }

    @app.post("/v1/embed", response_model=EmbedResponse)
    def embed(request_body: EmbedRequest, request: Request) -> EmbedResponse:
        """
        Encode a batch of texts into embedding vectors.

        Do this by:
        1. Validating the request body.
        2. Running the sentence-transformer model on the batch.
        3. Returning vectors with model metadata.
        """
        _ = request
        # Record the batch size.
        metrics.observe_batch_size(len(request_body.texts))
        # Try to encode the batch of texts.
        try:
            # Time the request.
            with metrics.time_request():
                embeddings = model.encode(request_body.texts)
            metrics.record_request("success")
        
        # If an exception is raised, record the failure and re-raise.
        except Exception as exc:
            metrics.record_request("failure")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        # Return the embeddings with model metadata.
        return EmbedResponse(
            embeddings=embeddings,
            model_name=model.model_name,
            dimension=model.dimension,
        )

    return app

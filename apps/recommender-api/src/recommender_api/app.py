"""FastAPI application factory for recommender-api."""

from __future__ import annotations

from fastapi import FastAPI
from recommender_api.runtime import InferenceRuntime
from recommender_api.routes.auth import create_auth_router
from recommender_api.settings import RecommenderApiSettings
from recommender_api.routes.movies import create_movies_router
from recommender_api.routes.ratings import create_ratings_router
from recommender_api.routes.recommend import create_recommend_router


def create_app(runtime: InferenceRuntime, settings: RecommenderApiSettings) -> FastAPI:
    """
    Create the recommender-api FastAPI application.

    Do this by:
    1. Registering health and versioned API routes.
    2. Wiring the loaded inference runtime into route handlers.

    ============================ Arguments ============================
    runtime: Startup-loaded inference dependencies.
    settings: Recommender API runtime settings.

    ============================ Returns ============================
    Configured FastAPI app ready for uvicorn.
    """
    app = FastAPI(title="recommender-api", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str | int | bool]:
        """
        ! Health check function.

        Returns ready when the hybrid model and CF cache are loaded in memory.
        """
        return {
            "status": "ok",
            "model_version": runtime.model_version,
            "cf_version": runtime.cf_cache.cf_version,
            "input_dim": runtime.model_config.input_dim,
            "cf_embeddings_loaded": len(runtime.cf_cache),
        }

    app.include_router(create_auth_router(runtime, settings))
    app.include_router(create_recommend_router(runtime, settings))
    app.include_router(create_ratings_router(runtime, settings))
    app.include_router(create_movies_router(runtime))

    return app

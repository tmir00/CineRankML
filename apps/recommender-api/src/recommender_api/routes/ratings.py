"""Standalone rating submission routes."""

from __future__ import annotations

from recommender_api.runtime import InferenceRuntime
from recommender_api.dependencies import get_current_user
from recommender_api.settings import RecommenderApiSettings
from fastapi import APIRouter, HTTPException, Request, status
from common.db.repositories.catalog import catalog_movie_exists
from recommender_api.schemas import SubmitRatingRequest, SubmitRatingResponse
from recommender_api.services.rating_publisher import build_api_rating_event, publish_rating_event


def create_ratings_router(runtime: InferenceRuntime, settings: RecommenderApiSettings) -> APIRouter:
    """
    Create the /v1/ratings route for publishing one rating event.

    ============================ Arguments ============================
    runtime: Startup-loaded inference runtime.
    settings: Recommender API settings.

    ============================ Returns ============================
    Configured FastAPI router.
    """
    router = APIRouter(prefix="/v1", tags=["ratings"])

    @router.post("/ratings", response_model=SubmitRatingResponse)
    def submit_rating(request: Request, request_body: SubmitRatingRequest) -> SubmitRatingResponse:
        """
        Publish one user rating to Kafka with optional recommendation lineage.

        Do this by:
        1. Requiring a valid session cookie.
        2. Validating that the movie exists in the catalog.
        3. Publishing a rating_created event to Kafka.
        """
        # Get the current user from the request.
        user = get_current_user(request, runtime, settings)
        session = runtime.session_factory()
        
        # Try to validate the movie and publish the rating event.
        try:
            # Check if the movie exists in the catalog.
            if not catalog_movie_exists(session, request_body.movie_id):
                # If the movie does not exist, record the error and raise an error.
                runtime.metrics.record_error("ratings", "invalid_movie")
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Movie {request_body.movie_id} does not exist in catalog",
                )

            # Build the rating event.
            event = build_api_rating_event(
                user_id=user.user_id,
                movie_id=request_body.movie_id,
                rating=request_body.rating,
                request_id=request_body.request_id,
                model_version=request_body.model_version,
                experiment_id=request_body.experiment_id,
            )
            # Publish the rating event to Kafka.
            publish_rating_event(runtime.kafka_producer, event)
            # Flush the Kafka producer.
            runtime.kafka_producer.flush()
            # Record the request.
            runtime.metrics.record_request("ratings", "success")
            # Return the submit rating response.
            return SubmitRatingResponse(status="queued")

        except HTTPException:
            raise
        
        except Exception:
            runtime.metrics.record_error("ratings", "internal_error")
            raise
        
        finally:
            session.close()

    return router

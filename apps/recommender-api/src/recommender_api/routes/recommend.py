"""Recommendation routes for recommender-api."""

from __future__ import annotations

from recommender_api.runtime import InferenceRuntime
from recommender_api.dependencies import get_current_user
from recommender_api.settings import RecommenderApiSettings
from fastapi import APIRouter, HTTPException, Request, status
from recommender_api.schemas import RecommendRequest, RecommendResponse
from recommender_api.services.inference import RecommendValidationError, run_recommendation


def create_recommend_router(runtime: InferenceRuntime, settings: RecommenderApiSettings) -> APIRouter:
    """
    Create the /v1/recommend route.

    ============================ Arguments ============================
    runtime: Startup-loaded inference runtime.
    settings: Recommender API settings.

    ============================ Returns ============================
    Configured FastAPI router.
    """
    router = APIRouter(prefix="/v1", tags=["recommend"])

    @router.post("/recommend", response_model=RecommendResponse)
    def recommend(request: Request, request_body: RecommendRequest) -> RecommendResponse:
        """
        Generate top-K movie recommendations for the authenticated user.

        Do this by:
        1. Requiring a valid session cookie.
        2. Publishing any inline ratings to Kafka.
        3. Running the full OpenSearch + hybrid ranker pipeline.
        """
        # Get the current user from the request.
        user = get_current_user(request, runtime, settings)
        top_k = request_body.top_k or settings.default_top_k

        session = runtime.session_factory()
        try:
            # Run the full OpenSearch + hybrid ranker pipeline.
            with runtime.metrics.time_recommend("recommend"):
                response = run_recommendation(
                    runtime=runtime,
                    session=session,
                    user_id=user.user_id,
                    new_ratings=request_body.ratings,
                    top_k=top_k,
                    refresh_token=request_body.refresh_token,
                )
            session.commit()
            runtime.metrics.record_request("recommend", "success")
            return response

        except RecommendValidationError as exc:
            session.rollback()
            runtime.metrics.record_error("recommend", "validation_error")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

        except HTTPException:
            session.rollback()
            raise

        except Exception:
            session.rollback()
            runtime.metrics.record_error("recommend", "internal_error")
            raise
        
        finally:
            session.close()

    return router

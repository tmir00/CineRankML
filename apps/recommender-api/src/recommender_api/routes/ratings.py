"""Rating list, submit, and delete routes."""

from __future__ import annotations

from recommender_api.runtime import InferenceRuntime
from recommender_api.dependencies import get_current_user
from recommender_api.settings import RecommenderApiSettings
from fastapi import APIRouter, HTTPException, Request, status
from common.db.repositories.catalog import catalog_movie_exists
from common.db.repositories.ratings import fetch_user_ratings_with_catalog, user_has_active_rating

from recommender_api.schemas import (
    DeleteRatingResponse,
    SubmitRatingRequest,
    SubmitRatingResponse,
    UserRatingItem,
    UserRatingsResponse,
)
from recommender_api.services.experiment_feedback import handle_recommendation_rating_feedback
from recommender_api.services.rating_publisher import (
    build_api_rating_deleted_event,
    build_api_rating_event,
    publish_rating_event,
)


def create_ratings_router(runtime: InferenceRuntime, settings: RecommenderApiSettings) -> APIRouter:
    """
    Create rating routes for listing, submitting, and deleting user ratings.

    ============================ Arguments ============================
    runtime: Startup-loaded inference runtime.
    settings: Recommender API settings.

    ============================ Returns ============================
    Configured FastAPI router.
    """
    router = APIRouter(prefix="/v1", tags=["ratings"])

    @router.get("/ratings", response_model=UserRatingsResponse)
    def list_ratings(request: Request) -> UserRatingsResponse:
        """
        Return the authenticated user's active ratings with catalog display metadata.

        Do this by:
        1. Requiring a valid session cookie.
        2. Loading active ratings from Postgres.
        3. Joining catalog title, poster, and genre fields for the frontend carousel.
        """
        user = get_current_user(request, runtime, settings)
        session = runtime.session_factory()

        try:
            rows = fetch_user_ratings_with_catalog(session, user.user_id)
            runtime.metrics.record_request("ratings_list", "success")
            return UserRatingsResponse(
                ratings=[
                    UserRatingItem(
                        movie_id=row.movie_id,
                        title=row.title,
                        year=row.year,
                        genres=row.genres,
                        poster_path=row.poster_path,
                        rating=row.rating,
                        rated_at=row.rated_at,
                    )
                    for row in rows
                ]
            )
        except Exception:
            runtime.metrics.record_error("ratings_list", "internal_error")
            raise
        finally:
            session.close()

    @router.post("/ratings", response_model=SubmitRatingResponse)
    def submit_rating(request: Request, request_body: SubmitRatingRequest) -> SubmitRatingResponse:
        """
        Publish one user rating to Kafka with optional recommendation lineage.

        Do this by:
        1. Requiring a valid session cookie.
        2. Validating that the movie exists in the catalog.
        3. Publishing a rating_created event to Kafka.
        """
        user = get_current_user(request, runtime, settings)
        session = runtime.session_factory()

        try:
            if not catalog_movie_exists(session, request_body.movie_id):
                runtime.metrics.record_error("ratings", "invalid_movie")
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Movie {request_body.movie_id} does not exist in catalog",
                )

            event = build_api_rating_event(
                user_id=user.user_id,
                movie_id=request_body.movie_id,
                rating=request_body.rating,
                request_id=request_body.request_id,
                model_version=request_body.model_version,
                experiment_id=request_body.experiment_id,
            )
            publish_rating_event(runtime.kafka_producer, event)
            runtime.kafka_producer.flush()

            if (
                request_body.request_id
                and request_body.model_version
                and request_body.model_role
                and request_body.experiment_id
            ):
                handle_recommendation_rating_feedback(
                    runtime=runtime,
                    session=session,
                    user_id=user.user_id,
                    request_id=request_body.request_id,
                    movie_id=request_body.movie_id,
                    model_version=request_body.model_version,
                    model_role=request_body.model_role,
                    experiment_id=request_body.experiment_id,
                    rating=request_body.rating,
                )
                session.commit()

            runtime.metrics.record_request("ratings", "success")
            return SubmitRatingResponse(status="queued")

        except HTTPException:
            session.rollback()
            raise

        except Exception:
            session.rollback()
            runtime.metrics.record_error("ratings", "internal_error")
            raise

        finally:
            session.close()

    @router.delete("/ratings/{movie_id}", response_model=DeleteRatingResponse)
    def delete_rating(request: Request, movie_id: int) -> DeleteRatingResponse:
        """
        Publish one rating_deleted event for an actively rated movie.

        Do this by:
        1. Requiring a valid session cookie.
        2. Verifying the movie exists and the user currently rates it.
        3. Publishing a rating_deleted event to Kafka.
        """
        user = get_current_user(request, runtime, settings)
        session = runtime.session_factory()

        try:
            if not catalog_movie_exists(session, movie_id):
                runtime.metrics.record_error("ratings_delete", "invalid_movie")
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Movie {movie_id} does not exist in catalog",
                )

            if not user_has_active_rating(session, user.user_id, movie_id):
                runtime.metrics.record_error("ratings_delete", "not_rated")
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Movie {movie_id} is not actively rated by this user",
                )

            event = build_api_rating_deleted_event(
                user_id=user.user_id,
                movie_id=movie_id,
            )
            publish_rating_event(runtime.kafka_producer, event)
            runtime.kafka_producer.flush()
            runtime.metrics.record_request("ratings_delete", "success")
            return DeleteRatingResponse(status="queued")

        except HTTPException:
            raise

        except Exception:
            runtime.metrics.record_error("ratings_delete", "internal_error")
            raise

        finally:
            session.close()

    return router

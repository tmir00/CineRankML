"""Movie search routes for onboarding and manual testing."""

from __future__ import annotations

from fastapi import APIRouter, Query
from recommender_api.runtime import InferenceRuntime
from common.opensearch.search import search_movies_by_title
from recommender_api.schemas import MovieSearchItem, MovieSearchResponse


def create_movies_router(runtime: InferenceRuntime) -> APIRouter:
    """
    Create movie search routes backed by OpenSearch.

    ============================ Arguments ============================
    runtime: Startup-loaded inference runtime.

    ============================ Returns ============================
    Configured FastAPI router.
    """
    router = APIRouter(prefix="/v1/movies", tags=["movies"])

    @router.get("/search", response_model=MovieSearchResponse)
    def search_movies(q: str = Query(min_length=1), limit: int = Query(default=20, ge=1, le=50)) -> MovieSearchResponse:
        """
        Search catalog movies by title for onboarding movie pickers.

        Do this by:
        1. Running a title multi_match query against OpenSearch.
        2. Returning basic movie metadata for each hit.
        """
        # Run a title multi_match query against OpenSearch.
        with runtime.metrics.time_opensearch():
            hits = search_movies_by_title(
                runtime.opensearch_client,
                runtime.opensearch_index_alias,
                q,
                limit=limit,
            )
        
        # Record the request.
        runtime.metrics.record_request("movies_search", "success")
        
        # Build the movie search items.
        movies = [
            MovieSearchItem(
                movie_id=hit.movie_id,
                title=hit.title,
                year=hit.year,
                genres=hit.genres,
                poster_path=hit.poster_path or None,
                poster_safe=hit.poster_safe,
                show_poster=hit.show_poster,
                certification_us=hit.certification_us,
            )
            for hit in hits
        ]
        return MovieSearchResponse(movies=movies)

    return router

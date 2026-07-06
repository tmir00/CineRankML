"""Orchestrate the full /recommend inference pipeline."""

from __future__ import annotations

import uuid
import torch

from datetime import UTC, datetime
from sqlalchemy.orm import Session
from common.features.schema import INPUT_DIM
from recommender_api.runtime import InferenceRuntime
from common.opensearch.retrieval import retrieve_candidate_pool
from common.db.repositories.ratings import fetch_user_ratings
from common.db.repositories.catalog import catalog_movie_exists, get_movie_genres_by_ids
from common.db.repositories.recommendations import ImpressionRow, insert_impressions
from common.recommendation.liked_genres import derive_liked_genres
from recommender_api.schemas import RatingInput, RecommendationItem, RecommendResponse
from recommender_api.services.rating_publisher import build_api_rating_event, publish_rating_event

from recommender_api.services.feature_builder import (
    build_feature_matrix,
    build_user_profiles,
    merge_user_ratings,
)


class RecommendValidationError(ValueError):
    """Raised when a recommend request fails business validation."""


def run_recommendation(*, runtime: InferenceRuntime, session: Session, user_id: int, \
                            new_ratings: list[RatingInput], top_k: int, \
                                kafka_producer_flush: bool = True) -> RecommendResponse:
    """
    Run the full online recommendation pipeline for one authenticated user.

    Do this by:
    1. Validating and publishing any new inline ratings to Kafka.
    2. Merging request ratings with Postgres history.
    3. Retrieving OpenSearch candidates and building the feature matrix.
    4. Scoring with the main hybrid model and logging impressions.

    ============================ Arguments ============================
    runtime: Startup-loaded inference dependencies.
    session: An open SQLAlchemy session inside a transaction.
    user_id: Authenticated app user id.
    new_ratings: Inline ratings from the request body.
    top_k: Number of recommendations to return.
    kafka_producer_flush: Whether to flush Kafka after publishing ratings.

    ============================ Returns ============================
    Ranked recommendations and request metadata.
    """
    # Validate each inline rating against the catalog.
    for rating in new_ratings:
        if not catalog_movie_exists(session, rating.movie_id):
            raise RecommendValidationError(f"Movie {rating.movie_id} does not exist in catalog")

    # Publish new ratings to Kafka before building features.
    for rating in new_ratings:
        event = build_api_rating_event(
            user_id=user_id,
            movie_id=rating.movie_id,
            rating=rating.rating,
        )
        publish_rating_event(runtime.kafka_producer, event)

    if kafka_producer_flush:
        runtime.kafka_producer.flush()

    # Load stored history and merge with request ratings.
    with runtime.metrics.time_postgres("user_ratings"):
        history_rows = fetch_user_ratings(session, user_id)
    history = {row.movie_id: row.rating for row in history_rows}
    merged_ratings = merge_user_ratings(history, new_ratings)

    # If the user has less than the minimum number of ratings, raise an error.
    if len(merged_ratings) < runtime.min_ratings_for_recommend:
        raise RecommendValidationError(
            f"Need at least {runtime.min_ratings_for_recommend} ratings; got {len(merged_ratings)}"
        )

    # Build user-side profiles and behavior stats.
    with runtime.metrics.time_feature_build():
        user_content_profile, user_cf_profile, user_behavior = build_user_profiles(
            session,
            merged_ratings,
            content_embedding_version=runtime.model_config.content_embedding_version,
            cf_cache=runtime.cf_cache,
            user_id=user_id,
            metrics=runtime.metrics,
        )

        # Retrieve candidate movies from OpenSearch using multi-bucket retrieval.
        exclude_movie_ids = set(merged_ratings.keys())
        with runtime.metrics.time_postgres("movie_genres"):
            movie_genres = get_movie_genres_by_ids(session, list(merged_ratings.keys()))
        liked_genres = derive_liked_genres(
            merged_ratings,
            movie_genres,
            top_n=runtime.retrieval.liked_genre_count,
        )
        with runtime.metrics.time_opensearch():
            candidates = retrieve_candidate_pool(
                client=runtime.opensearch_client,
                index_alias=runtime.opensearch_index_alias,
                query_vector=user_content_profile,
                liked_genres=liked_genres,
                exclude_movie_ids=exclude_movie_ids,
                user_id=user_id,
                settings=runtime.retrieval,
            )

        # Observe the number of candidates retrieved from OpenSearch.
        runtime.metrics.observe_candidates_retrieved(len(candidates))

        if not candidates:
            raise RecommendValidationError("No candidate movies retrieved from OpenSearch")

        # Build the feature matrix.
        feature_matrix = build_feature_matrix(
            user_content_profile=user_content_profile,
            user_cf_profile=user_cf_profile,
            user_behavior=user_behavior,
            candidates=candidates,
            cf_cache=runtime.cf_cache,
            model_config=runtime.model_config,
            user_id=user_id,
            metrics=runtime.metrics,
        )

    # If the model input dimension does not match the schema, raise an error.
    if runtime.model_config.input_dim != INPUT_DIM:
        raise RecommendValidationError(
            f"Model input_dim {runtime.model_config.input_dim} does not match schema {INPUT_DIM}"
        )

    # Score candidates with the main hybrid model.
    runtime.model.eval()
    with torch.no_grad(), runtime.metrics.time_model_inference("main"):
        scores = runtime.model(feature_matrix.to(runtime.device)).detach().cpu().numpy()

    # Get the ranked indices of the candidates.
    ranked_indices = scores.argsort()[::-1][:top_k]
    # Generate a request id.
    request_id = f"rec-{uuid.uuid4()}"
    shown_at = datetime.now(tz=UTC)

    # Create lists to store the recommendations and impression rows.
    recommendations: list[RecommendationItem] = []
    impression_rows: list[ImpressionRow] = []

    # Iterate over the ranked indices and build the recommendations and impression rows.
    for rank_position, candidate_index in enumerate(ranked_indices, start=1):
        # Get the candidate from the candidates list.
        candidate = candidates[int(candidate_index)]
        # Get the predicted score for the candidate.
        predicted_score = float(scores[int(candidate_index)])
        # Build the recommendation item.
        recommendations.append(
            RecommendationItem(
                movie_id=candidate.movie_id,
                title=candidate.title,
                year=candidate.year,
                genres=candidate.genres,
                poster_path=candidate.poster_path or None,
                predicted_score=predicted_score,
                rank_position=rank_position,
            )
        )
        # Build the impression row.
        impression_rows.append(
            ImpressionRow(
                request_id=request_id,
                user_id=user_id,
                movie_id=candidate.movie_id,
                rank_position=rank_position,
                model_version=runtime.model_version,
                model_role="main",
                experiment_id="none",
                predicted_score=predicted_score,
                shown_at=shown_at,
                retrieval_source=candidate.retrieval_source,
            )
        )

    # Insert the impression rows into the database.
    with runtime.metrics.time_postgres("impressions"):
        insert_impressions(session, impression_rows)

    # Observe the number of recommendations returned.
    runtime.metrics.observe_recommendations_returned(len(recommendations))

    # Return the response.
    return RecommendResponse(
        request_id=request_id,
        model_version=runtime.model_version,
        recommendations=recommendations,
    )

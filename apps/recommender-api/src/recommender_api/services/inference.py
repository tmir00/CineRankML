"""Orchestrate the full /recommend inference pipeline."""

from __future__ import annotations

import uuid
import torch

from datetime import UTC, datetime
from sqlalchemy.orm import Session
from common.features.schema import INPUT_DIM
from recommender_api.runtime import InferenceRuntime, LoadedHybridModel
from common.opensearch.retrieval import retrieve_candidate_pool
from common.db.repositories.ratings import fetch_user_ratings
from common.db.repositories.catalog import catalog_movie_exists, get_movie_genres_by_ids
from common.db.repositories.recommendations import ImpressionRow, insert_impressions
from common.recommendation.merge_recommendations import RankedCandidate, merge_ranked_recommendations
from common.recommendation.split_policy import SplitFractions, allocate_model_slots
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


def _score_candidates(
    *,
    runtime: InferenceRuntime,
    bundle: LoadedHybridModel,
    feature_matrix: torch.Tensor,
    candidates: list,
    model_role: str,
) -> list[RankedCandidate]:
    """
    Score all candidates with one hybrid model and return ranked rows.

    ============================ Arguments ============================
    runtime: Startup-loaded inference runtime.
    bundle: Loaded model bundle (main or candidate).
    feature_matrix: Feature matrix built for the candidate pool.
    candidates: OpenSearch candidate metadata rows.
    model_role: Label used in metrics and impressions (main or candidate).

    ============================ Returns ============================
    Candidates ranked best-first with scores attached.
    """
    bundle.model.eval()
    with torch.no_grad(), runtime.metrics.time_model_inference(model_role):
        scores = bundle.model(feature_matrix.to(runtime.device)).detach().cpu().numpy()

    ranked_indices = scores.argsort()[::-1]
    ranked: list[RankedCandidate] = []
    for candidate_index in ranked_indices:
        idx = int(candidate_index)
        candidate = candidates[idx]
        ranked.append(
            RankedCandidate(
                movie_id=candidate.movie_id,
                predicted_score=float(scores[idx]),
                model_role=model_role,
                model_version=bundle.model_version,
                candidate_index=idx,
                title=candidate.title,
                year=candidate.year,
                genres=candidate.genres,
                poster_path=candidate.poster_path or None,
                poster_safe=candidate.poster_safe,
                show_poster=candidate.show_poster,
                certification_us=candidate.certification_us,
                retrieval_source=candidate.retrieval_source,
            )
        )
    return ranked


def run_recommendation(*, runtime: InferenceRuntime, session: Session, user_id: int, \
                            new_ratings: list[RatingInput], top_k: int, \
                                kafka_producer_flush: bool = True) -> RecommendResponse:
    """
    Run the full online recommendation pipeline for one authenticated user.

    Do this by:
    1. Validating and publishing any new inline ratings to Kafka.
    2. Merging request ratings with Postgres history.
    3. Retrieving OpenSearch candidates and building the feature matrix.
    4. Scoring with main and candidate models, then merging by split fractions.
    5. Logging impressions with experiment lineage.

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
    for rating in new_ratings:
        if not catalog_movie_exists(session, rating.movie_id):
            raise RecommendValidationError(f"Movie {rating.movie_id} does not exist in catalog")

    for rating in new_ratings:
        event = build_api_rating_event(
            user_id=user_id,
            movie_id=rating.movie_id,
            rating=rating.rating,
        )
        publish_rating_event(runtime.kafka_producer, event)

    if kafka_producer_flush:
        runtime.kafka_producer.flush()

    with runtime.metrics.time_postgres("user_ratings"):
        history_rows = fetch_user_ratings(session, user_id)
    history = {row.movie_id: row.rating for row in history_rows}
    merged_ratings = merge_user_ratings(history, new_ratings)

    if len(merged_ratings) < runtime.min_ratings_for_recommend:
        raise RecommendValidationError(
            f"Need at least {runtime.min_ratings_for_recommend} ratings; got {len(merged_ratings)}"
        )

    with runtime.metrics.time_feature_build():
        user_content_profile, user_cf_profile, user_behavior = build_user_profiles(
            session,
            merged_ratings,
            content_embedding_version=runtime.main.model_config.content_embedding_version,
            cf_cache=runtime.main.cf_cache,
            user_id=user_id,
            metrics=runtime.metrics,
        )

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

        runtime.metrics.observe_candidates_retrieved(len(candidates))

        if not candidates:
            raise RecommendValidationError("No candidate movies retrieved from OpenSearch")

        feature_matrix = build_feature_matrix(
            user_content_profile=user_content_profile,
            user_cf_profile=user_cf_profile,
            user_behavior=user_behavior,
            candidates=candidates,
            cf_cache=runtime.main.cf_cache,
            model_config=runtime.main.model_config,
            user_id=user_id,
            metrics=runtime.metrics,
        )

    if runtime.main.model_config.input_dim != INPUT_DIM:
        raise RecommendValidationError(
            f"Model input_dim {runtime.main.model_config.input_dim} does not match schema {INPUT_DIM}"
        )

    fractions = SplitFractions(
        main=runtime.experiment.main_split_fraction,
        candidate=runtime.experiment.candidate_split_fraction,
    )
    has_candidate = runtime.candidate is not None and fractions.candidate > 0
    main_slots, candidate_slots = allocate_model_slots(top_k, fractions, has_candidate=has_candidate)

    main_ranked = _score_candidates(
        runtime=runtime,
        bundle=runtime.main,
        feature_matrix=feature_matrix,
        candidates=candidates,
        model_role="main",
    )

    candidate_ranked: list[RankedCandidate] = []
    if has_candidate and candidate_slots > 0 and runtime.candidate is not None:
        candidate_feature_matrix = feature_matrix
        if runtime.candidate.model_config.cf_version != runtime.main.model_config.cf_version:
            _, candidate_cf_profile, _ = build_user_profiles(
                session,
                merged_ratings,
                content_embedding_version=runtime.candidate.model_config.content_embedding_version,
                cf_cache=runtime.candidate.cf_cache,
                user_id=user_id,
                metrics=runtime.metrics,
            )
            candidate_feature_matrix = build_feature_matrix(
                user_content_profile=user_content_profile,
                user_cf_profile=candidate_cf_profile,
                user_behavior=user_behavior,
                candidates=candidates,
                cf_cache=runtime.candidate.cf_cache,
                model_config=runtime.candidate.model_config,
                user_id=user_id,
                metrics=runtime.metrics,
            )

        candidate_ranked = _score_candidates(
            runtime=runtime,
            bundle=runtime.candidate,
            feature_matrix=candidate_feature_matrix,
            candidates=candidates,
            model_role="candidate",
        )

    merged = merge_ranked_recommendations(
        main_ranked=main_ranked,
        candidate_ranked=candidate_ranked,
        main_slots=main_slots,
        candidate_slots=candidate_slots,
    )[:top_k]

    request_id = f"rec-{uuid.uuid4()}"
    shown_at = datetime.now(tz=UTC)
    experiment_id = runtime.experiment_id

    recommendations: list[RecommendationItem] = []
    impression_rows: list[ImpressionRow] = []

    for rank_position, row in enumerate(merged, start=1):
        recommendations.append(
            RecommendationItem(
                movie_id=row.movie_id,
                title=row.title,
                year=row.year,
                genres=row.genres,
                poster_path=row.poster_path,
                poster_safe=row.poster_safe,
                show_poster=row.show_poster,
                certification_us=row.certification_us,
                predicted_score=row.predicted_score,
                rank_position=rank_position,
                model_role=row.model_role,
                model_version=row.model_version,
            )
        )
        impression_rows.append(
            ImpressionRow(
                request_id=request_id,
                user_id=user_id,
                movie_id=row.movie_id,
                rank_position=rank_position,
                model_version=row.model_version,
                model_role=row.model_role,
                experiment_id=experiment_id,
                predicted_score=row.predicted_score,
                shown_at=shown_at,
                retrieval_source=row.retrieval_source,
            )
        )

    with runtime.metrics.time_postgres("impressions"):
        insert_impressions(session, impression_rows)

    runtime.metrics.observe_recommendations_returned(len(recommendations))

    return RecommendResponse(
        request_id=request_id,
        model_version=runtime.main.model_version,
        experiment_id=experiment_id,
        recommendations=recommendations,
    )

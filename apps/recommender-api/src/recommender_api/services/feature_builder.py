"""Online hybrid feature matrix assembly for recommender-api."""

from __future__ import annotations

import torch
import numpy as np
import logging

from numpy.typing import NDArray
from sqlalchemy.orm import Session
from recommender_api.schemas import RatingInput
from common.opensearch.search import CandidateMovieDoc
from common.features.vector import build_feature_vector
from common.features.behavior import compute_user_behavior
from common.features.similarity import weighted_embedding_mean
from common.storage.cf_embedding_cache import CfEmbeddingCache
from common.db.repositories.embeddings import get_movie_embeddings
from common.features.normalization import normalize_candidate_metadata
from common.schemas.hybrid_ranker_artifact_manifest import HybridModelConfig
from common.features.schema import CF_EMBEDDING_DIM, CONTENT_EMBEDDING_DIM, INPUT_DIM
from common.metrics.recommender import RecommenderMetrics


logger = logging.getLogger(__name__)


def _record_feature_issue(metrics: RecommenderMetrics | None, error_type: str) -> None:
    """Increment recommend error metrics when feature assembly degrades."""
    if metrics is not None:
        metrics.record_error("recommend", error_type)


def merge_user_ratings(history: dict[int, float], new_ratings: list[RatingInput],) -> dict[int, float]:
    """
    Merge Postgres rating history with inline request ratings.

    Do this by:
    1. Starting from the stored history keyed by movie_id.
    2. Overwriting or adding entries from the request body.

    ============================ Arguments ============================
    history: Existing ratings keyed by movie_id.
    new_ratings: Inline ratings sent with the current request.

    ============================ Returns ============================
    Combined latest rating per movie.
    """
    merged = dict(history)
    for rating in new_ratings:
        merged[rating.movie_id] = rating.rating
    return merged


def build_user_profiles(session: Session, merged_ratings: dict[int, float], *,
                            content_embedding_version: str, cf_cache: CfEmbeddingCache,
                            user_id: int | None = None,
                            metrics: RecommenderMetrics | None = None) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """
    Build user content profile, CF profile, and behavior stats from ratings.

    Do this by:
    1. Loading content embeddings for rated movies from Postgres.
    2. Looking up CF embeddings from the startup cache.
    3. Computing weighted means and the five behavior statistics.

    ============================ Arguments ============================
    session: An open SQLAlchemy session.
    merged_ratings: Latest rating per movie for the user.
    content_embedding_version: Embedding version required by the deployed model.
    cf_cache: Startup CF embedding cache.
    user_id: Optional app user id for degraded-feature logging.
    metrics: Optional Prometheus metrics helper.

    ============================ Returns ============================
    Tuple of (user_content_profile, user_cf_profile, user_behavior).
    """
    # Get the movie ids from the merged ratings.
    movie_ids = list(merged_ratings.keys())
    # Get the content embeddings from the database.
    content_rows = get_movie_embeddings(session, movie_ids, content_embedding_version)

    # Create lists to store the content and CF embeddings.
    content_vectors: list[NDArray[np.float32]] = []
    cf_vectors: list[NDArray[np.float32]] = []
    weights: list[float] = []
    missing_content_movie_ids: list[int] = []
    invalid_content_shape_movie_ids: list[int] = []
    missing_cf_movie_ids: list[int] = []

    # Iterate over the movie ids and get the content and CF embeddings.
    for movie_id in movie_ids:
        rating = merged_ratings[movie_id]
        weights.append(rating)

        # Get the content embedding from the database.
        stored = content_rows.get(movie_id)
        
        # If the content embedding is found, add it to the list.
        if stored is not None:
            content_embedding = np.asarray(stored.embedding, dtype=np.float32)
            if content_embedding.shape != (CONTENT_EMBEDDING_DIM,):
                invalid_content_shape_movie_ids.append(movie_id)
                content_vectors.append(np.zeros(CONTENT_EMBEDDING_DIM, dtype=np.float32))
            else:
                content_vectors.append(content_embedding)
        else:
            missing_content_movie_ids.append(movie_id)
            content_vectors.append(np.zeros(CONTENT_EMBEDDING_DIM, dtype=np.float32))


        if not cf_cache.has_movie(movie_id):
            missing_cf_movie_ids.append(movie_id)
        cf_vectors.append(cf_cache.get(movie_id))

    # If there are missing content embeddings, log the warning.
    if missing_content_movie_ids:
        logger.warning(
            "Missing content embeddings for rated movies",
            extra={
                "user_id": user_id,
                "missing_count": len(missing_content_movie_ids),
                "rated_count": len(movie_ids),
                "content_embedding_version": content_embedding_version,
                "sample_movie_ids": missing_content_movie_ids[:10],
            },
        )
        _record_feature_issue(metrics, "missing_content_embedding")

    # If there are invalid content embeddings, log the warning.
    if invalid_content_shape_movie_ids:
        logger.warning(
            "Invalid content embedding shape for rated movies",
            extra={
                "user_id": user_id,
                "invalid_count": len(invalid_content_shape_movie_ids),
                "expected_dim": CONTENT_EMBEDDING_DIM,
                "sample_movie_ids": invalid_content_shape_movie_ids[:10],
            },
        )
        _record_feature_issue(metrics, "invalid_content_embedding_shape")

    # If there are missing CF embeddings, log the warning.
    if missing_cf_movie_ids:
        logger.warning(
            "Missing CF embeddings for rated movies",
            extra={
                "user_id": user_id,
                "missing_count": len(missing_cf_movie_ids),
                "rated_count": len(movie_ids),
                "cf_version": cf_cache.cf_version,
                "sample_movie_ids": missing_cf_movie_ids[:10],
            },
        )
        _record_feature_issue(metrics, "missing_cf_embedding")

    # Compute the user content profile.
    user_content_profile = weighted_embedding_mean(content_vectors, weights)
    # Compute the user CF profile.
    user_cf_profile = weighted_embedding_mean(cf_vectors, weights)

    # If the user content profile is invalid, log the warning.
    if user_content_profile.shape != (CONTENT_EMBEDDING_DIM,):
        logger.warning(
            "Reset invalid user content profile shape",
            extra={
                "user_id": user_id,
                "actual_shape": tuple(user_content_profile.shape),
                "expected_dim": CONTENT_EMBEDDING_DIM,
            },
        )
        _record_feature_issue(metrics, "invalid_user_content_profile_shape")
        user_content_profile = np.zeros(CONTENT_EMBEDDING_DIM, dtype=np.float32)

    # If the user CF profile is invalid, log the warning.
    if user_cf_profile.shape != (CF_EMBEDDING_DIM,):
        logger.warning(
            "Reset invalid user CF profile shape",
            extra={
                "user_id": user_id,
                "actual_shape": tuple(user_cf_profile.shape),
                "expected_dim": CF_EMBEDDING_DIM,
            },
        )
        _record_feature_issue(metrics, "invalid_user_cf_profile_shape")
        user_cf_profile = np.zeros(CF_EMBEDDING_DIM, dtype=np.float32)

    # Compute the user behavior.
    user_behavior = compute_user_behavior(list(merged_ratings.values()))
    # Return the user content profile, CF profile, and behavior.
    return user_content_profile, user_cf_profile, user_behavior


def build_feature_matrix(*, user_content_profile: NDArray[np.float32], user_cf_profile: NDArray[np.float32],
                                user_behavior: NDArray[np.float32], candidates: list[CandidateMovieDoc],
                                    cf_cache: CfEmbeddingCache, model_config: HybridModelConfig,
                                    user_id: int | None = None,
                                    metrics: RecommenderMetrics | None = None) -> torch.Tensor:
    """
    Build the [num_candidates, 1356] feature matrix for hybrid ranker scoring.

    Do this by:
    1. Normalizing candidate metadata with train-fit stats from model_config.
    2. Looking up candidate CF embeddings from the startup cache.
    3. Assembling one fixed-size vector per candidate.

    ============================ Arguments ============================
    user_content_profile: User content embedding profile.
    user_cf_profile: User CF embedding profile.
    user_behavior: Five user behavior statistics.
    candidates: OpenSearch candidate documents.
    cf_cache: Startup CF embedding cache.
    model_config: Deployed hybrid model configuration.
    user_id: Optional app user id for degraded-feature logging.
    metrics: Optional Prometheus metrics helper.

    ============================ Returns ============================
    Feature tensor with shape [num_candidates, INPUT_DIM].
    """
    # Create a list to store the feature vectors.
    rows: list[NDArray[np.float32]] = []
    metadata_stats = model_config.metadata_normalization

    # Iterate over the candidates and build the feature vectors.
    for candidate in candidates:
        # Get the candidate content embedding.
        candidate_content = np.asarray(candidate.content_embedding, dtype=np.float32)
        # Get the candidate CF embedding from the cache.
        candidate_cf = cf_cache.get(candidate.movie_id)
        # Normalize the candidate metadata.
        metadata_values = normalize_candidate_metadata(
            year=candidate.year,
            runtime=candidate.runtime,
            tmdb_popularity=candidate.tmdb_popularity,
            tmdb_vote_average=candidate.tmdb_vote_average,
            tmdb_vote_count=candidate.tmdb_vote_count,
            stats=metadata_stats,
        )
        # Convert the metadata values to a numpy array.
        candidate_metadata = np.asarray(metadata_values, dtype=np.float32)
        # Build the feature vector.
        rows.append(
            build_feature_vector(
                user_content_profile=user_content_profile,
                candidate_content_embedding=candidate_content,
                user_cf_profile=user_cf_profile,
                candidate_cf_embedding=candidate_cf,
                user_behavior=user_behavior,
                candidate_metadata=candidate_metadata,
            )
        )

    # If there are no feature vectors, log the warning.
    if not rows:
        logger.warning(
            "Built empty feature matrix",
            extra={
                "user_id": user_id,
                "candidate_count": len(candidates),
                "input_dim": INPUT_DIM,
            },
        )
        _record_feature_issue(metrics, "empty_feature_matrix")
        return torch.empty((0, INPUT_DIM), dtype=torch.float32)

    matrix = np.stack(rows, axis=0)
    return torch.from_numpy(matrix)

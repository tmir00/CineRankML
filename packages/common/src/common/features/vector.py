"""Assemble the fixed-size hybrid ranker input vector."""

from __future__ import annotations

import numpy as np

from numpy.typing import NDArray
from common.features.schema import (
    CANDIDATE_METADATA_DIM,
    CF_EMBEDDING_DIM,
    CONTENT_EMBEDDING_DIM,
    INPUT_DIM,
    OFFSET_CANDIDATE_CF_EMBEDDING,
    OFFSET_CANDIDATE_CONTENT_EMBEDDING,
    OFFSET_CANDIDATE_METADATA,
    OFFSET_CF_COSINE,
    OFFSET_CF_PRODUCT,
    OFFSET_CONTENT_COSINE,
    OFFSET_CONTENT_PRODUCT,
    OFFSET_USER_BEHAVIOR,
    OFFSET_USER_CF_PROFILE,
    OFFSET_USER_CONTENT_PROFILE,
    USER_BEHAVIOR_DIM,
)
from common.features.similarity import cosine_similarity, elementwise_product


def build_feature_vector(*, user_content_profile: NDArray[np.float32], candidate_content_embedding: NDArray[np.float32],
                            user_cf_profile: NDArray[np.float32], candidate_cf_embedding: NDArray[np.float32], user_behavior: NDArray[np.float32],
                                candidate_metadata: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Concatenate all hybrid ranker input slots into one float32 vector.

    Do this by:
    1. Computing cosine similarities and elementwise products for content and CF blocks.
    2. Writing each slot into its fixed offset in a pre-sized zero vector.

    ============================ Arguments ============================
    user_content_profile: Rating-weighted mean content embedding for user history.
    candidate_content_embedding: Content embedding for the candidate movie.
    user_cf_profile: Rating-weighted mean CF embedding for user history.
    candidate_cf_embedding: CF embedding for the candidate movie.
    user_behavior: Five user behavior statistics from temporal history.
    candidate_metadata: Five normalized candidate metadata values.

    ============================ Returns ============================
    Feature vector of length INPUT_DIM (1356).
    """
    if user_content_profile.shape != (CONTENT_EMBEDDING_DIM,):
        raise ValueError(f"user_content_profile must have shape ({CONTENT_EMBEDDING_DIM},)")
    if candidate_content_embedding.shape != (CONTENT_EMBEDDING_DIM,):
        raise ValueError(f"candidate_content_embedding must have shape ({CONTENT_EMBEDDING_DIM},)")
    if user_cf_profile.shape != (CF_EMBEDDING_DIM,):
        raise ValueError(f"user_cf_profile must have shape ({CF_EMBEDDING_DIM},)")
    if candidate_cf_embedding.shape != (CF_EMBEDDING_DIM,):
        raise ValueError(f"candidate_cf_embedding must have shape ({CF_EMBEDDING_DIM},)")
    if user_behavior.shape != (USER_BEHAVIOR_DIM,):
        raise ValueError(f"user_behavior must have shape ({USER_BEHAVIOR_DIM},)")
    if candidate_metadata.shape != (CANDIDATE_METADATA_DIM,):
        raise ValueError(f"candidate_metadata must have shape ({CANDIDATE_METADATA_DIM},)")

    # Initialize a zero vector of the correct length.
    features = np.zeros(INPUT_DIM, dtype=np.float32)

    # Compute the cosine similarities for the content and CF blocks.
    content_cosine = cosine_similarity(user_content_profile, candidate_content_embedding)
    cf_cosine = cosine_similarity(user_cf_profile, candidate_cf_embedding)

    # Write the content and CF blocks into the feature vector.
    features[OFFSET_USER_CONTENT_PROFILE:OFFSET_CANDIDATE_CONTENT_EMBEDDING] = user_content_profile
    features[OFFSET_CANDIDATE_CONTENT_EMBEDDING:OFFSET_CONTENT_COSINE] = candidate_content_embedding
    features[OFFSET_CONTENT_COSINE] = content_cosine
    features[OFFSET_CONTENT_PRODUCT:OFFSET_USER_CF_PROFILE] = elementwise_product(
        user_content_profile,
        candidate_content_embedding,
    )
    # Write the CF block into the feature vector.
    features[OFFSET_USER_CF_PROFILE:OFFSET_CANDIDATE_CF_EMBEDDING] = user_cf_profile
    features[OFFSET_CANDIDATE_CF_EMBEDDING:OFFSET_CF_COSINE] = candidate_cf_embedding
    features[OFFSET_CF_COSINE] = cf_cosine
    features[OFFSET_CF_PRODUCT:OFFSET_USER_BEHAVIOR] = elementwise_product(
        user_cf_profile,
        candidate_cf_embedding,
    )
    features[OFFSET_USER_BEHAVIOR:OFFSET_CANDIDATE_METADATA] = user_behavior.astype(np.float32)
    features[OFFSET_CANDIDATE_METADATA:INPUT_DIM] = candidate_metadata.astype(np.float32)

    return features

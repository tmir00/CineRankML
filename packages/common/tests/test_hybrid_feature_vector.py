"""Tests for hybrid ranker feature vector assembly."""

from __future__ import annotations

import numpy as np

from common.features.schema import INPUT_DIM, OFFSET_CF_COSINE, OFFSET_CONTENT_COSINE
from common.features.similarity import cosine_similarity, elementwise_product, weighted_embedding_mean
from common.features.vector import build_feature_vector


def test_build_feature_vector_has_expected_length_and_slots() -> None:
    """The concatenated vector should be 1356 dims with cosine values in the right slots."""
    user_content = np.full(384, 0.5, dtype=np.float32)
    candidate_content = np.full(384, 1.0, dtype=np.float32)
    user_cf = np.full(64, 0.25, dtype=np.float32)
    candidate_cf = np.full(64, 0.5, dtype=np.float32)
    user_behavior = np.array([1, 4, 0, 1, 0], dtype=np.float32)
    candidate_metadata = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)

    features = build_feature_vector(
        user_content_profile=user_content,
        candidate_content_embedding=candidate_content,
        user_cf_profile=user_cf,
        candidate_cf_embedding=candidate_cf,
        user_behavior=user_behavior,
        candidate_metadata=candidate_metadata,
    )

    assert features.shape == (INPUT_DIM,)
    assert features.dtype == np.float32
    assert features[OFFSET_CONTENT_COSINE] == cosine_similarity(user_content, candidate_content)
    assert features[OFFSET_CF_COSINE] == cosine_similarity(user_cf, candidate_cf)


def test_weighted_embedding_mean_uses_rating_weights() -> None:
    """Weighted mean should favor higher-rated history embeddings."""
    low = np.array([0.0, 0.0], dtype=np.float32)
    high = np.array([1.0, 1.0], dtype=np.float32)

    mean = weighted_embedding_mean([low, high], [1.0, 3.0])

    np.testing.assert_allclose(mean, [0.75, 0.75], rtol=1e-5)


def test_elementwise_product_matches_numpy_multiply() -> None:
    """Elementwise product helper should match plain numpy multiplication."""
    left = np.array([1.0, 2.0], dtype=np.float32)
    right = np.array([3.0, 4.0], dtype=np.float32)

    result = elementwise_product(left, right)

    np.testing.assert_array_equal(result, left * right)

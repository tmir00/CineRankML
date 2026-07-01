"""Embedding similarity and aggregation helpers for hybrid features."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def weighted_embedding_mean(embeddings: list[NDArray[np.float32]], weights: list[float]) -> NDArray[np.float32]:
    """
    Aggregate a list of embedding vectors into a single weighted mean embedding vector.

    Do this by:
    1. Returning a zero vector when there is no history.
    2. Otherwise summing weight * embedding and dividing by total weight.

    ============================ Arguments ============================
    embeddings: One embedding vector per rated movie in user history.
    weights: Rating weights aligned with embeddings.

    ============================ Returns ============================
    Weighted mean embedding as float32.
    """
    # Return a zero vector when there is no history.
    if not embeddings:
        dim = 0
        if weights:
            raise ValueError("weights must be empty when embeddings is empty")
        return np.zeros(0, dtype=np.float32)

    # Return a zero vector when there are no weights.
    dim = embeddings[0].shape[0]
    if not weights:
        return np.zeros(dim, dtype=np.float32)

    # Return a zero vector when the total weight is zero or negative.
    total_weight = float(sum(weights))
    if total_weight <= 0:
        return np.zeros(dim, dtype=np.float32)

    # Accumulate the weighted sum of embeddings.
    weighted_sum = np.zeros(dim, dtype=np.float64)
    for embedding, weight in zip(embeddings, weights, strict=True):
        weighted_sum += float(weight) * embedding.astype(np.float64)

    # Return the weighted mean embedding vector.
    return (weighted_sum / total_weight).astype(np.float32)


def cosine_similarity(left: NDArray[np.float32], right: NDArray[np.float32]) -> float:
    """
    Compute cosine similarity between two vectors.

    Returns 0.0 when either vector has zero norm.

    ============================ Arguments ============================
    left: First embedding vector.
    right: Second embedding vector.

    ============================ Returns ============================
    Cosine similarity in [-1, 1], or 0.0 for degenerate inputs.
    """
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    # Return the cosine similarity.
    return float(np.dot(left, right) / (left_norm * right_norm))


def elementwise_product(left: NDArray[np.float32], right: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return the elementwise product of two same-length vectors."""
    return (left * right).astype(np.float32)

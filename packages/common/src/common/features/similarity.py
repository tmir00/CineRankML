"""Embedding similarity and aggregation helpers for hybrid features."""

from __future__ import annotations

import numpy as np

from dataclasses import dataclass, field
from numpy.typing import NDArray


def _profile_from_running_totals(dim: int, weighted_sum: NDArray[np.float64], total_weight: float) -> NDArray[np.float32]:
    """
    Build one weighted-mean embedding profile from running totals.

    Do this by:
    1. Returning a zero vector when there is no positive total weight.
    2. Otherwise dividing the weighted sum by the total weight.


    ============================ Arguments ============================
    dim: Expected embedding dimension.
    weighted_sum: Running sum of weight times embedding.
    total_weight: Running sum of rating weights.

    ============================ Returns ============================
    Weighted mean embedding as float32.
    """
    # If every weight was zero or negative, the profile is a zero vector.
    if total_weight <= 0:
        return np.zeros(dim, dtype=np.float32)

    # Return the weighted mean embedding vector.
    return (weighted_sum / total_weight).astype(np.float32)


@dataclass
class WeightedEmbeddingAccumulator:
    """
    Track a rating-weighted embedding profile incrementally as history grows.

    Do this by:
    1. Keeping a running weighted sum and total weight.
    2. Adding weight * embedding each time a new rated movie is observed.
    3. Building the current profile as weighted_sum / total_weight.

    Batch training uses this while walking each user's rating timeline.
    Online inference can still call weighted_embedding_mean() on a full history list.
    """

    dim: int
    total_weight: float = 0.0
    weighted_sum: NDArray[np.float64] = field(init=False)

    def __post_init__(self) -> None:
        # Start with a zero running sum in float64 for numerical stability.
        self.weighted_sum = np.zeros(self.dim, dtype=np.float64)

    def observe(self, embedding: NDArray[np.float32], weight: float) -> None:
        """
        Add one rated movie embedding to the running profile totals.

        ============================ Arguments ============================
        embedding: The movie embedding vector.
        weight: The rating value used as the weight.
        """
        # Add this movie's contribution to the running weighted sum.
        self.weighted_sum += float(weight) * embedding.astype(np.float64)
        self.total_weight += float(weight)

    def profile(self) -> NDArray[np.float32]:
        """
        Build the current weighted-mean embedding profile.

        ============================ Returns ============================
        Weighted mean embedding as float32.
        """
        return _profile_from_running_totals(self.dim, self.weighted_sum, self.total_weight)


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

    # Feed every rated movie through the same accumulator used during batch training.
    accumulator = WeightedEmbeddingAccumulator(dim=dim)
    for embedding, weight in zip(embeddings, weights, strict=True):
        accumulator.observe(embedding, weight)
    return accumulator.profile()


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

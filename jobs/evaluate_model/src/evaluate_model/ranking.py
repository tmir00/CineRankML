"""Ranking metric helpers for hybrid ranker evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class UserRankingExample:
    """One user's candidate list with predicted scores and relevance labels."""

    ratings: list[float]
    predictions: list[float]


@dataclass
class RankingMetricAccumulator:
    """Macro-averaged ranking metrics across users."""

    precision_at_5: list[float] = field(default_factory=list)
    precision_at_10: list[float] = field(default_factory=list)
    recall_at_5: list[float] = field(default_factory=list)
    recall_at_10: list[float] = field(default_factory=list)
    ndcg_at_5: list[float] = field(default_factory=list)
    ndcg_at_10: list[float] = field(default_factory=list)
    mrr_at_10: list[float] = field(default_factory=list)

    @property
    def num_users_evaluated(self) -> int:
        return len(self.precision_at_5)

    def averages(self) -> dict[str, float]:
        """Return macro-averaged metric values across evaluated users."""
        if not self.precision_at_5:
            return {
                "precision_at_5": 0.0,
                "precision_at_10": 0.0,
                "recall_at_5": 0.0,
                "recall_at_10": 0.0,
                "ndcg_at_5": 0.0,
                "ndcg_at_10": 0.0,
                "mrr_at_10": 0.0,
            }

        def mean(values: list[float]) -> float:
            return sum(values) / len(values)

        return {
            "precision_at_5": mean(self.precision_at_5),
            "precision_at_10": mean(self.precision_at_10),
            "recall_at_5": mean(self.recall_at_5),
            "recall_at_10": mean(self.recall_at_10),
            "ndcg_at_5": mean(self.ndcg_at_5),
            "ndcg_at_10": mean(self.ndcg_at_10),
            "mrr_at_10": mean(self.mrr_at_10),
        }


def _is_relevant(rating: float, *, relevance_threshold: float) -> bool:
    """ 
    Check if a rating is relevant based on the relevance threshold. 

    A rating is relevant if it is greater than or equal to the relevance threshold.
    """
    return rating >= relevance_threshold


def _dcg(relevances: list[float], k: int) -> float:
    """
    Out of the top k movie recommendations, how well are they ordered by relevance?
    """
    total = 0.0
    # Calculate the DCG at position k.
    for index, relevance in enumerate(relevances[:k]):
        total += relevance / math.log2(index + 2)
    return total


def _precision_at_k(relevances: list[float], k: int) -> float:
    """
    Out of the top k movie recommendations, how many are relevant?
    """
    if k == 0:
        return 0.0
    return sum(relevances[:k]) / k


def _recall_at_k(relevances: list[float], k: int, total_relevant: int) -> float:
    """
    Of all the movies the user liked in holdout, how many did we successfully recover in the top K?
    """
    if total_relevant == 0:
        return 0.0
    return sum(relevances[:k]) / total_relevant


def _ndcg_at_k(relevances: list[float], k: int) -> float:
    """
    Did we rank the relevant movies near the top?
    """
    dcg = _dcg(relevances, k)
    ideal = sorted(relevances, reverse=True)
    idcg = _dcg(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def _mrr_at_k(relevances: list[float], k: int) -> float:
    """
    At what rank did we find the first relevant movie?
    """
    for index, relevance in enumerate(relevances[:k]):
        if relevance > 0.0:
            return 1.0 / (index + 1)
    return 0.0


def compute_user_ranking_metrics(example: UserRankingExample, *, relevance_threshold: float = 4.0) -> dict[str, float] | None:
    """
    Compute ranking metrics for one user's candidate list.

    Do this by:
    1. Sorting candidates by predicted score descending.
    2. Marking items with rating >= relevance_threshold as relevant.
    3. Computing Precision@K, Recall@K, NDCG@K, and MRR@10.

    ============================ Arguments ============================
    example: Ratings and predictions for one user's candidates.
    relevance_threshold: Minimum rating treated as relevant (default 4.0).

    ============================ Returns ============================
    Metric dict, or None when the user has fewer than two candidates.
    """
    # Check if the user has fewer than two candidates, return None if so.
    if len(example.ratings) < 2:
        return None

    # Sort the candidates by predicted score descending.
    ranked_pairs = sorted(
        zip(example.predictions, example.ratings),
        key=lambda pair: pair[0],
        reverse=True,
    )
    
    # Mark items with rating >= relevance_threshold as relevant.
    relevances = [
        1.0 if _is_relevant(rating, relevance_threshold=relevance_threshold) else 0.0
        for _, rating in ranked_pairs
    ]
    # Count the total number of relevant items.
    total_relevant = sum(relevances)
    # If there are no relevant items, return None.
    if total_relevant == 0.0:
        return None

    return {
        "precision_at_5": _precision_at_k(relevances, 5),
        "precision_at_10": _precision_at_k(relevances, 10),
        "recall_at_5": _recall_at_k(relevances, 5, int(total_relevant)),
        "recall_at_10": _recall_at_k(relevances, 10, int(total_relevant)),
        "ndcg_at_5": _ndcg_at_k(relevances, 5),
        "ndcg_at_10": _ndcg_at_k(relevances, 10),
        "mrr_at_10": _mrr_at_k(relevances, 10),
    }


def accumulate_user_metrics(accumulator: RankingMetricAccumulator, user_metrics: dict[str, float]) -> None:
    """ Add this user's metrics to the running lists used to compute final average ranking metrics. """
    accumulator.precision_at_5.append(user_metrics["precision_at_5"])
    accumulator.precision_at_10.append(user_metrics["precision_at_10"])
    accumulator.recall_at_5.append(user_metrics["recall_at_5"])
    accumulator.recall_at_10.append(user_metrics["recall_at_10"])
    accumulator.ndcg_at_5.append(user_metrics["ndcg_at_5"])
    accumulator.ndcg_at_10.append(user_metrics["ndcg_at_10"])
    accumulator.mrr_at_10.append(user_metrics["mrr_at_10"])

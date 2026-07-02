"""Unit tests for ranking metric helpers."""

from __future__ import annotations

from evaluate_model.ranking import UserRankingExample, compute_user_ranking_metrics


def test_compute_user_ranking_metrics_perfect_ranking() -> None:
    example = UserRankingExample(
        ratings=[5.0, 1.0, 4.5, 2.0],
        predictions=[0.9, 0.1, 0.8, 0.2],
    )
    metrics = compute_user_ranking_metrics(example, relevance_threshold=4.0)

    assert metrics is not None
    assert metrics["precision_at_5"] == 0.4
    assert metrics["precision_at_10"] == 0.2
    assert metrics["recall_at_5"] == 1.0
    assert metrics["mrr_at_10"] == 1.0


def test_compute_user_ranking_metrics_skips_single_candidate_users() -> None:
    example = UserRankingExample(ratings=[5.0], predictions=[0.9])
    assert compute_user_ranking_metrics(example) is None


def test_compute_user_ranking_metrics_skips_users_without_relevant_items() -> None:
    example = UserRankingExample(
        ratings=[1.0, 2.0, 3.0],
        predictions=[0.9, 0.8, 0.7],
    )
    assert compute_user_ranking_metrics(example, relevance_threshold=4.0) is None

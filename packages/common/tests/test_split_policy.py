"""Tests for adaptive online experiment split policy."""

from __future__ import annotations

from common.recommendation.merge_recommendations import RankedCandidate, merge_ranked_recommendations
from common.recommendation.split_policy import (
    SplitFractions,
    SplitPolicySettings,
    adjust_split_after_rating,
    allocate_model_slots,
)


def _settings() -> SplitPolicySettings:
    return SplitPolicySettings()


def test_allocate_model_slots_70_30_for_top_20() -> None:
    fractions = SplitFractions(main=0.70, candidate=0.30)
    main_slots, candidate_slots = allocate_model_slots(20, fractions, has_candidate=True)
    assert main_slots == 14
    assert candidate_slots == 6


def test_high_rating_increases_candidate_share() -> None:
    current = SplitFractions(main=0.70, candidate=0.30)
    result = adjust_split_after_rating(
        current=current,
        model_role="candidate",
        rating=5.0,
        settings=_settings(),
        candidate_rating_count=1,
        candidate_avg_rating=5.0,
    )
    assert result.changed
    assert result.fractions.candidate > current.candidate
    assert result.fractions.main < current.main


def test_low_rating_decreases_candidate_share() -> None:
    current = SplitFractions(main=0.70, candidate=0.30)
    result = adjust_split_after_rating(
        current=current,
        model_role="candidate",
        rating=2.0,
        settings=_settings(),
        candidate_rating_count=1,
        candidate_avg_rating=2.0,
    )
    assert result.changed
    assert result.fractions.candidate < current.candidate


def test_promotion_when_candidate_hits_80_percent_with_enough_ratings() -> None:
    current = SplitFractions(main=0.20, candidate=0.80)
    result = adjust_split_after_rating(
        current=current,
        model_role="candidate",
        rating=3.5,
        settings=_settings(),
        candidate_rating_count=10,
        candidate_avg_rating=4.2,
    )
    assert result.should_promote


def test_merge_deduplicates_by_movie_id() -> None:
    main_ranked = [
        RankedCandidate(
            movie_id=1,
            predicted_score=4.0,
            model_role="main",
            model_version="main-v1",
            candidate_index=0,
            title="A",
            year=2000,
            genres=["Drama"],
            poster_path=None,
            retrieval_source="knn",
        ),
        RankedCandidate(
            movie_id=2,
            predicted_score=3.0,
            model_role="main",
            model_version="main-v1",
            candidate_index=1,
            title="B",
            year=2001,
            genres=["Comedy"],
            poster_path=None,
            retrieval_source="knn",
        ),
    ]
    candidate_ranked = [
        RankedCandidate(
            movie_id=1,
            predicted_score=4.5,
            model_role="candidate",
            model_version="cand-v1",
            candidate_index=0,
            title="A",
            year=2000,
            genres=["Drama"],
            poster_path=None,
            retrieval_source="knn",
        ),
    ]
    merged = merge_ranked_recommendations(
        main_ranked=main_ranked,
        candidate_ranked=candidate_ranked,
        main_slots=1,
        candidate_slots=1,
    )
    movie_ids = [row.movie_id for row in merged]
    assert movie_ids.count(1) == 1
    assert merged[0].movie_id == 1
    assert merged[0].predicted_score == 4.5

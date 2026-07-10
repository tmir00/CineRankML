"""Tests for online feature builder helpers."""

from __future__ import annotations

from recommender_api.schemas import RatingInput
from recommender_api.services.feature_builder import merge_user_ratings


def test_merge_user_ratings_request_overrides_history() -> None:
    history = {1: 3.0, 2: 4.0}
    new_ratings = [RatingInput(movie_id=2, rating=5.0), RatingInput(movie_id=3, rating=2.5)]
    merged = merge_user_ratings(history, new_ratings)
    assert merged == {1: 3.0, 2: 5.0, 3: 2.5}

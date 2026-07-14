"""Tests for client exclude_movie_ids merge/cap behavior."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from recommender_api.schemas import RecommendRequest
from recommender_api.services.inference import merge_retrieval_exclude_ids


def test_recommend_request_accepts_exclude_movie_ids() -> None:
    body = RecommendRequest(exclude_movie_ids=[1, 2, 3], top_k=5)
    assert body.exclude_movie_ids == [1, 2, 3]


def test_recommend_request_defaults_exclude_movie_ids_empty() -> None:
    body = RecommendRequest()
    assert body.exclude_movie_ids == []


def test_recommend_request_rejects_more_than_sixty_exclude_ids() -> None:
    with pytest.raises(ValidationError):
        RecommendRequest(exclude_movie_ids=list(range(61)))


def test_merge_retrieval_exclude_ids_unions_rated_and_client() -> None:
    result = merge_retrieval_exclude_ids({10, 20}, [20, 30, 40])
    assert result == {10, 20, 30, 40}


def test_merge_retrieval_exclude_ids_dedupes_and_caps_client_ids() -> None:
    client_ids = list(range(100)) + [0, 1]  # duplicates + overflow
    result = merge_retrieval_exclude_ids({999}, client_ids, max_client_ids=60)
    assert 999 in result
    assert len(result) == 61  # 60 client + rated 999
    for movie_id in range(60):
        assert movie_id in result
    assert 60 not in result


def test_merge_retrieval_exclude_ids_handles_none_and_empty() -> None:
    assert merge_retrieval_exclude_ids({1, 2}, None) == {1, 2}
    assert merge_retrieval_exclude_ids({1, 2}, []) == {1, 2}

"""Tests that OpenSearch query builders exclude poster_safe=false movies."""

from __future__ import annotations

import numpy as np

from common.opensearch.search import (
    _build_knn_query_body,
    _parse_candidate_hit,
    _poster_safe_filter,
    build_popular_by_genres_query_body,
    build_random_by_genres_query_body,
    build_title_search_query_body,
)

POSTER_SAFE_TERM = {"term": {"poster_safe": True}}


def _assert_contains_poster_safe_filter(filters: list[dict]) -> None:
    assert POSTER_SAFE_TERM in filters


def test_poster_safe_filter_clause() -> None:
    assert _poster_safe_filter() == POSTER_SAFE_TERM


def test_knn_query_body_always_filters_poster_safe() -> None:
    vector = np.zeros(384, dtype=np.float32)
    body = _build_knn_query_body(vector, k=10, exclude_movie_ids=set())
    bool_query = body["query"]["bool"]
    _assert_contains_poster_safe_filter(bool_query["filter"])
    assert "must_not" not in bool_query


def test_knn_query_body_keeps_excludes_and_poster_safe() -> None:
    vector = np.zeros(384, dtype=np.float32)
    body = _build_knn_query_body(vector, k=10, exclude_movie_ids={1, 2})
    bool_query = body["query"]["bool"]
    _assert_contains_poster_safe_filter(bool_query["filter"])
    assert bool_query["must_not"] == [{"terms": {"movie_id": [1, 2]}}]


def test_popular_by_genres_query_body_filters_poster_safe() -> None:
    body = build_popular_by_genres_query_body(
        ["Action"],
        size=20,
        exclude_movie_ids=set(),
        min_vote_count=100,
        min_vote_average=6.0,
    )
    _assert_contains_poster_safe_filter(body["query"]["bool"]["filter"])


def test_random_by_genres_query_body_filters_poster_safe() -> None:
    body = build_random_by_genres_query_body(
        ["Drama"],
        size=20,
        exclude_movie_ids={5},
        random_seed=42,
        min_vote_count=100,
        min_vote_average=6.0,
    )
    bool_query = body["query"]["function_score"]["query"]["bool"]
    _assert_contains_poster_safe_filter(bool_query["filter"])
    assert bool_query["must_not"] == [{"terms": {"movie_id": [5]}}]


def test_title_search_query_body_filters_poster_safe() -> None:
    body = build_title_search_query_body("Inception", limit=10)
    bool_query = body["query"]["bool"]
    _assert_contains_poster_safe_filter(bool_query["filter"])
    assert bool_query["must"][0]["multi_match"]["query"] == "Inception"


def test_parse_candidate_hit_drops_unsafe_poster() -> None:
    hit = {
        "_source": {
            "movie_id": 42,
            "title": "Unsafe",
            "poster_safe": False,
            "content_embedding": [0.0] * 384,
        }
    }
    assert (
        _parse_candidate_hit(
            hit,
            exclude_movie_ids=set(),
            retrieval_source="knn",
        )
        is None
    )


def test_parse_candidate_hit_keeps_safe_poster() -> None:
    hit = {
        "_source": {
            "movie_id": 42,
            "title": "Safe",
            "poster_safe": True,
            "content_embedding": [0.1] * 384,
        }
    }
    doc = _parse_candidate_hit(
        hit,
        exclude_movie_ids=set(),
        retrieval_source="knn",
    )
    assert doc is not None
    assert doc.movie_id == 42
    assert doc.poster_safe is True

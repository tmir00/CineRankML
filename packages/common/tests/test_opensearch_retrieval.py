"""Tests for multi-bucket OpenSearch retrieval helpers."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

import numpy as np

from common.opensearch.retrieval import RetrievalSettings, merge_candidates, retrieve_candidate_pool
from common.opensearch.search import (
    RETRIEVAL_SOURCE_KNN,
    RETRIEVAL_SOURCE_POPULAR,
    RETRIEVAL_SOURCE_RANDOM_GENRE,
    RETRIEVAL_SOURCE_RANDOM_KNN,
    CandidateMovieDoc,
    build_popular_by_genres_query_body,
    build_random_by_genres_query_body,
    sample_random_knn_candidates,
)

_EMBEDDING = [0.1] * 384


def _doc(movie_id: int, source: str) -> CandidateMovieDoc:
    return CandidateMovieDoc(
        movie_id=movie_id,
        title=f"Movie {movie_id}",
        year=2000,
        genres=["Sci-Fi"],
        runtime=120,
        tmdb_popularity=10.0,
        tmdb_vote_average=7.5,
        tmdb_vote_count=500,
        content_embedding=_EMBEDDING,
        retrieval_source=source,
    )


def test_merge_candidates_prefers_knn_over_popular() -> None:
    knn_bucket = [_doc(1, RETRIEVAL_SOURCE_KNN)]
    popular_bucket = [replace(_doc(1, RETRIEVAL_SOURCE_POPULAR), title="Duplicate")]
    merged = merge_candidates([knn_bucket, popular_bucket], max_candidates=10)
    assert len(merged) == 1
    assert merged[0].retrieval_source == RETRIEVAL_SOURCE_KNN
    assert merged[0].title == "Movie 1"


def test_merge_candidates_respects_max_candidates() -> None:
    buckets = [[_doc(1, RETRIEVAL_SOURCE_KNN), _doc(2, RETRIEVAL_SOURCE_KNN)]]
    merged = merge_candidates(buckets, max_candidates=1)
    assert len(merged) == 1


def test_build_popular_by_genres_query_body_includes_filters_and_sort() -> None:
    body = build_popular_by_genres_query_body(
        ["Sci-Fi", "Action"],
        50,
        {99},
        min_vote_count=100,
        min_vote_average=6.0,
    )
    filters = body["query"]["bool"]["filter"]
    assert {"terms": {"genres": ["Sci-Fi", "Action"]}} in filters
    assert {"range": {"tmdb_vote_count": {"gte": 100}}} in filters
    assert body["sort"] == [{"tmdb_popularity": "desc"}]
    assert body["query"]["bool"]["must_not"] == [{"terms": {"movie_id": [99]}}]


def test_build_random_by_genres_query_body_uses_random_score() -> None:
    body = build_random_by_genres_query_body(
        ["Sci-Fi"],
        75,
        set(),
        12345,
        min_vote_count=100,
        min_vote_average=6.0,
    )
    function_score = body["query"]["function_score"]
    assert function_score["random_score"]["seed"] == 12345
    assert function_score["boost_mode"] == "replace"


def test_sample_random_knn_candidates_tags_source() -> None:
    pool = [_doc(i, RETRIEVAL_SOURCE_KNN) for i in range(200)]
    sampled = sample_random_knn_candidates(
        pool,
        skip_top=150,
        sample_size=5,
        random_seed=42,
    )
    assert len(sampled) == 5
    assert all(doc.retrieval_source == RETRIEVAL_SOURCE_RANDOM_KNN for doc in sampled)


def test_retrieve_candidate_pool_no_genres_uses_extended_knn_only() -> None:
    client = MagicMock()
    hits = [
        {
            "_source": {
                "movie_id": i,
                "title": f"M{i}",
                "genres": ["Sci-Fi"],
                "content_embedding": _EMBEDDING,
            }
        }
        for i in range(1, 201)
    ]
    client.search.return_value = {"hits": {"hits": hits}}

    settings = RetrievalSettings(
        knn_size=10,
        knn_pool_size=200,
        random_knn_skip_top=150,
        random_knn_size=5,
        max_candidates=20,
    )
    vector = np.zeros(384, dtype=np.float32)

    candidates = retrieve_candidate_pool(
        client,
        "movies",
        vector,
        liked_genres=[],
        exclude_movie_ids=set(),
        user_id=7,
        settings=settings,
    )

    client.msearch.assert_not_called()
    client.search.assert_called_once()
    sources = {doc.retrieval_source for doc in candidates}
    assert RETRIEVAL_SOURCE_KNN in sources
    assert RETRIEVAL_SOURCE_RANDOM_KNN in sources
    assert RETRIEVAL_SOURCE_POPULAR not in sources


def test_retrieve_candidate_pool_with_genres_uses_msearch() -> None:
    client = MagicMock()

    def _hit(movie_id: int) -> dict:
        return {
            "_source": {
                "movie_id": movie_id,
                "title": f"M{movie_id}",
                "genres": ["Sci-Fi"],
                "content_embedding": _EMBEDDING,
            }
        }

    client.msearch.return_value = {
        "responses": [
            {"hits": {"hits": [_hit(1), _hit(2)]}},
            {"hits": {"hits": [_hit(3)]}},
            {"hits": {"hits": [_hit(4)]}},
        ]
    }

    settings = RetrievalSettings(knn_size=2, popular_size=1, random_genre_size=1, max_candidates=10)
    vector = np.zeros(384, dtype=np.float32)

    candidates = retrieve_candidate_pool(
        client,
        "movies",
        vector,
        liked_genres=["Sci-Fi"],
        exclude_movie_ids=set(),
        user_id=3,
        settings=settings,
    )

    client.msearch.assert_called_once()
    client.search.assert_not_called()
    assert len(candidates) == 4
    assert {doc.movie_id for doc in candidates} == {1, 2, 3, 4}

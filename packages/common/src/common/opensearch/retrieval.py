"""Multi-bucket OpenSearch candidate retrieval for online recommendations."""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
from numpy.typing import NDArray
from opensearchpy import OpenSearch

from common.opensearch.search import (
    RETRIEVAL_SOURCE_KNN,
    RETRIEVAL_SOURCE_POPULAR,
    RETRIEVAL_SOURCE_RANDOM_GENRE,
    RETRIEVAL_SOURCE_RANDOM_KNN,
    CandidateMovieDoc,
    _build_knn_query_body,
    _parse_hits,
    build_popular_by_genres_query_body,
    build_random_by_genres_query_body,
    sample_random_knn_candidates,
    search_similar_movies_extended,
)

# Lower number = higher merge priority when the same movie appears in multiple buckets.
_MERGE_PRIORITY: dict[str, int] = {
    RETRIEVAL_SOURCE_KNN: 0,
    RETRIEVAL_SOURCE_POPULAR: 1,
    RETRIEVAL_SOURCE_RANDOM_GENRE: 2,
    RETRIEVAL_SOURCE_RANDOM_KNN: 3,
}


@dataclass(frozen=True)
class RetrievalSettings:
    """Configurable sizes and quality thresholds for multi-bucket retrieval."""

    knn_size: int = 250
    popular_size: int = 50
    random_genre_size: int = 75
    random_knn_size: int = 75
    knn_pool_size: int = 400
    random_knn_skip_top: int = 150
    max_candidates: int = 300
    liked_genre_count: int = 4
    min_vote_count: int = 100
    min_vote_average: float = 6.0


def stable_retrieval_seed(user_id: int, when: datetime | None = None) -> int:
    """
    Build a stable random seed for one user on one UTC day.

    Do this by:
    1. Formatting user_id and the UTC calendar date into a short string.
    2. Hashing that string with adler32 so OpenSearch random_score stays stable for the day.

    ============================ Arguments ============================
    user_id: Authenticated app user id.
    when: Optional timestamp; defaults to now in UTC.

    ============================ Returns ============================
    Positive integer seed for random_score and numpy sampling.
    """
    moment = when or datetime.now(tz=UTC)
    day = moment.astimezone(UTC).date().isoformat()
    raw = f"{user_id}:{day}".encode()
    return int(zlib.adler32(raw) & 0x7FFFFFFF)


def merge_candidates(
    buckets: list[list[CandidateMovieDoc]],
    *,
    max_candidates: int,
) -> list[CandidateMovieDoc]:
    """
    Merge multiple retrieval buckets into one deduplicated candidate list.

    Do this by:
    1. Walking buckets in order so earlier buckets win on duplicate movie_id.
    2. Within the same movie_id, keeping the copy with the better (lower) merge priority.
    3. Truncating to max_candidates when the merged list grows too large.

    ============================ Arguments ============================
    buckets: Ordered lists of candidates, highest-priority bucket first.
    max_candidates: Maximum number of unique movies to return.

    ============================ Returns ============================
    Deduplicated candidates preserving bucket priority.
    """
    merged: dict[int, CandidateMovieDoc] = {}
    order: list[int] = []

    for bucket in buckets:
        for doc in bucket:
            existing = merged.get(doc.movie_id)
            if existing is None:
                merged[doc.movie_id] = doc
                order.append(doc.movie_id)
                continue

            # Keep the copy from the higher-priority bucket when duplicates appear.
            existing_rank = _MERGE_PRIORITY.get(existing.retrieval_source, 99)
            new_rank = _MERGE_PRIORITY.get(doc.retrieval_source, 99)
            if new_rank < existing_rank:
                merged[doc.movie_id] = doc

    results = [merged[movie_id] for movie_id in order]
    if max_candidates > 0:
        return results[:max_candidates]
    return results


def _msearch_retrieval_buckets(
    client: OpenSearch,
    index_alias: str,
    query_vector: NDArray[np.float32],
    liked_genres: list[str],
    exclude_movie_ids: set[int],
    random_seed: int,
    settings: RetrievalSettings,
) -> tuple[list[CandidateMovieDoc], list[CandidateMovieDoc], list[CandidateMovieDoc]]:
    """
    Run kNN, popular, and random genre queries in one OpenSearch msearch round-trip.

    ============================ Returns ============================
    Tuple of (knn_bucket, popular_bucket, random_genre_bucket).
    """
    knn_body = _build_knn_query_body(query_vector, settings.knn_size, exclude_movie_ids)
    popular_body = build_popular_by_genres_query_body(
        liked_genres,
        settings.popular_size,
        exclude_movie_ids,
        min_vote_count=settings.min_vote_count,
        min_vote_average=settings.min_vote_average,
    )
    random_body = build_random_by_genres_query_body(
        liked_genres,
        settings.random_genre_size,
        exclude_movie_ids,
        random_seed,
        min_vote_count=settings.min_vote_count,
        min_vote_average=settings.min_vote_average,
    )

    header = {"index": index_alias}
    response = client.msearch(
        body=[
            header,
            knn_body,
            header,
            popular_body,
            header,
            random_body,
        ]
    )

    responses = response.get("responses", [])
    knn_hits = responses[0].get("hits", {}).get("hits", []) if len(responses) > 0 else []
    popular_hits = responses[1].get("hits", {}).get("hits", []) if len(responses) > 1 else []
    random_hits = responses[2].get("hits", {}).get("hits", []) if len(responses) > 2 else []

    knn_bucket = _parse_hits(
        knn_hits,
        exclude_movie_ids=exclude_movie_ids,
        retrieval_source=RETRIEVAL_SOURCE_KNN,
        limit=settings.knn_size,
    )
    popular_bucket = _parse_hits(
        popular_hits,
        exclude_movie_ids=exclude_movie_ids,
        retrieval_source=RETRIEVAL_SOURCE_POPULAR,
        limit=settings.popular_size,
    )
    random_bucket = _parse_hits(
        random_hits,
        exclude_movie_ids=exclude_movie_ids,
        retrieval_source=RETRIEVAL_SOURCE_RANDOM_GENRE,
        limit=settings.random_genre_size,
    )
    return knn_bucket, popular_bucket, random_bucket


def retrieve_candidate_pool(
    client: OpenSearch,
    index_alias: str,
    query_vector: NDArray[np.float32],
    liked_genres: list[str],
    exclude_movie_ids: set[int],
    user_id: int,
    settings: RetrievalSettings,
) -> list[CandidateMovieDoc]:
    """
    Retrieve and merge candidates from all active OpenSearch buckets.

    Do this by:
    1. When liked genres exist, running kNN, popular, and random genre queries in parallel.
    2. When liked genres are empty, fetching one extended kNN pool and sampling random_knn from the tail.
    3. Merging buckets with priority knn -> popular -> random_genre/random_knn and capping the list.

    ============================ Arguments ============================
    client: The OpenSearch client.
    index_alias: Stable movies alias such as movies.
    query_vector: User content profile vector with shape (384,).
    liked_genres: Top genres derived from the user's ratings (may be empty).
    exclude_movie_ids: Rated movie ids to filter out of every bucket.
    user_id: Authenticated user id used for stable daily random seeds.
    settings: Bucket sizes and quality thresholds.

    ============================ Returns ============================
    Deduplicated candidate movies ready for feature matrix construction.
    """
    random_seed = stable_retrieval_seed(user_id)

    if liked_genres:
        knn_bucket, popular_bucket, random_bucket = _msearch_retrieval_buckets(
            client,
            index_alias,
            query_vector,
            liked_genres,
            exclude_movie_ids,
            random_seed,
            settings,
        )
        return merge_candidates(
            [knn_bucket, popular_bucket, random_bucket],
            max_candidates=settings.max_candidates,
        )

    # No liked genres: one extended kNN query, then sample exploration from the tail.
    extended_pool = search_similar_movies_extended(
        client,
        index_alias,
        query_vector,
        settings.knn_pool_size,
        exclude_movie_ids,
    )
    knn_bucket = extended_pool[: settings.knn_size]
    random_knn_bucket = sample_random_knn_candidates(
        extended_pool,
        skip_top=settings.random_knn_skip_top,
        sample_size=settings.random_knn_size,
        random_seed=random_seed,
    )
    return merge_candidates(
        [knn_bucket, random_knn_bucket],
        max_candidates=settings.max_candidates,
    )

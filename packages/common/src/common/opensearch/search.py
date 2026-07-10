"""OpenSearch query helpers for online candidate retrieval."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from opensearchpy import OpenSearch

RETRIEVAL_SOURCE_KNN = "knn"
RETRIEVAL_SOURCE_POPULAR = "popular"
RETRIEVAL_SOURCE_RANDOM_GENRE = "random_genre"
RETRIEVAL_SOURCE_RANDOM_KNN = "random_knn"

CONTENT_EMBEDDING_DIM = 384


@dataclass(frozen=True)
class CandidateMovieDoc:
    """One candidate movie returned from OpenSearch."""

    movie_id: int
    title: str
    year: int | None
    genres: list[str]
    runtime: int | None
    tmdb_popularity: float | None
    tmdb_vote_average: float | None
    tmdb_vote_count: int | None
    content_embedding: list[float]
    poster_path: str | None = None
    poster_safe: bool = True
    show_poster: bool = True
    certification_us: str | None = None
    retrieval_source: str = RETRIEVAL_SOURCE_KNN


def _parse_candidate_hit(
    hit: dict,
    *,
    exclude_movie_ids: set[int],
    retrieval_source: str,
    require_embedding: bool = True,
) -> CandidateMovieDoc | None:
    """
    Turn one OpenSearch hit into a CandidateMovieDoc when it has required fields.

    Do this by:
    1. Reading movie_id and content_embedding from _source.
    2. Skipping hits the user already rated or that lack required fields.
    3. Building a CandidateMovieDoc with the requested retrieval_source tag.

    ============================ Arguments ============================
    hit: One search hit dict from OpenSearch.
    exclude_movie_ids: Rated movie ids to drop from the candidate pool.
    retrieval_source: Bucket label such as knn, popular, or random_genre.
    require_embedding: When True, skip hits with no content_embedding.

    ============================ Returns ============================
    Parsed candidate document, or None when the hit should be skipped.
    """
    source = hit.get("_source", {})
    movie_id = source.get("movie_id")
    if movie_id is None:
        return None

    movie_id_int = int(movie_id)
    if movie_id_int in exclude_movie_ids:
        return None

    embedding = source.get("content_embedding")
    if require_embedding and embedding is None:
        return None

    if embedding is None:
        embedding = [0.0] * CONTENT_EMBEDDING_DIM

    return CandidateMovieDoc(
        movie_id=movie_id_int,
        title=str(source.get("title") or ""),
        year=source.get("year"),
        genres=list(source.get("genres") or []),
        runtime=source.get("runtime"),
        tmdb_popularity=source.get("tmdb_popularity"),
        tmdb_vote_average=source.get("tmdb_vote_average"),
        tmdb_vote_count=source.get("tmdb_vote_count"),
        content_embedding=list(embedding),
        poster_path=source.get("poster_path"),
        poster_safe=bool(source.get("poster_safe", True)),
        show_poster=bool(source.get("show_poster", True)),
        certification_us=source.get("certification_us"),
        retrieval_source=retrieval_source,
    )


def _build_knn_query_body(
    query_vector: NDArray[np.float32],
    k: int,
    exclude_movie_ids: set[int],
) -> dict:
    """
    Build an OpenSearch query body for kNN retrieval on content_embedding.

    ============================ Arguments ============================
    query_vector: User content profile vector with shape (384,).
    k: Maximum number of neighbors OpenSearch should consider.
    exclude_movie_ids: Rated movie ids to filter out of results.

    ============================ Returns ============================
    Query body dict ready for client.search or client.msearch.
    """
    knn_clause: dict = {
        "knn": {
            "content_embedding": {
                "vector": query_vector.astype(float).tolist(),
                "k": max(k, len(exclude_movie_ids) + k),
            }
        }
    }

    if exclude_movie_ids:
        return {
            "size": k,
            "query": {
                "bool": {
                    "must": [knn_clause],
                    "must_not": [{"terms": {"movie_id": sorted(exclude_movie_ids)}}],
                }
            },
        }

    return {"size": k, "query": knn_clause}


def _build_genre_quality_filters(
    genres: list[str],
    min_vote_count: int,
    min_vote_average: float,
) -> list[dict]:
    """Build shared genre and quality filter clauses for bucket B and C queries."""
    return [
        {"terms": {"genres": genres}},
        {"range": {"tmdb_vote_count": {"gte": min_vote_count}}},
        {"range": {"tmdb_vote_average": {"gte": min_vote_average}}},
    ]


def build_popular_by_genres_query_body(
    genres: list[str],
    size: int,
    exclude_movie_ids: set[int],
    *,
    min_vote_count: int,
    min_vote_average: float,
) -> dict:
    """
    Build a popular-movies query filtered to liked genres.

    Do this by:
    1. Filtering to movies that match at least one liked genre.
    2. Applying minimum vote count and average quality floors.
    3. Sorting by tmdb_popularity descending.

    ============================ Arguments ============================
    genres: Liked genre names from the user's rating history.
    size: How many movies to return.
    exclude_movie_ids: Rated movie ids to filter out.
    min_vote_count: Minimum tmdb_vote_count for quality filtering.
    min_vote_average: Minimum tmdb_vote_average for quality filtering.

    ============================ Returns ============================
    Query body dict for OpenSearch search or msearch.
    """
    bool_query: dict = {
        "filter": _build_genre_quality_filters(genres, min_vote_count, min_vote_average),
    }
    if exclude_movie_ids:
        bool_query["must_not"] = [{"terms": {"movie_id": sorted(exclude_movie_ids)}}]

    return {
        "size": size,
        "query": {"bool": bool_query},
        "sort": [{"tmdb_popularity": "desc"}],
    }


def build_random_by_genres_query_body(
    genres: list[str],
    size: int,
    exclude_movie_ids: set[int],
    random_seed: int,
    *,
    min_vote_count: int,
    min_vote_average: float,
) -> dict:
    """
    Build a random exploration query filtered to liked genres.

    Do this by:
    1. Filtering to movies that match at least one liked genre with quality floors.
    2. Replacing the relevance score with random_score so results shuffle within the filter.

    ============================ Arguments ============================
    genres: Liked genre names from the user's rating history.
    size: How many movies to return.
    exclude_movie_ids: Rated movie ids to filter out.
    random_seed: Stable seed for random_score (e.g. per user per day).
    min_vote_count: Minimum tmdb_vote_count for quality filtering.
    min_vote_average: Minimum tmdb_vote_average for quality filtering.

    ============================ Returns ============================
    Query body dict for OpenSearch search or msearch.
    """
    bool_query: dict = {
        "filter": _build_genre_quality_filters(genres, min_vote_count, min_vote_average),
    }
    if exclude_movie_ids:
        bool_query["must_not"] = [{"terms": {"movie_id": sorted(exclude_movie_ids)}}]

    return {
        "size": size,
        "query": {
            "function_score": {
                "query": {"bool": bool_query},
                "random_score": {
                    "seed": random_seed,
                    "field": "movie_id",
                },
                "boost_mode": "replace",
            }
        },
    }


def _parse_hits(
    hits: list[dict],
    *,
    exclude_movie_ids: set[int],
    retrieval_source: str,
    limit: int,
    require_embedding: bool = True,
) -> list[CandidateMovieDoc]:
    """Parse up to limit candidate docs from a list of OpenSearch hits."""
    candidates: list[CandidateMovieDoc] = []
    for hit in hits:
        doc = _parse_candidate_hit(
            hit,
            exclude_movie_ids=exclude_movie_ids,
            retrieval_source=retrieval_source,
            require_embedding=require_embedding,
        )
        if doc is None:
            continue
        candidates.append(doc)
        if len(candidates) >= limit:
            break
    return candidates


def search_similar_movies(
    client: OpenSearch,
    index_alias: str,
    query_vector: NDArray[np.float32],
    k: int,
    exclude_movie_ids: set[int],
    *,
    retrieval_source: str = RETRIEVAL_SOURCE_KNN,
) -> list[CandidateMovieDoc]:
    """
    Retrieve candidate movies with kNN search on content embeddings.

    Do this by:
    1. Building a kNN query using the user's content profile vector.
    2. Excluding movies the user has already rated.
    3. Parsing movie metadata and embeddings from each hit.

    ============================ Arguments ============================
    client: The OpenSearch client.
    index_alias: Stable movies alias such as movies.
    query_vector: User content profile vector with shape (384,).
    k: Maximum number of candidates to return.
    exclude_movie_ids: Rated movie ids to filter out of results.
    retrieval_source: Bucket label stored on each returned candidate.

    ============================ Returns ============================
    Candidate movie documents ordered by vector similarity.
    """
    if query_vector.shape != (CONTENT_EMBEDDING_DIM,):
        raise ValueError(f"query_vector must have shape ({CONTENT_EMBEDDING_DIM},)")

    query_body = _build_knn_query_body(query_vector, k, exclude_movie_ids)
    response = client.search(index=index_alias, body=query_body)
    hits = response.get("hits", {}).get("hits", [])
    return _parse_hits(
        hits,
        exclude_movie_ids=exclude_movie_ids,
        retrieval_source=retrieval_source,
        limit=k,
    )


def search_similar_movies_extended(
    client: OpenSearch,
    index_alias: str,
    query_vector: NDArray[np.float32],
    k: int,
    exclude_movie_ids: set[int],
) -> list[CandidateMovieDoc]:
    """
    Retrieve a large kNN pool for the no-genre fallback path.

    Do this by:
    1. Running the same kNN query as search_similar_movies with a larger k.
    2. Returning the full ordered hit list for client-side bucket splitting.

    ============================ Arguments ============================
    client: The OpenSearch client.
    index_alias: Stable movies alias such as movies.
    query_vector: User content profile vector with shape (384,).
    k: Total neighbors to fetch from OpenSearch.
    exclude_movie_ids: Rated movie ids to filter out of results.

    ============================ Returns ============================
    Candidate movie documents ordered by vector similarity.
    """
    return search_similar_movies(
        client,
        index_alias,
        query_vector,
        k,
        exclude_movie_ids,
        retrieval_source=RETRIEVAL_SOURCE_KNN,
    )


def search_popular_by_genres(
    client: OpenSearch,
    index_alias: str,
    genres: list[str],
    size: int,
    exclude_movie_ids: set[int],
    *,
    min_vote_count: int,
    min_vote_average: float,
) -> list[CandidateMovieDoc]:
    """
    Retrieve popular high-quality movies within the user's liked genres.

    ============================ Arguments ============================
    client: The OpenSearch client.
    index_alias: Stable movies alias such as movies.
    genres: Liked genre names from rating history.
    size: How many candidates to return.
    exclude_movie_ids: Rated movie ids to filter out.
    min_vote_count: Minimum tmdb_vote_count for quality filtering.
    min_vote_average: Minimum tmdb_vote_average for quality filtering.

    ============================ Returns ============================
    Candidate movies sorted by tmdb_popularity descending.
    """
    if not genres:
        return []

    body = build_popular_by_genres_query_body(
        genres,
        size,
        exclude_movie_ids,
        min_vote_count=min_vote_count,
        min_vote_average=min_vote_average,
    )
    response = client.search(index=index_alias, body=body)
    hits = response.get("hits", {}).get("hits", [])
    return _parse_hits(
        hits,
        exclude_movie_ids=exclude_movie_ids,
        retrieval_source=RETRIEVAL_SOURCE_POPULAR,
        limit=size,
    )


def search_random_by_genres(
    client: OpenSearch,
    index_alias: str,
    genres: list[str],
    size: int,
    exclude_movie_ids: set[int],
    random_seed: int,
    *,
    min_vote_count: int,
    min_vote_average: float,
) -> list[CandidateMovieDoc]:
    """
    Retrieve random exploration candidates within the user's liked genres.

    ============================ Arguments ============================
    client: The OpenSearch client.
    index_alias: Stable movies alias such as movies.
    genres: Liked genre names from rating history.
    size: How many candidates to return.
    exclude_movie_ids: Rated movie ids to filter out.
    random_seed: Stable seed for random_score.
    min_vote_count: Minimum tmdb_vote_count for quality filtering.
    min_vote_average: Minimum tmdb_vote_average for quality filtering.

    ============================ Returns ============================
    Candidate movies in random order within the genre and quality filters.
    """
    if not genres:
        return []

    body = build_random_by_genres_query_body(
        genres,
        size,
        exclude_movie_ids,
        random_seed,
        min_vote_count=min_vote_count,
        min_vote_average=min_vote_average,
    )
    response = client.search(index=index_alias, body=body)
    hits = response.get("hits", {}).get("hits", [])
    return _parse_hits(
        hits,
        exclude_movie_ids=exclude_movie_ids,
        retrieval_source=RETRIEVAL_SOURCE_RANDOM_GENRE,
        limit=size,
    )


def sample_random_knn_candidates(
    pool: list[CandidateMovieDoc],
    *,
    skip_top: int,
    sample_size: int,
    random_seed: int,
) -> list[CandidateMovieDoc]:
    """
    Sample exploration candidates from the tail of an extended kNN pool.

    Do this by:
    1. Skipping the top skip_top similarity hits so we do not duplicate bucket A head.
    2. Uniformly sampling sample_size movies without replacement from the remainder.
    3. Tagging each returned doc with retrieval_source random_knn.

    ============================ Arguments ============================
    pool: Full ordered kNN candidate list from search_similar_movies_extended.
    skip_top: How many head results to skip before sampling.
    sample_size: How many random_knn candidates to return.
    random_seed: Stable seed for reproducible sampling within a day.

    ============================ Returns ============================
    Random subset of the kNN tail, tagged as random_knn.
    """
    if sample_size <= 0 or not pool:
        return []

    tail = pool[skip_top:]
    if not tail:
        return []

    rng = np.random.default_rng(random_seed)
    count = min(sample_size, len(tail))
    indices = rng.choice(len(tail), size=count, replace=False)

    sampled: list[CandidateMovieDoc] = []
    for index in indices:
        doc = tail[int(index)]
        sampled.append(replace(doc, retrieval_source=RETRIEVAL_SOURCE_RANDOM_KNN))
    return sampled


def search_movies_by_title(client: OpenSearch, index_alias: str, query: str, \
                            limit: int = 20) -> list[CandidateMovieDoc]:
    """
    Search catalog movies by title for onboarding movie pickers.

    Do this by:
    1. Running a multi_match query on title fields.
    2. Returning basic metadata without requiring a content embedding.

    ============================ Arguments ============================
    client: The OpenSearch client.
    index_alias: Stable movies alias such as movies.
    query: Free-text title search string.
    limit: Maximum number of movies to return.

    ============================ Returns ============================
    Matching movie documents ordered by relevance.
    """
    # Trim the query and check if it is empty.
    trimmed = query.strip()
    if not trimmed:
        return []

    # Build the query body.
    body = {
        "size": limit,
        "query": {
            "multi_match": {
                "query": trimmed,
                "fields": ["title", "title.keyword"],
                "type": "best_fields",
            }
        },
    }

    # Execute the query and get the hits.
    response = client.search(index=index_alias, body=body)
    hits = response.get("hits", {}).get("hits", [])

    return _parse_hits(
        hits,
        exclude_movie_ids=set(),
        retrieval_source=RETRIEVAL_SOURCE_KNN,
        limit=limit,
        require_embedding=False,
    )

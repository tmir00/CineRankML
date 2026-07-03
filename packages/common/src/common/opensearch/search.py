"""OpenSearch query helpers for online candidate retrieval."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from opensearchpy import OpenSearch


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


def search_similar_movies(client: OpenSearch, index_alias: str, query_vector: NDArray[np.float32], k: int, \
                            exclude_movie_ids: set[int]) -> list[CandidateMovieDoc]:
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

    ============================ Returns ============================
    Candidate movie documents ordered by vector similarity.
    """
    # Validate the query vector shape.
    if query_vector.shape != (384,):
        raise ValueError("query_vector must have shape (384,)")

    # Build the kNN query clause.
    knn_clause: dict = {
        "knn": {
            "content_embedding": {
                "vector": query_vector.astype(float).tolist(),
                "k": max(k, len(exclude_movie_ids) + k),
            }
        }
    }

    # This is the query body for if we have movies to exclude (user already rated these movies)
    if exclude_movie_ids:
        query_body = {
            "size": k,
            "query": {
                "bool": {
                    "must": [knn_clause],
                    "must_not": [{"terms": {"movie_id": sorted(exclude_movie_ids)}}],
                }
            },
        }
    # This is the query body for if we don't have movies to exclude.
    else:
        query_body = {"size": k, "query": knn_clause}

    # Execute the query and get the hits.
    response = client.search(index=index_alias, body=query_body)
    hits = response.get("hits", {}).get("hits", [])

    # Parse the hits and create a list of candidate movie documents.
    candidates: list[CandidateMovieDoc] = []

    # Iterate over the hits and create a list of candidate movie documents.
    for hit in hits:
        source = hit.get("_source", {})
        movie_id = source.get("movie_id")
        embedding = source.get("content_embedding")

        # If the movie id or embedding is None, skip the hit.
        if movie_id is None or embedding is None:
            continue

        # If the movie id is in the exclude movie ids, skip the hit.
        if int(movie_id) in exclude_movie_ids:
            continue

        # Create a candidate movie document and add it to the list.
        candidates.append(
            CandidateMovieDoc(
                movie_id=int(movie_id),
                title=str(source.get("title") or ""),
                year=source.get("year"),
                genres=list(source.get("genres") or []),
                runtime=source.get("runtime"),
                tmdb_popularity=source.get("tmdb_popularity"),
                tmdb_vote_average=source.get("tmdb_vote_average"),
                tmdb_vote_count=source.get("tmdb_vote_count"),
                content_embedding=list(embedding),
            )
        )

        # If we have reached the maximum number of candidates, break.
        if len(candidates) >= k:
            break

    return candidates


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

    # Parse the hits and create a list of candidate movie documents.
    results: list[CandidateMovieDoc] = []

    # Iterate over the hits and create a list of candidate movie documents.
    for hit in hits:
        source = hit.get("_source", {})
        movie_id = source.get("movie_id")

        # If the movie id is None, skip the hit.
        if movie_id is None:
            continue
        
        embedding = source.get("content_embedding") or [0.0] * 384

        # Create a candidate movie document and add it to the list.
        results.append(
            CandidateMovieDoc(
                movie_id=int(movie_id),
                title=str(source.get("title") or ""),
                year=source.get("year"),
                genres=list(source.get("genres") or []),
                runtime=source.get("runtime"),
                tmdb_popularity=source.get("tmdb_popularity"),
                tmdb_vote_average=source.get("tmdb_vote_average"),
                tmdb_vote_count=source.get("tmdb_vote_count"),
                content_embedding=list(embedding),
            )
        )
    return results

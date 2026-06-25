"""Build OpenSearch movie documents from catalog rows."""

from __future__ import annotations

from typing import Any, Protocol


class MovieLike(Protocol):
    """Minimal movie fields needed to build an OpenSearch document."""

    movie_id: int
    title: str
    year: int | None
    genres: list[str] | None
    overview: str | None
    tagline: str | None
    original_language: str | None
    tmdb_keywords: list[str] | None
    runtime: int | None
    tmdb_popularity: float | None
    tmdb_vote_average: float | None
    tmdb_vote_count: int | None
    tmdb_id: int | None
    imdb_id: str | None


def build_movie_document(movie: MovieLike, tags: list[str], embedding: list[float], \
                            embedding_version: str) -> dict[str, Any]:
    """
    Build one OpenSearch document for a catalog movie.

    Do this by:
    1. Copying searchable metadata from the catalog row.
    2. Attaching top user tags and the content embedding vector.
    3. Returning a dict ready for bulk indexing.

    ============================ Arguments ============================
    movie: Catalog movie metadata.
    tags: Top tags for this movie ordered by popularity.
    embedding: Content embedding vector for vector search.
    embedding_version: Version label stored with the document.

    ============================ Returns ============================
    A document body for OpenSearch bulk indexing.
    """
    return {
        "movie_id": movie.movie_id,
        "title": movie.title,
        "year": movie.year,
        "genres": movie.genres or [],
        "tags": tags,
        "overview": movie.overview,
        "tagline": movie.tagline,
        "original_language": movie.original_language,
        "tmdb_keywords": movie.tmdb_keywords or [],
        "runtime": movie.runtime,
        "tmdb_popularity": movie.tmdb_popularity,
        "tmdb_vote_average": movie.tmdb_vote_average,
        "tmdb_vote_count": movie.tmdb_vote_count,
        "tmdb_id": movie.tmdb_id,
        "imdb_id": movie.imdb_id,
        "embedding_version": embedding_version,
        "content_embedding": embedding,
    }

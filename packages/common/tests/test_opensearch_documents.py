"""Unit tests for OpenSearch document builders."""

from __future__ import annotations

from dataclasses import dataclass

from common.opensearch.documents import build_movie_document


@dataclass
class _Movie:
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


def test_build_movie_document_includes_embedding_vector() -> None:
    movie = _Movie(
        movie_id=1,
        title="Toy Story",
        year=1995,
        genres=["Animation"],
        overview="Overview",
        tagline=None,
        original_language="en",
        tmdb_keywords=["toys"],
        runtime=81,
        tmdb_popularity=10.5,
        tmdb_vote_average=7.7,
        tmdb_vote_count=1000,
        tmdb_id=862,
        imdb_id="tt0114709",
    )
    embedding = [0.1, 0.2, 0.3]
    doc = build_movie_document(movie, ["funny"], embedding, "content-v1")
    assert doc["movie_id"] == 1
    assert doc["title"] == "Toy Story"
    assert doc["tags"] == ["funny"]
    assert doc["content_embedding"] == embedding
    assert doc["embedding_version"] == "content-v1"

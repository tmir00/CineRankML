"""Unit tests for embedding text helpers."""

from __future__ import annotations

from dataclasses import dataclass

from common.embeddings.text import build_embedding_text, embedding_text_hash


@dataclass
class _Movie:
    title: str
    overview: str | None
    tagline: str | None
    genres: list[str] | None
    tmdb_keywords: list[str] | None


def test_build_embedding_text_includes_title_genres_and_tags() -> None:
    movie = _Movie(
        title="Toy Story",
        overview="A toy adventure.",
        tagline="Buzz and Woody",
        genres=["Animation", "Children"],
        tmdb_keywords=["toys"],
    )
    text = build_embedding_text(movie, ["funny", "pixar"])
    assert "Toy Story" in text
    assert "A toy adventure." in text
    assert "Animation" in text
    assert "funny" in text


def test_embedding_text_hash_is_stable() -> None:
    first = embedding_text_hash("same text")
    second = embedding_text_hash("same text")
    third = embedding_text_hash("different text")
    assert first == second
    assert first != third

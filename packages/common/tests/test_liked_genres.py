"""Tests for liked-genre derivation from rating history."""

from __future__ import annotations

from common.recommendation.liked_genres import derive_liked_genres


def test_derive_liked_genres_weights_high_ratings_more() -> None:
    merged_ratings = {1: 5.0, 2: 2.0}
    movie_genres = {
        1: ["Sci-Fi", "Action"],
        2: ["Drama"],
    }
    genres = derive_liked_genres(merged_ratings, movie_genres, top_n=3)
    assert genres[0] == "Sci-Fi"
    assert genres[1] == "Action"
    assert "Drama" not in genres


def test_derive_liked_genres_returns_empty_when_no_genres() -> None:
    merged_ratings = {1: 5.0}
    movie_genres: dict[int, list[str]] = {}
    assert derive_liked_genres(merged_ratings, movie_genres) == []


def test_derive_liked_genres_aggregates_across_movies() -> None:
    merged_ratings = {1: 5.0, 2: 4.5}
    movie_genres = {
        1: ["Sci-Fi", "Action"],
        2: ["Sci-Fi", "Thriller"],
    }
    genres = derive_liked_genres(merged_ratings, movie_genres, top_n=2)
    assert genres == ["Sci-Fi", "Action"]

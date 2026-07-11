"""Tests for questionable movie detection."""

from __future__ import annotations

from common.poster_safety.questionable import is_questionable_movie


def test_is_questionable_movie_flags_adult_movies() -> None:
    assert is_questionable_movie(adult=True, certification_us=None) is True


def test_is_questionable_movie_flags_r_and_nc17() -> None:
    assert is_questionable_movie(adult=False, certification_us="R") is True
    assert is_questionable_movie(adult=False, certification_us="NC-17") is True


def test_is_questionable_movie_ignores_lower_certifications() -> None:
    assert is_questionable_movie(adult=False, certification_us="PG-13") is False

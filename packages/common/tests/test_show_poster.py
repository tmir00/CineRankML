"""Tests for computed poster display rules."""

from __future__ import annotations

from common.poster_safety.show_poster import compute_show_poster


def test_compute_show_poster_hides_adult_movies() -> None:
    assert (
        compute_show_poster(
            poster_path="/poster.jpg",
            poster_safe=True,
            poster_checked=True,
            adult=True,
            certification_us="R",
        )
        is False
    )


def test_compute_show_poster_hides_unchecked_r_rated_movies() -> None:
    assert (
        compute_show_poster(
            poster_path="/poster.jpg",
            poster_safe=True,
            poster_checked=False,
            adult=False,
            certification_us="R",
        )
        is False
    )


def test_compute_show_poster_shows_checked_safe_posters() -> None:
    assert (
        compute_show_poster(
            poster_path="/poster.jpg",
            poster_safe=True,
            poster_checked=True,
            adult=False,
            certification_us="R",
        )
        is True
    )

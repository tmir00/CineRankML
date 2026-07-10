"""Compute whether the frontend should display a movie poster."""

from __future__ import annotations

from common.tmdb.certifications import QUESTIONABLE_US_CERTIFICATIONS


def compute_show_poster(
    *,
    poster_path: str | None,
    poster_safe: bool,
    poster_checked: bool,
    adult: bool,
    certification_us: str | None,
) -> bool:
    """
    Decide if the UI should render the TMDB poster image.

    This reads precomputed catalog fields only — no request-time moderation.

    Do this by:
    1. Hiding posters for adult movies and movies without a poster path.
    2. Hiding unchecked R/NC-17 posters until the offline script runs.
    3. Honoring poster_safe for all other cases.

    ============================ Arguments ============================
    poster_path: TMDB poster path from catalog_movies.
    poster_safe: Offline safety result (default true until checked).
    poster_checked: Whether the offline script has evaluated this poster.
    adult: TMDB adult flag.
    certification_us: US certification from TMDB release_dates.

    ============================ Returns ============================
    True when the frontend should show the poster image.
    """
    if adult:
        return False
    if not poster_path:
        return False
    if not poster_checked and certification_us in QUESTIONABLE_US_CERTIFICATIONS:
        return False
    return poster_safe is not False

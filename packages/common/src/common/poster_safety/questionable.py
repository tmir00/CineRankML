"""Identify catalog movies that need offline poster safety checks."""

from __future__ import annotations

from common.tmdb.certifications import QUESTIONABLE_US_CERTIFICATIONS


def is_questionable_movie(*, adult: bool, certification_us: str | None) -> bool:
    """
    Return True when a movie should be checked by the offline poster safety script.

    Questionable movies are adult-flagged or carry a mature US certification.

    ============================ Arguments ============================
    adult: TMDB adult flag stored on catalog_movies.
    certification_us: US certification from TMDB release_dates.

    ============================ Returns ============================
    True when the movie is risky enough to warrant a poster check.
    """
    if adult:
        return True
    if certification_us and certification_us in QUESTIONABLE_US_CERTIFICATIONS:
        return True
    return False

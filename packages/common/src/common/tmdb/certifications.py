"""Parse US movie certifications from TMDB release_dates payloads."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

QUESTIONABLE_US_CERTIFICATIONS = frozenset({"R", "NC-17"})


def extract_us_certification(release_dates_payload: dict | None) -> str | None:
    """
    Extract the first non-empty US certification from a TMDB release_dates block.

    TMDB returns release_dates via append_to_response=release_dates or
    GET /movie/{id}/release_dates. The US entry uses iso_3166_1 == "US".

    ============================ Arguments ============================
    release_dates_payload: The release_dates object from TMDB JSON.

    ============================ Returns ============================
    Certification string such as G, PG, PG-13, R, or NC-17; None when unavailable.
    """
    if not release_dates_payload or not isinstance(release_dates_payload, dict):
        return None

    results = release_dates_payload.get("results")
    if not isinstance(results, list):
        return None

    for country in results:
        if not isinstance(country, dict):
            continue
        if country.get("iso_3166_1") != "US":
            continue

        release_dates = country.get("release_dates")
        if not isinstance(release_dates, list):
            continue

        for entry in release_dates:
            if not isinstance(entry, dict):
                continue
            certification = entry.get("certification")
            if certification is None:
                continue
            cert_str = str(certification).strip()
            if cert_str:
                return cert_str

    return None


def parse_us_certification_from_movie_payload(
    payload: dict,
    *,
    tmdb_id: int | None = None,
) -> str | None:
    """
    Safely extract US certification from a full TMDB movie JSON response.

    Logs a warning when release_dates is present but cannot be parsed.

    ============================ Arguments ============================
    payload: Decoded JSON from GET /3/movie/{id}.
    tmdb_id: Optional TMDB id for error logging.

    ============================ Returns ============================
    US certification string or None.
    """
    try:
        release_dates_block = payload.get("release_dates")
        return extract_us_certification(release_dates_block)
    except (KeyError, TypeError, ValueError):
        logger.warning(
            "Failed to parse US certification from release_dates",
            extra={"tmdb_id": tmdb_id},
        )
        return None

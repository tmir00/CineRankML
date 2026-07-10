"""Build TMDB poster image URLs for offline safety checks."""

from __future__ import annotations

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p"


def build_tmdb_poster_url(poster_path: str, *, size: str = "w342") -> str:
    """
    Build a TMDB poster image URL from a poster_path value.

    ============================ Arguments ============================
    poster_path: TMDB poster path such as /abc123.jpg.
    size: TMDB image size token (default w342 for small downloads).

    ============================ Returns ============================
    Full HTTPS URL for the poster image.
    """
    if not poster_path.startswith("/"):
        poster_path = f"/{poster_path}"
    return f"{TMDB_IMAGE_BASE}/{size}{poster_path}"

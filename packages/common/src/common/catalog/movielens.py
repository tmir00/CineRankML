"""Parse MovieLens CSV fields into catalog-ready values."""

from __future__ import annotations

import re

# Regular expression to match titles like 'Toy Story (1995)'
_TITLE_YEAR_RE = re.compile(r"^(.*)\s+\((\d{4})\)$")


def parse_title_year(raw_title: str) -> tuple[str, int | None]:
    """
    Split a MovieLens title like 'Toy Story (1995)' into clean title and year.

    ============================ Arguments ============================
    raw_title: The title column value from movies.csv. Example: 'Toy Story (1995)'

    ============================ Returns ============================
    A tuple of (title, year). Year is None when the title has no trailing year.
    """
    # Remove whitespace and parentheses from the title
    match = _TITLE_YEAR_RE.match(raw_title.strip())

    # If no match, return the original title and None for the year
    if match is None:
        return raw_title.strip(), None

    # If match, return the title and year
    return match.group(1).strip(), int(match.group(2))


def parse_genres(raw_genres: str) -> list[str]:
    """
    Split MovieLens pipe-separated genres into a list of genre names.

    ============================ Arguments ============================
    raw_genres: The genres column value from movies.csv.

    ============================ Returns ============================
    A list of genre strings, or an empty list when genres are missing.
    """
    if not raw_genres or raw_genres == "(no genres listed)":
        return []
    return [genre.strip() for genre in raw_genres.split("|") if genre.strip()]


def format_imdb_id(raw_imdb_id: str) -> str | None:
    """
    Format a MovieLens imdbId into a standard tt-prefixed imdb id.

    ============================ Arguments ============================
    raw_imdb_id: The imdbId column value from links.csv (e.g. 0114709).

    ============================ Returns ============================
    A string like tt0114709, or None when the value is empty.
    """
    # Remove whitespace from the imdbId
    cleaned = raw_imdb_id.strip()

    # If the imdbId is empty, return None
    if not cleaned:
        return None
    
    # If the imdbId starts with 'tt', return it
    if cleaned.startswith("tt"):
        return cleaned
        
    # If the imdbId does not start with 'tt', return it prefixed with 'tt'
    return f"tt{cleaned}"

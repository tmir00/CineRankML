"""Build canonical embedding text from catalog movie metadata."""

from __future__ import annotations

import hashlib
from typing import Protocol

TEXT_TEMPLATE_VERSION = "v1"
MAX_TAGS = 10


class MovieTextLike(Protocol):
    """Minimal catalog fields used to build embedding text."""

    title: str
    overview: str | None
    tagline: str | None
    genres: list[str] | None
    tmdb_keywords: list[str] | None


def build_embedding_text(movie: MovieTextLike, tags: list[str]) -> str:
    """
    Build one canonical text blob for content embedding.

    Do this by:
    1. Starting with the movie title.
    2. Appending overview, tagline, genres, TMDB keywords, and top tags.
    3. Joining sections with newlines for a stable MiniLM input.

    ============================ Arguments ============================
    movie: Catalog movie metadata.
    tags: Top tags ordered by popularity.

    ============================ Returns ============================
    Canonical text passed to the embedder model.
    """
    # Build the list of sections to join. Start with the movie title.
    sections: list[str] = [movie.title.strip()]

    # Append the rest of the fields to go into the embedding text.
    if movie.overview:
        sections.append(movie.overview.strip())
    if movie.tagline:
        sections.append(movie.tagline.strip())
    if movie.genres:
        sections.append("Genres: " + ", ".join(movie.genres))
    if movie.tmdb_keywords:
        sections.append("Keywords: " + ", ".join(movie.tmdb_keywords))
    if tags:
        sections.append("Tags: " + ", ".join(tags[:MAX_TAGS]))

    # Join the sections with newlines for a stable MiniLM input.
    return "\n".join(section for section in sections if section)


def embedding_text_hash(text: str) -> str:
    """
    Return a stable hash for one embedding text blob.

    ============================ Arguments ============================
    text: Canonical embedding text.

    ============================ Returns ============================
    Hex digest used to detect stale embeddings.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

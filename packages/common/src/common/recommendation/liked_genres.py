"""Derive liked genres from a user's rated movies."""

from __future__ import annotations


def derive_liked_genres(merged_ratings: dict[int, float], movie_genres: dict[int, list[str]], *, top_n: int = 4) -> list[str]:
    """
    Build a ranked list of genres the user seems to enjoy from their ratings.

    Do this by:
    1. Walking every rated movie and looking up its genre list.
    2. Adding a weight of max(0, rating - 3.0) to each genre on that movie.
    3. Sorting genres by total weight and returning the top few names.

    ============================ Arguments ============================
    merged_ratings: Latest rating per movie_id for the user.
    movie_genres: Genre lists keyed by movie_id from catalog_movies.
    top_n: How many top genres to return.

    ============================ Returns ============================
    Up to top_n genre names, highest weight first. Empty when no genres found.
    """
    if top_n <= 0:
        return []

    # Accumulate a taste score per genre name.
    genre_scores: dict[str, float] = {}

    # Walk each rated movie and spread its rating weight across its genres.
    for movie_id, rating in merged_ratings.items():
        weight = max(0.0, rating - 3.0)
        if weight <= 0.0:
            continue

        for genre in movie_genres.get(movie_id, []):
            genre_scores[genre] = genre_scores.get(genre, 0.0) + weight

    if not genre_scores:
        return []

    # Highest total weight first.
    ranked = sorted(genre_scores.items(), key=lambda item: item[1], reverse=True)
    return [genre for genre, _ in ranked[:top_n]]

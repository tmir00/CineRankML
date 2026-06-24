"""Load MovieLens CSV files and upsert catalog_movies rows."""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterator

from common.catalog.movielens import format_imdb_id, parse_genres, parse_title_year
from common.config.settings import CatalogSeedSettings
from common.db.repositories.catalog import CatalogSeedRow

logger = logging.getLogger(__name__)


def load_movies_by_id(settings: CatalogSeedSettings) -> dict[int, dict]:
    """
    Read movies.csv into a movieId-indexed lookup map.

    Do this by:
    1. Streaming movies.csv row by row.
    2. Parsing title/year and genres for each movieId.
    3. Returning a dict keyed by movieId.

    ============================ Arguments ============================
    settings: Paths to the MovieLens CSV files.

    ============================ Returns ============================
    A dict mapping movieId to {title, year, genres}.
    """
    # Initialize an empty dictionary to store the movies.
    movies: dict[int, dict] = {}

    # Open the movies.csv file and read the rows.
    with open(settings.movies_csv_path, newline="", encoding="utf-8") as handle:
        # Create a CSV reader for the movies.csv file.
        reader = csv.DictReader(handle)

        # Iterate over the rows in the movies.csv file.
        for row in reader:
            # Get the movieId from the row.
            movie_id = int(row["movieId"])
            # Parse the title and year from the row.
            title, year = parse_title_year(row["title"])
            # Parse the genres from the row.
            genres = parse_genres(row.get("genres", ""))
            # Add the movie to the dictionary.
            movies[movie_id] = {
                "title": title,
                "year": year,
                "genres": genres,
            }

    logger.info(
        "Loaded movies.csv",
        extra={"path": settings.movies_csv_path, "count": len(movies)},
    )
    return movies


def iter_seed_rows(settings: CatalogSeedSettings) -> Iterator[CatalogSeedRow]:
    """
    Join links.csv with movies.csv then return an iterator of CatalogSeedRow objects.
    This row can be used to bulk upsert into catalog_movies.

    Do this by:
    1. Loading movies.csv into memory keyed by movieId.
    2. Streaming links.csv and joining each row to its movie metadata.
    3. Formatting imdb_id and yielding CatalogSeedRow objects.

    ============================ Arguments ============================
    settings: Paths to the MovieLens CSV files.

    ============================ Returns ============================
    An iterator of CatalogSeedRow ready for bulk upsert. E.g:
        CatalogSeedRow(
            movie_id=1,
            title="The Dark Knight",
            year=2008,
            genres=["Action", "Crime", "Drama"],
            tmdb_id=155,
            imdb_id="tt0468569",
        )
    """
    movies_by_id = load_movies_by_id(settings)

    # Open the links.csv file and read the rows.
    with open(settings.links_csv_path, newline="", encoding="utf-8") as handle:
        # Create a CSV reader for the links.csv file.
        reader = csv.DictReader(handle)

        # Iterate over the rows in the links.csv file.
        for row in reader:
            # Get the movieId from the row.
            movie_id = int(row["movieId"])
            # Get the movie from the dictionary.
            movie = movies_by_id.get(movie_id)
            if movie is None:
                # Log a warning if the movie is not found.
                logger.warning(
                    "Skipping link row with no matching movie",
                    extra={"movie_id": movie_id},
                )
                continue

            # Get the tmdbId from the row.
            tmdb_raw = row.get("tmdbId", "").strip()
            # Convert the tmdbId to an integer.
            tmdb_id = int(tmdb_raw) if tmdb_raw else None

            # Yield the CatalogSeedRow object.
            yield CatalogSeedRow(
                movie_id=movie_id,
                title=movie["title"],
                year=movie["year"],
                genres=movie["genres"],
                tmdb_id=tmdb_id,
                imdb_id=format_imdb_id(row.get("imdbId", "")),
            )

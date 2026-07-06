"""Fetch movie details from The Movie Database (TMDB) API."""

from __future__ import annotations

import time
import httpx
import logging

from dataclasses import dataclass
from common.config.settings import TmdbSettings


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TmdbMovieDetails:
    """ TMDB movie fields mapped to catalog_movies columns. """

    overview: str | None
    tagline: str | None
    original_language: str | None
    runtime: int | None
    tmdb_popularity: float | None
    tmdb_vote_average: float | None
    tmdb_vote_count: int | None
    tmdb_keywords: list[str]
    poster_path: str | None


class TmdbClient:
    """
    HTTP client for TMDB movie detail lookups.
    This is used to fetch movie details from TMDB by tmdb_id.

    Do this by:
    1. Sending authenticated GET requests to /3/movie/{tmdb_id}.
    2. Sleeping between requests to respect rate limits.
    3. Retrying transient HTTP failures with exponential backoff.
    """

    def __init__(self, settings: TmdbSettings) -> None:
        """
        Create a TMDB client from application settings.

        ============================ Arguments ============================
        settings: TMDB API key, base URL, timeout, and rate-limit settings.
        """
        # Initialize the client with the settings
        self._settings = settings
        # Initialize the minimum interval between requests
        self._min_interval = 1.0 / max(0.1, settings.tmdb_requests_per_second)
        self._last_request_at = 0.0

    def fetch_movie(self, tmdb_id: int) -> tuple[TmdbMovieDetails | None, str | None]:
        """
        Fetch one movie's details from TMDB by tmdb_id.

        Do this by:
        1. Waiting until the configured requests-per-second limit allows another call.
        2. Calling GET /3/movie/{tmdb_id}?append_to_response=keywords.
        3. Mapping the JSON response into TmdbMovieDetails.
        4. Returning (None, error_type) when the movie is missing or the request fails.

        ============================ Arguments ============================
        tmdb_id: The TMDB movie identifier from links.csv.

        ============================ Returns ============================
        A tuple of (details, error_type). details is set on success; error_type is
        set on failure (e.g. not_found, http_error, parse_error).
        """
        # Wait for the rate limit to be allowed
        self._wait_for_rate_limit()

        # Build the URL, parameters, and headers for the TMDB movie details request.
        url = f"{self._settings.tmdb_base_url.rstrip('/')}/3/movie/{tmdb_id}"
        params = {"append_to_response": "keywords", "language": "en-US"}
        headers = {"Authorization": f"Bearer {self._settings.tmdb_api_key}"}

        # Retry the request up to the maximum number of retries
        for attempt in range(self._settings.tmdb_max_retries + 1):
            # Try to get the movie details
            try:
                # Create a new HTTP client with the timeout
                with httpx.Client(timeout=self._settings.tmdb_timeout_seconds) as client:
                    # Send the GET request to the TMDB API
                    response = client.get(url, params=params, headers=headers)

                # If the movie is not found, return None and the error type
                if response.status_code == 404:
                    return None, "not_found"

                # If the request is rate limited or the server is erroring, retry the request
                if response.status_code == 429 or response.status_code >= 500:
                    # If the attempt is less than the maximum number of retries, retry the request
                    if attempt < self._settings.tmdb_max_retries:
                        # Calculate the backoff time
                        backoff = min(2**attempt, 30)
                        # Log the error
                        logger.warning(
                            "TMDB transient error, retrying",
                            extra={
                                "tmdb_id": tmdb_id,
                                "status_code": response.status_code,
                                "attempt": attempt + 1,
                                "backoff_seconds": backoff,
                            },
                        )
                        # Sleep for the backoff time
                        time.sleep(backoff)
                        continue
                    # If the attempt is greater than the maximum number of retries, return None and the error type
                    return None, "http_error"

                # If the request is a client error, return None and the error type
                if response.status_code >= 400:
                    return None, "http_error"

                # Parse the response body as JSON
                payload = response.json()
                # Return the parsed movie details and None for the error type
                return self._parse_movie_payload(payload), None

            except httpx.HTTPError:
                # If the attempt is less than the maximum number of retries, retry the request
                if attempt < self._settings.tmdb_max_retries:
                    # Calculate the backoff time
                    backoff = min(2**attempt, 30)
                    # Sleep for the backoff time
                    time.sleep(backoff)
                    # Retry the request
                    continue
                return None, "http_error"
            except (KeyError, TypeError, ValueError):
                # If the attempt is greater than the maximum number of retries, return None and the error type
                return None, "parse_error"

        return None, "http_error"

    def _wait_for_rate_limit(self) -> None:
        """Sleep until the next request is allowed under requests-per-second."""
        # Get the current time
        now = time.monotonic()
        
        # Calculate the elapsed time since the last request
        elapsed = now - self._last_request_at
        
        # If the elapsed time is less than the minimum interval, sleep for the minimum interval
        if elapsed < self._min_interval:
            # Sleep for the minimum interval
            time.sleep(self._min_interval - elapsed)
        
        # Update the last request time
        self._last_request_at = time.monotonic()

    def _parse_movie_payload(self, payload: dict) -> TmdbMovieDetails:
        """
        Map a TMDB movie JSON object into catalog column values.

        ============================ Arguments ============================
        payload: The decoded JSON body from GET /3/movie/{id}.

        ============================ Returns ============================
        A TmdbMovieDetails instance ready for catalog_movies.
        """
        # Get the keywords block from the payload
        keywords_block = payload.get("keywords") or {}
        # Get the keyword rows from the keywords block
        keyword_rows = keywords_block.get("keywords") or []
        # Get the keywords from the keyword rows
        keywords = [
            str(row["name"])
            for row in keyword_rows
            if isinstance(row, dict) and row.get("name")
        ]

        # Get the runtime from the payload
        runtime = payload.get("runtime")
        # Get the vote count from the payload
        vote_count = payload.get("vote_count")

        # Return the parsed movie details
        return TmdbMovieDetails(
            overview=payload.get("overview") or None,
            tagline=payload.get("tagline") or None,
            original_language=payload.get("original_language") or None,
            runtime=int(runtime) if runtime is not None else None,
            tmdb_popularity=float(payload["popularity"]) if payload.get("popularity") is not None else None,
            tmdb_vote_average=float(payload["vote_average"]) if payload.get("vote_average") is not None else None,
            tmdb_vote_count=int(vote_count) if vote_count is not None else None,
            tmdb_keywords=keywords,
            poster_path=payload.get("poster_path") or None,
        )

"""Train-fit metadata normalization for hybrid ranker candidate features."""

from __future__ import annotations

import math

from pydantic import BaseModel


class MetadataNormalizationStats(BaseModel):
    """Min/max stats fit on train-split candidates for metadata normalization."""

    year_min: float
    year_max: float
    runtime_min: float
    runtime_max: float
    tmdb_popularity_log_min: float
    tmdb_popularity_log_max: float
    tmdb_vote_average_min: float
    tmdb_vote_average_max: float
    tmdb_vote_count_log_min: float
    tmdb_vote_count_log_max: float


def min_max_norm(value: float | None, minimum: float, maximum: float) -> float:
    """
    Scale one numeric value into [0, 1] using train-fit min and max.

    Returns 0.0 when value is missing or when min equals max.

    ============================ Arguments ============================
    value: Raw candidate metadata value.
    minimum: Train-fit minimum.
    maximum: Train-fit maximum.

    ============================ Returns ============================
    Normalized value in [0, 1].
    """
    if value is None or math.isnan(value):
        return 0.0
    if maximum <= minimum:
        return 0.0
    scaled = (float(value) - minimum) / (maximum - minimum)
    return max(0.0, min(1.0, scaled))


def log1p_or_zero(value: float | None) -> float:
    """Apply log1p to a non-negative value; missing values become 0."""
    if value is None or math.isnan(value) or value < 0:
        return 0.0
    return math.log1p(float(value))


def normalize_candidate_metadata(*, year: float | None, runtime: float | None, tmdb_popularity: float | None, tmdb_vote_average: float | None, \
                                        tmdb_vote_count: float | None, stats: MetadataNormalizationStats) -> tuple[float, float, float, float, float]:
    """
    Normalize five candidate metadata fields using training-fit normalization stats.

    Do this by:
    1. Applying log1p to popularity and vote_count before min-max scaling.
    2. Min-max scaling year, runtime, vote_average, and the log fields.

    ============================ Arguments ============================
    year: Candidate release year.
    runtime: Candidate runtime in minutes.
    tmdb_popularity: Raw TMDB popularity.
    tmdb_vote_average: Raw TMDB vote average.
    tmdb_vote_count: Raw TMDB vote count.
    stats: Train-fit normalization bounds from the hybrid dataset manifest.

    ============================ Returns ============================
    Tuple of five normalized metadata features in schema order.
    """
    popularity_log = log1p_or_zero(tmdb_popularity)
    vote_count_log = log1p_or_zero(tmdb_vote_count)

    return (
        min_max_norm(year, stats.year_min, stats.year_max),
        min_max_norm(runtime, stats.runtime_min, stats.runtime_max),
        min_max_norm(popularity_log, stats.tmdb_popularity_log_min, stats.tmdb_popularity_log_max),
        min_max_norm(tmdb_vote_average, stats.tmdb_vote_average_min, stats.tmdb_vote_average_max),
        min_max_norm(vote_count_log, stats.tmdb_vote_count_log_min, stats.tmdb_vote_count_log_max),
    )

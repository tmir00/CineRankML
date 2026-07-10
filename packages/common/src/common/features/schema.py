"""Hybrid ranker feature vector layout and version constants."""

from __future__ import annotations

FEATURE_SCHEMA_VERSION = "hybrid-v1"
INPUT_DIM = 1356

CONTENT_EMBEDDING_DIM = 384
CF_EMBEDDING_DIM = 64
USER_BEHAVIOR_DIM = 5
CANDIDATE_METADATA_DIM = 5

HIGH_RATED_THRESHOLD = 4.0
LOW_RATED_THRESHOLD = 2.0

# Slot offsets in the concatenated feature vector.
OFFSET_USER_CONTENT_PROFILE = 0
OFFSET_CANDIDATE_CONTENT_EMBEDDING = 384
OFFSET_CONTENT_COSINE = 768
OFFSET_CONTENT_PRODUCT = 769
OFFSET_USER_CF_PROFILE = 1153
OFFSET_CANDIDATE_CF_EMBEDDING = 1217
OFFSET_CF_COSINE = 1281
OFFSET_CF_PRODUCT = 1282
OFFSET_USER_BEHAVIOR = 1346
OFFSET_CANDIDATE_METADATA = 1351

USER_BEHAVIOR_FIELDS = (
    "num_user_ratings",
    "user_avg_rating",
    "user_rating_std",
    "num_high_rated_movies",
    "num_low_rated_movies",
)

CANDIDATE_METADATA_FIELDS = (
    "candidate_year_norm",
    "candidate_runtime_norm",
    "candidate_tmdb_popularity_log_norm",
    "candidate_tmdb_vote_average_norm",
    "candidate_tmdb_vote_count_log_norm",
)

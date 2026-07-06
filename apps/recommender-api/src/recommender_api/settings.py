"""Runtime settings for the recommender-api service."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RecommenderApiSettings(BaseSettings):
    """Environment-backed settings for recommender-api."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    recommender_host: str = "0.0.0.0"
    recommender_port: int = 8090
    metrics_port: int = 9111

    session_cookie_name: str = "session_id"
    session_ttl_days: int = 7

    default_top_k: int = 20
    min_ratings_for_recommend: int = 5

    retrieval_knn_size: int = 250
    retrieval_popular_size: int = 50
    retrieval_random_genre_size: int = 75
    retrieval_random_knn_size: int = 75
    retrieval_knn_pool_size: int = 400
    retrieval_random_knn_skip_top: int = 150
    retrieval_max_candidates: int = 300
    retrieval_liked_genre_count: int = 4
    retrieval_min_vote_count: int = 100
    retrieval_min_vote_average: float = 6.0

    model_version: str | None = None
    cf_version: str | None = None

    cors_allow_origins: list[str] = Field(
        default=["http://localhost:5173"],
        validation_alias="CORS_ALLOW_ORIGINS",
    )

    s3_endpoint_url: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "cinerankml"

    @property
    def session_ttl_seconds(self) -> int:
        """Return session cookie max-age in seconds."""
        return self.session_ttl_days * 24 * 60 * 60

"""Runtime settings for the recommender-api service."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class RecommenderApiSettings(BaseSettings):
    """Environment-backed settings for recommender-api."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    recommender_host: str = "0.0.0.0"
    recommender_port: int = 8090
    metrics_port: int = 9111

    session_cookie_name: str = "session_id"
    session_ttl_days: int = 7

    candidate_pool_size: int = 200
    default_top_k: int = 20
    min_ratings_for_recommend: int = 5

    model_version: str | None = None
    cf_version: str | None = None

    s3_endpoint_url: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "cinerankml"

    @property
    def session_ttl_seconds(self) -> int:
        """Return session cookie max-age in seconds."""
        return self.session_ttl_days * 24 * 60 * 60

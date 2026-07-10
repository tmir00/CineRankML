"""Runtime settings for the recommender-api service."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from common.recommendation.split_policy import SplitPolicySettings


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

    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        validation_alias="MLFLOW_TRACKING_URI",
    )
    mlflow_registered_model_name: str = Field(
        default="hybrid_ranker",
        validation_alias="MLFLOW_REGISTERED_MODEL_NAME",
    )
    experiment_id: str = Field(
        default="exp-main-vs-candidate",
        validation_alias="EXPERIMENT_ID",
    )

    initial_main_split: float = Field(default=0.70, validation_alias="INITIAL_MAIN_SPLIT")
    initial_candidate_split: float = Field(default=0.30, validation_alias="INITIAL_CANDIDATE_SPLIT")
    split_adjust_step: float = Field(default=0.02, validation_alias="SPLIT_ADJUST_STEP")
    split_high_rating_threshold: float = Field(default=4.0, validation_alias="SPLIT_HIGH_RATING_THRESHOLD")
    split_low_rating_threshold: float = Field(default=3.0, validation_alias="SPLIT_LOW_RATING_THRESHOLD")
    split_max_fraction: float = Field(default=0.80, validation_alias="SPLIT_MAX_FRACTION")
    split_min_fraction: float = Field(default=0.00, validation_alias="SPLIT_MIN_FRACTION")
    promotion_min_ratings: int = Field(default=10, validation_alias="PROMOTION_MIN_RATINGS")
    promotion_min_avg_rating: float = Field(default=4.0, validation_alias="PROMOTION_MIN_AVG_RATING")

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

    @property
    def split_policy(self) -> SplitPolicySettings:
        """Build split policy thresholds from environment-backed settings."""
        return SplitPolicySettings(
            initial_main_split=self.initial_main_split,
            initial_candidate_split=self.initial_candidate_split,
            adjust_step=self.split_adjust_step,
            high_rating_threshold=self.split_high_rating_threshold,
            low_rating_threshold=self.split_low_rating_threshold,
            max_fraction=self.split_max_fraction,
            min_fraction=self.split_min_fraction,
            promotion_min_ratings=self.promotion_min_ratings,
            promotion_min_avg_rating=self.promotion_min_avg_rating,
        )

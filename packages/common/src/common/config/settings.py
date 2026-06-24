from typing import Self
from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Postgres connection settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://cinerankml:cinerankml@localhost:5432/cinerankml"
    
    # Set the number of persistent connections to the database (5 for default)
    pool_size: int = 5
    # If all connections are in use, allow up to max_overflow additional connections to be created (10 default, 15 total)
    max_overflow: int = 10


class KafkaSettings(BaseSettings):
    """Kafka broker and topic settings."""

    # Load settings from .env file and ignore extra env variables not in KafkaSettings.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    kafka_bootstrap_servers: str = "localhost:9092"
    ratings_topic: str = "ratings"
    tags_topic: str = "tags"
    ratings_consumer_group: str = "ratings-consumer"
    tags_consumer_group: str = "tags-consumer"


class ProducerSettings(BaseSettings):
    """CSV producer throttle and Postgres checkpoint settings."""

    # Load settings from .env file and ignore extra env variables not in ProducerSettings.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    csv_path: str = "/data/ratings.csv"
    rows_per_second: int = 1000
    row_delay_seconds: float = 1.5
    row_limit: int | None = None
    source_file: str | None = None
    start_row: int = 0
    producer_log_every_n: int = 1000
    checkpoint_every_n: int = 1000

    @model_validator(mode="after")
    def _default_source_file(self) -> Self:
        """
        Use the CSV file name as source_file when SOURCE_FILE is not set.

        This keeps checkpoint keys stable across environments (e.g. ratings.csv)
        even when CSV_PATH differs between local and Docker mounts.
        """
        if not self.source_file:
            self.source_file = Path(self.csv_path).name
        return self

    # Run the validator before the 'row_limit' field is set.
    @field_validator("row_limit", mode="before")
    @classmethod # Pydantic validators are class methods since they do not belong to an instance (instance not yet created), instead to a class.
    def _empty_row_limit(cls, value: object) -> object:
        """ Before Pydantic parses row_limit, if the raw value is empty or None, set it to None. """
        if value == "" or value is None:
            return None
        return value


class WorkerMetricsSettings(BaseSettings):
    """Prometheus metrics HTTP server settings."""

    # Load settings from .env file and ignore extra env variables not in WorkerMetricsSettings.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    metrics_port: int = 9100
    worker_name: str = "worker"


class TmdbSettings(BaseSettings):
    """TMDB API client settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    tmdb_api_key: str = ""
    tmdb_base_url: str = "https://api.themoviedb.org"
    tmdb_requests_per_second: float = 3.0
    tmdb_timeout_seconds: float = 30.0
    tmdb_max_retries: int = 3


class CatalogSeedSettings(BaseSettings):
    """MovieLens CSV paths and seed batch size."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    movies_csv_path: str = "/data/movies.csv"
    links_csv_path: str = "/data/links.csv"
    seed_batch_size: int = 1000


class EnrichmentSettings(BaseSettings):
    """TMDB enrichment batch job settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    enrichment_batch_size: int = 50
    enrichment_limit: int | None = None
    enrichment_log_every_n: int = 100

    @field_validator("enrichment_limit", mode="before")
    @classmethod
    def _empty_enrichment_limit(cls, value: object) -> object:
        """Treat empty env values as no limit."""
        if value == "" or value is None:
            return None
        return value


def get_database_settings() -> DatabaseSettings:
    return DatabaseSettings()


def get_kafka_settings() -> KafkaSettings:
    return KafkaSettings()


def get_producer_settings() -> ProducerSettings:
    return ProducerSettings()


def producer_row_delay_seconds(settings: ProducerSettings) -> float:
    """
    Return how long to wait between publishing two CSV rows.

    When ROW_DELAY_SECONDS is greater than zero, that value is used directly.
    Otherwise the wait is derived from ROWS_PER_SECOND (1 / rows_per_second).

    ============================ Arguments ============================
    settings: The producer configuration.

    ============================ Returns ============================
    Seconds to sleep after each published row.
    """
    if settings.row_delay_seconds > 0:
        return settings.row_delay_seconds
    return 1.0 / max(1, settings.rows_per_second)


def get_worker_metrics_settings() -> WorkerMetricsSettings:
    return WorkerMetricsSettings()


def get_tmdb_settings() -> TmdbSettings:
    return TmdbSettings()


def get_catalog_seed_settings() -> CatalogSeedSettings:
    return CatalogSeedSettings()


def get_enrichment_settings() -> EnrichmentSettings:
    return EnrichmentSettings()


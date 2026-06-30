from typing import Self
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator, model_validator
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


class OpenSearchSettings(BaseSettings):
    """OpenSearch connection and index settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    opensearch_host: str = "localhost"
    opensearch_port: int = 9200
    opensearch_use_ssl: bool = False
    opensearch_verify_certs: bool = False
    opensearch_timeout_seconds: float = 30.0
    opensearch_index_alias: str = "movies"
    opensearch_bulk_batch_size: int = 100


class EmbedderSettings(BaseSettings):
    """HTTP client settings for the embedder-api service."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    embedder_base_url: str = "http://localhost:8080"
    embedder_timeout_seconds: float = 60.0
    embedding_version: str = "content-v1"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    embedding_text_template_version: str = "v1"


class SyncSettings(BaseSettings):
    """OpenSearch sync batch job settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    sync_batch_size: int = 50
    sync_limit: int | None = None
    rebuild_index: bool = False
    sync_log_every_n: int = 100
    job_name: str = "opensearch_sync"

    @field_validator("sync_limit", mode="before")
    @classmethod
    def _empty_sync_limit(cls, value: object) -> object:
        """Treat empty env values as no limit."""
        if value == "" or value is None:
            return None
        return value


def get_opensearch_settings() -> OpenSearchSettings:
    return OpenSearchSettings()


def get_embedder_settings() -> EmbedderSettings:
    return EmbedderSettings()


def get_sync_settings() -> SyncSettings:
    return SyncSettings()


class SnapshotSettings(BaseSettings):
    """Snapshot-to-S3 batch job and MinIO settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    s3_endpoint_url: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "cinerankml"
    snapshot_batch_size: int = 50_000
    snapshot_id: str | None = None
    job_name: str = "snapshot_to_s3"
    metrics_job_name: str = "s3_snapshot"
    pushgateway_url: str = "http://localhost:9091"

    @field_validator("snapshot_id", mode="before")
    @classmethod
    def _empty_snapshot_id(cls, value: object) -> object:
        """Treat empty env values as auto-generated snapshot id."""
        if value == "" or value is None:
            return None
        return value


def get_snapshot_settings() -> SnapshotSettings:
    return SnapshotSettings()


class CfDatasetSettings(BaseSettings):
    """CF dataset prep batch job and MinIO settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    s3_endpoint_url: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "cinerankml"
    snapshot_id: str | None = None
    cf_dataset_version: str | None = None
    cf_shuffle_seed: int = 42
    train_fraction: float = Field(
        default=0.8,
        validation_alias=AliasChoices("CF_TRAIN_FRACTION", "TRAIN_FRACTION"),
    )
    validation_fraction: float = Field(
        default=0.1,
        validation_alias=AliasChoices("CF_VALIDATION_FRACTION"),
    )
    test_fraction: float = Field(
        default=0.1,
        validation_alias=AliasChoices("CF_TEST_FRACTION"),
    )
    cf_part_row_limit: int = 500_000
    job_name: str = "prepare_cf_dataset"
    metrics_job_name: str = Field(
        default="prepare_cf_dataset",
        validation_alias=AliasChoices("CF_DATASET_METRICS_JOB_NAME", "METRICS_JOB_NAME"),
    )
    pushgateway_url: str = "http://localhost:9091"

    @field_validator("snapshot_id", "cf_dataset_version", mode="before")
    @classmethod
    def _empty_optional_ids(cls, value: object) -> object:
        """Treat empty env values as auto-generated ids."""
        if value == "" or value is None:
            return None
        return value

    @model_validator(mode="after")
    def _split_fractions_sum_to_one(self) -> Self:
        """Train, validation, and test fractions must sum to 1.0."""
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                "CF split fractions must sum to 1.0 "
                f"(train={self.train_fraction}, validation={self.validation_fraction}, "
                f"test={self.test_fraction})"
            )
        return self


def get_cf_dataset_settings() -> CfDatasetSettings:
    return CfDatasetSettings()


class CfTrainingSettings(BaseSettings):
    """CF PyTorch training batch job, MinIO, MLflow, and metrics settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    s3_endpoint_url: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "cinerankml"
    cf_dataset_version: str | None = None
    cf_version: str | None = None
    embedding_dim: int = Field(default=64, validation_alias=AliasChoices("CF_EMBEDDING_DIM"))
    num_epochs: int = Field(default=20, validation_alias=AliasChoices("CF_EPOCHS"))
    batch_size: int = Field(default=4096, validation_alias=AliasChoices("CF_BATCH_SIZE"))
    learning_rate: float = Field(default=0.01, validation_alias=AliasChoices("CF_LEARNING_RATE"))
    early_stopping_patience: int = Field(
        default=3,
        validation_alias=AliasChoices("CF_EARLY_STOPPING_PATIENCE"),
    )
    shuffle_seed: int = Field(default=42, validation_alias=AliasChoices("CF_SHUFFLE_SEED"))
    device: str = Field(default="auto", validation_alias=AliasChoices("CF_DEVICE"))
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        validation_alias=AliasChoices("MLFLOW_TRACKING_URI"),
    )
    mlflow_experiment_name: str = Field(
        default="collaborative_filtering",
        validation_alias=AliasChoices("MLFLOW_EXPERIMENT_NAME"),
    )
    job_name: str = "train_cf"
    metrics_job_name: str = Field(
        default="train_cf",
        validation_alias=AliasChoices("CF_METRICS_JOB_NAME", "METRICS_JOB_NAME"),
    )
    pushgateway_url: str = "http://localhost:9091"

    @field_validator("cf_dataset_version", "cf_version", mode="before")
    @classmethod
    def _empty_optional_ids(cls, value: object) -> object:
        """ Set unset env values as None instead of an empty string. """
        if value == "" or value is None:
            return None
        return value


def get_cf_training_settings() -> CfTrainingSettings:
    return CfTrainingSettings()


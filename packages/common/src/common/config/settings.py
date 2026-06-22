from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Postgres connection settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://cinerankml:cinerankml@localhost:5432/cinerankml"
    
    # Set the number of persistent connections to the database (5 for default)
    pool_size: int = 5
    # If all connections are in use, allow up to max_overflow additional connections to be created (10 default, 15 total)
    max_overflow: int = 10


def get_database_settings() -> DatabaseSettings:
    return DatabaseSettings()

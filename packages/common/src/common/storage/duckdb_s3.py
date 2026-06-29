"""DuckDB connection helpers for reading and writing Parquet on MinIO."""

from __future__ import annotations

import duckdb

from urllib.parse import urlparse
from common.config.settings import CfDatasetSettings


def _endpoint_host_port(endpoint_url: str) -> str:
    """
    Convert an S3 endpoint URL to a host:port string for DuckDB httpfs.
    DuckDb needs httpfs to be able to read and write Parquet files from S3.
    """
    parsed = urlparse(endpoint_url)
    if parsed.port is not None:
        return f"{parsed.hostname}:{parsed.port}"
    return parsed.hostname or endpoint_url


def create_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """
    Create an in-memory DuckDB connection with httpfs loaded.

    ============================ Returns ============================
    A DuckDB connection ready for S3 Parquet reads and writes.
    """
    conn = duckdb.connect(database=":memory:")
    conn.execute("INSTALL httpfs;")
    conn.execute("LOAD httpfs;")
    return conn


def configure_duckdb_s3(conn: duckdb.DuckDBPyConnection, settings: CfDatasetSettings) -> None:
    """
    Point DuckDB httpfs at the MinIO endpoint used by this project.

    Do this by:
    1. Parsing the endpoint host and port from settings.
    2. Setting S3 credentials, path-style URLs, and SSL flags for MinIO.

    ============================ Arguments ============================
    conn: An open DuckDB connection with httpfs loaded.
    settings: CF dataset prep configuration with S3 endpoint and credentials.
    """
    endpoint = _endpoint_host_port(settings.s3_endpoint_url)
    use_ssl = settings.s3_endpoint_url.startswith("https://")

    # Configure DuckDB to talk to MinIO instead of AWS S3.
    conn.execute(f"SET s3_endpoint='{endpoint}';")
    conn.execute(f"SET s3_access_key_id='{settings.s3_access_key}';")
    conn.execute(f"SET s3_secret_access_key='{settings.s3_secret_key}';")
    conn.execute("SET s3_url_style='path';")
    conn.execute("SET s3_region='us-east-1';")
    conn.execute(f"SET s3_use_ssl={'true' if use_ssl else 'false'};")

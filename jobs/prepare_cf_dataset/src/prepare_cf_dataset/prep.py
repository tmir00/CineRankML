"""DuckDB pipeline that builds CF train/validation/test datasets from snapshot Parquet."""

from __future__ import annotations

import duckdb
import logging

from typing import Callable
from datetime import UTC, datetime
from botocore.client import BaseClient
from common.storage.s3 import (
    cf_dataset_manifest_object_key,
    cf_dataset_movie_map_object_key,
    cf_dataset_test_part_object_key,
    cf_dataset_train_part_object_key,
    cf_dataset_user_map_object_key,
    cf_dataset_validation_part_object_key,
    put_json,
)
from dataclasses import dataclass, field
from common.config.settings import CfDatasetSettings
from common.schemas.cf_dataset_manifest import CfDatasetPartEntry
from prepare_cf_dataset.manifest import build_complete_manifest, utc_now
from common.storage.duckdb_s3 import configure_duckdb_s3, create_duckdb_connection
from common.storage.snapshot_reader import resolve_snapshot_id, snapshot_table_glob_uri


logger = logging.getLogger(__name__)


@dataclass
class CfDatasetPrepStats:
    """Counters and metadata collected during one CF dataset prep run."""

    snapshot_id: str
    cf_dataset_version: str
    train_row_count: int = 0
    validation_row_count: int = 0
    test_row_count: int = 0
    num_users: int = 0
    num_movies: int = 0
    train_parts: list[CfDatasetPartEntry] = field(default_factory=list)
    validation_parts: list[CfDatasetPartEntry] = field(default_factory=list)
    test_parts: list[CfDatasetPartEntry] = field(default_factory=list)
    user_id_map_key: str = ""
    movie_id_map_key: str = ""


def resolve_cf_dataset_version(settings: CfDatasetSettings) -> str:
    """
    Return the CF dataset version from settings or generate a UTC timestamp id.

    ============================ Arguments ============================
    settings: CF dataset prep configuration.

    ============================ Returns ============================
    Version string like 2026-06-25T121500Z.
    """
    if settings.cf_dataset_version:
        return settings.cf_dataset_version
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H%M%SZ")


def _s3_uri(bucket: str, object_key: str) -> str:
    """Build an s3:// URI for DuckDB COPY targets."""
    return f"s3://{bucket}/{object_key}"


def _write_partitioned_split(
    conn: duckdb.DuckDBPyConnection,
    bucket: str,
    cf_dataset_version: str,
    *,
    source_table: str,
    order_clause: str,
    part_key_builder: Callable[[str, int], str],
    part_row_limit: int,
) -> list[CfDatasetPartEntry]:
    """
    Take one DuckDB table, split it into multiple Parquet files, and upload them to MinIO.

    Do this by:
    1. Assigning each row to a part based on its ordered row number and part_row_limit.
    2. Writing each part to its own Parquet object in MinIO.

    ============================ Arguments ============================
    conn: Open DuckDB connection with httpfs configured.
    bucket: Target MinIO/S3 bucket name.
    cf_dataset_version: CF dataset version identifier.
    source_table: DuckDB table with user_idx, movie_idx, and rating columns.
    order_clause: SQL ORDER BY expression used before assigning rows to parts.
    part_key_builder: Function that builds the object key for a given version and part index.
    part_row_limit: Maximum rows per output Parquet file.

    ============================ Returns ============================
    Metadata for each written Parquet part file.
    """
    parts_table = f"{source_table}_parts"
    conn.execute(f"DROP TABLE IF EXISTS {parts_table}")
    conn.execute(
        f"""
        CREATE TABLE {parts_table} AS
        SELECT
            user_idx,
            movie_idx,
            rating,
            CAST(FLOOR((ROW_NUMBER() OVER (ORDER BY {order_clause}) - 1) / {part_row_limit}) AS INTEGER) AS part_idx
        FROM {source_table}
        """
    )

    part_count = conn.execute(
        f"SELECT COALESCE(MAX(part_idx), -1) + 1 FROM {parts_table}"
    ).fetchone()[0]
    parts: list[CfDatasetPartEntry] = []

    for part_index in range(part_count):
        object_key = part_key_builder(cf_dataset_version, part_index)
        uri = _s3_uri(bucket, object_key)
        conn.execute(
            f"""
            COPY (
                SELECT user_idx, movie_idx, rating
                FROM {parts_table}
                WHERE part_idx = {part_index}
                ORDER BY part_idx
            ) TO '{uri}' (FORMAT PARQUET)
            """
        )
        row_count = conn.execute(
            f"SELECT COUNT(*) FROM {parts_table} WHERE part_idx = {part_index}"
        ).fetchone()[0]
        parts.append(CfDatasetPartEntry(object_key=object_key, row_count=row_count))

    conn.execute(f"DROP TABLE IF EXISTS {parts_table}")
    return parts


def run_cf_dataset_prep(
    client: BaseClient,
    settings: CfDatasetSettings,
    pipeline_run_id: str,
) -> CfDatasetPrepStats:
    """
    Build a versioned CF dataset from snapshot Parquet on MinIO.

    Do this by:
    1. Resolving the source snapshot and opening a DuckDB connection.
    2. Running temporal 80/10/10 split, id mapping, and deterministic train shuffle.
    3. Writing maps, train/validation/test parts, and manifest.json last.

    ============================ Arguments ============================
    client: boto3 S3 client for snapshot resolution and manifest upload.
    settings: CF dataset prep configuration.
    pipeline_run_id: pipeline_runs.run_id for manifest provenance.

    ============================ Returns ============================
    CfDatasetPrepStats with row counts and part metadata.
    """
    snapshot_id = resolve_snapshot_id(client, settings.s3_bucket, settings.snapshot_id)
    cf_dataset_version = resolve_cf_dataset_version(settings)
    created_at = utc_now()
    stats = CfDatasetPrepStats(snapshot_id=snapshot_id, cf_dataset_version=cf_dataset_version)

    ratings_glob = snapshot_table_glob_uri(settings.s3_bucket, snapshot_id, "ratings_events")
    catalog_glob = snapshot_table_glob_uri(settings.s3_bucket, snapshot_id, "catalog_movies")

    conn = create_duckdb_connection()
    configure_duckdb_s3(conn, settings)

    train_fraction = settings.train_fraction
    validation_fraction = settings.validation_fraction
    validation_upper = train_fraction + validation_fraction

    try:
        conn.execute(
            f"""
            CREATE TABLE ratings AS
            SELECT user_id, movie_id, rating, rating_timestamp, id
            FROM read_parquet('{ratings_glob}')
            """
        )
        conn.execute(
            f"""
            CREATE TABLE catalog AS
            SELECT movie_id
            FROM read_parquet('{catalog_glob}')
            """
        )

        raw_rating_count = conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]
        catalog_movie_count = conn.execute("SELECT COUNT(*) FROM catalog").fetchone()[0]
        logger.info(
            "Loaded snapshot inputs",
            extra={
                "snapshot_id": snapshot_id,
                "ratings_rows": raw_rating_count,
                "catalog_movies": catalog_movie_count,
            },
        )

        # Stage 2: temporal train/validation/test split on rating_timestamp, id.
        conn.execute(
            f"""
            CREATE TABLE split_ratings AS
            WITH numbered AS (
                SELECT
                    user_id,
                    movie_id,
                    rating,
                    rating_timestamp,
                    id,
                    ROW_NUMBER() OVER (ORDER BY rating_timestamp, id) AS rn,
                    COUNT(*) OVER () AS total
                FROM ratings
            )
            SELECT
                user_id,
                movie_id,
                rating,
                rating_timestamp,
                id,
                CASE
                    WHEN rn <= CAST(FLOOR(total * {train_fraction}) AS BIGINT) THEN 'train'
                    WHEN rn <= CAST(FLOOR(total * {validation_upper}) AS BIGINT) THEN 'validation'
                    ELSE 'test'
                END AS split
            FROM numbered
            """
        )

        conn.execute(
            """
            CREATE TABLE user_id_map AS
            SELECT
                user_id,
                CAST(ROW_NUMBER() OVER (ORDER BY user_id) - 1 AS INTEGER) AS user_idx
            FROM (
                SELECT DISTINCT user_id
                FROM split_ratings
                WHERE split = 'train'
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE movie_id_map AS
            SELECT
                movie_id,
                CAST(ROW_NUMBER() OVER (ORDER BY movie_id) - 1 AS INTEGER) AS movie_idx
            FROM catalog
            """
        )

        stats.num_users = conn.execute("SELECT COUNT(*) FROM user_id_map").fetchone()[0]
        stats.num_movies = conn.execute("SELECT COUNT(*) FROM movie_id_map").fetchone()[0]

        conn.execute(
            """
            CREATE TABLE train_mapped AS
            SELECT
                u.user_idx,
                m.movie_idx,
                s.rating,
                s.user_id,
                s.movie_id,
                s.rating_timestamp,
                s.id
            FROM split_ratings s
            INNER JOIN user_id_map u ON s.user_id = u.user_id
            INNER JOIN movie_id_map m ON s.movie_id = m.movie_id
            WHERE s.split = 'train'
            """
        )
        conn.execute(
            """
            CREATE TABLE validation_mapped AS
            SELECT
                u.user_idx,
                m.movie_idx,
                s.rating,
                s.rating_timestamp,
                s.id
            FROM split_ratings s
            INNER JOIN user_id_map u ON s.user_id = u.user_id
            INNER JOIN movie_id_map m ON s.movie_id = m.movie_id
            WHERE s.split = 'validation'
            """
        )
        conn.execute(
            """
            CREATE TABLE test_mapped AS
            SELECT
                u.user_idx,
                m.movie_idx,
                s.rating,
                s.rating_timestamp,
                s.id
            FROM split_ratings s
            INNER JOIN user_id_map u ON s.user_id = u.user_id
            INNER JOIN movie_id_map m ON s.movie_id = m.movie_id
            WHERE s.split = 'test'
            """
        )

        mapped_train_count = conn.execute("SELECT COUNT(*) FROM train_mapped").fetchone()[0]
        mapped_validation_count = conn.execute("SELECT COUNT(*) FROM validation_mapped").fetchone()[0]
        mapped_test_count = conn.execute("SELECT COUNT(*) FROM test_mapped").fetchone()[0]
        dropped_count = raw_rating_count - mapped_train_count - mapped_validation_count - mapped_test_count
        if dropped_count > 0:
            logger.warning(
                "Dropped ratings that could not be mapped to catalog movies or train users",
                extra={
                    "snapshot_id": snapshot_id,
                    "dropped_rows": dropped_count,
                },
            )

        conn.execute(
            f"""
            CREATE TABLE train_shuffled AS
            SELECT
                user_idx,
                movie_idx,
                rating,
                hash(
                    CAST(user_id AS VARCHAR) || '-' ||
                    CAST(movie_id AS VARCHAR) || '-' ||
                    CAST(rating_timestamp AS VARCHAR) || '-' ||
                    '{settings.cf_shuffle_seed}'
                ) AS shuffle_key
            FROM train_mapped
            ORDER BY shuffle_key
            """
        )

        stats.train_row_count = mapped_train_count
        stats.validation_row_count = mapped_validation_count
        stats.test_row_count = mapped_test_count

        user_map_key = cf_dataset_user_map_object_key(cf_dataset_version)
        movie_map_key = cf_dataset_movie_map_object_key(cf_dataset_version)
        stats.user_id_map_key = user_map_key
        stats.movie_id_map_key = movie_map_key

        conn.execute(
            f"COPY (SELECT user_id, user_idx FROM user_id_map ORDER BY user_idx) "
            f"TO '{_s3_uri(settings.s3_bucket, user_map_key)}' (FORMAT PARQUET)"
        )
        conn.execute(
            f"COPY (SELECT movie_id, movie_idx FROM movie_id_map ORDER BY movie_idx) "
            f"TO '{_s3_uri(settings.s3_bucket, movie_map_key)}' (FORMAT PARQUET)"
        )

        stats.train_parts = _write_partitioned_split(
            conn,
            settings.s3_bucket,
            cf_dataset_version,
            source_table="train_shuffled",
            order_clause="shuffle_key",
            part_key_builder=cf_dataset_train_part_object_key,
            part_row_limit=settings.cf_part_row_limit,
        )
        stats.validation_parts = _write_partitioned_split(
            conn,
            settings.s3_bucket,
            cf_dataset_version,
            source_table="validation_mapped",
            order_clause="rating_timestamp, id",
            part_key_builder=cf_dataset_validation_part_object_key,
            part_row_limit=settings.cf_part_row_limit,
        )
        stats.test_parts = _write_partitioned_split(
            conn,
            settings.s3_bucket,
            cf_dataset_version,
            source_table="test_mapped",
            order_clause="rating_timestamp, id",
            part_key_builder=cf_dataset_test_part_object_key,
            part_row_limit=settings.cf_part_row_limit,
        )

        finished_at = utc_now()
        manifest = build_complete_manifest(
            snapshot_id=snapshot_id,
            cf_dataset_version=cf_dataset_version,
            pipeline_run_id=pipeline_run_id,
            created_at=created_at,
            finished_at=finished_at,
            train_row_count=stats.train_row_count,
            validation_row_count=stats.validation_row_count,
            test_row_count=stats.test_row_count,
            num_users=stats.num_users,
            num_movies=stats.num_movies,
            train_fraction=settings.train_fraction,
            validation_fraction=settings.validation_fraction,
            test_fraction=settings.test_fraction,
            shuffle_seed=settings.cf_shuffle_seed,
            user_id_map_key=user_map_key,
            movie_id_map_key=movie_map_key,
            train_parts=stats.train_parts,
            validation_parts=stats.validation_parts,
            test_parts=stats.test_parts,
        )
        manifest_key = cf_dataset_manifest_object_key(cf_dataset_version)
        put_json(client, settings.s3_bucket, manifest_key, manifest.model_dump(mode="json"))

        logger.info(
            "CF dataset prep complete",
            extra={
                "snapshot_id": snapshot_id,
                "cf_dataset_version": cf_dataset_version,
                "train_row_count": stats.train_row_count,
                "validation_row_count": stats.validation_row_count,
                "test_row_count": stats.test_row_count,
                "num_users": stats.num_users,
                "num_movies": stats.num_movies,
                "manifest_key": manifest_key,
            },
        )
        return stats
    finally:
        conn.close()

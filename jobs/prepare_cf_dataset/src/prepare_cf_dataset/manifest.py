"""Build CF dataset manifest payloads."""

from __future__ import annotations

from datetime import UTC, datetime

from common.schemas.cf_dataset_manifest import CfDatasetManifest, CfDatasetPartEntry


def build_complete_manifest(
    *,
    snapshot_id: str,
    cf_dataset_version: str,
    pipeline_run_id: str,
    created_at: datetime,
    finished_at: datetime,
    train_row_count: int,
    validation_row_count: int,
    test_row_count: int,
    num_users: int,
    num_movies: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    shuffle_seed: int,
    user_id_map_key: str,
    movie_id_map_key: str,
    train_parts: list[CfDatasetPartEntry],
    validation_parts: list[CfDatasetPartEntry],
    test_parts: list[CfDatasetPartEntry],
) -> CfDatasetManifest:
    """
    Build a complete CF dataset manifest after all parts are uploaded.

    ============================ Arguments ============================
    snapshot_id: Source snapshot identifier.
    cf_dataset_version: Version id for this CF dataset.
    pipeline_run_id: pipeline_runs.run_id for this job.
    created_at: When prep started.
    finished_at: When all parts and metadata were ready.
    train_row_count: Rows written to train/.
    validation_row_count: Rows written to validation/.
    test_row_count: Rows written to test/.
    num_users: Size of user_id_map.
    num_movies: Size of movie_id_map.
    train_fraction: Temporal train split ratio.
    validation_fraction: Temporal validation split ratio.
    test_fraction: Temporal test split ratio (locked for hybrid eval).
    shuffle_seed: Seed used for deterministic train shuffle.
    user_id_map_key: S3 object key for user_id_map.parquet.
    movie_id_map_key: S3 object key for movie_id_map.parquet.
    train_parts: Metadata for each train part file.
    validation_parts: Metadata for each validation part file.
    test_parts: Metadata for each test part file.

    ============================ Returns ============================
    A CfDatasetManifest with status complete.
    """
    return CfDatasetManifest(
        snapshot_id=snapshot_id,
        cf_dataset_version=cf_dataset_version,
        status="complete",
        train_row_count=train_row_count,
        validation_row_count=validation_row_count,
        test_row_count=test_row_count,
        num_users=num_users,
        num_movies=num_movies,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        shuffle_seed=shuffle_seed,
        created_at=created_at,
        finished_at=finished_at,
        pipeline_run_id=pipeline_run_id,
        user_id_map_key=user_id_map_key,
        movie_id_map_key=movie_id_map_key,
        train_parts=train_parts,
        validation_parts=validation_parts,
        test_parts=test_parts,
    )


def utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(tz=UTC)

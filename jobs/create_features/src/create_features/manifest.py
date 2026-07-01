"""Build hybrid ranker dataset manifest payloads."""

from __future__ import annotations

from datetime import UTC, datetime

from common.features.normalization import MetadataNormalizationStats
from common.features.schema import FEATURE_SCHEMA_VERSION, INPUT_DIM
from common.schemas.hybrid_ranker_dataset_manifest import HybridRankerDatasetManifest, HybridRankerPartEntry


def build_complete_hybrid_manifest(
    *,
    dataset_version: str,
    snapshot_id: str,
    cf_dataset_version: str,
    cf_version: str,
    content_embedding_version: str,
    feature_schema_version: str,
    pipeline_run_id: str,
    created_at: datetime,
    finished_at: datetime,
    train_row_count: int,
    validation_row_count: int,
    test_row_count: int,
    metadata_normalization: MetadataNormalizationStats,
    train_parts: list[HybridRankerPartEntry],
    validation_parts: list[HybridRankerPartEntry],
    test_parts: list[HybridRankerPartEntry],
) -> HybridRankerDatasetManifest:
    """
    Build a complete hybrid ranker dataset manifest after all parts are uploaded.

    ============================ Arguments ============================
    dataset_version: Version id for this hybrid feature dataset.
    snapshot_id: Source snapshot identifier.
    cf_dataset_version: CF dataset version used for split membership.
    cf_version: CF artifact version used for movie CF embeddings.
    content_embedding_version: Content embedding version from the snapshot.
    feature_schema_version: Feature layout version (defaults to hybrid-v1).
    pipeline_run_id: pipeline_runs.run_id for this job.
    created_at: When feature generation started.
    finished_at: When all parts and metadata were ready.
    train_row_count: Rows written to train/.
    validation_row_count: Rows written to validation/.
    test_row_count: Rows written to test/.
    metadata_normalization: Train-fit metadata normalization stats.
    train_parts: Metadata for each train part file.
    validation_parts: Metadata for each validation part file.
    test_parts: Metadata for each test part file.

    ============================ Returns ============================
    A HybridRankerDatasetManifest with status complete.
    """
    return HybridRankerDatasetManifest(
        dataset_version=dataset_version,
        snapshot_id=snapshot_id,
        cf_dataset_version=cf_dataset_version,
        cf_version=cf_version,
        content_embedding_version=content_embedding_version,
        feature_schema_version=feature_schema_version or FEATURE_SCHEMA_VERSION,
        input_dim=INPUT_DIM,
        status="complete",
        train_row_count=train_row_count,
        validation_row_count=validation_row_count,
        test_row_count=test_row_count,
        metadata_normalization=metadata_normalization,
        created_at=created_at,
        finished_at=finished_at,
        pipeline_run_id=pipeline_run_id,
        train_parts=train_parts,
        validation_parts=validation_parts,
        test_parts=test_parts,
    )


def utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(tz=UTC)

"""Pydantic models for hybrid ranker dataset manifest.json written to MinIO."""

from __future__ import annotations

from typing import Literal
from datetime import datetime
from pydantic import BaseModel
from common.features.normalization import MetadataNormalizationStats


class HybridRankerPartEntry(BaseModel):
    """One Parquet part file for train, validation, or test."""

    object_key: str
    row_count: int


class HybridRankerDatasetManifest(BaseModel):
    """Complete hybrid ranker dataset manifest; consumers require status=complete."""

    dataset_version: str
    snapshot_id: str
    cf_dataset_version: str
    cf_version: str
    content_embedding_version: str
    feature_schema_version: str
    input_dim: int
    status: Literal["complete", "failed"]
    train_row_count: int
    validation_row_count: int
    test_row_count: int
    metadata_normalization: MetadataNormalizationStats
    created_at: datetime
    finished_at: datetime | None = None
    pipeline_run_id: str
    train_parts: list[HybridRankerPartEntry]
    validation_parts: list[HybridRankerPartEntry]
    test_parts: list[HybridRankerPartEntry]

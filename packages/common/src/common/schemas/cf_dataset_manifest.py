"""Pydantic models for CF dataset manifest.json written to MinIO."""

from __future__ import annotations

from typing import Literal
from datetime import datetime
from pydantic import BaseModel


class CfDatasetPartEntry(BaseModel):
    """One Parquet part file for train or holdout."""

    object_key: str
    row_count: int


class CfDatasetManifest(BaseModel):
    """Complete CF dataset manifest; consumers require status=complete."""

    snapshot_id: str
    cf_dataset_version: str
    status: Literal["complete", "failed"]
    train_row_count: int
    holdout_row_count: int
    num_users: int
    num_movies: int
    train_fraction: float
    shuffle_seed: int
    created_at: datetime
    finished_at: datetime | None = None
    pipeline_run_id: str
    user_id_map_key: str
    movie_id_map_key: str
    train_parts: list[CfDatasetPartEntry]
    holdout_parts: list[CfDatasetPartEntry]

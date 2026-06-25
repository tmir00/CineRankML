"""Pydantic models for snapshot manifest.json written to MinIO."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class SnapshotPartEntry(BaseModel):
    """ This describes one Parquet part file for a snapshot table. """

    object_key: str
    row_count: int
    sha256: str


class SnapshotTableEntry(BaseModel):
    """ This describes one exported table entry. """

    prefix: str
    row_count: int
    part_count: int
    parts: list[SnapshotPartEntry]


class SnapshotManifest(BaseModel):
    """ This describes the whole snapshot; consumers require status=complete."""

    snapshot_id: str
    status: Literal["complete", "failed"]
    created_at: datetime
    finished_at: datetime
    pipeline_run_id: str
    tables: dict[str, SnapshotTableEntry]

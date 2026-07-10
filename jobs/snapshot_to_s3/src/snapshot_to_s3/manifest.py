"""Build snapshot manifest payloads."""

from __future__ import annotations

from datetime import UTC, datetime
from common.schemas.snapshot_manifest import SnapshotManifest, SnapshotTableEntry


def build_complete_manifest(snapshot_id: str, pipeline_run_id: str, created_at: datetime, 
                            finished_at: datetime, tables: dict[str, SnapshotTableEntry]) -> SnapshotManifest:
    """
    Build a complete snapshot manifest after all parts are uploaded.

    ============================ Arguments ============================
    snapshot_id: UTC snapshot identifier.
    pipeline_run_id: pipeline_runs.run_id for this job.
    created_at: When the export started.
    finished_at: When all parts and metadata were ready.
    tables: Per-table export metadata including part files.

    ============================ Returns ============================
    A SnapshotManifest with status complete.
    """
    return SnapshotManifest(
        snapshot_id=snapshot_id,
        status="complete",
        created_at=created_at,
        finished_at=finished_at,
        pipeline_run_id=pipeline_run_id,
        tables=tables,
    )


def utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(tz=UTC)

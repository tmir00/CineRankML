"""Export Postgres tables to MinIO as partitioned Parquet snapshots."""

from __future__ import annotations

import hashlib
import logging
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq

from pathlib import Path

from common.storage.s3 import (
    ensure_bucket_exists,
    manifest_object_key,
    part_object_key,
    put_json,
    table_prefix,
    upload_file,
)

from datetime import UTC, datetime
from botocore.client import BaseClient
from dataclasses import dataclass, field
from sqlalchemy.orm import Session, sessionmaker
from common.config.settings import SnapshotSettings
from snapshot_to_s3.manifest import build_complete_manifest, utc_now
from common.schemas.snapshot_manifest import SnapshotPartEntry, SnapshotTableEntry
from common.db.repositories.snapshot_export import EXPORT_TABLE_ORDER, TABLE_EXPORT_ITERATORS


logger = logging.getLogger(__name__)


@dataclass
class SnapshotStats:
    """ These counters are collected during one snapshot export run. """

    snapshot_id: str
    table_row_counts: dict[str, int] = field(default_factory=dict)
    table_part_counts: dict[str, int] = field(default_factory=dict)
    upload_failures: int = 0


def _sha256_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of one file."""
    # Initialize the SHA-256 digest.
    digest = hashlib.sha256()

    # Open the file and read it in chunks.
    with path.open("rb") as file_handle:
        # Read the file in chunks of 1MB.
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            # Update the digest with the chunk.
            digest.update(chunk)

    return digest.hexdigest()


def _write_batch_parquet(batch: list[dict], path: Path) -> None:
    """ Write one batch of row dicts to a single Parquet file. """
    # Convert the batch to a PyArrow table.
    table = pa.Table.from_pylist(batch)
    # Write the table to the file.
    pq.write_table(table, path)


def _export_table(session: Session, client: BaseClient, settings: SnapshotSettings, 
                snapshot_id: str, table_name: str, temp_dir: Path, stats: SnapshotStats) -> SnapshotTableEntry:
    """
    Export one Postgres table as partitioned Parquet parts in MinIO.

    Do this by:
    1. Reading rows from Postgres in batches.
    2. Writing each batch to a local part file and uploading it.
    3. Recording per-part metadata for the manifest.

    ============================ Arguments ============================
    session: SQLAlchemy session for reading Postgres.
    client: boto3 S3 client.
    settings: Snapshot job configuration.
    snapshot_id: UTC snapshot identifier.
    table_name: Postgres table to export.
    temp_dir: Directory for temporary part files.

    ============================ Returns ============================
    SnapshotTableEntry with part metadata and total row count.
    """
    # Get the iterator for the table.
    iterator = TABLE_EXPORT_ITERATORS[table_name]
    # Get the prefix for the table.
    prefix = table_prefix(snapshot_id, table_name)
    
    # Initialize the list of parts.
    parts: list[SnapshotPartEntry] = []
    part_index = 0
    table_row_count = 0

    # Loop through the batches.
    for batch in iterator(session, settings.snapshot_batch_size):
        # If there are no more batches, break out of the loop.
        if not batch:
            continue

        # If there are batches, get the local path for the part.
        local_path = temp_dir / f"{table_name}-part-{part_index:05d}.parquet"
        # Construct the object key for the part.
        object_key = part_object_key(snapshot_id, table_name, part_index)

        # Write the batch to a local Parquet part file.
        _write_batch_parquet(batch, local_path)
        checksum = _sha256_file(local_path)

        # Upload the part to S3.
        try:
            upload_file(client, settings.s3_bucket, object_key, local_path)
        except Exception:
            # If the upload fails, increment the upload failures counter.
            stats.upload_failures += 1
            raise
        finally:
            # Delete the local part file.
            local_path.unlink(missing_ok=True)

        # Get the row count for the batch.
        row_count = len(batch)

        # Append the part to the list of parts.
        parts.append(
            SnapshotPartEntry(
                object_key=object_key,
                row_count=row_count,
                sha256=checksum,
            )
        )
        # Increment the part index.
        part_index += 1
        table_row_count += row_count

        # Log the progress every 10 parts.
        if part_index % 10 == 0:
            logger.info(
                f"Exported {part_index} parts for {table_name} with {table_row_count} rows",
                extra={
                    "snapshot_id": snapshot_id,
                    "table": table_name,
                    "parts_written": part_index,
                    "rows_exported": table_row_count,
                },
            )

    # Return the table entry.
    return SnapshotTableEntry(
        prefix=prefix,
        row_count=table_row_count,
        part_count=len(parts),
        parts=parts,
    )


def resolve_snapshot_id(settings: SnapshotSettings) -> str:
    """
    Return the snapshot id from settings or generate a UTC timestamp id.

    ============================ Arguments ============================
    settings: Snapshot job configuration.

    ============================ Returns ============================
    Snapshot id string like 2026-06-25T120000Z.
    """
    if settings.snapshot_id:
        return settings.snapshot_id
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H%M%SZ")


def run_snapshot_export(session_factory: sessionmaker[Session], client: BaseClient, 
                        settings: SnapshotSettings, pipeline_run_id: str) -> SnapshotStats:
    """
    Export all snapshot tables to MinIO and write manifest.json last.

    Do this by:
    1. Resolving snapshot_id and exporting each table as part files.
    2. Building a complete manifest from per-table metadata.
    3. Uploading manifest.json after all parts succeed.

    ============================ Arguments ============================
    session_factory: Factory for short-lived SQLAlchemy sessions.
    client: boto3 S3 client.
    settings: Snapshot job configuration.
    pipeline_run_id: pipeline_runs.run_id for manifest provenance.

    ============================ Returns ============================
    SnapshotStats with per-table row and part counts.
    """
    # Resolve the snapshot id and get the creation time.
    snapshot_id = resolve_snapshot_id(settings)
    created_at = utc_now()
    # Initialize the snapshot stats.
    stats = SnapshotStats(snapshot_id=snapshot_id)
    # Initialize the dictionary of table entries.
    table_entries: dict[str, SnapshotTableEntry] = {}

    # Ensure the bucket exists.
    ensure_bucket_exists(client, settings.s3_bucket)

    # Create a temporary directory for the parts.
    with tempfile.TemporaryDirectory(prefix="snapshot-export-") as temp_dir_path:
        temp_dir = Path(temp_dir_path)
        
        # Create a new SQLAlchemy session to read from the database.
        session = session_factory()

        try:
            # Loop through the tables to export.
            for table_name in EXPORT_TABLE_ORDER:
                
                # Log the progress.
                logger.info(
                    "Exporting snapshot table",
                    extra={"snapshot_id": snapshot_id, "table": table_name},
                )

                # Export the table.
                entry = _export_table(
                    session,
                    client,
                    settings,
                    snapshot_id,
                    table_name,
                    temp_dir,
                    stats,
                )

                # Add the table entry to the dictionary.
                table_entries[table_name] = entry
                # Add the row count to the stats.
                stats.table_row_counts[table_name] = entry.row_count
                # Add the part count to the stats.
                stats.table_part_counts[table_name] = entry.part_count
        
        finally:
            session.close()

    # Get the finished time.
    finished_at = utc_now()
    # Build the complete manifest.
    manifest = build_complete_manifest(
        snapshot_id=snapshot_id,
        pipeline_run_id=pipeline_run_id,
        created_at=created_at,
        finished_at=finished_at,
        tables=table_entries,
    )

    # Construct the object key for the manifest.
    manifest_key = manifest_object_key(snapshot_id)
    # Upload the manifest to S3.
    try:
        put_json(client, settings.s3_bucket, manifest_key, manifest.model_dump(mode="json"))
    except Exception:
        # If the upload fails, increment the upload failures counter.
        stats.upload_failures += 1
        logger.exception(
            "Failed to upload snapshot manifest",
            extra={"snapshot_id": snapshot_id, "manifest_key": manifest_key},
        )
        raise


    logger.info(
        "Snapshot export complete",
        extra={
            "snapshot_id": snapshot_id,
            "manifest_key": manifest_key,
            "table_row_counts": stats.table_row_counts,
            "table_part_counts": stats.table_part_counts,
        },
    )

    return stats

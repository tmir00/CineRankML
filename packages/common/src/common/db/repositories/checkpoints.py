"""Read and write CSV producer checkpoints in Postgres."""

from __future__ import annotations

from sqlalchemy import select
from dataclasses import dataclass
from datetime import UTC, datetime
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert
from common.db.models.checkpoints import CsvIngestionCheckpoint


@dataclass
class CsvCheckpoint:
    """ Resume state for one CSV source file. """
    last_row_number: int
    last_event_id: str | None


def get_checkpoint(session: Session, source_file: str) -> CsvCheckpoint | None:
    """
    Load the saved ingestion progress for one CSV source file.

    Do this by:
    1. Looking up the row keyed by source_file.
    2. Returning the row numbers and last event id when a row exists.

    ============================ Arguments ============================
    session: An open SQLAlchemy session.
    source_file: Stable logical file name, e.g. ratings.csv.

    ============================ Returns ============================
    The saved checkpoint, or None when this file has not been ingested yet.
    """
    # Execute the select statement.
    row = session.execute(
        select(CsvIngestionCheckpoint).where(CsvIngestionCheckpoint.source_file == source_file)
    ).scalar_one_or_none()

    # Return the row if it exists, otherwise return None.
    if row is None:
        return None

    # Return the checkpoint.
    return CsvCheckpoint(
        last_row_number=row.last_row_number,
        last_event_id=row.last_event_id,
    )


def upsert_checkpoint(session: Session, source_file: str, last_row_number: int, last_event_id: str) -> None:
    """
    Save ingestion progress for one CSV source file.

    Do this by:
    1. Inserting a new checkpoint row when source_file is new.
    2. Updating last_row_number, last_event_id, and updated_at when the row already exists.

    ============================ Arguments ============================
    session: An open SQLAlchemy session inside a transaction.
    source_file: Stable logical file name, e.g. ratings.csv.
    last_row_number: How many data rows have been published so far.
    last_event_id: The event_id string of the last published event.
    """
    # Get the current timestamp.
    now = datetime.now(tz=UTC)
    
    # Create the insert statement.
    stmt = (
        insert(CsvIngestionCheckpoint)
        .values(
            source_file=source_file,
            last_row_number=last_row_number,
            last_event_id=last_event_id,
            updated_at=now,
        )
        .on_conflict_do_update(
            index_elements=["source_file"],
            set_={
                "last_row_number": last_row_number,
                "last_event_id": last_event_id,
                "updated_at": now,
            },
        )
    )
    
    # Execute the statement.
    session.execute(stmt)

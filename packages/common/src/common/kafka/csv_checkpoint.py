"""Postgres-backed CSV producer checkpoint helpers."""

from __future__ import annotations

import logging

from common.db.session import get_session_factory
from common.db.repositories.checkpoints import CsvCheckpoint, get_checkpoint, upsert_checkpoint


logger = logging.getLogger(__name__)


def read_csv_checkpoint(source_file: str, default_row: int = 0) -> CsvCheckpoint:
    """
    Load saved CSV ingestion progress from Postgres.

    Do this by:
    1. Opening a short database session.
    2. Reading the checkpoint row for source_file when it exists.
    3. Returning the default row when no checkpoint has been saved yet.

    ============================ Arguments ============================
    source_file: Stable logical file name, e.g. ratings.csv.
    default_row: Row count to use when no checkpoint exists yet.

    ============================ Returns ============================
    The resume state with last_row_number and optional last_event_id.
    """
    session_factory = get_session_factory()
    session = session_factory()
    try:
        saved = get_checkpoint(session, source_file)
        if saved is None:
            return CsvCheckpoint(last_row_number=default_row, last_event_id=None)
        return saved
    finally:
        session.close()


def save_csv_checkpoint(source_file: str, last_row_number: int, last_event_id: str) -> None:
    """
    Persist CSV ingestion progress to Postgres.

    Do this by:
    1. Opening a database session and starting a transaction.
    2. Inserting or updating the checkpoint row for source_file.
    3. Committing when the write succeeds.

    ============================ Arguments ============================
    source_file: Stable logical file name, e.g. ratings.csv.
    last_row_number: How many data rows have been published so far.
    last_event_id: The event_id string of the last published event.
    """
    session_factory = get_session_factory()
    session = session_factory()
    try:
        upsert_checkpoint(session, source_file, last_row_number, last_event_id)
        session.commit()
    except Exception:
        session.rollback()
        logger.exception(
            "Failed to save CSV ingestion checkpoint",
            extra={"source_file": source_file, "last_row_number": last_row_number},
        )
        raise
    finally:
        session.close()

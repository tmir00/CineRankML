"""Pipeline run tracking in pipeline_runs."""

from __future__ import annotations

from datetime import UTC, datetime
from sqlalchemy.orm import Session
from common.db.models.pipeline import PipelineRun


def start_pipeline_run(session: Session, run_id: str, job_name: str) -> None:
    """
    Insert a new pipeline_runs row when a batch job starts.

    ============================ Arguments ============================
    session: An open SQLAlchemy session inside a transaction.
    run_id: Unique identifier for this job run (e.g. UUID string).
    job_name: Logical job name, e.g. seed_catalog or tmdb_enrichment.
    """
    now = datetime.now(tz=UTC)
    session.add(
        PipelineRun(
            run_id=run_id,
            job_name=job_name,
            status="running",
            started_at=now,
        )
    )


def finish_pipeline_run(session: Session, run_id: str, status: str, records_processed: int | None = None, \
                        records_failed: int | None = None, error_message: str | None = None) -> None:
    """
    Update a pipeline_runs row when a batch job finishes.

    ============================ Arguments ============================
    session: An open SQLAlchemy session inside a transaction.
    run_id: The run_id passed to start_pipeline_run.
    status: Final status, e.g. success or failure.
    records_processed: Optional count of records handled successfully.
    records_failed: Optional count of records that failed.
    error_message: Optional error text when status is failure.
    """
    now = datetime.now(tz=UTC)
    # Get the pipeline run row by the run_id.
    run = session.get(PipelineRun, run_id)
    if run is None:
        return

    # Update the pipeline run row with the new status, finished at, records processed, records failed, and error message.
    run.status = status
    run.finished_at = now
    run.records_processed = records_processed
    run.records_failed = records_failed
    run.error_message = error_message

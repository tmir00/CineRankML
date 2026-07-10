"""Read and update online recommendation experiment split state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from common.db.models.recommendations import RecommendationExperiment, RecommendationRating


@dataclass(frozen=True)
class ExperimentState:
    """ Active experiment row used by the recommender API at request time. """

    experiment_id: str
    main_model_version: str
    candidate_model_version: str | None
    main_split_fraction: float
    candidate_split_fraction: float
    status: str


def get_active_experiment(session: Session, experiment_id: str) -> ExperimentState | None:
    """
    Load one active experiment row by id.

    ============================ Arguments ============================
    session: An open SQLAlchemy session.
    experiment_id: Primary key for the experiment.

    ============================ Returns ============================
    The experiment state, or None when no active row exists.
    """
    row = session.get(RecommendationExperiment, experiment_id)
    if row is None or row.status != "active":
        return None
    return _to_state(row)


def get_or_create_active_experiment(*, session: Session, experiment_id: str, main_model_version: str, candidate_model_version: str | None, 
                                        initial_main_split: float, initial_candidate_split: float) -> ExperimentState:
    """
    Return the active experiment row, creating it with default splits when missing.

    Do this by:
    1. Loading an existing active row when present.
    2. Syncing candidate version when MLflow reports a new challenger.
    3. Inserting a new row with the configured 70/30 starting split otherwise.

    ============================ Arguments ============================
    session: An open SQLAlchemy session.
    experiment_id: Primary key for the experiment.
    main_model_version: Current main MinIO model version.
    candidate_model_version: Current candidate version, if any.
    initial_main_split: Starting fraction for main (e.g. 0.70).
    initial_candidate_split: Starting fraction for candidate (e.g. 0.30).

    ============================ Returns ============================
    The active experiment state row.
    """
    existing = session.get(RecommendationExperiment, experiment_id)
    now = datetime.now(tz=UTC)

    # If an active experiment already exists, keep using it.
    # This preserves the current online traffic split unless MLflow reports a new candidate model.
    if existing is not None and existing.status == "active":
        # If MLflow now points the candidate alias to a different model version,
        # treat that as a new challenger and restart the experiment split.
        if (
            candidate_model_version is not None
            and candidate_model_version != existing.candidate_model_version
        ):
            # Reset traffic allocation for the new main/candidate comparison.
            # Example: main gets 70% of requests, candidate gets 30%.
            existing.candidate_model_version = candidate_model_version
            existing.main_model_version = main_model_version
            existing.main_split_fraction = initial_main_split
            existing.candidate_split_fraction = initial_candidate_split
            existing.updated_at = now
            session.flush()
        return _to_state(existing)

    # If a row exists but is not active, retire it before creating a fresh active row.
    # This keeps old experiment history while ensuring only the new row is used by the API.
    if existing is not None:
        existing.status = "retired"
        existing.updated_at = now

    # No active experiment exists, so create the default experiment state.
    # This is the row the recommender API will use to decide main vs candidate traffic.
    row = RecommendationExperiment(
        experiment_id=experiment_id,
        main_model_version=main_model_version,
        candidate_model_version=candidate_model_version,
        main_split_fraction=initial_main_split,
        candidate_split_fraction=initial_candidate_split,
        status="active",
        created_at=now,
        updated_at=now,
    )
    session.add(row)
    session.flush()
    return _to_state(row)


def update_split_fractions(*, session: Session, experiment_id: str, main_split_fraction: float, 
                                candidate_split_fraction: float) -> ExperimentState:
    """
    Persist new main/candidate split fractions for one experiment.

    ============================ Arguments ============================
    session: An open SQLAlchemy session.
    experiment_id: Primary key for the experiment.
    main_split_fraction: Updated fraction for main recommendations.
    candidate_split_fraction: Updated fraction for candidate recommendations.

    ============================ Returns ============================
    The updated experiment state.
    """
    # Load the experiment row whose traffic split we want to change.
    row = session.get(RecommendationExperiment, experiment_id)
    if row is None:
        raise ValueError(f"Experiment {experiment_id} not found")

    # Update the percentage of recommendation traffic sent to each model.
    # Example:
    # main_split_fraction = 0.80 means 80% of requests use the main model.
    # candidate_split_fraction = 0.20 means 20% of requests use the candidate model.
    row.main_split_fraction = main_split_fraction
    row.candidate_split_fraction = candidate_split_fraction
    row.updated_at = datetime.now(tz=UTC)

    # Flush so the update is sent to Postgres inside the current transaction.
    # The caller can still decide when to commit/rollback the transaction.
    session.flush()

    return _to_state(row)


def record_promotion(*, session: Session, experiment_id: str, new_main_version: str) -> ExperimentState:
    """
    Record in database in the table recommendation_experiments that the candidate model was promoted to main.

    Do this by:
    1. Moving the candidate version into main_model_version.
    2. Clearing the candidate version and resetting split to 100% main.

    ============================ Arguments ============================
    session: An open SQLAlchemy session.
    experiment_id: Primary key for the experiment.
    new_main_version: MinIO model version that is now main.

    ============================ Returns ============================
    The updated experiment state.
    """
    # Load the experiment row to update.
    row = session.get(RecommendationExperiment, experiment_id)
    if row is None:
        raise ValueError(f"Experiment {experiment_id} not found")

    # Update the experiment row to reflect the promotion.
    now = datetime.now(tz=UTC)
    row.main_model_version = new_main_version
    row.candidate_model_version = None
    row.main_split_fraction = 1.0
    row.candidate_split_fraction = 0.0
    row.status = "active"
    row.updated_at = now
    session.flush()
    return _to_state(row)


def fetch_candidate_rating_stats(session: Session, *, experiment_id: str, model_role: str = "candidate") -> tuple[int, float]:
    """
    Count ratings and average score for one model role in an experiment.

    ============================ Arguments ============================
    session: An open SQLAlchemy session.
    experiment_id: Experiment to filter on.
    model_role: Model role label (main or candidate).

    ============================ Returns ============================
    A tuple of (rating_count, average_rating). Average is 0.0 when count is 0.
    """
    count, average = session.execute(
        select(
            func.count(RecommendationRating.id),
            func.coalesce(func.avg(RecommendationRating.rating), 0.0),
        ).where(
            RecommendationRating.experiment_id == experiment_id,
            RecommendationRating.model_role == model_role,
        )
    ).one()
    return int(count), float(average)


def _to_state(row: RecommendationExperiment) -> ExperimentState:
    return ExperimentState(
        experiment_id=row.experiment_id,
        main_model_version=row.main_model_version,
        candidate_model_version=row.candidate_model_version,
        main_split_fraction=row.main_split_fraction,
        candidate_split_fraction=row.candidate_split_fraction,
        status=row.status,
    )

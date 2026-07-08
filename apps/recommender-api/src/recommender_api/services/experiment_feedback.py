"""Handle online experiment split updates after recommendation ratings."""

from __future__ import annotations

import logging

from common.recommendation.split_policy import SplitFractions, adjust_split_after_rating
from common.db.repositories.experiments import (
    fetch_candidate_rating_stats,
    record_promotion,
    update_split_fractions,
)

from datetime import UTC, datetime
from sqlalchemy.orm import Session
from recommender_api.runtime import InferenceRuntime
from common.mlflow.registry import promote_candidate_to_main
from common.db.repositories.recommendations import RecommendationRatingRow, insert_recommendation_rating


logger = logging.getLogger(__name__)


def handle_recommendation_rating_feedback(*, runtime: InferenceRuntime, session: Session, user_id: int, request_id: str, \
                                            movie_id: int, model_version: str, model_role: str, experiment_id: str, \
                                                rating: float) -> None:
    """
    Handle one recommendation rating and update online experiment split.

    Persist one recommendation rating and adjust the online experiment split.

    Do this by:
    1. Inserting a row into recommendation_ratings with full lineage.
    2. Nudging main/candidate split fractions based on the rating value.
    3. Promoting the candidate to main in MLflow when thresholds are met.

    ============================ Arguments ============================
    runtime: Startup-loaded inference runtime with experiment state.
    session: An open SQLAlchemy session inside a transaction.
    user_id: Authenticated user id.
    request_id: Recommendation request id from the impression.
    movie_id: Rated movie id.
    model_version: Model version that produced the recommendation.
    model_role: Model role label (main or candidate).
    experiment_id: Active experiment id.
    rating: User rating value.
    """
    # Save when the user gave feedback so online metrics have an event timestamp.
    rated_at = datetime.now(tz=UTC)

    # Store the recommendation rating with full lineage.
    # This tells us:
    # - which user rated the recommendation
    # - which request produced it
    # - which movie was rated
    # - which model version/model role produced it
    # - which online experiment it belongs to
    insert_recommendation_rating(
        session,
        RecommendationRatingRow(
            request_id=request_id,
            user_id=user_id,
            movie_id=movie_id,
            model_version=model_version,
            model_role=model_role,
            experiment_id=experiment_id,
            rating=rating,
            rated_at=rated_at,
        ),
    )

    # Read the current traffic split from the in-memory runtime.
    # Example: main=0.70 and candidate=0.30.
    current = SplitFractions(
        main=runtime.experiment.main_split_fraction,
        candidate=runtime.experiment.candidate_split_fraction,
    )

    # Get the candidate model's feedback so far.
    # This is used to decide whether the candidate has enough strong ratings to be promoted.
    candidate_count, candidate_avg = fetch_candidate_rating_stats(
        session,
        experiment_id=experiment_id,
        model_role="candidate",
    )

    # Decide whether this rating should move the online traffic split.
    # High ratings increase the rated model's share.
    # Low ratings decrease the rated model's share.
    # Neutral ratings leave the split unchanged.
    result = adjust_split_after_rating(
        current=current,
        model_role=model_role,
        rating=rating,
        settings=runtime.split_policy,
        candidate_rating_count=candidate_count,
        candidate_avg_rating=candidate_avg,
    )

    # If the policy changed the split, persist the new fractions in Postgres.
    # Also update the in-memory runtime so future requests use the latest split immediately.
    if result.changed:
        updated = update_split_fractions(
            session,
            experiment_id=experiment_id,
            main_split_fraction=result.fractions.main,
            candidate_split_fraction=result.fractions.candidate,
        )
        # Keep the API runtime in sync with the database row.
        runtime.experiment = updated

         # Update Prometheus/Grafana gauges so monitoring shows the latest rollout split.
        runtime.metrics.set_experiment_split_fraction("main", updated.main_split_fraction)
        runtime.metrics.set_experiment_split_fraction("candidate", updated.candidate_split_fraction)

    # If candidate has reached promotion criteria, promote it to main.
    # This usually means it reached max rollout and has enough strong feedback.
    if result.should_promote:
        promoted_version = promote_candidate_to_main(
            tracking_uri=runtime.mlflow_tracking_uri,
            registered_model_name=runtime.mlflow_registered_model_name,
        )

        # If MLflow promotion succeeded, update local experiment state too.
        if promoted_version:
            updated = record_promotion(
                session,
                experiment_id=experiment_id,
                new_main_version=promoted_version,
            )

            # Store the promoted experiment state in memory.
            runtime.experiment = updated

            # The candidate model object is now the main model object.
            # If for some reason candidate is missing, keep the existing main model as a fallback.
            runtime.main = runtime.candidate or runtime.main

            # Candidate slot is cleared because there is no challenger after promotion.
            runtime.candidate = None

            # After promotion, 100% of traffic goes to main.
            runtime.metrics.set_experiment_split_fraction("main", 1.0)
            runtime.metrics.set_experiment_split_fraction("candidate", 0.0)
            logger.info(
                "Promoted candidate model to main from online feedback",
                extra={"model_version": promoted_version},
            )

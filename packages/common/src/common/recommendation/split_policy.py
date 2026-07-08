"""Adaptive traffic split rules for main vs candidate online recommendations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SplitPolicySettings:
    """Environment-backed thresholds for online experiment split adjustments."""

    initial_main_split: float = 0.70
    initial_candidate_split: float = 0.30
    adjust_step: float = 0.02
    high_rating_threshold: float = 4.0
    low_rating_threshold: float = 3.0
    max_fraction: float = 0.80
    min_fraction: float = 0.00
    promotion_min_ratings: int = 10
    promotion_min_avg_rating: float = 4.0


@dataclass(frozen=True)
class SplitFractions:
    """Main and candidate recommendation slot fractions."""

    main: float
    candidate: float


@dataclass(frozen=True)
class SplitAdjustmentResult:
    """Outcome after applying one rating-based split adjustment."""

    fractions: SplitFractions
    changed: bool
    should_promote: bool


def normalize_fractions(main: float, candidate: float) -> SplitFractions:
    """
    Clamp fractions and make them sum to 1.0.

    ============================ Arguments ============================
    main: Raw main fraction.
    candidate: Raw candidate fraction.

    ============================ Returns ============================
    Normalized main/candidate fractions.
    """
    total = main + candidate
    if total <= 0:
        return SplitFractions(main=1.0, candidate=0.0)
    return SplitFractions(main=main / total, candidate=candidate / total)


def clamp_fraction(value: float, settings: SplitPolicySettings) -> float:
    """Clamp one model fraction to the configured min/max bounds."""
    return max(settings.min_fraction, min(settings.max_fraction, value))


def adjust_split_after_rating(*, current: SplitFractions, model_role: str, rating: float, settings: SplitPolicySettings, 
                                candidate_rating_count: int, candidate_avg_rating: float) -> SplitAdjustmentResult:
    """
    Nudge main/candidate split fractions after one recommendation rating.

    Do this by:
    1. Leaving neutral ratings unchanged.
    2. Increasing the rated model's share on high ratings.
    3. Decreasing the rated model's share on low ratings.
    4. Checking whether the candidate qualifies for promotion to main.

    ============================ Arguments ============================
    current: Current main/candidate fractions from Postgres.
    model_role: Which model produced the rated recommendation (main or candidate).
    rating: User rating value (0.5 to 5.0).
    settings: Thresholds and step size from environment config.
    candidate_rating_count: Total candidate ratings logged so far in this experiment.
    candidate_avg_rating: Average rating on candidate recommendations so far.

    ============================ Returns ============================
    Updated fractions, whether they changed, and whether to promote candidate to main.
    """
    # Only main/candidate recommendations should affect the online split.
    if model_role not in {"main", "candidate"}:
        return SplitAdjustmentResult(fractions=current, changed=False, should_promote=False)

    # Convert the user's rating into a small traffic adjustment.
    # High rating  -> give this model more traffic.
    # Low rating   -> give this model less traffic.
    # Neutral      -> leave the split unchanged.
    delta = 0.0
    if rating >= settings.high_rating_threshold:
        delta = settings.adjust_step
    elif rating <= settings.low_rating_threshold:
        delta = -settings.adjust_step
    else:
        # Neutral ratings do not move traffic, but we still check promotion.
        # The candidate may already have enough strong feedback to be promoted.
        should_promote = _should_promote_candidate(
            current=current,
            settings=settings,
            candidate_rating_count=candidate_rating_count,
            candidate_avg_rating=candidate_avg_rating,
        )
        return SplitAdjustmentResult(fractions=current, changed=False, should_promote=should_promote)

    # Apply the traffic nudge to whichever model received the rating.
    # If the main model gets a good rating, main share goes up.
    # If the main model gets a bad rating, main share goes down.
    if model_role == "main":
        new_main = clamp_fraction(current.main + delta, settings)
        new_candidate = clamp_fraction(1.0 - new_main, settings)

    # Same idea for candidate:
    # good candidate rating -> candidate gets more traffic
    # bad candidate rating  -> candidate gets less traffic
    else:
        new_candidate = clamp_fraction(current.candidate + delta, settings)
        new_main = clamp_fraction(1.0 - new_candidate, settings)

    # Re-normalize so the two fractions still add up cleanly to 1.0.
    updated = normalize_fractions(new_main, new_candidate)

    # After changing the split, check whether candidate has reached promotion criteria.
    # Usually this means candidate is at/near max rollout and has enough good ratings.
    should_promote = _should_promote_candidate(
        current=updated,
        settings=settings,
        candidate_rating_count=candidate_rating_count,
        candidate_avg_rating=candidate_avg_rating,
    )

    # Mark changed only when the effective saved fractions actually moved.
    # This avoids unnecessary DB writes when clamping keeps the split the same.
    changed = abs(updated.main - current.main) > 1e-9 or abs(updated.candidate - current.candidate) > 1e-9
    return SplitAdjustmentResult(fractions=updated, changed=changed, should_promote=should_promote)


def allocate_model_slots(top_k: int, fractions: SplitFractions, *, has_candidate: bool) -> tuple[int, int]:
    """
    Convert split fractions into integer slot counts for one recommend response.

    ============================ Arguments ============================
    top_k: Number of recommendations to return.
    fractions: Current main/candidate fractions.
    has_candidate: Whether a candidate model is loaded and eligible.

    ============================ Returns ============================
    Tuple of (main_slots, candidate_slots) that sum to top_k.
    """
    if not has_candidate or fractions.candidate <= 0:
        return top_k, 0

    candidate_slots = int(round(top_k * fractions.candidate))
    candidate_slots = max(0, min(top_k, candidate_slots))
    main_slots = top_k - candidate_slots
    return main_slots, candidate_slots


def _should_promote_candidate(*, current: SplitFractions, settings: SplitPolicySettings, candidate_rating_count: int, \
                                candidate_avg_rating: float) -> bool:
    # Do not promote until candidate has reached the maximum allowed rollout.
    # This prevents promoting too early after only a small traffic test.
    if current.candidate < settings.max_fraction - 1e-9:
        return False

    # Do not promote until candidate has received enough user feedback.
    # This avoids promoting based on a tiny number of lucky ratings.
    if candidate_rating_count < settings.promotion_min_ratings:
        return False
    return candidate_avg_rating >= settings.promotion_min_avg_rating

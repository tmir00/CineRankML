"""Online user behavior statistics for hybrid ranker features."""

from __future__ import annotations

import math
import numpy as np

from dataclasses import dataclass
from numpy.typing import NDArray
from common.features.schema import HIGH_RATED_THRESHOLD, LOW_RATED_THRESHOLD, USER_BEHAVIOR_DIM


def _behavior_vector_from_running_stats(count: int, total: float, sum_sq: float, num_high: int, num_low: int) -> NDArray[np.float32]:
    """
    Build the five user behavior features from running counters.

    Do this by:
    1. Returning zeros when there are no ratings yet.
    2. Computing the average rating from total / count.
    3. Computing population standard deviation from the running sum of squares.
    4. Packing count, average, std, high count, and low count into one vector.

    ============================ Arguments ============================
    count: How many ratings are in the running history.
    total: Sum of those rating values.
    sum_sq: Sum of each rating value squared.
    num_high: How many ratings are at or above the high-rated threshold.
    num_low: How many ratings are at or below the low-rated threshold.

    ============================ Returns ============================
    Behavior vector with shape (5,).
    """
    # If the user has no ratings yet, every behavior feature is zero.
    if count == 0:
        return np.zeros(USER_BEHAVIOR_DIM, dtype=np.float32)

    # Compute the average rating.
    mean = total / count

    # Population variance from running stats:
    #   var = (sum_sq / count) - (mean * mean)
    # Tiny negative values can appear from float rounding, so clamp at zero.
    variance = max(0.0, (sum_sq / count) - (mean * mean))

    # Population std needs at least two ratings; one rating has zero spread.
    rating_std = math.sqrt(variance) if count > 1 else 0.0

    return np.array(
        [float(count), float(mean), float(rating_std), float(num_high), float(num_low)],
        dtype=np.float32,
    )


@dataclass
class UserBehaviorAccumulator:
    """
    Track user behavior stats incrementally as ratings arrive in time order.

    Do this by:
    1. Keeping running counters for count, sum, sum-of-squares, and high/low counts.
    2. Updating those counters each time a new rating is observed.
    3. Building a behavior vector snapshot from the counters at any moment.

    Batch training uses this while walking each user's rating timeline.
    Online inference can still call compute_user_behavior() on a full rating list.
    """

    count: int = 0
    total: float = 0.0
    sum_sq: float = 0.0
    num_high: int = 0
    num_low: int = 0

    def observe(self, rating: float) -> None:
        """
        Add one rating to the running history.

        ============================ Arguments ============================
        rating: The rating value to include in the running counters.
        """
        # Update the basic running totals.
        self.count += 1
        self.total += rating
        self.sum_sq += rating * rating

        # Count high and low ratings using the same thresholds as online inference.
        if rating >= HIGH_RATED_THRESHOLD:
            self.num_high += 1
        if rating <= LOW_RATED_THRESHOLD:
            self.num_low += 1

    def snapshot(self) -> NDArray[np.float32]:
        """
        Build the current behavior vector from the running counters.

        ============================ Returns ============================
        Behavior vector with shape (5,).
        """
        return _behavior_vector_from_running_stats(
            self.count,
            self.total,
            self.sum_sq,
            self.num_high,
            self.num_low,
        )


def compute_user_behavior(ratings: list[float]) -> NDArray[np.float32]:
    """
    Compute the five user behavior features from a rating history.

    Do this by:
    1. Counting total ratings and high/low-rated movies.
    2. Computing average rating and population standard deviation.

    ============================ Arguments ============================
    ratings: Rating values for the user's distinct rated movies.

    ============================ Returns ============================
    Behavior vector with shape (5,).
    """
    if not ratings:
        return np.zeros(USER_BEHAVIOR_DIM, dtype=np.float32)

    # Feed every rating through the same accumulator used during batch training.
    accumulator = UserBehaviorAccumulator()
    for rating in ratings:
        accumulator.observe(rating)
    return accumulator.snapshot()

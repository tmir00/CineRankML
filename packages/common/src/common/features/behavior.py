"""Online user behavior statistics for hybrid ranker features."""

from __future__ import annotations

import statistics
import numpy as np

from numpy.typing import NDArray
from common.features.schema import HIGH_RATED_THRESHOLD, LOW_RATED_THRESHOLD, USER_BEHAVIOR_DIM


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

    num_ratings = float(len(ratings))
    avg_rating = float(sum(ratings) / len(ratings))
    rating_std = float(statistics.pstdev(ratings)) if len(ratings) > 1 else 0.0
    num_high = float(sum(1 for rating in ratings if rating >= HIGH_RATED_THRESHOLD))
    num_low = float(sum(1 for rating in ratings if rating <= LOW_RATED_THRESHOLD))

    return np.array(
        [num_ratings, avg_rating, rating_std, num_high, num_low],
        dtype=np.float32,
    )

"""Tests for online user behavior feature helpers."""

from __future__ import annotations

import numpy as np
import pytest

from common.features.behavior import UserBehaviorAccumulator, compute_user_behavior
from common.features.schema import USER_BEHAVIOR_DIM


def test_compute_user_behavior_empty_history() -> None:
    behavior = compute_user_behavior([])
    assert behavior.shape == (USER_BEHAVIOR_DIM,)
    assert np.allclose(behavior, 0.0)


def test_compute_user_behavior_counts_high_and_low_ratings() -> None:
    behavior = compute_user_behavior([5.0, 4.5, 3.0, 2.0, 1.0])
    assert behavior[0] == 5.0
    assert behavior[1] == 3.1
    assert behavior[3] == 2.0
    assert behavior[4] == 2.0


def test_compute_user_behavior_population_std() -> None:
    """Population std for [5.0, 4.5, 3.0, 2.0, 1.0] should be about 1.49666."""
    behavior = compute_user_behavior([5.0, 4.5, 3.0, 2.0, 1.0])
    assert behavior[2] == pytest.approx(1.49666, rel=1e-4)


def test_user_behavior_accumulator_matches_compute_user_behavior() -> None:
    """Each accumulator snapshot should match compute_user_behavior on the same prefix."""
    histories = [
        [],
        [4.0],
        [4.0, 2.5],
        [5.0, 4.5, 3.0, 2.0, 1.0],
    ]

    for history in histories:
        accumulator = UserBehaviorAccumulator()
        prefix: list[float] = []

        # Before any rating is observed, the snapshot should match an empty prefix.
        assert np.allclose(accumulator.snapshot(), compute_user_behavior(prefix))

        for rating in history:
            assert np.allclose(accumulator.snapshot(), compute_user_behavior(prefix))
            accumulator.observe(rating)
            prefix.append(rating)

        assert np.allclose(accumulator.snapshot(), compute_user_behavior(history))

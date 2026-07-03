"""Tests for online user behavior feature helpers."""

from __future__ import annotations

import numpy as np

from common.features.behavior import compute_user_behavior
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

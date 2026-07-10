"""Tests for streaming validation evaluation metrics."""

from __future__ import annotations

import torch

from train_cf.evaluate import StreamingRegressionMetrics


def test_streaming_regression_metrics_rmse_and_mae() -> None:
    """Running counters should produce correct RMSE and MAE."""
    metrics = StreamingRegressionMetrics()
    predictions = torch.tensor([3.0, 4.0])
    targets = torch.tensor([1.0, 4.0])

    metrics.update(predictions, targets)

    assert metrics.count == 2
    assert metrics.mae == 1.0
    assert metrics.rmse == (2.0 ** 0.5)

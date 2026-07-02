"""Unit tests for streaming regression metrics."""

from __future__ import annotations

import torch

from train_hybrid_ranker.evaluate import StreamingRegressionMetrics


def test_streaming_regression_metrics() -> None:
    metrics = StreamingRegressionMetrics()
    predictions = torch.tensor([3.0, 4.0, 5.0])
    targets = torch.tensor([3.0, 2.0, 5.0])

    metrics.update(predictions, targets)

    assert metrics.count == 3
    assert metrics.mae == (0.0 + 2.0 + 0.0) / 3
    assert metrics.rmse == ((0.0 + 4.0 + 0.0) / 3) ** 0.5

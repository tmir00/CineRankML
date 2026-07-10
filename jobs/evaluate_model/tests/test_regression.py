"""Unit tests for streaming regression metrics used during evaluation."""

from __future__ import annotations

import torch

from train_hybrid_ranker.evaluate import StreamingRegressionMetrics


def test_streaming_regression_metrics_zero_error() -> None:
    metrics = StreamingRegressionMetrics()
    values = torch.tensor([3.0, 4.0, 5.0])
    metrics.update(values, values)

    assert metrics.rmse == 0.0
    assert metrics.mae == 0.0
    assert metrics.count == 3

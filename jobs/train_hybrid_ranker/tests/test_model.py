"""Unit tests for HybridRankerMLP."""

from __future__ import annotations

import torch

from train_hybrid_ranker.model import HybridRankerMLP


def test_forward_pass_shape() -> None:
    model = HybridRankerMLP(input_dim=1356, hidden_dims=[512, 256, 64], dropout=0.0)
    features = torch.randn(8, 1356)
    predictions = model(features)
    assert predictions.shape == (8,)


def test_dropout_changes_training_mode_output_distribution() -> None:
    model = HybridRankerMLP(input_dim=16, hidden_dims=[8, 4], dropout=0.5)
    features = torch.ones(4, 16)

    model.train()
    train_outputs = [model(features) for _ in range(5)]

    model.eval()
    eval_outputs = [model(features) for _ in range(5)]

    assert not all(torch.allclose(train_outputs[0], out) for out in train_outputs[1:])
    assert all(torch.allclose(eval_outputs[0], out) for out in eval_outputs[1:])

"""Tests for the dot-product CF model."""

from __future__ import annotations

import torch

from train_cf.model import DotProductCF


def test_dot_product_cf_forward_shape() -> None:
    """Forward pass should return one prediction per user/movie pair."""
    model = DotProductCF(num_users=5, num_movies=7, embedding_dim=64)
    user_idx = torch.tensor([0, 1, 2])
    movie_idx = torch.tensor([3, 4, 5])

    predictions = model(user_idx, movie_idx)

    assert predictions.shape == (3,)


def test_dot_product_cf_matches_manual_dot_product() -> None:
    """Predictions should equal the dot product of looked-up embeddings."""
    torch.manual_seed(0)
    model = DotProductCF(num_users=2, num_movies=2, embedding_dim=4)
    user_idx = torch.tensor([0])
    movie_idx = torch.tensor([1])

    prediction = model(user_idx, movie_idx)
    expected = torch.dot(model.user_emb.weight[0], model.movie_emb.weight[1])

    assert torch.allclose(prediction, expected.unsqueeze(0))

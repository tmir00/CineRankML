"""MLP hybrid ranker model definition."""

from __future__ import annotations

import torch

from torch import nn
from common.features.schema import INPUT_DIM


class HybridRankerMLP(nn.Module):
    """
    Predict user ratings from a 1356-dimensional hybrid feature vector.

    Do this by:
    1. Passing the feature vector through three ReLU hidden layers (512, 256, 64).
    2. Applying dropout between hidden layers during training.
    3. Returning a single linear output as the predicted rating.
    """

    def __init__(self, input_dim: int = INPUT_DIM, hidden_dims: list[int] | None = None, dropout: float = 0.1) -> None:
        """
        Create the MLP stack for hybrid ranker inference.

        ============================ Arguments ============================
        input_dim: Feature vector size (1356 for hybrid-v1).
        hidden_dims: Hidden layer sizes; defaults to [512, 256, 64].
        dropout: Dropout probability applied after each hidden ReLU layer.
        """
        super().__init__()
        dims = hidden_dims or [512, 256, 64]
        layers: list[nn.Module] = []
        in_features = input_dim

        for hidden_dim in dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            in_features = hidden_dim

        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict ratings for a batch of feature vectors.

        ============================ Arguments ============================
        features: Batch of feature tensors with shape [batch_size, input_dim].

        ============================ Returns ============================
        Predicted rating tensor with shape [batch_size].
        """
        return self.network(features).squeeze(-1)

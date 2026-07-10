"""Dot-product collaborative filtering model."""

from __future__ import annotations

import torch
from torch import nn


class DotProductCF(nn.Module):
    """
    Learn user and movie embeddings so their dot product predicts ratings.

    Do this by:
    1. Looking up a user embedding vector for each user_idx.
    2. Looking up a movie embedding vector for each movie_idx.
    3. Returning the elementwise dot product as the predicted rating.
    """

    def __init__(self, num_users: int, num_movies: int, embedding_dim: int) -> None:
        """
        Create embedding tables for users and movies.

        ============================ Arguments ============================
        num_users: Number of user embedding rows (train-split users only).
        num_movies: Number of movie embedding rows (full catalog).
        embedding_dim: Embedding size (64 for the hybrid ranker).
        """
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.movie_emb = nn.Embedding(num_movies, embedding_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.movie_emb.weight, std=0.01)

    def forward(self, user_idx: torch.Tensor, movie_idx: torch.Tensor) -> torch.Tensor:
        """
        Predict ratings for a batch of user/movie index pairs.

        ============================ Arguments ============================
        user_idx: Batch of user embedding indices.
        movie_idx: Batch of movie embedding indices.

        ============================ Returns ============================
        Predicted rating tensor with shape [batch_size].
        """
        user_vectors = self.user_emb(user_idx)
        movie_vectors = self.movie_emb(movie_idx)
        return (user_vectors * movie_vectors).sum(dim=1)

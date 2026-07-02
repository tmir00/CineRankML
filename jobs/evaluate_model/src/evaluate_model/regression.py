"""Streaming test regression metrics for hybrid ranker evaluation."""

from __future__ import annotations

import torch
import torch.nn as nn

from collections.abc import Sequence
from botocore.client import BaseClient
from torch.utils.data import DataLoader

from evaluate_model.ranking import (
    RankingMetricAccumulator,
    UserRankingExample,
    accumulate_user_metrics,
    compute_user_ranking_metrics,
)

from train_hybrid_ranker.evaluate import StreamingRegressionMetrics
from train_hybrid_ranker.dataset import HybridParquetIterableDataset
from common.schemas.hybrid_ranker_dataset_manifest import HybridRankerPartEntry


def _collate_eval_batch(batch: list[tuple[list[float], float, int, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Turn a list of feature rows with ids into batched tensors."""
    features, ratings, user_ids, movie_ids = zip(*batch)
    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(ratings, dtype=torch.float32),
        torch.tensor(user_ids, dtype=torch.long),
        torch.tensor(movie_ids, dtype=torch.long),
    )


def evaluate_test_split(model: nn.Module, client: BaseClient, bucket: str, test_parts: Sequence[HybridRankerPartEntry], \
                            *, batch_size: int, device: torch.device, \
                            relevance_threshold: float = 4.0) -> tuple[float, float, RankingMetricAccumulator, int]:
    """
    Compute test RMSE/MAE and ranking metrics by streaming test Parquet parts.

    Do this by:
    1. Streaming test rows with user_id and movie_id included.
    2. Accumulating regression errors across all rows.
    3. Grouping predictions by user_id for ranking metrics.

    ============================ Arguments ============================
    model: Trained hybrid ranker MLP.
    client: The boto3 S3 client.
    bucket: Source MinIO/S3 bucket.
    test_parts: Test part metadata from the dataset manifest.
    batch_size: Evaluation batch size.
    device: Torch device for inference.
    relevance_threshold: Minimum rating treated as relevant for ranking metrics.

    ============================ Returns ============================
    Tuple of (test_rmse, test_mae, ranking_accumulator, num_test_rows).
    """
    # Create the dataset and loader.
    dataset = HybridParquetIterableDataset(
        client,
        bucket,
        test_parts,
        shuffle_within_part=False,
        include_ids=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=_collate_eval_batch,
    )

    # Initialize the regression metrics accumulator.
    regression_metrics = StreamingRegressionMetrics()
    # Initialize the dictionary to store user examples.
    user_examples: dict[int, UserRankingExample] = {}

    # Set the model to evaluation mode.
    model.eval()
    with torch.no_grad():
        # Iterate over the loader.
        for features, rating, user_id_tensor, _movie_id_tensor in loader:
            # Move the features and target to the device.
            feature_tensor = features.to(device)
            target = rating.to(device, dtype=torch.float32)
            # Make predictions.
            predictions = model(feature_tensor)
            # Update the regression metrics.
            regression_metrics.update(predictions, target)

            # Iterate over the predictions, actual ratings, and user ids.
            for prediction, actual_rating, user_id in zip(
                predictions.detach().cpu().tolist(),
                target.detach().cpu().tolist(),
                user_id_tensor.detach().cpu().tolist(),
            ):
                # Create a new user example if it doesn't exist, otherwise append to the existing one.
                example = user_examples.setdefault(int(user_id), UserRankingExample(ratings=[], predictions=[]))
                example.ratings.append(float(actual_rating))
                example.predictions.append(float(prediction))

    # Initialize the ranking metrics accumulator.
    ranking_accumulator = RankingMetricAccumulator()
    # Iterate over the user examples and compute the ranking metrics.
    for example in user_examples.values():
        # Compute the ranking metrics for the user example.
        user_metrics = compute_user_ranking_metrics(example, relevance_threshold=relevance_threshold)
        # If the user metrics are not None, add them to the ranking accumulator.
        if user_metrics is not None:
            # Add the user metrics to the ranking accumulator.
            accumulate_user_metrics(ranking_accumulator, user_metrics)
    
    # Return the regression metrics, ranking metrics, and number of test rows.
    return (
        regression_metrics.rmse,
        regression_metrics.mae,
        ranking_accumulator,
        regression_metrics.count,
    )

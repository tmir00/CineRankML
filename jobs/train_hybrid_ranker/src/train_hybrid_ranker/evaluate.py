"""Streaming validation evaluation metrics for hybrid ranker training."""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

from collections.abc import Sequence
from botocore.client import BaseClient
from torch.utils.data import DataLoader
from train_hybrid_ranker.dataset import HybridParquetIterableDataset
from common.schemas.hybrid_ranker_dataset_manifest import HybridRankerPartEntry


logger = logging.getLogger(__name__)


class StreamingRegressionMetrics:
    """Running RMSE/MAE counters for one evaluation pass."""

    def __init__(self) -> None:
        self.sum_squared_error = 0.0
        self.sum_absolute_error = 0.0
        self.count = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate batch errors into running totals."""
        diff = predictions - targets
        self.sum_squared_error += float(torch.sum(diff * diff).item())
        self.sum_absolute_error += float(torch.sum(torch.abs(diff)).item())
        self.count += int(targets.numel())

    @property
    def rmse(self) -> float:
        """Root mean squared error across all accumulated rows."""
        if self.count == 0:
            return 0.0
        return (self.sum_squared_error / self.count) ** 0.5

    @property
    def mae(self) -> float:
        """Mean absolute error across all accumulated rows."""
        if self.count == 0:
            return 0.0
        return self.sum_absolute_error / self.count


def _collate_features_batch(batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn a list of (features, rating) rows into batched tensors."""
    features, ratings = zip(*batch)
    if isinstance(features[0], np.ndarray):
        return (
            torch.from_numpy(np.stack(features, axis=0)),
            torch.from_numpy(np.asarray(ratings, dtype=np.float32)),
        )
    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(ratings, dtype=torch.float32),
    )


def evaluate_validation_streaming(model: nn.Module, client: BaseClient, bucket: str, validation_parts: Sequence[HybridRankerPartEntry], *, \
                                    batch_size: int, device: torch.device) -> tuple[float, float]:
    """
    Compute validation RMSE and MAE by streaming validation Parquet parts.

    Do this by:
    1. Building an IterableDataset over validation parts without shuffling.
    2. Running the model in eval mode on each batch.
    3. Accumulating running squared and absolute errors only.

    ============================ Arguments ============================
    model: Trained hybrid ranker MLP.
    client: The boto3 S3 client.
    bucket: Source MinIO/S3 bucket.
    validation_parts: Validation part metadata from the dataset manifest.
    batch_size: Evaluation batch size.
    device: Torch device for inference.

    ============================ Returns ============================
    A tuple of (validation_rmse, validation_mae).
    """
    logger.info(
        "Starting validation evaluation parts=%d batch_size=%d device=%s",
        len(validation_parts),
        batch_size,
        device,
    )

    dataset = HybridParquetIterableDataset(
        client,
        bucket,
        validation_parts,
        shuffle_within_part=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=_collate_features_batch,
    )
    metrics = StreamingRegressionMetrics()

    model.eval()
    with torch.no_grad():
        for batch_index, (features, rating) in enumerate(loader, start=1):
            feature_tensor = features.to(device)
            target = rating.to(device, dtype=torch.float32)
            predictions = model(feature_tensor)
            metrics.update(predictions, target)
            if batch_index % 50 == 0:
                logger.info(
                    "Validation progress batches=%d rows=%d running_rmse=%.6f",
                    batch_index,
                    metrics.count,
                    metrics.rmse,
                )

    logger.info(
        "Validation evaluation complete rows=%d rmse=%.6f mae=%.6f",
        metrics.count,
        metrics.rmse,
        metrics.mae,
    )

    return metrics.rmse, metrics.mae

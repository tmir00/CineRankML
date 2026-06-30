"""Streaming validation evaluation metrics for CF training."""

from __future__ import annotations

import torch
import torch.nn as nn

from dataclasses import dataclass
from collections.abc import Sequence
from botocore.client import BaseClient
from torch.utils.data import DataLoader
from train_cf.dataset import CfParquetIterableDataset
from common.schemas.cf_dataset_manifest import CfDatasetPartEntry


@dataclass
class StreamingRegressionMetrics:
    """Running RMSE/MAE counters for one evaluation pass."""

    sum_squared_error: float = 0.0
    sum_absolute_error: float = 0.0
    count: int = 0

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


def evaluate_validation_streaming(
    model: nn.Module,
    client: BaseClient,
    bucket: str,
    validation_parts: Sequence[CfDatasetPartEntry],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    """
    Compute validation RMSE and MAE by streaming validation Parquet parts.

    Do this by:
    1. Building an IterableDataset over validation parts without shuffling.
    2. Running the model in eval mode on each batch.
    3. Accumulating running squared and absolute errors only.

    ============================ Arguments ============================
    model: Trained dot-product CF model.
    client: The boto3 S3 client.
    bucket: Source MinIO/S3 bucket.
    validation_parts: Validation part metadata from the CF dataset manifest.
    batch_size: Evaluation batch size.
    device: Torch device for inference.

    ============================ Returns ============================
    A tuple of (validation_rmse, validation_mae).
    """
    dataset = CfParquetIterableDataset(client, bucket, validation_parts, shuffle_within_part=False)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    metrics = StreamingRegressionMetrics()

    model.eval()
    with torch.no_grad():
        for user_idx, movie_idx, rating in loader:
            user_tensor = user_idx.to(device)
            movie_tensor = movie_idx.to(device)
            target = rating.to(device, dtype=torch.float32)
            predictions = model(user_tensor, movie_tensor)
            metrics.update(predictions, target)

    return metrics.rmse, metrics.mae

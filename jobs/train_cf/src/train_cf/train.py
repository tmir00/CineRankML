"""Core CF PyTorch training loop and artifact upload."""

from __future__ import annotations

import time
import json
import torch
import mlflow
import logging
import random
import tempfile
import torch.nn as nn

from pathlib import Path
from copy import deepcopy
from datetime import UTC, datetime

from common.storage.s3 import (
    cf_artifact_manifest_object_key,
    cf_config_object_key,
    cf_metrics_object_key,
    cf_model_object_key,
    cf_movie_embeddings_object_key,
    cf_training_curve_object_key,
    put_json,
    upload_file,
)
from common.mlflow.cf_run import (
    log_cf_artifacts,
    log_epoch_metrics,
    log_final_metrics,
    start_cf_training_run,
)

from botocore.client import BaseClient
from torch.utils.data import DataLoader
from train_cf.model import DotProductCF
from dataclasses import dataclass, field
from train_cf.device import resolve_device
from train_cf.plot import save_training_curve
from train_cf.version import resolve_cf_version
from common.config.settings import CfTrainingSettings
from train_cf.dataset import CfParquetIterableDataset
from train_cf.evaluate import evaluate_validation_streaming
from train_cf.manifest import build_complete_cf_manifest
from common.schemas.cf_artifact_manifest import CfTrainingConfig
from train_cf.export import export_movie_embeddings, finalize_training_metrics
from common.storage.cf_dataset_reader import load_cf_dataset_manifest, resolve_cf_dataset_version


logger = logging.getLogger(__name__)


@dataclass
class CfTrainingStats:
    """Counters and metadata collected during one CF training run."""

    cf_version: str
    cf_dataset_version: str
    snapshot_id: str
    best_validation_rmse: float | None = None
    train_losses: list[float] = field(default_factory=list)
    validation_rmses: list[float] = field(default_factory=list)
    validation_maes: list[float] = field(default_factory=list)


def _collate_batch(batch: list[tuple[int, int, float]],) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Takes a list of training examples and turns them into PyTorch tensors so the model can train on them.
    """
    user_idx, movie_idx, rating = zip(*batch)
    return (
        torch.tensor(user_idx, dtype=torch.long),
        torch.tensor(movie_idx, dtype=torch.long),
        torch.tensor(rating, dtype=torch.float32),
    )


def _train_one_epoch(model: DotProductCF, loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: nn.Module,
                        device: torch.device, rated_movie_indices: set[int]) -> float:
    """
    Run one training epoch and return the average MSE loss.

    ============================ Arguments ============================
    model: Dot-product CF model in train mode.
    loader: DataLoader over shuffled train parts.
    optimizer: Optimizer for embedding weights.
    loss_fn: Regression loss (MSELoss).
    device: Torch device for training.
    rated_movie_indices: Mutable set collecting movie indices seen in train data.

    ============================ Returns ============================
    Average training loss for the epoch.
    """
    # Set the model to train mode.
    model.train()
    # Initialize the total loss and the number of rows.
    total_loss = 0.0
    total_rows = 0

    # Iterate over the training data.
    for user_idx, movie_idx, rating in loader:
        # Convert the ratings,user and movie indices to PyTorch tensors.
        user_tensor = user_idx.to(device)
        movie_tensor = movie_idx.to(device)
        target = rating.to(device)

        # Add the movie indices to the set of rated movie indices.
        for movie_index in movie_tensor.detach().cpu().tolist():
            rated_movie_indices.add(int(movie_index))

        # Run the model forward and compute the loss.
        optimizer.zero_grad(set_to_none=True)
        predictions = model(user_tensor, movie_tensor)
        loss = loss_fn(predictions, target)
        loss.backward()
        optimizer.step()

        # Update the total loss and the number of rows.
        batch_size = int(target.numel())
        total_loss += float(loss.item()) * batch_size  # Turn MSELoss into total loss by multiplying by batch size.
        total_rows += batch_size #

    return total_loss / max(1, total_rows)  # Return the average loss for the epoch.


def _upload_artifacts(client: BaseClient, bucket: str, cf_version: str, local_dir: Path) -> None:
    """
    Upload all CF artifact files to MinIO except manifest.json.

    ============================ Arguments ============================
    client: The boto3 S3 client.
    bucket: Target MinIO/S3 bucket.
    cf_version: CF artifact version identifier.
    local_dir: Local directory containing artifact files.
    """
    upload_file(client, bucket, cf_movie_embeddings_object_key(cf_version), local_dir / "movie_cf_embeddings.parquet")
    upload_file(client, bucket, cf_model_object_key(cf_version), local_dir / "cf_model.pt")
    upload_file(client, bucket, cf_config_object_key(cf_version), local_dir / "cf_config.json")
    upload_file(client, bucket, cf_metrics_object_key(cf_version), local_dir / "cf_metrics.json")
    upload_file(client, bucket, cf_training_curve_object_key(cf_version), local_dir / "training_curve.png")


def run_cf_training(client: BaseClient, settings: CfTrainingSettings, *, pipeline_run_id: str) -> CfTrainingStats:
    """
    Train the CF model, export artifacts, and write the MinIO manifest.

    Do this by:
    1. Resolving the input CF dataset and output cf_version.
    2. Training with streaming IterableDataset over train parts.
    3. Evaluating validation RMSE/MAE each epoch with optional early stopping.
    4. Exporting embeddings and uploading artifacts (manifest last).
    5. Logging metrics and artifacts to MLflow.

    ============================ Arguments ============================
    client: The boto3 S3 client.
    settings: CF training configuration.
    pipeline_run_id: pipeline_runs.run_id for lineage.

    ============================ Returns ============================
    CfTrainingStats with version ids and best validation RMSE.
    """
    created_at = datetime.now(tz=UTC)

    # Resolve the CF dataset version.
    cf_dataset_version = resolve_cf_dataset_version(
        client,
        settings.s3_bucket,
        settings.cf_dataset_version,
    )
    # Load the CF dataset manifest.
    dataset_manifest = load_cf_dataset_manifest(client, settings.s3_bucket, cf_dataset_version)
    # Check if the dataset is complete.
    if dataset_manifest.status != "complete":
        raise ValueError(f"CF dataset {cf_dataset_version} is not complete")

    # Resolve the CF model version.
    cf_version = resolve_cf_version(settings)
    # Resolve the device to train on.
    device = resolve_device(settings.device)

    stats = CfTrainingStats(
        cf_version=cf_version,
        cf_dataset_version=cf_dataset_version,
        snapshot_id=dataset_manifest.snapshot_id,
    )

    # Initialize the CF model.
    model = DotProductCF(
        dataset_manifest.num_users,
        dataset_manifest.num_movies,
        settings.embedding_dim,
    ).to(device)
    # Initialize the optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    # Initialize the loss function.
    loss_fn = nn.MSELoss()

    best_state_dict: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_rmse = float("inf")
    best_mae = float("inf")
    epochs_without_improvement = 0
    rated_movie_indices: set[int] = set()

    # Start the MLflow CF training run.
    with start_cf_training_run(
        settings,
        cf_version=cf_version,
        cf_dataset_version=cf_dataset_version,
        snapshot_id=dataset_manifest.snapshot_id,
        train_fraction=dataset_manifest.train_fraction,
        validation_fraction=dataset_manifest.validation_fraction,
        test_fraction=dataset_manifest.test_fraction,
    ):
        # Log the device and early stopping patience.
        mlflow.log_param("device", str(device))
        mlflow.log_param("early_stopping_patience", settings.early_stopping_patience)

        # Iterate over the epochs.
        for epoch in range(settings.num_epochs):
            # Start the epoch timer.
            epoch_start = time.perf_counter()

            # Shuffle the train parts.
            train_parts = list(dataset_manifest.train_parts)
            rng = random.Random(settings.shuffle_seed + epoch)
            rng.shuffle(train_parts)

            # Create the train dataset.
            train_dataset = CfParquetIterableDataset(
                client,
                settings.s3_bucket,
                train_parts,
                shuffle_within_part=True,
                seed=settings.shuffle_seed + epoch,
            )

            # Create the train data loader.
            train_loader = DataLoader(
                train_dataset,
                batch_size=settings.batch_size,
                num_workers=0,
                collate_fn=_collate_batch,
            )

            # Train the model for one epoch.
            train_loss = _train_one_epoch(
                model,
                train_loader,
                optimizer,
                loss_fn,
                device,
                rated_movie_indices,
            )

            # Evaluate the model on the validation set.
            val_rmse, val_mae = evaluate_validation_streaming(
                model,
                client,
                settings.s3_bucket,
                dataset_manifest.validation_parts,
                batch_size=settings.batch_size,
                device=device,
            )
            # Calculate the epoch duration.
            epoch_duration = time.perf_counter() - epoch_start

            # Update the training statistics.
            stats.train_losses.append(train_loss)
            stats.validation_rmses.append(val_rmse)
            stats.validation_maes.append(val_mae)
            log_epoch_metrics(
                epoch=epoch,
                train_loss=train_loss,
                validation_rmse=val_rmse,
                validation_mae=val_mae,
                epoch_duration_seconds=epoch_duration,
            )

            logger.info(
                "CF training epoch complete",
                extra={
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "validation_rmse": val_rmse,
                    "validation_mae": val_mae,
                    "epoch_duration_seconds": epoch_duration,
                },
            )

            # Update the best validation RMSE and epoch.
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_mae = val_mae
                best_epoch = epoch
                best_state_dict = deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Check if early stopping is triggered.
            if (
                settings.early_stopping_patience > 0
                and epochs_without_improvement >= settings.early_stopping_patience
            ):
                # Log the early stopping trigger.
                logger.info(
                    "Early stopping triggered",
                    extra={
                        "epoch": epoch + 1,
                        "best_epoch": best_epoch + 1,
                        "patience": settings.early_stopping_patience,
                    },
                )
                break

        # Check if a best checkpoint was found.
        if best_state_dict is None:
            raise RuntimeError("Training finished without a best checkpoint")

        # Load the best checkpoint.
        model.load_state_dict(best_state_dict)
        # Update the training statistics.
        stats.best_validation_rmse = best_rmse

        # Create a temporary directory for the artifacts.
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir)
            embeddings_path = artifact_dir / "movie_cf_embeddings.parquet"
            embedding_metrics = export_movie_embeddings(
                model,
                client,
                settings.s3_bucket,
                cf_dataset_version,
                embeddings_path,
                num_movies=dataset_manifest.num_movies,
                rated_movie_indices=rated_movie_indices,
            )

            # Finalize the training metrics.
            final_metrics = finalize_training_metrics(
                embedding_metrics,
                best_epoch=best_epoch + 1,
                best_validation_rmse=best_rmse,
                best_validation_mae=best_mae,
                num_train_rows=dataset_manifest.train_row_count,
                num_validation_rows=dataset_manifest.validation_row_count,
                num_users=dataset_manifest.num_users,
                num_movies=dataset_manifest.num_movies,
            )

            # Check if the exported embeddings contain any NaN rows.
            if final_metrics.nan_embedding_count != 0:
                raise RuntimeError(
                    f"Exported embeddings contain {final_metrics.nan_embedding_count} NaN rows"
                )

            # Create the CF training config.
            config = CfTrainingConfig(
                cf_version=cf_version,
                cf_dataset_version=cf_dataset_version,
                snapshot_id=dataset_manifest.snapshot_id,
                embedding_dim=settings.embedding_dim,
                num_epochs=settings.num_epochs,
                batch_size=settings.batch_size,
                learning_rate=settings.learning_rate,
                early_stopping_patience=settings.early_stopping_patience,
                shuffle_seed=settings.shuffle_seed,
                train_fraction=dataset_manifest.train_fraction,
                validation_fraction=dataset_manifest.validation_fraction,
                test_fraction=dataset_manifest.test_fraction,
                device=str(device),
                pipeline_run_id=pipeline_run_id,
                created_at=created_at,
            )
            # Save the model state dictionary.
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_users": dataset_manifest.num_users,
                    "num_movies": dataset_manifest.num_movies,
                    "embedding_dim": settings.embedding_dim,
                    "best_epoch": best_epoch + 1,
                },
                artifact_dir / "cf_model.pt",
            )
            # Save the CF training config.
            (artifact_dir / "cf_config.json").write_text(
                json.dumps(config.model_dump(mode="json"), indent=2),
                encoding="utf-8",
            )
            # Save the CF training metrics.
            (artifact_dir / "cf_metrics.json").write_text(
                json.dumps(final_metrics.model_dump(mode="json"), indent=2),
                encoding="utf-8",
            )
            # Save the training curve.
            save_training_curve(
                artifact_dir / "training_curve.png",
                train_losses=stats.train_losses,
                validation_rmses=stats.validation_rmses,
                validation_maes=stats.validation_maes,
            )

            # Upload the artifacts to MinIO.
            _upload_artifacts(client, settings.s3_bucket, cf_version, artifact_dir)

            # Build the complete CF manifest.
            finished_at = datetime.now(tz=UTC)
            manifest = build_complete_cf_manifest(
                cf_version=cf_version,
                cf_dataset_version=cf_dataset_version,
                snapshot_id=dataset_manifest.snapshot_id,
                embedding_dim=settings.embedding_dim,
                pipeline_run_id=pipeline_run_id,
                created_at=created_at,
                finished_at=finished_at,
                metrics=final_metrics,
            )

            # Save the complete CF manifest.
            put_json(
                client,
                settings.s3_bucket,
                cf_artifact_manifest_object_key(cf_version),
                manifest.model_dump(mode="json"),
            )

            log_final_metrics(final_metrics)
            log_cf_artifacts(artifact_dir)

    return stats

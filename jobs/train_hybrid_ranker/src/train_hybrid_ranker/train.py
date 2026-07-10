"""Core hybrid ranker PyTorch training loop and artifact upload."""

from __future__ import annotations

import json
import time
import torch
import mlflow
import random
import logging
import tempfile
import torch.nn as nn

from pathlib import Path
from copy import deepcopy
from datetime import UTC, datetime
from botocore.client import BaseClient
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from train_hybrid_ranker.device import resolve_device
from train_hybrid_ranker.model import HybridRankerMLP
from common.config.settings import HybridTrainingSettings
from train_hybrid_ranker.plot import save_training_curve
from train_hybrid_ranker.version import resolve_model_version
from train_hybrid_ranker.dataset import HybridParquetIterableDataset
from common.features.schema import FEATURE_SCHEMA_VERSION, INPUT_DIM
from train_hybrid_ranker.evaluate import _collate_features_batch, evaluate_validation_streaming

from common.mlflow.hybrid_run import (
    log_epoch_metrics,
    log_final_training_metrics,
    log_hybrid_artifacts,
    log_training_params,
    start_hybrid_training_run,
)

from common.schemas.hybrid_ranker_artifact_manifest import (
    MODEL_ARCHITECTURE,
    HybridModelConfig,
    HybridTrainingMetrics,
)

from common.storage.hybrid_ranker_dataset_reader import (
    load_hybrid_ranker_dataset_manifest,
    resolve_hybrid_ranker_dataset_version,
)

from common.storage.s3 import (
    hybrid_ranker_model_config_object_key,
    hybrid_ranker_model_object_key,
    hybrid_ranker_training_curve_object_key,
    hybrid_ranker_training_metrics_object_key,
    upload_file,
)


logger = logging.getLogger(__name__)

_TRAIN_BATCH_LOG_INTERVAL = 50


@dataclass
class HybridTrainingStats:
    """Counters and metadata collected during one hybrid training run."""

    model_version: str
    dataset_version: str
    snapshot_id: str
    cf_version: str
    best_validation_rmse: float | None = None
    train_losses: list[float] = field(default_factory=list)
    validation_rmses: list[float] = field(default_factory=list)
    validation_maes: list[float] = field(default_factory=list)


def _train_one_epoch(model: HybridRankerMLP, loader: DataLoader, optimizer: torch.optim.Optimizer, \
                        loss_fn: nn.Module, device: torch.device, *, epoch: int) -> float:
    """
    Run one training epoch and return the average MSE loss.

    ============================ Arguments ============================
    model: Hybrid ranker MLP in train mode.
    loader: DataLoader over shuffled train parts.
    optimizer: Optimizer for model weights.
    loss_fn: Regression loss (MSELoss).
    device: Torch device for training.

    ============================ Returns ============================
    Average training loss for the epoch.
    """
    # Set the model to training mode.
    model.train()
    # Initialize counters for total loss and total rows.
    total_loss = 0.0
    total_rows = 0
    batch_index = 0

    # Iterate over the loader and train the model on each batch.
    for batch_index, (features, rating) in enumerate(loader, start=1):
        # Move the features and rating to the device.
        feature_tensor = features.to(device)
        target = rating.to(device)

        # Zero the gradients.
        optimizer.zero_grad(set_to_none=True)
        # Forward pass to get predictions.
        predictions = model(feature_tensor)
        # Compute the loss.
        loss = loss_fn(predictions, target)
        loss.backward()
        optimizer.step()

        # Accumulate the loss and total rows.
        batch_size = int(target.numel())
        total_loss += float(loss.item()) * batch_size
        total_rows += batch_size

        if batch_index % _TRAIN_BATCH_LOG_INTERVAL == 0:
            logger.info(
                "Training epoch %d progress batches=%d rows=%d avg_loss=%.6f",
                epoch + 1,
                batch_index,
                total_rows,
                total_loss / max(1, total_rows),
            )

    logger.info(
        "Training epoch %d finished batches=%d rows=%d avg_loss=%.6f",
        epoch + 1,
        batch_index,
        total_rows,
        total_loss / max(1, total_rows),
    )

    return total_loss / max(1, total_rows)


def _upload_training_artifacts(client: BaseClient, bucket: str, model_version: str, local_dir: Path) -> None:
    """
    Upload hybrid ranker training artifacts to MinIO (manifest excluded).

    ============================ Arguments ============================
    client: The boto3 S3 client.
    bucket: Target MinIO/S3 bucket.
    model_version: Model version identifier.
    local_dir: Local directory containing artifact files.
    """
    # Upload the model state dictionary.
    logger.info(
        "Uploading hybrid ranker artifacts bucket=%s model_version=%s",
        bucket,
        model_version,
    )
    upload_file(
        client,
        bucket,
        hybrid_ranker_model_object_key(model_version),
        local_dir / "hybrid_ranker_model.pt",
    )
    # Upload the model config.
    upload_file(
        client,
        bucket,
        hybrid_ranker_model_config_object_key(model_version),
        local_dir / "model_config.json",
    )
    # Upload the training metrics.
    upload_file(
        client,
        bucket,
        hybrid_ranker_training_metrics_object_key(model_version),
        local_dir / "training_metrics.json",
    )
    # Upload the training curve.
    upload_file(
        client,
        bucket,
        hybrid_ranker_training_curve_object_key(model_version),
        local_dir / "training_curve.png",
    )
    logger.info(
        "Uploaded hybrid ranker artifacts bucket=%s model_version=%s",
        bucket,
        model_version,
    )


def run_hybrid_training(client: BaseClient, settings: HybridTrainingSettings, *, pipeline_run_id: str) -> HybridTrainingStats:
    """
    Train the hybrid ranker, export artifacts, and log metrics to MLflow.

    Do this by:
    1. Resolving the input hybrid dataset and output model_version.
    2. Training with streaming IterableDataset over train parts.
    3. Evaluating validation RMSE/MAE each epoch with optional early stopping.
    4. Uploading model artifacts (manifest written later by evaluate_model).
    5. Logging metrics and artifacts to MLflow.

    ============================ Arguments ============================
    client: The boto3 S3 client.
    settings: Hybrid training configuration.
    pipeline_run_id: pipeline_runs.run_id for lineage.

    ============================ Returns ============================
    HybridTrainingStats with version ids and best validation RMSE.
    """
    created_at = datetime.now(tz=UTC)

    # Resolve the dataset version and manifest.
    logger.info("Resolving hybrid ranker dataset version requested=%s", settings.dataset_version)
    dataset_version = resolve_hybrid_ranker_dataset_version(
        client,
        settings.s3_bucket,
        settings.dataset_version,
    )
    # Load the dataset manifest.
    dataset_manifest = load_hybrid_ranker_dataset_manifest(client, settings.s3_bucket, dataset_version)
    logger.info(
        "Resolved hybrid dataset version=%s status=%s snapshot_id=%s cf_version=%s",
        dataset_version,
        dataset_manifest.status,
        dataset_manifest.snapshot_id,
        dataset_manifest.cf_version,
    )
    
    # Check if the dataset is complete.
    if dataset_manifest.status != "complete":
        raise ValueError(f"Hybrid dataset {dataset_version} is not complete")
    
    # Check if the feature schema is supported.
    if dataset_manifest.feature_schema_version != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported feature schema {dataset_manifest.feature_schema_version}; "
            f"expected {FEATURE_SCHEMA_VERSION}"
        )

    # Check if the input dimension matches the expected dimension.
    if dataset_manifest.input_dim != INPUT_DIM:
        raise ValueError(
            f"Dataset input_dim {dataset_manifest.input_dim} does not match expected {INPUT_DIM}"
        )

    # Resolve the model version.
    model_version = resolve_model_version(settings)
    logger.info(
        "Hybrid dataset ready train_rows=%d validation_rows=%d train_parts=%d validation_parts=%d "
        "input_dim=%d feature_schema=%s",
        dataset_manifest.train_row_count,
        dataset_manifest.validation_row_count,
        len(dataset_manifest.train_parts),
        len(dataset_manifest.validation_parts),
        dataset_manifest.input_dim,
        dataset_manifest.feature_schema_version,
    )
    # Resolve the device.
    device = resolve_device(settings.device)
    if device.type == "cuda":
        logger.info(
            "Hybrid ranker training compute device=cuda gpu=%s device_setting=%s",
            torch.cuda.get_device_name(device),
            settings.device,
        )
    else:
        logger.info(
            "Hybrid ranker training compute device=cpu device_setting=%s",
            settings.device,
        )

    # Initialize the training statistics.
    stats = HybridTrainingStats(
        model_version=model_version,
        dataset_version=dataset_version,
        snapshot_id=dataset_manifest.snapshot_id,
        cf_version=dataset_manifest.cf_version,
    )

    # Create the model.
    model = HybridRankerMLP(
        input_dim=dataset_manifest.input_dim,
        hidden_dims=settings.hidden_dims,
        dropout=settings.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
    loss_fn = nn.MSELoss()
    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    logger.info(
        "Hybrid model initialized model_version=%s parameters=%d hidden_dims=%s dropout=%g",
        model_version,
        num_parameters,
        settings.hidden_dims,
        settings.dropout,
    )
    logger.info(
        "Training hyperparameters epochs=%d batch_size=%d learning_rate=%g "
        "early_stopping_patience=%d shuffle_seed=%d",
        settings.num_epochs,
        settings.batch_size,
        settings.learning_rate,
        settings.early_stopping_patience,
        settings.shuffle_seed,
    )

    # Initialize the best state dictionary, epoch, RMSE, MAE, and epochs without improvement.
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_rmse = float("inf")
    best_mae = float("inf")
    epochs_without_improvement = 0

    # Start the hybrid training run.
    with start_hybrid_training_run(
        settings,
        model_version=model_version,
        dataset_version=dataset_version,
        snapshot_id=dataset_manifest.snapshot_id,
        cf_dataset_version=dataset_manifest.cf_dataset_version,
        cf_version=dataset_manifest.cf_version,
        content_embedding_version=dataset_manifest.content_embedding_version,
        feature_schema_version=dataset_manifest.feature_schema_version,
        input_dim=dataset_manifest.input_dim,
        model_architecture=MODEL_ARCHITECTURE,
        learning_rate=settings.learning_rate,
        batch_size=settings.batch_size,
        dropout=settings.dropout,
        num_epochs=settings.num_epochs,
        shuffle_seed=settings.shuffle_seed,
    ):
        # Get the MLflow run ID.
        mlflow_run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
        logger.info("MLflow training run started run_id=%s", mlflow_run_id)
        # Log the training parameters.
        log_training_params(
            {
                "device": str(device),
                "early_stopping_patience": settings.early_stopping_patience,
                "hidden_dims": settings.hidden_dims,
            }
        )

        # Train the model for the number of epochs.
        for epoch in range(settings.num_epochs):
            # Start the epoch timer.
            epoch_start = time.perf_counter()

            # Shuffle the train parts.
            train_parts = list(dataset_manifest.train_parts)
            rng = random.Random(settings.shuffle_seed + epoch)
            rng.shuffle(train_parts)
            logger.info(
                "Starting epoch %d/%d train_parts=%d",
                epoch + 1,
                settings.num_epochs,
                len(train_parts),
            )

            # Create the train dataset.
            train_dataset = HybridParquetIterableDataset(
                client,
                settings.s3_bucket,
                train_parts,
                shuffle_within_part=True,
                seed=settings.shuffle_seed + epoch,
            )
            # Create the train loader.
            train_loader = DataLoader(
                train_dataset,
                batch_size=settings.batch_size,
                num_workers=0,
                collate_fn=_collate_features_batch,
            )

            # Train the model for one epoch.
            train_loss = _train_one_epoch(
                model,
                train_loader,
                optimizer,
                loss_fn,
                device,
                epoch=epoch,
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
            
            # Append the training loss, validation RMSE, and validation MAE to the statistics.
            stats.train_losses.append(train_loss)
            stats.validation_rmses.append(val_rmse)
            stats.validation_maes.append(val_mae)

            # Log the epoch metrics.
            log_epoch_metrics(
                epoch=epoch,
                train_loss=train_loss,
                validation_rmse=val_rmse,
                validation_mae=val_mae,
                epoch_duration_seconds=epoch_duration,
            )

            logger.info(
                "Hybrid training epoch %d/%d complete train_loss=%.6f validation_rmse=%.6f "
                "validation_mae=%.6f duration_seconds=%.1f",
                epoch + 1,
                settings.num_epochs,
                train_loss,
                val_rmse,
                val_mae,
                epoch_duration,
            )

            # Check if the validation RMSE is the best so far.
            if val_rmse < best_rmse:
                previous_best_rmse = best_rmse
                best_rmse = val_rmse
                best_mae = val_mae
                best_epoch = epoch
                best_state_dict = deepcopy(model.state_dict())
                epochs_without_improvement = 0
                logger.info(
                    "New best checkpoint epoch=%d validation_rmse=%.6f validation_mae=%.6f "
                    "previous_best_rmse=%.6f",
                    epoch + 1,
                    val_rmse,
                    val_mae,
                    previous_best_rmse,
                )
            else:
                epochs_without_improvement += 1
                logger.info(
                    "No validation improvement epoch=%d validation_rmse=%.6f best_rmse=%.6f "
                    "epochs_without_improvement=%d/%d",
                    epoch + 1,
                    val_rmse,
                    best_rmse,
                    epochs_without_improvement,
                    settings.early_stopping_patience,
                )

            # Check if the early stopping patience has been reached.
            if (
                settings.early_stopping_patience > 0
                and epochs_without_improvement >= settings.early_stopping_patience
            ):
                logger.info(
                    "Early stopping triggered epoch=%d best_epoch=%d best_validation_rmse=%.6f "
                    "best_validation_mae=%.6f patience=%d",
                    epoch + 1,
                    best_epoch + 1,
                    best_rmse,
                    best_mae,
                    settings.early_stopping_patience,
                )
                break

        logger.info(
            "Training loop finished best_epoch=%d best_validation_rmse=%.6f best_validation_mae=%.6f "
            "epochs_run=%d",
            best_epoch + 1,
            best_rmse,
            best_mae,
            len(stats.train_losses),
        )

        # Check if the best state dictionary is None, raise an error if it is.
        if best_state_dict is None:
            raise RuntimeError("Training finished without a best checkpoint")

        # Load the best state dictionary into the model.
        model.load_state_dict(best_state_dict)
        # Set the best validation RMSE in the statistics.
        stats.best_validation_rmse = best_rmse

        # Create the final training metrics.
        final_metrics = HybridTrainingMetrics(
            best_epoch=best_epoch + 1,
            best_validation_rmse=best_rmse,
            best_validation_mae=best_mae,
            num_train_rows=dataset_manifest.train_row_count,
            num_validation_rows=dataset_manifest.validation_row_count,
            model_architecture=MODEL_ARCHITECTURE,
        )

        # Create the model config.
        config = HybridModelConfig(
            model_version=model_version,
            dataset_version=dataset_version,
            snapshot_id=dataset_manifest.snapshot_id,
            cf_dataset_version=dataset_manifest.cf_dataset_version,
            cf_version=dataset_manifest.cf_version,
            content_embedding_version=dataset_manifest.content_embedding_version,
            feature_schema_version=dataset_manifest.feature_schema_version,
            input_dim=dataset_manifest.input_dim,
            model_architecture=MODEL_ARCHITECTURE,
            hidden_dims=settings.hidden_dims,
            dropout=settings.dropout,
            num_epochs=settings.num_epochs,
            batch_size=settings.batch_size,
            learning_rate=settings.learning_rate,
            early_stopping_patience=settings.early_stopping_patience,
            shuffle_seed=settings.shuffle_seed,
            device=str(device),
            pipeline_run_id=pipeline_run_id,
            mlflow_run_id=mlflow_run_id,
            metadata_normalization=dataset_manifest.metadata_normalization,
            created_at=created_at,
        )

        # Create a temporary directory to store the artifacts.
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": dataset_manifest.input_dim,
                    "hidden_dims": settings.hidden_dims,
                    "dropout": settings.dropout,
                    "best_epoch": best_epoch + 1,
                    "model_architecture": MODEL_ARCHITECTURE,
                },
                artifact_dir / "hybrid_ranker_model.pt",
            )
            (artifact_dir / "model_config.json").write_text(
                json.dumps(config.model_dump(mode="json"), indent=2),
                encoding="utf-8",
            )
            (artifact_dir / "training_metrics.json").write_text(
                json.dumps(final_metrics.model_dump(mode="json"), indent=2),
                encoding="utf-8",
            )

            # Save the training curv to the artifact directory.
            save_training_curve(
                artifact_dir / "training_curve.png",
                train_losses=stats.train_losses,
                validation_rmses=stats.validation_rmses,
                validation_maes=stats.validation_maes,
            )
            logger.info(
                "Prepared local artifacts model_version=%s best_epoch=%d",
                model_version,
                best_epoch + 1,
            )

            # Upload the artifacts to the S3 bucket.
            _upload_training_artifacts(client, settings.s3_bucket, model_version, artifact_dir)
            # Log the final training metrics.
            log_final_training_metrics(final_metrics)
            # Log the hybrid artifacts.
            log_hybrid_artifacts(artifact_dir)
            logger.info(
                "Hybrid training complete model_version=%s dataset_version=%s mlflow_run_id=%s",
                model_version,
                dataset_version,
                mlflow_run_id,
            )

    return stats

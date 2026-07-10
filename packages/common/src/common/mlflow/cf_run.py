"""MLflow logging helpers for collaborative filtering training runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow

from common.config.settings import CfTrainingSettings
from common.schemas.cf_artifact_manifest import CfTrainingMetrics


def configure_mlflow(settings: CfTrainingSettings) -> None:
    """
    Point the MLflow client at the configured tracking URI and experiment.

    ============================ Arguments ============================
    settings: CF training configuration with MLflow URI and experiment name.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)


def start_cf_training_run(
    settings: CfTrainingSettings,
    *,
    cf_version: str,
    cf_dataset_version: str,
    snapshot_id: str,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> mlflow.ActiveRun:
    """
    Start one MLflow run and set all required CF training tags.

    Do this by:
    1. Configuring the tracking URI and experiment.
    2. Starting a run named after cf_version.
    3. Setting lineage and hyperparameter tags once at run start.

    ============================ Arguments ============================
    settings: CF training configuration.
    cf_version: Version id for this CF artifact bundle.
    cf_dataset_version: Input CF dataset version used for training.
    snapshot_id: Snapshot lineage from the CF dataset manifest.
    train_fraction: Train split fraction from the CF dataset manifest.
    validation_fraction: Validation split fraction from the CF dataset manifest.
    test_fraction: Locked test split fraction (lineage only; not used in CF training).

    ============================ Returns ============================
    The active MLflow run context manager.
    """
    configure_mlflow(settings)
    run = mlflow.start_run(run_name=cf_version)
    log_cf_tags(
        cf_version=cf_version,
        cf_dataset_version=cf_dataset_version,
        snapshot_id=snapshot_id,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        embedding_dim=settings.embedding_dim,
        learning_rate=settings.learning_rate,
        batch_size=settings.batch_size,
        num_epochs=settings.num_epochs,
        optimizer="Adam",
        loss_function="MSELoss",
        shuffle_seed=settings.shuffle_seed,
        model_type="dot_product_cf",
    )
    return run


def log_cf_tags(**tags: str | int | float) -> None:
    """
    Set CF training lineage and hyperparameter tags on the active MLflow run.

    ============================ Arguments ============================
    tags: Tag name/value pairs required by the observability contract.
    """
    for key, value in tags.items():
        mlflow.set_tag(key, str(value))


def log_epoch_metrics(*, epoch: int, train_loss: float, validation_rmse: float, validation_mae: float, 
                        epoch_duration_seconds: float) -> None:
    """
    Log per-epoch CF training metrics to the active MLflow run.

    ============================ Arguments ============================
    epoch: Zero-based epoch index.
    train_loss: Average training MSE loss for the epoch.
    validation_rmse: Validation RMSE for the epoch.
    validation_mae: Validation MAE for the epoch.
    epoch_duration_seconds: Wall-clock time for the epoch.
    """
    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("validation_rmse", validation_rmse, step=epoch)
    mlflow.log_metric("validation_mae", validation_mae, step=epoch)
    mlflow.log_metric("epoch_duration_seconds", epoch_duration_seconds, step=epoch)


def log_final_metrics(metrics: CfTrainingMetrics) -> None:
    """
    Log final CF training metrics once after training completes.

    ============================ Arguments ============================
    metrics: Final metric summary validated by CfTrainingMetrics.
    """
    payload = metrics.model_dump()
    for key, value in payload.items():
        mlflow.log_metric(key, value)


def log_cf_artifacts(local_dir: Path) -> None:
    """
    Upload all CF training artifacts from a local directory to MLflow.

    ============================ Arguments ============================
    local_dir: Directory containing cf_model.pt, parquet, json, and png files.
    """
    mlflow.log_artifacts(str(local_dir))


def log_training_params(extra: dict[str, Any] | None = None) -> None:
    """
    Log optional extra MLflow params on the active run.

    ============================ Arguments ============================
    extra: Additional param key/value pairs to log.
    """
    if extra:
        mlflow.log_params({k: str(v) for k, v in extra.items()})

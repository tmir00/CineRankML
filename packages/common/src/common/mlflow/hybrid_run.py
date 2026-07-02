"""MLflow logging helpers for hybrid ranker training and evaluation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import mlflow

from common.schemas.hybrid_ranker_artifact_manifest import HybridTestMetrics, HybridTrainingMetrics


class HybridMlflowSettings(Protocol):
    """Minimal settings shape for hybrid ranker MLflow configuration."""

    mlflow_tracking_uri: str
    mlflow_experiment_name: str


def configure_mlflow(settings: HybridMlflowSettings) -> None:
    """
    Point the MLflow client at the configured tracking URI and experiment.

    ============================ Arguments ============================
    settings: Hybrid training configuration with MLflow URI and experiment name.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)


def start_hybrid_training_run(settings: HybridMlflowSettings, *, model_version: str, dataset_version: str, snapshot_id: str, \
                                    cf_dataset_version: str, cf_version: str, content_embedding_version: str, \
                                        feature_schema_version: str, input_dim: int, model_architecture: str, \
                                            learning_rate: float, batch_size: int, dropout: float, \
                                                num_epochs: int, shuffle_seed: int) -> mlflow.ActiveRun:
    """
    Start one MLflow run and set all required hybrid ranker training tags.

    Do this by:
    1. Configuring the tracking URI and experiment.
    2. Starting a run named after model_version.
    3. Setting lineage and hyperparameter tags once at run start.

    ============================ Arguments ============================
    settings: Hybrid training configuration.
    model_version: Version id for this model artifact bundle.
    dataset_version: Input hybrid feature dataset version.
    snapshot_id: Snapshot lineage from the dataset manifest.
    cf_dataset_version: CF dataset version from the dataset manifest.
    cf_version: CF artifact version from the dataset manifest.
    content_embedding_version: Content embedding version from the dataset manifest.
    feature_schema_version: Feature schema version from the dataset manifest.
    input_dim: Model input dimension (1356).
    model_architecture: Human-readable layer description.
    learning_rate: Optimizer learning rate.
    batch_size: Training batch size.
    dropout: Dropout rate between hidden layers.
    num_epochs: Maximum training epochs.
    shuffle_seed: RNG seed for part shuffling.

    ============================ Returns ============================
    The active MLflow run context manager.
    """
    configure_mlflow(settings)
    run = mlflow.start_run(run_name=model_version)
    log_hybrid_tags(
        model_version=model_version,
        dataset_version=dataset_version,
        snapshot_id=snapshot_id,
        cf_dataset_version=cf_dataset_version,
        cf_version=cf_version,
        content_embedding_version=content_embedding_version,
        feature_schema_version=feature_schema_version,
        input_dim=input_dim,
        model_architecture=model_architecture,
        learning_rate=learning_rate,
        batch_size=batch_size,
        dropout=dropout,
        num_epochs=num_epochs,
        optimizer="AdamW",
        loss_function="MSELoss",
        shuffle_seed=shuffle_seed,
        model_type="hybrid_ranker_mlp",
    )
    return run


def resume_hybrid_mlflow_run(mlflow_run_id: str) -> mlflow.ActiveRun:
    """
    Resume an existing MLflow run so evaluation metrics land on the training run.

    ============================ Arguments ============================
    mlflow_run_id: MLflow run id stored in model_config.json.

    ============================ Returns ============================
    The resumed active MLflow run context manager.
    """
    return mlflow.start_run(run_id=mlflow_run_id)


def log_hybrid_tags(**tags: str | int | float) -> None:
    """
    Set hybrid ranker lineage and hyperparameter tags on the active MLflow run.

    ============================ Arguments ============================
    tags: Tag name/value pairs required by the observability contract.
    """
    for key, value in tags.items():
        mlflow.set_tag(key, str(value))


def log_epoch_metrics(*, epoch: int, train_loss: float, validation_rmse: float, validation_mae: float, \
                        epoch_duration_seconds: float) -> None:
    """
    Log per-epoch hybrid training metrics to the active MLflow run.

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


def log_final_training_metrics(metrics: HybridTrainingMetrics) -> None:
    """
    Log final hybrid training metrics once after training completes.

    ============================ Arguments ============================
    metrics: Final metric summary validated by HybridTrainingMetrics.
    """
    payload = metrics.model_dump()
    for key, value in payload.items():
        # Skip the model architecture key. This is not a metric it's a string description.
        if key == "model_architecture":
            continue
        mlflow.log_metric(key, value)


def log_evaluation_metrics(metrics: HybridTestMetrics) -> None:
    """
    Log test regression and ranking metrics after evaluation completes.

    ============================ Arguments ============================
    metrics: Final test metric summary validated by HybridTestMetrics.
    """
    payload = metrics.model_dump()
    for key, value in payload.items():
        # Skip the num_* keys. These are not model quality scores.
        if key.startswith("num_"):
            continue
        mlflow.log_metric(key, value)


def log_hybrid_artifacts(local_dir: Path) -> None:
    """
    Upload all hybrid ranker artifacts from a local directory to MLflow.

    ============================ Arguments ============================
    local_dir: Directory containing model checkpoint, json, and png files.
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

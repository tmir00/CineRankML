"""Pydantic models for CF training artifact manifests and metric payloads."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field


class CfTrainingMetrics(BaseModel):
    """Final CF training metrics mirrored in cf_metrics.json and MLflow."""

    best_epoch: int
    best_validation_rmse: float
    best_validation_mae: float
    num_train_rows: int
    num_validation_rows: int
    num_users: int
    num_movies: int
    movie_embedding_coverage: float
    default_embedding_count: int
    nan_embedding_count: int
    embedding_norm_mean: float
    embedding_norm_std: float


class CfTrainingConfig(BaseModel):
    """Full CF training run configuration written to cf_config.json."""

    cf_version: str
    cf_dataset_version: str
    snapshot_id: str
    embedding_dim: int
    num_epochs: int
    batch_size: int
    learning_rate: float
    early_stopping_patience: int
    shuffle_seed: int
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    optimizer: str = "Adam"
    loss_function: str = "MSELoss"
    model_type: str = "dot_product_cf"
    device: str
    pipeline_run_id: str
    created_at: datetime


class CfArtifactManifest(BaseModel):
    """Complete CF artifact manifest; consumers require status=complete."""

    cf_version: str
    cf_dataset_version: str
    snapshot_id: str
    embedding_dim: int
    status: Literal["complete", "failed"]
    movie_cf_embeddings_path: str
    cf_model_path: str
    cf_config_path: str
    cf_metrics_path: str
    training_curve_path: str
    metrics: CfTrainingMetrics
    created_at: datetime
    finished_at: datetime | None = None
    pipeline_run_id: str
    extra: dict[str, Any] = Field(default_factory=dict)

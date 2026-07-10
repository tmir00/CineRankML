"""Pydantic models for hybrid ranker model artifacts written to MinIO."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field
from common.features.normalization import MetadataNormalizationStats


MODEL_ARCHITECTURE = "1356->512->256->64->1"


class HybridTrainingMetrics(BaseModel):
    """Final hybrid training metrics mirrored in training_metrics.json and MLflow."""

    best_epoch: int
    best_validation_rmse: float
    best_validation_mae: float
    num_train_rows: int
    num_validation_rows: int
    model_architecture: str = MODEL_ARCHITECTURE


class HybridTestMetrics(BaseModel):
    """Test-split regression and ranking metrics written by evaluate_model."""

    test_rmse: float
    test_mae: float
    precision_at_5: float
    precision_at_10: float
    recall_at_5: float
    recall_at_10: float
    ndcg_at_5: float
    ndcg_at_10: float
    mrr_at_10: float
    num_test_rows: int
    num_users_evaluated: int


class HybridModelConfig(BaseModel):
    """Full hybrid ranker training run configuration written to model_config.json."""

    model_version: str
    dataset_version: str
    snapshot_id: str
    cf_dataset_version: str
    cf_version: str
    content_embedding_version: str
    feature_schema_version: str
    input_dim: int
    model_architecture: str = MODEL_ARCHITECTURE
    hidden_dims: list[int] = Field(default_factory=lambda: [512, 256, 64])
    dropout: float
    num_epochs: int
    batch_size: int
    learning_rate: float
    early_stopping_patience: int
    shuffle_seed: int
    optimizer: str = "AdamW"
    loss_function: str = "MSELoss"
    model_type: str = "hybrid_ranker_mlp"
    device: str
    pipeline_run_id: str
    mlflow_run_id: str | None = None
    metadata_normalization: MetadataNormalizationStats
    created_at: datetime


class HybridRankerArtifactManifest(BaseModel):
    """Complete hybrid ranker model manifest; consumers require status=complete."""

    model_version: str
    dataset_version: str
    snapshot_id: str
    cf_dataset_version: str
    cf_version: str
    content_embedding_version: str
    feature_schema_version: str
    input_dim: int
    status: Literal["complete", "failed"]
    hybrid_ranker_model_path: str
    model_config_path: str
    training_metrics_path: str
    test_metrics_path: str
    training_curve_path: str
    training_metrics: HybridTrainingMetrics
    test_metrics: HybridTestMetrics
    created_at: datetime
    finished_at: datetime | None = None
    pipeline_run_id: str
    extra: dict[str, Any] = Field(default_factory=dict)

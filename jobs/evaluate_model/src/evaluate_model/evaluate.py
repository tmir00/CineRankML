"""Core hybrid ranker evaluation orchestration."""

from __future__ import annotations

import json
import torch
import logging
import tempfile

from pathlib import Path
from dataclasses import dataclass
from datetime import UTC, datetime
from botocore.client import BaseClient
from train_hybrid_ranker.device import resolve_device
from train_hybrid_ranker.model import HybridRankerMLP
from common.config.settings import EvaluateModelSettings
from evaluate_model.regression import evaluate_test_split
from evaluate_model.manifest import build_complete_hybrid_manifest
from common.schemas.hybrid_ranker_artifact_manifest import HybridTestMetrics
from common.features.schema import FEATURE_SCHEMA_VERSION, HIGH_RATED_THRESHOLD, INPUT_DIM

from common.storage.s3 import (
    hybrid_ranker_model_manifest_object_key,
    hybrid_ranker_test_metrics_object_key,
    put_json,
)


from common.mlflow.hybrid_run import (
    configure_mlflow,
    log_evaluation_metrics,
    resume_hybrid_mlflow_run,
    start_hybrid_training_run,
)


from common.storage.hybrid_ranker_dataset_reader import (
    load_hybrid_ranker_dataset_manifest,
    resolve_hybrid_ranker_dataset_version,
)

from common.storage.hybrid_ranker_model_reader import (
    load_hybrid_model_checkpoint_dict,
    load_hybrid_model_config,
    load_hybrid_training_metrics,
    resolve_hybrid_ranker_model_version,
)


logger = logging.getLogger(__name__)


@dataclass
class HybridEvaluationStats:
    """Summary returned after one evaluation run."""

    model_version: str
    dataset_version: str
    test_rmse: float
    test_mae: float


def _load_model_from_checkpoint(checkpoint: dict, device: torch.device) -> HybridRankerMLP:
    """
    Rebuild the hybrid ranker MLP from a saved checkpoint dict.

    ============================ Arguments ============================
    checkpoint: Parsed hybrid_ranker_model.pt payload.
    device: Torch device for evaluation.

    ============================ Returns ============================
    The loaded model in eval mode on the target device.
    """
    model = HybridRankerMLP(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dims=list(checkpoint["hidden_dims"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _validate_lineage(model_config_dataset_version: str, dataset_manifest_version: str) -> None:
    """Raise when the evaluation dataset does not match the trained model lineage."""
    if model_config_dataset_version != dataset_manifest_version:
        raise ValueError(
            f"Dataset version mismatch: model was trained on {model_config_dataset_version}, "
            f"but evaluation requested {dataset_manifest_version}"
        )


def run_hybrid_evaluation(client: BaseClient, settings: EvaluateModelSettings, *, pipeline_run_id: str) -> HybridEvaluationStats:
    """
    Evaluate a trained hybrid ranker on the held-out test split.

    Do this by:
    1. Loading the trained model and dataset manifests from MinIO.
    2. Streaming test rows for regression and ranking metrics.
    3. Writing test_metrics.json and manifest.json last.
    4. Logging final metrics to MLflow.

    ============================ Arguments ============================
    client: The boto3 S3 client.
    settings: Evaluation job configuration.
    pipeline_run_id: pipeline_runs.run_id for lineage.

    ============================ Returns ============================
    HybridEvaluationStats with test regression metrics.
    """
    # Resolve the model version.
    model_version = resolve_hybrid_ranker_model_version(
        client,
        settings.s3_bucket,
        settings.model_version,
    )
    # Load the model config based on the model version.
    model_config = load_hybrid_model_config(client, settings.s3_bucket, model_version)
    # Load the training metrics based on the model version.
    training_metrics = load_hybrid_training_metrics(client, settings.s3_bucket, model_version)
    
    # Resolve the dataset version and validate the lineage (make sure the dataset version is the same as the model version).
    dataset_version = settings.dataset_version or model_config.dataset_version
    _validate_lineage(model_config.dataset_version, dataset_version)

    # Load the dataset manifest based on the dataset version.
    dataset_manifest = load_hybrid_ranker_dataset_manifest(client, settings.s3_bucket, dataset_version)
    # Check if the dataset manifest is complete.
    if dataset_manifest.status != "complete":
        raise ValueError(f"Hybrid dataset {dataset_version} is not complete")
    # Check if the feature schema version is supported.
    if dataset_manifest.feature_schema_version != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported feature schema {dataset_manifest.feature_schema_version}; "
            f"expected {FEATURE_SCHEMA_VERSION}"
        )
    # Check if the input dimension is correct.
    if dataset_manifest.input_dim != INPUT_DIM:
        raise ValueError(
            f"Dataset input_dim {dataset_manifest.input_dim} does not match expected {INPUT_DIM}"
        )

    # Check if the cf version is correct.
    if model_config.cf_version != dataset_manifest.cf_version:
        raise ValueError("Model cf_version does not match dataset cf_version")
    # Check if the feature schema version is correct.
    if model_config.feature_schema_version != dataset_manifest.feature_schema_version:
        raise ValueError("Model feature_schema_version does not match dataset feature_schema_version")
    # Check if the input dimension is correct.
    if model_config.input_dim != dataset_manifest.input_dim:
        raise ValueError("Model input_dim does not match dataset input_dim")

    # Resolve the device.
    device = resolve_device(settings.device)
    # Load the model checkpoint based on the model version.
    checkpoint = load_hybrid_model_checkpoint_dict(client, settings.s3_bucket, model_version)
    # Load the model from the checkpoint.
    model = _load_model_from_checkpoint(checkpoint, device)
    
    # Evaluate the test split.
    test_rmse, test_mae, ranking_accumulator, num_test_rows = evaluate_test_split(
        model,
        client,
        settings.s3_bucket,
        dataset_manifest.test_parts,
        batch_size=settings.batch_size,
        device=device,
        relevance_threshold=HIGH_RATED_THRESHOLD,
    )
    # Compute the ranking averages.
    ranking_averages = ranking_accumulator.averages()
    
    # Build the test metrics to be written to the manifest.
    test_metrics = HybridTestMetrics(
        test_rmse=test_rmse,
        test_mae=test_mae,
        precision_at_5=ranking_averages["precision_at_5"],
        precision_at_10=ranking_averages["precision_at_10"],
        recall_at_5=ranking_averages["recall_at_5"],
        recall_at_10=ranking_averages["recall_at_10"],
        ndcg_at_5=ranking_averages["ndcg_at_5"],
        ndcg_at_10=ranking_averages["ndcg_at_10"],
        mrr_at_10=ranking_averages["mrr_at_10"],
        num_test_rows=num_test_rows,
        num_users_evaluated=ranking_accumulator.num_users_evaluated,
    )

    # Configure MLflow.
    configure_mlflow(settings)
    # Check if the MLflow run id is present.
    if model_config.mlflow_run_id:
        # Resume the MLflow run.
        mlflow_context = resume_hybrid_mlflow_run(model_config.mlflow_run_id)
    else:
        # Start a new MLflow run.
        mlflow_context = start_hybrid_training_run(
            settings,
            model_version=model_version,
            dataset_version=dataset_version,
            snapshot_id=dataset_manifest.snapshot_id,
            cf_dataset_version=dataset_manifest.cf_dataset_version,
            cf_version=dataset_manifest.cf_version,
            content_embedding_version=dataset_manifest.content_embedding_version,
            feature_schema_version=dataset_manifest.feature_schema_version,
            input_dim=dataset_manifest.input_dim,
            model_architecture=model_config.model_architecture,
            learning_rate=model_config.learning_rate,
            batch_size=model_config.batch_size,
            dropout=model_config.dropout,
            num_epochs=model_config.num_epochs,
            shuffle_seed=model_config.shuffle_seed,
        )

    with mlflow_context:
        # Log the evaluation metrics.
        log_evaluation_metrics(test_metrics)
        # Create a temporary directory to store the artifacts.

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir)
            (artifact_dir / "test_metrics.json").write_text(
                json.dumps(test_metrics.model_dump(mode="json"), indent=2),
                encoding="utf-8",
            )

            # Write the test metrics to the S3 bucket.
            put_json(
                client,
                settings.s3_bucket,
                hybrid_ranker_test_metrics_object_key(model_version),
                test_metrics.model_dump(mode="json"),
            )

            # Build the complete hybrid manifest.
            finished_at = datetime.now(tz=UTC)
            manifest = build_complete_hybrid_manifest(
                model_version=model_version,
                dataset_version=dataset_version,
                snapshot_id=dataset_manifest.snapshot_id,
                cf_dataset_version=dataset_manifest.cf_dataset_version,
                cf_version=dataset_manifest.cf_version,
                content_embedding_version=dataset_manifest.content_embedding_version,
                feature_schema_version=dataset_manifest.feature_schema_version,
                input_dim=dataset_manifest.input_dim,
                pipeline_run_id=pipeline_run_id,
                created_at=model_config.created_at,
                finished_at=finished_at,
                training_metrics=training_metrics,
                test_metrics=test_metrics,
            )

            # Write the complete hybrid manifest to the S3 bucket.
            put_json(
                client,
                settings.s3_bucket,
                hybrid_ranker_model_manifest_object_key(model_version),
                manifest.model_dump(mode="json"),
            )

    # Log the evaluation complete.
    logger.info(
        "Hybrid evaluation complete",
        extra={
            "model_version": model_version,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "num_users_evaluated": ranking_accumulator.num_users_evaluated,
        },
    )

    return HybridEvaluationStats(
        model_version=model_version,
        dataset_version=dataset_version,
        test_rmse=test_rmse,
        test_mae=test_mae,
    )

"""Build hybrid ranker artifact manifest payloads."""

from __future__ import annotations

from datetime import datetime

from common.storage.s3 import (
    hybrid_ranker_model_config_object_key,
    hybrid_ranker_model_manifest_object_key,
    hybrid_ranker_model_object_key,
    hybrid_ranker_test_metrics_object_key,
    hybrid_ranker_training_curve_object_key,
    hybrid_ranker_training_metrics_object_key,
)

from common.schemas.hybrid_ranker_artifact_manifest import (
    HybridRankerArtifactManifest,
    HybridTestMetrics,
    HybridTrainingMetrics,
)

def build_complete_hybrid_manifest(
    *,
    model_version: str,
    dataset_version: str,
    snapshot_id: str,
    cf_dataset_version: str,
    cf_version: str,
    content_embedding_version: str,
    feature_schema_version: str,
    input_dim: int,
    pipeline_run_id: str,
    created_at: datetime,
    finished_at: datetime,
    training_metrics: HybridTrainingMetrics,
    test_metrics: HybridTestMetrics,
) -> HybridRankerArtifactManifest:
    """
    Build a complete hybrid ranker model manifest after evaluation finishes.

    ============================ Arguments ============================
    model_version: Version id for this model artifact bundle.
    dataset_version: Input hybrid feature dataset version.
    snapshot_id: Snapshot lineage from the dataset manifest.
    cf_dataset_version: CF dataset version from lineage.
    cf_version: CF artifact version from lineage.
    content_embedding_version: Content embedding version from lineage.
    feature_schema_version: Feature schema version from lineage.
    input_dim: Model input dimension.
    pipeline_run_id: pipeline_runs.run_id for the evaluation job.
    created_at: When model training started.
    finished_at: When evaluation completed and manifest was written.
    training_metrics: Validation metrics from the training job.
    test_metrics: Test regression and ranking metrics from evaluation.

    ============================ Returns ============================
    A HybridRankerArtifactManifest with status complete.
    """
    return HybridRankerArtifactManifest(
        model_version=model_version,
        dataset_version=dataset_version,
        snapshot_id=snapshot_id,
        cf_dataset_version=cf_dataset_version,
        cf_version=cf_version,
        content_embedding_version=content_embedding_version,
        feature_schema_version=feature_schema_version,
        input_dim=input_dim,
        status="complete",
        hybrid_ranker_model_path=hybrid_ranker_model_object_key(model_version),
        model_config_path=hybrid_ranker_model_config_object_key(model_version),
        training_metrics_path=hybrid_ranker_training_metrics_object_key(model_version),
        test_metrics_path=hybrid_ranker_test_metrics_object_key(model_version),
        training_curve_path=hybrid_ranker_training_curve_object_key(model_version),
        training_metrics=training_metrics,
        test_metrics=test_metrics,
        created_at=created_at,
        finished_at=finished_at,
        pipeline_run_id=pipeline_run_id,
        extra={"manifest_path": hybrid_ranker_model_manifest_object_key(model_version)},
    )

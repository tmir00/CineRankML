"""Build CF artifact manifest payloads."""

from __future__ import annotations

from datetime import datetime
from common.storage.s3 import (
    cf_artifact_manifest_object_key,
    cf_config_object_key,
    cf_metrics_object_key,
    cf_model_object_key,
    cf_movie_embeddings_object_key,
    cf_training_curve_object_key,
)
from common.schemas.cf_artifact_manifest import CfArtifactManifest, CfTrainingMetrics



def build_complete_cf_manifest(*, cf_version: str, cf_dataset_version: str, snapshot_id: str, 
                                    embedding_dim: int, pipeline_run_id: str, created_at: datetime, 
                                    finished_at: datetime, metrics: CfTrainingMetrics) -> CfArtifactManifest:
    """
    Build a complete CF artifact manifest after all files are uploaded.

    ============================ Arguments ============================
    cf_version: Version id for this CF artifact bundle.
    cf_dataset_version: Input CF dataset version used for training.
    snapshot_id: Snapshot lineage from the CF dataset manifest.
    embedding_dim: Embedding size used by the trained model.
    pipeline_run_id: pipeline_runs.run_id for this job.
    created_at: When training started.
    finished_at: When all artifacts were ready.
    metrics: Final training and embedding quality metrics.

    ============================ Returns ============================
    A CfArtifactManifest with status complete.
    """
    return CfArtifactManifest(
        cf_version=cf_version,
        cf_dataset_version=cf_dataset_version,
        snapshot_id=snapshot_id,
        embedding_dim=embedding_dim,
        status="complete",
        movie_cf_embeddings_path=cf_movie_embeddings_object_key(cf_version),
        cf_model_path=cf_model_object_key(cf_version),
        cf_config_path=cf_config_object_key(cf_version),
        cf_metrics_path=cf_metrics_object_key(cf_version),
        training_curve_path=cf_training_curve_object_key(cf_version),
        metrics=metrics,
        created_at=created_at,
        finished_at=finished_at,
        pipeline_run_id=pipeline_run_id,
        extra={"manifest_path": cf_artifact_manifest_object_key(cf_version)},
    )

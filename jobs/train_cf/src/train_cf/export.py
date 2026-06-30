"""Export movie CF embeddings and compute embedding quality metrics."""

from __future__ import annotations

import torch
import tempfile
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pathlib import Path
from botocore.client import BaseClient
from train_cf.model import DotProductCF
from common.schemas.cf_artifact_manifest import CfTrainingMetrics
from common.storage.s3 import cf_dataset_movie_map_object_key, download_file


DEFAULT_EMBEDDING_NORM_THRESHOLD = 0.05


def export_movie_embeddings(model: DotProductCF, client: BaseClient, bucket: str, cf_dataset_version: str, output_path: Path, *,
                                num_movies: int, rated_movie_indices: set[int]) -> CfTrainingMetrics:
    """
    Write movie_cf_embeddings.parquet and compute final embedding metrics.

    Do this by:
    1. Loading movie_id_map.parquet from the CF dataset prefix.
    2. Extracting movie embedding weights from the trained model.
    3. Writing movie_id and cf_embedding columns to Parquet.
    4. Computing coverage and norm-based quality metrics.

    ============================ Arguments ============================
    model: Trained dot-product CF model with best-epoch weights loaded.
    client: The boto3 S3 client.
    bucket: Source MinIO/S3 bucket.
    cf_dataset_version: CF dataset version containing movie_id_map.parquet.
    output_path: Local path for movie_cf_embeddings.parquet.
    num_movies: Total catalog movie count from the CF dataset manifest.
    rated_movie_indices: Movie indices observed at least once in train data.

    ============================ Returns ============================
    CfTrainingMetrics with embedding quality fields populated.
    """
    # Download the movie_id_mapping file from the CF dataset prefix.
    with tempfile.TemporaryDirectory() as temp_dir:
        map_path = Path(temp_dir) / "movie_id_map.parquet"
        download_file(
            client,
            bucket,
            cf_dataset_movie_map_object_key(cf_dataset_version),
            map_path,
        )
        movie_map = pq.read_table(map_path).to_pandas()

    # Extract the movie embedding weights from the model.
    weights = model.movie_emb.weight.detach().cpu().numpy()
    # Compute the norms of the movie embedding weights.
    norms = np.linalg.norm(weights, axis=1)
    # Count the number of movie embeddings that are NaN.
    nan_count = int(np.isnan(weights).any(axis=1).sum())
    # Count the number of movie embeddings that are less than the default norm threshold.
    default_count = int(np.sum(norms < DEFAULT_EMBEDDING_NORM_THRESHOLD))

    # Sort the movie map by movie_idx.
    movie_map = movie_map.sort_values("movie_idx").reset_index(drop=True)
    if len(movie_map) != num_movies:
        raise ValueError(
            f"movie_id_map row count {len(movie_map)} does not match manifest num_movies {num_movies}"
        )

    # Extract the movie embeddings from the model weights.
    embeddings = [weights[int(row.movie_idx)].astype(np.float32).tolist() for row in movie_map.itertuples()]
    # Create a DataFrame with the movie_id and cf_embedding columns.
    frame = pd.DataFrame(
        {
            "movie_id": movie_map["movie_id"].astype(int),
            "cf_embedding": embeddings,
        }
    )

    # Create a PyArrow table from the DataFrame.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write the table to the output path.
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, output_path)

    # Compute the coverage of the movie embeddings.
    coverage = len(rated_movie_indices) / max(1, num_movies)
    # Return the CfTrainingMetrics object.
    return CfTrainingMetrics(
        best_epoch=0,
        best_validation_rmse=0.0,
        best_validation_mae=0.0,
        num_train_rows=0,
        num_validation_rows=0,
        num_users=0,
        num_movies=num_movies,
        movie_embedding_coverage=coverage,
        default_embedding_count=default_count,
        nan_embedding_count=nan_count,
        embedding_norm_mean=float(norms.mean()),
        embedding_norm_std=float(norms.std()),
    )


def finalize_training_metrics(
    embedding_metrics: CfTrainingMetrics,
    *,
    best_epoch: int,
    best_validation_rmse: float,
    best_validation_mae: float,
    num_train_rows: int,
    num_validation_rows: int,
    num_users: int,
    num_movies: int,
) -> CfTrainingMetrics:
    """
    Merge training-run counters into the embedding quality metrics object.

    ============================ Arguments ============================
    embedding_metrics: Metrics computed during embedding export.
    best_epoch: Epoch with lowest validation RMSE.
    best_validation_rmse: Best validation RMSE across epochs.
    best_validation_mae: Validation MAE at the best epoch.
    num_train_rows: Train split row count from the CF dataset manifest.
    num_validation_rows: Validation split row count from the CF dataset manifest.
    num_users: User embedding table size.
    num_movies: Movie embedding table size.

    ============================ Returns ============================
    A complete CfTrainingMetrics instance.
    """
    return embedding_metrics.model_copy(
        update={
            "best_epoch": best_epoch,
            "best_validation_rmse": best_validation_rmse,
            "best_validation_mae": best_validation_mae,
            "num_train_rows": num_train_rows,
            "num_validation_rows": num_validation_rows,
            "num_users": num_users,
            "num_movies": num_movies,
        }
    )

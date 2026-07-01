"""DuckDB pipeline that builds hybrid ranker feature datasets from frozen artifacts."""

from __future__ import annotations

import logging
import tempfile

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal
from botocore.client import BaseClient
from dataclasses import dataclass, field
from common.storage.s3 import (
    cf_movie_embeddings_object_key,
    hybrid_ranker_manifest_object_key,
    hybrid_ranker_test_part_object_key,
    hybrid_ranker_train_part_object_key,
    hybrid_ranker_validation_part_object_key,
    put_json,
    upload_file,
)
from common.features.vector import build_feature_vector
from common.config.settings import CreateFeaturesSettings
from create_features.version import resolve_dataset_version
from common.features.similarity import weighted_embedding_mean
from create_features.manifest import build_complete_hybrid_manifest, utc_now
from common.schemas.hybrid_ranker_dataset_manifest import HybridRankerPartEntry
from common.storage.duckdb_s3 import configure_duckdb_s3, create_duckdb_connection
from common.schemas.cf_dataset_manifest import CfDatasetManifest, CfDatasetPartEntry
from common.storage.snapshot_reader import resolve_snapshot_id, snapshot_table_glob_uri
from common.storage.cf_artifact_reader import load_cf_artifact_manifest, resolve_cf_version
from common.storage.cf_dataset_reader import load_cf_dataset_manifest, resolve_cf_dataset_version
from common.features.normalization import MetadataNormalizationStats, normalize_candidate_metadata
from common.features.schema import CF_EMBEDDING_DIM, CONTENT_EMBEDDING_DIM, HIGH_RATED_THRESHOLD, LOW_RATED_THRESHOLD
from common.metrics.hybrid_feature_stats import HybridFeatureQualityStats, compute_row_count_delta


logger = logging.getLogger(__name__)
SplitName = Literal["train", "validation", "test"]


@dataclass
class HybridFeaturePrepStats:
    """Counters and metadata collected during one hybrid feature generation run."""

    snapshot_id: str
    cf_dataset_version: str
    cf_version: str
    dataset_version: str
    content_embedding_version: str
    feature_schema_version: str
    train_row_count: int = 0
    validation_row_count: int = 0
    test_row_count: int = 0
    metadata_normalization: MetadataNormalizationStats | None = None
    train_parts: list[HybridRankerPartEntry] = field(default_factory=list)
    validation_parts: list[HybridRankerPartEntry] = field(default_factory=list)
    test_parts: list[HybridRankerPartEntry] = field(default_factory=list)
    quality: HybridFeatureQualityStats | None = None


@dataclass
class CatalogRow:
    """Raw candidate metadata fields from catalog_movies."""

    year: float | None
    runtime: float | None
    tmdb_popularity: float | None
    tmdb_vote_average: float | None
    tmdb_vote_count: float | None


def _s3_uri(bucket: str, object_key: str) -> str:
    """Build an s3:// URI for DuckDB reads and writes."""
    return f"s3://{bucket}/{object_key}"


def _parquet_uri_list(bucket: str, object_keys: list[str]) -> str:
    """Build a DuckDB read_parquet list literal from object keys."""
    uris = ", ".join(f"'{_s3_uri(bucket, key)}'" for key in object_keys)
    return f"[{uris}]"


def _part_keys(parts: list[CfDatasetPartEntry]) -> list[str]:
    """Return object keys from CF dataset part entries."""
    return [part.object_key for part in parts]


def _normalize_rating_timestamp(ts: datetime | int | float) -> int:
    """
    Convert rating_timestamp values into epoch seconds for candidate key matching.

    Do this by:
    1. Returning epoch seconds when DuckDB gives a datetime from snapshot Parquet.
    2. Casting integer epoch values directly when they are already numeric.

    ============================ Arguments ============================
    ts: rating_timestamp from DuckDB or an already-normalized epoch value.

    ============================ Returns ============================
    Unix epoch seconds as an integer.
    """
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return int(ts.timestamp())
    return int(ts)


def _validate_lineage(*, snapshot_id: str, cf_dataset_manifest: CfDatasetManifest, cf_version: str, \
                        cf_artifact_snapshot_id: str, cf_artifact_dataset_version: str) -> None:
    """
    Ensure selected snapshot, CF dataset, and CF artifacts refer to the same lineage.

    Raises ValueError when snapshot or CF dataset versions do not align.
    """
    if cf_dataset_manifest.snapshot_id != snapshot_id:
        raise ValueError(
            f"CF dataset snapshot_id={cf_dataset_manifest.snapshot_id} "
            f"does not match resolved snapshot_id={snapshot_id}"
        )
    if cf_artifact_snapshot_id != snapshot_id:
        raise ValueError(
            f"CF artifact snapshot_id={cf_artifact_snapshot_id} "
            f"does not match resolved snapshot_id={snapshot_id}"
        )
    if cf_artifact_dataset_version != cf_dataset_manifest.cf_dataset_version:
        raise ValueError(
            f"CF artifact cf_dataset_version={cf_artifact_dataset_version} "
            f"does not match CF dataset version={cf_dataset_manifest.cf_dataset_version}"
        )


def _fit_metadata_normalization(conn: duckdb.DuckDBPyConnection) -> MetadataNormalizationStats:
    """
    Fit min/max metadata normalization stats on train-split candidates.

    Do this by:
    1. Joining train candidates to catalog metadata.
    2. Applying log1p to popularity and vote_count before min/max aggregation.

    ============================ Arguments ============================
    conn: Open DuckDB connection with train_candidates and catalog tables loaded.

    ============================ Returns ============================
    Train-fit MetadataNormalizationStats for manifest persistence.
    """
    # SELECT the min/max values for the catalog metadata columns.
    row = conn.execute(
        """
        SELECT
            COALESCE(MIN(c.year), 0.0),
            COALESCE(MAX(c.year), 0.0),
            COALESCE(MIN(c.runtime), 0.0),
            COALESCE(MAX(c.runtime), 0.0),
            COALESCE(MIN(LN(COALESCE(c.tmdb_popularity, 0.0) + 1.0)), 0.0),
            COALESCE(MAX(LN(COALESCE(c.tmdb_popularity, 0.0) + 1.0)), 0.0),
            COALESCE(MIN(c.tmdb_vote_average), 0.0),
            COALESCE(MAX(c.tmdb_vote_average), 0.0),
            COALESCE(MIN(LN(COALESCE(c.tmdb_vote_count, 0.0) + 1.0)), 0.0),
            COALESCE(MAX(LN(COALESCE(c.tmdb_vote_count, 0.0) + 1.0)), 0.0)
        FROM candidates cand
        INNER JOIN catalog c ON cand.movie_id = c.movie_id
        WHERE cand.split = 'train'
        """
    ).fetchone()

    # Return the MetadataNormalizationStats object.
    return MetadataNormalizationStats(
        year_min=float(row[0]),
        year_max=float(row[1]),
        runtime_min=float(row[2]),
        runtime_max=float(row[3]),
        tmdb_popularity_log_min=float(row[4]),
        tmdb_popularity_log_max=float(row[5]),
        tmdb_vote_average_min=float(row[6]),
        tmdb_vote_average_max=float(row[7]),
        tmdb_vote_count_log_min=float(row[8]),
        tmdb_vote_count_log_max=float(row[9]),
    )


def _load_embedding_map(conn: duckdb.DuckDBPyConnection, query: str, dim: int) -> dict[int, np.ndarray]:
    """
    Load movie_id -> embedding vectors from one DuckDB query.

    ============================ Arguments ============================
    conn: Open DuckDB connection.
    query: SQL returning movie_id and embedding list column.
    dim: Expected embedding dimension.

    ============================ Returns ============================
    Mapping from movie_id to float32 embedding vector.
    """
    # Initialize the mapping from movie_id to embedding vector.
    embedding_map: dict[int, np.ndarray] = {}
    # Execute the query and fetch the results.
    rows = conn.execute(query).fetchall()
    
    # Iterate over the results and add them to the mapping.
    for movie_id, embedding in rows:
        # Convert the embedding to a numpy array.
        vector = np.asarray(embedding, dtype=np.float32)
        # Check if the embedding has the expected dimension.
        if vector.shape != (dim,):
            vector = vector.reshape(-1)
        # Check if the embedding has the expected dimension.
        if vector.shape[0] != dim:
            logger.warning(
                "Skipping embedding with unexpected dimension",
                extra={"movie_id": movie_id, "expected_dim": dim, "actual_dim": vector.shape[0]},
            )
            continue
        embedding_map[int(movie_id)] = vector
    return embedding_map


def _load_catalog_map(conn: duckdb.DuckDBPyConnection) -> dict[int, CatalogRow]:
    """ Load catalog metadata keyed by movie_id. """

    # Initialize the mapping from movie_id to catalog metadata.
    catalog: dict[int, CatalogRow] = {}

    # Execute the query and fetch the results.
    rows = conn.execute(
        """
        SELECT
            movie_id,
            year,
            runtime,
            tmdb_popularity,
            tmdb_vote_average,
            tmdb_vote_count
        FROM catalog
        """
    ).fetchall()

    # Iterate over the results and add them to the mapping.
    for movie_id, year, runtime, popularity, vote_average, vote_count in rows:
        catalog[int(movie_id)] = CatalogRow(
            year=float(year) if year is not None else None,
            runtime=float(runtime) if runtime is not None else None,
            tmdb_popularity=float(popularity) if popularity is not None else None,
            tmdb_vote_average=float(vote_average) if vote_average is not None else None,
            tmdb_vote_count=float(vote_count) if vote_count is not None else None,
        )

    return catalog


def _zero_embedding(dim: int) -> np.ndarray:
    """Return a zero embedding vector for missing lookup keys."""
    return np.zeros(dim, dtype=np.float32)


def _flush_output_part(
    client: BaseClient,
    bucket: str,
    dataset_version: str,
    split: SplitName,
    part_index: int,
    rows: list[dict],
    part_key_builder: Callable[[str, int], str],
) -> HybridRankerPartEntry:
    """
    Write one output Parquet part to MinIO from buffered rows.

    ============================ Arguments ============================
    client: boto3 S3 client.
    bucket: Target bucket name.
    dataset_version: Hybrid dataset version identifier.
    split: Split name for logging only.
    part_index: Zero-based part index within the split.
    rows: Buffered output rows with user_id, movie_id, rating, features.
    part_key_builder: Function that builds the object key for a split part.

    ============================ Returns ============================
    HybridRankerPartEntry metadata for the uploaded part.
    """
    object_key = part_key_builder(dataset_version, part_index)
    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = Path(temp_dir) / f"{split}-part-{part_index:05d}.parquet"
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, local_path)
        upload_file(client, bucket, object_key, local_path)

    return HybridRankerPartEntry(object_key=object_key, row_count=len(rows))


def _generate_features_for_users(
    conn: duckdb.DuckDBPyConnection,
    client: BaseClient,
    settings: CreateFeaturesSettings,
    dataset_version: str,
    *,
    content_embeddings: dict[int, np.ndarray],
    cf_embeddings: dict[int, np.ndarray],
    catalog: dict[int, CatalogRow],
    metadata_stats: MetadataNormalizationStats,
    candidate_keys: set[tuple[int, int, float, int, int, str]],
    join_dropped_candidates: int,
) -> tuple[dict[SplitName, int], dict[SplitName, list[HybridRankerPartEntry]], HybridFeatureQualityStats]:
    """
    Build hybrid ranker feature rows and upload them as train/validation/test Parquet parts.

    Big idea:
    For each user, walk through their ratings in time order. The current rating is the
    prediction target only if it exists in candidate_keys. When it is a candidate row,
    build the feature vector from the user's history BEFORE this rating, then use the
    current rating as the label.

    This prevents leakage:
    - The model can use movies the user rated before the candidate.
    - The model cannot use the candidate movie/rating inside the user profile.
    - After the row is written, the current rating is added to history for future rows.

    Do this by:
    1. Converting candidate_keys into a fast lookup map.
    2. Processing each user's ratings chronologically.
    3. Writing feature rows only for candidate rows.
    4. Buffering rows and flushing each split to Parquet when the buffer is full.

    ============================ Arguments ============================
    conn: DuckDB connection with ratings_with_behavior and candidates_enriched tables loaded.
    client: boto3 S3 client for uploads.
    settings: Hybrid feature generation configuration.
    dataset_version: Output hybrid dataset version.
    content_embeddings: movie_id -> content embedding lookup.
    cf_embeddings: movie_id -> CF embedding lookup.
    catalog: movie_id -> raw candidate metadata lookup.
    metadata_stats: Train-fit metadata normalization stats.
    candidate_keys: Candidate rows with split labels:
        (user_id, movie_id, rating, rating_timestamp, event_id, split)
    join_dropped_candidates: CF rows that failed to join back to ratings_events.

    ============================ Returns ============================
    Tuple containing:
    1. Row counts for train/validation/test.
    2. Uploaded Parquet part metadata for train/validation/test.
    3. Data-quality counters for Prometheus.
    """

    # Count how many feature rows we write for each split.
    split_row_counts: dict[SplitName, int] = {"train": 0, "validation": 0, "test": 0}

    # Track uploaded Parquet parts. These entries go into manifest.json later.
    split_parts: dict[SplitName, list[HybridRankerPartEntry]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    # Hold rows in memory before writing a Parquet part.
    # This avoids writing one tiny file per training example.
    split_buffers: dict[SplitName, list[dict]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    # Track the next part number for each split:
    # train/part-00000.parquet, train/part-00001.parquet, etc.
    split_part_indexes: dict[SplitName, int] = {"train": 0, "validation": 0, "test": 0}

    # Each split writes to a different S3 prefix.
    part_key_builders: dict[SplitName, Callable[[str, int], str]] = {
        "train": hybrid_ranker_train_part_object_key,
        "validation": hybrid_ranker_validation_part_object_key,
        "test": hybrid_ranker_test_part_object_key,
    }

    # Convert candidate_keys into a dictionary so we can check candidate rows quickly.
    # Key: (user_id, movie_id, rating, rating_timestamp, event_id)
    # Value: split name (train, validation, test)
    candidate_split_map: dict[tuple[int, int, float, int, int], SplitName] = {
        (uid, mid, rt, ts, eid): split  # type: ignore[misc]
        for uid, mid, rt, ts, eid, split in candidate_keys
    }

    # We only need to process users who have at least one candidate row.
    # Non-candidate ratings for these users are still loaded later because they
    # help build history before future candidate rows.
    # So here, we get the list of all user ids that have at least one candidate row.
    user_ids = [
        int(row[0])
        for row in conn.execute(
            """
            SELECT DISTINCT user_id
            FROM candidates_enriched
            ORDER BY user_id
            """
        ).fetchall()
    ]

    # Data-quality counters collected while emitting candidate rows.
    cold_start_rows = 0
    missing_content_embedding_rows = 0
    missing_cf_embedding_rows = 0
    users_processed = 0

    for user_id in user_ids:
        users_processed += 1
        # Load this user's full rating timeline, not just candidate rows.
        #
        # Why full timeline?
        # A rating might not be a training target itself, but it can still be
        # part of the user's history for a later target.
        rating_rows = conn.execute(
            """
            SELECT
                b.movie_id,
                b.rating,
                b.rating_timestamp,
                b.id,
                b.num_user_ratings,
                b.user_avg_rating,
                b.user_rating_std,
                b.num_high_rated_movies,
                b.num_low_rated_movies
            FROM ratings_with_behavior b
            WHERE b.user_id = ?
            ORDER BY b.rating_timestamp, b.id
            """,
            [user_id],
        ).fetchall()

        # Growing user profile.
        #
        # At the start, the user has no history.
        # As we walk forward in time, each past rating adds:
        # - the movie's content embedding
        # - the movie's CF embedding
        # - the rating value as the weight
        history_content: list[np.ndarray] = []
        history_cf: list[np.ndarray] = []
        history_weights: list[float] = []

        # Iterate over the rating rows and build the feature vectors.
        for (
            movie_id,
            rating,
            rating_timestamp,
            event_id,
            num_user_ratings,
            user_avg_rating,
            user_rating_std,
            num_high_rated,
            num_low_rated,
        ) in rating_rows:
            movie_id = int(movie_id)
            rating = float(rating)
            rating_timestamp = _normalize_rating_timestamp(rating_timestamp)
            event_id = int(event_id)

            # This identifies the exact rating event currently being processed.
            # event_id avoids ambiguity if the same user/movie/rating appears more than once.
            candidate_key = (
                user_id,
                movie_id,
                rating,
                rating_timestamp,
                event_id,
            )

            # Check if this exact rating row is one of the train/validation/test targets.
            # If it is, split_name will be "train", "validation", or "test".
            # If it is not, split_name will be None.
            split_name = candidate_split_map.get(candidate_key)

            # Look up embeddings once for this movie.
            # If missing, use zero vectors so feature generation can continue.
            missing_content = movie_id not in content_embeddings
            missing_cf = movie_id not in cf_embeddings
            movie_content = content_embeddings.get(movie_id, _zero_embedding(CONTENT_EMBEDDING_DIM))
            movie_cf = cf_embeddings.get(movie_id, _zero_embedding(CF_EMBEDDING_DIM))

            if split_name is None:
                # This rating is not a prediction target.
                # It only becomes part of the user's history for future rows.
                history_weights.append(rating)
                history_content.append(movie_content)
                history_cf.append(movie_cf)
                continue

            # This rating is a prediction target.
            # Build user profiles from history BEFORE the current rating.
            #
            # Important:
            # The current movie has not been added to history yet.
            # That is what prevents leakage.
            user_content_profile = weighted_embedding_mean(history_content, history_weights)
            user_cf_profile = weighted_embedding_mean(history_cf, history_weights)

            # If the user has no valid history yet, use zero profile vectors.
            if user_content_profile.shape != (CONTENT_EMBEDDING_DIM,):
                user_content_profile = _zero_embedding(CONTENT_EMBEDDING_DIM)

            if user_cf_profile.shape != (CF_EMBEDDING_DIM,):
                user_cf_profile = _zero_embedding(CF_EMBEDDING_DIM)

            # Candidate movie metadata.
            # This describes the movie we are trying to predict the user's rating for.
            catalog_row = catalog.get(movie_id, CatalogRow(None, None, None, None, None))

            # Normalize metadata using train-fit stats, so train/validation/test/inference
            # all use the same scaling.
            metadata_values = normalize_candidate_metadata(
                year=catalog_row.year,
                runtime=catalog_row.runtime,
                tmdb_popularity=catalog_row.tmdb_popularity,
                tmdb_vote_average=catalog_row.tmdb_vote_average,
                tmdb_vote_count=catalog_row.tmdb_vote_count,
                stats=metadata_stats,
            )

            # Point-in-time user behavior stats.
            # These were computed in DuckDB using only ratings before this row.
            user_behavior = np.array(
                [
                    float(num_user_ratings or 0),
                    float(user_avg_rating or 0),
                    float(user_rating_std or 0),
                    float(num_high_rated or 0),
                    float(num_low_rated or 0),
                ],
                dtype=np.float32,
            )

            if float(num_user_ratings or 0) == 0.0:
                cold_start_rows += 1
            if missing_content:
                missing_content_embedding_rows += 1
            if missing_cf:
                missing_cf_embedding_rows += 1

            candidate_metadata = np.array(metadata_values, dtype=np.float32)

            # Final model input vector.
            #
            # Combines:
            # - user content profile
            # - candidate content embedding
            # - user CF profile
            # - candidate CF embedding
            # - point-in-time user behavior
            # - normalized candidate metadata
            feature_vector = build_feature_vector(
                user_content_profile=user_content_profile,
                candidate_content_embedding=movie_content,
                user_cf_profile=user_cf_profile,
                candidate_cf_embedding=movie_cf,
                user_behavior=user_behavior,
                candidate_metadata=candidate_metadata,
            )

            # Write one supervised example:
            #   features -> rating
            #
            # The split decides whether it goes to train, validation, or test.
            split_buffers[split_name].append(
                {
                    "user_id": user_id,
                    "movie_id": movie_id,
                    "rating": rating,
                    "features": feature_vector.astype(float).tolist(),
                }
            )
            split_row_counts[split_name] += 1

            # Flush this split if the buffer has reached the configured part size.
            if len(split_buffers[split_name]) >= settings.hybrid_part_row_limit:
                part_entry = _flush_output_part(
                    client,
                    settings.s3_bucket,
                    dataset_version,
                    split_name,
                    split_part_indexes[split_name],
                    split_buffers[split_name],
                    part_key_builders[split_name],
                )
                split_parts[split_name].append(part_entry)
                split_part_indexes[split_name] += 1
                split_buffers[split_name] = []

            # Add the current rating to history AFTER writing the feature row.
            #
            # This lets the current movie help predict future ratings,
            # but not itself.
            history_weights.append(rating)
            history_content.append(movie_content)
            history_cf.append(movie_cf)

    # Flush any leftover rows that did not fill a complete part.
    for split_name in ("train", "validation", "test"):
        if split_buffers[split_name]:
            part_entry = _flush_output_part(
                client,
                settings.s3_bucket,
                dataset_version,
                split_name,
                split_part_indexes[split_name],
                split_buffers[split_name],
                part_key_builders[split_name],
            )
            split_parts[split_name].append(part_entry)

    total_emitted = (
        split_row_counts["train"]
        + split_row_counts["validation"]
        + split_row_counts["test"]
    )
    cold_start_fraction = cold_start_rows / total_emitted if total_emitted > 0 else 0.0

    quality = HybridFeatureQualityStats(
        cf_expected_row_count=0,
        row_count_delta_vs_cf=0,
        cold_start_rows=cold_start_rows,
        cold_start_fraction=cold_start_fraction,
        users_processed=users_processed,
        train_parts=len(split_parts["train"]),
        validation_parts=len(split_parts["validation"]),
        test_parts=len(split_parts["test"]),
        missing_content_embedding_rows=missing_content_embedding_rows,
        missing_cf_embedding_rows=missing_cf_embedding_rows,
        join_dropped_candidates=join_dropped_candidates,
    )

    return split_row_counts, split_parts, quality


def run_hybrid_feature_generation(
    client: BaseClient,
    settings: CreateFeaturesSettings,
    pipeline_run_id: str,
) -> HybridFeaturePrepStats:
    """
    Build the hybrid ranker feature dataset and write it to MinIO/S3.

    Big idea:
    This job takes frozen artifacts from previous pipeline stages and turns them into
    train/validation/test feature rows for the final hybrid ranker model.

    The final model will learn:

        features -> rating

    where features include:
    - user content profile
    - candidate movie content embedding
    - user CF profile
    - candidate movie CF embedding
    - point-in-time user behavior stats
    - normalized candidate metadata

    Do this by:
    1. Resolving the snapshot, CF dataset, CF model artifacts, and output dataset version.
    2. Checking lineage so we do not mix incompatible snapshots/features/models.
    3. Loading snapshot tables and CF artifacts into DuckDB temporary tables.
    4. Reconstructing the candidate rating rows for train/validation/test.
    5. Computing user behavior stats using only ratings before each candidate row.
    6. Fitting metadata normalization stats on train candidates only.
    7. Generating feature vectors and uploading Parquet parts.
    8. Writing manifest.json last to mark the dataset complete.

    ============================ Arguments ============================
    client: boto3 S3 client used to read/write MinIO/S3 objects.
    settings: Hybrid feature generation configuration.
    pipeline_run_id: pipeline_runs.run_id used for lineage/provenance.

    ============================ Returns ============================
    HybridFeaturePrepStats containing output version ids, row counts, normalization stats,
    and uploaded train/validation/test part metadata.
    """

    # Resolve the exact input/output versions for this run.
    #
    # snapshot_id:
    #   Frozen copy of raw/operational data, such as ratings_events and catalog_movies.
    #
    # cf_dataset_version:
    #   Versioned CF train/validation/test split with user_idx/movie_idx mappings.
    #
    # cf_version:
    #   Versioned CF artifacts, especially movie_cf_embeddings.parquet.
    #
    # dataset_version:
    #   New output version for this hybrid feature dataset.
    snapshot_id = resolve_snapshot_id(client, settings.s3_bucket, settings.snapshot_id)
    cf_dataset_version = resolve_cf_dataset_version(
        client,
        settings.s3_bucket,
        settings.cf_dataset_version,
    )
    cf_version = resolve_cf_version(client, settings.s3_bucket, settings.cf_version)
    dataset_version = resolve_dataset_version(settings)

    # Load manifests for the selected CF dataset and CF artifacts.
    #
    # These manifests tell us:
    # - which snapshot the CF dataset came from
    # - which snapshot the CF model came from
    # - where the train/validation/test parts are
    # - where user/movie maps are
    cf_dataset_manifest = load_cf_dataset_manifest(client, settings.s3_bucket, cf_dataset_version)
    cf_artifact_manifest = load_cf_artifact_manifest(client, settings.s3_bucket, cf_version)

    # Safety check:
    # Make sure the snapshot, CF dataset, and CF embeddings all came from the same lineage.
    #
    # Without this, we could accidentally combine:
    # - ratings from snapshot A
    # - CF split from snapshot B
    # - CF embeddings from snapshot C
    #
    # That would make the training data inconsistent and hard to reproduce.
    _validate_lineage(
        snapshot_id=snapshot_id,
        cf_dataset_manifest=cf_dataset_manifest,
        cf_version=cf_version,
        cf_artifact_snapshot_id=cf_artifact_manifest.snapshot_id,
        cf_artifact_dataset_version=cf_artifact_manifest.cf_dataset_version,
    )

    # Start time for the output manifest.
    created_at = utc_now()

    # This object collects metadata while the job runs.
    # At the end, it contains row counts, uploaded parts, and normalization stats.
    stats = HybridFeaturePrepStats(
        snapshot_id=snapshot_id,
        cf_dataset_version=cf_dataset_version,
        cf_version=cf_version,
        dataset_version=dataset_version,
        content_embedding_version=settings.content_embedding_version,
        feature_schema_version=settings.feature_schema_version,
    )

    # Build S3/MinIO locations for the input snapshot tables and artifacts.
    ratings_glob = snapshot_table_glob_uri(settings.s3_bucket, snapshot_id, "ratings_events")
    catalog_glob = snapshot_table_glob_uri(settings.s3_bucket, snapshot_id, "catalog_movies")
    content_glob = snapshot_table_glob_uri(settings.s3_bucket, snapshot_id, "movie_content_embeddings")

    # CF movie embeddings were produced by the CF training job.
    cf_embeddings_uri = _s3_uri(settings.s3_bucket, cf_movie_embeddings_object_key(cf_version))

    # These maps translate between original MovieLens IDs and dense model indices.
    user_map_uri = _s3_uri(settings.s3_bucket, cf_dataset_manifest.user_id_map_key)
    movie_map_uri = _s3_uri(settings.s3_bucket, cf_dataset_manifest.movie_id_map_key)

    # Build DuckDB-readable Parquet URI lists for the CF train/validation/test parts.
    train_parquet = _parquet_uri_list(settings.s3_bucket, _part_keys(cf_dataset_manifest.train_parts))
    validation_parquet = _parquet_uri_list(settings.s3_bucket, _part_keys(cf_dataset_manifest.validation_parts))
    test_parquet = _parquet_uri_list(settings.s3_bucket, _part_keys(cf_dataset_manifest.test_parts))

    # Create DuckDB connection and configure it so DuckDB can read Parquet files from MinIO/S3.
    conn = create_duckdb_connection()
    configure_duckdb_s3(conn, settings)

    try:
        # Load the frozen ratings snapshot into DuckDB.
        #
        # This is the full rating event table needed for:
        # - reconstructing candidate timestamps
        # - walking user histories
        # - calculating point-in-time behavior stats
        conn.execute(
            f"""
            CREATE TABLE ratings AS
            SELECT user_id, movie_id, rating, rating_timestamp, id
            FROM read_parquet('{ratings_glob}')
            """
        )

        # Load candidate movie metadata from the frozen catalog snapshot.
        #
        # These fields become candidate_metadata features later.
        conn.execute(
            f"""
            CREATE TABLE catalog AS
            SELECT
                movie_id,
                year,
                runtime,
                tmdb_popularity,
                tmdb_vote_average,
                tmdb_vote_count
            FROM read_parquet('{catalog_glob}')
            """
        )

        # Load content embeddings for the requested content embedding version.
        #
        # Example:
        # movie_id -> 384-dim MiniLM embedding
        conn.execute(
            f"""
            CREATE TABLE content_embeddings AS
            SELECT movie_id, embedding
            FROM read_parquet('{content_glob}')
            WHERE embedding_version = '{settings.content_embedding_version}'
            """
        )

        # Load CF movie embeddings produced by the CF training job.
        #
        # Example:
        # movie_id -> 64-dim collaborative filtering embedding
        conn.execute(
            f"""
            CREATE TABLE cf_embeddings AS
            SELECT movie_id, cf_embedding AS embedding
            FROM read_parquet('{cf_embeddings_uri}')
            """
        )

        # Load maps that convert dense CF indices back to original user/movie IDs.
        #
        # CF datasets use user_idx/movie_idx because embedding models need dense indices.
        # Hybrid feature generation needs original user_id/movie_id to join back to ratings.
        conn.execute(f"CREATE TABLE user_id_map AS SELECT * FROM read_parquet('{user_map_uri}')")
        conn.execute(f"CREATE TABLE movie_id_map AS SELECT * FROM read_parquet('{movie_map_uri}')")

        # Combine the CF train/validation/test rows into one table.
        #
        # These are the rating rows that should become hybrid ranker examples.
        # At this point they still use user_idx/movie_idx, not original IDs.
        conn.execute(
            f"""
            CREATE TABLE cf_candidates AS
            SELECT user_idx, movie_idx, rating, 'train' AS split
            FROM read_parquet({train_parquet})
            UNION ALL
            SELECT user_idx, movie_idx, rating, 'validation' AS split
            FROM read_parquet({validation_parquet})
            UNION ALL
            SELECT user_idx, movie_idx, rating, 'test' AS split
            FROM read_parquet({test_parquet})
            """
        )

        # Convert CF candidate rows back to original user_id/movie_id.
        #
        # Also join back to ratings to recover:
        # - rating_timestamp
        # - rating event id
        #
        # We need those because feature generation must walk each user's timeline
        # in exact temporal order and avoid leakage.
        conn.execute(
            f"""
            CREATE TABLE candidates AS
            SELECT
                c.user_idx,
                c.movie_idx,
                c.rating,
                c.split,
                u.user_id,
                m.movie_id,
                r.rating_timestamp,
                r.id
            FROM cf_candidates c
            INNER JOIN user_id_map u ON c.user_idx = u.user_idx
            INNER JOIN movie_id_map m ON c.movie_idx = m.movie_idx
            INNER JOIN ratings r
                ON u.user_id = r.user_id
               AND m.movie_id = r.movie_id
               AND c.rating = r.rating
            """
        )

        cf_rows, joined_rows = conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM cf_candidates) AS cf_rows,
                (SELECT COUNT(*) FROM candidates) AS joined_rows
            """
        ).fetchone()
        join_dropped_candidates = int(cf_rows) - int(joined_rows)
        if join_dropped_candidates > 0:
            logger.warning(
                "Dropped CF candidate rows that could not join back to ratings_events",
                extra={
                    "snapshot_id": snapshot_id,
                    "cf_dataset_version": cf_dataset_version,
                    "dropped_rows": join_dropped_candidates,
                },
            )

        # Compute point-in-time user behavior stats for every rating row.
        #
        # The key part is:
        #   ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        #
        # That means:
        #   "For this rating, calculate stats using only previous ratings from this user."
        #
        # So if we are predicting a user's rating for Movie C, these stats do not include
        # Movie C's rating. This prevents leakage.
        conn.execute(
            f"""
            CREATE TABLE ratings_with_behavior AS
            SELECT
                user_id,
                movie_id,
                rating,
                rating_timestamp,
                id,
                COUNT(*) OVER history_window AS num_user_ratings,
                COALESCE(AVG(rating) OVER history_window, 0.0) AS user_avg_rating,
                COALESCE(STDDEV_SAMP(rating) OVER history_window, 0.0) AS user_rating_std,
                COALESCE(
                    SUM(CASE WHEN rating >= {HIGH_RATED_THRESHOLD} THEN 1 ELSE 0 END)
                        OVER history_window,
                    0
                ) AS num_high_rated_movies,
                COALESCE(
                    SUM(CASE WHEN rating <= {LOW_RATED_THRESHOLD} THEN 1 ELSE 0 END)
                        OVER history_window,
                    0
                ) AS num_low_rated_movies
            FROM ratings
            WINDOW history_window AS (
                PARTITION BY user_id
                ORDER BY rating_timestamp, id
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            )
            """
        )

        # Add the point-in-time behavior stats onto the candidate rows.
        #
        # candidates:
        #   rows we want to turn into train/validation/test examples
        #
        # ratings_with_behavior:
        #   every rating row with stats from before that rating
        #
        # candidates_enriched:
        #   candidate rows + split + user behavior stats
        conn.execute(
            """
            CREATE TABLE candidates_enriched AS
            SELECT
                c.user_id,
                c.movie_id,
                c.rating,
                c.rating_timestamp,
                c.id,
                c.split,
                b.num_user_ratings,
                b.user_avg_rating,
                b.user_rating_std,
                b.num_high_rated_movies,
                b.num_low_rated_movies
            FROM candidates c
            INNER JOIN ratings_with_behavior b
                ON c.user_id = b.user_id
               AND c.movie_id = b.movie_id
               AND c.rating = b.rating
               AND c.rating_timestamp = b.rating_timestamp
               AND c.id = b.id
            """
        )

        # Fit metadata normalization stats using train candidates only.
        #
        # These stats are saved into the manifest and reused for validation/test/inference.
        # This avoids leaking validation/test distribution into training preprocessing.
        stats.metadata_normalization = _fit_metadata_normalization(conn)

        # Load movie embedding lookup maps into memory.
        #
        # This is okay because movie-level tables are small compared to ratings.
        # These maps let _generate_features_for_users quickly look up embeddings by movie_id.
        content_embeddings = _load_embedding_map(
            conn,
            "SELECT movie_id, embedding FROM content_embeddings",
            CONTENT_EMBEDDING_DIM,
        )
        cf_embeddings = _load_embedding_map(
            conn,
            "SELECT movie_id, embedding FROM cf_embeddings",
            CF_EMBEDDING_DIM,
        )

        # Load candidate metadata lookup:
        # movie_id -> CatalogRow(year, runtime, popularity, vote stats)
        catalog = _load_catalog_map(conn)

        # Build the set of exact candidate rows.
        #
        # Each key identifies one rating event and tells us which split it belongs to:
        #   (user_id, movie_id, rating, rating_timestamp, event_id, split)
        #
        # _generate_features_for_users uses this to decide:
        #   "When walking this user's full timeline, is this row a train/val/test target?"
        candidate_key_rows = conn.execute(
            """
            SELECT user_id, movie_id, rating, rating_timestamp, id, split
            FROM candidates_enriched
            """
        ).fetchall()
        candidate_keys = {
            (
                int(uid),
                int(mid),
                float(rating),
                _normalize_rating_timestamp(ts),
                int(eid),
                str(split),
            )
            for uid, mid, rating, ts, eid, split in candidate_key_rows
        }

        # Generate the actual hybrid feature vectors and upload Parquet parts.
        #
        # This is where the function walks user timelines, builds user profiles from
        # strict past history, and writes train/validation/test rows.
        split_row_counts, split_parts, generation_quality = _generate_features_for_users(
            conn,
            client,
            settings,
            dataset_version,
            content_embeddings=content_embeddings,
            cf_embeddings=cf_embeddings,
            catalog=catalog,
            metadata_stats=stats.metadata_normalization,
            candidate_keys=candidate_keys,
            join_dropped_candidates=join_dropped_candidates,
        )

        # Store output counts and part metadata on the stats object.
        stats.train_row_count = split_row_counts["train"]
        stats.validation_row_count = split_row_counts["validation"]
        stats.test_row_count = split_row_counts["test"]
        stats.train_parts = split_parts["train"]
        stats.validation_parts = split_parts["validation"]
        stats.test_parts = split_parts["test"]

        cf_expected_row_count = (
            cf_dataset_manifest.train_row_count
            + cf_dataset_manifest.validation_row_count
            + cf_dataset_manifest.test_row_count
        )
        hybrid_total = stats.train_row_count + stats.validation_row_count + stats.test_row_count
        generation_quality.cf_expected_row_count = cf_expected_row_count
        generation_quality.row_count_delta_vs_cf = compute_row_count_delta(
            hybrid_total,
            cf_expected_row_count,
        )
        stats.quality = generation_quality

        # Build and write the manifest last.
        #
        # Writing manifest.json last is important:
        # - if parts fail halfway, there is no complete manifest
        # - downstream jobs only trust datasets with a complete manifest
        finished_at = utc_now()
        manifest = build_complete_hybrid_manifest(
            dataset_version=dataset_version,
            snapshot_id=snapshot_id,
            cf_dataset_version=cf_dataset_version,
            cf_version=cf_version,
            content_embedding_version=settings.content_embedding_version,
            feature_schema_version=settings.feature_schema_version,
            pipeline_run_id=pipeline_run_id,
            created_at=created_at,
            finished_at=finished_at,
            train_row_count=stats.train_row_count,
            validation_row_count=stats.validation_row_count,
            test_row_count=stats.test_row_count,
            metadata_normalization=stats.metadata_normalization,
            train_parts=stats.train_parts,
            validation_parts=stats.validation_parts,
            test_parts=stats.test_parts,
        )

        manifest_key = hybrid_ranker_manifest_object_key(dataset_version)
        put_json(client, settings.s3_bucket, manifest_key, manifest.model_dump(mode="json"))

        logger.info(
            "Hybrid feature generation complete",
            extra={
                "dataset_version": dataset_version,
                "snapshot_id": snapshot_id,
                "cf_dataset_version": cf_dataset_version,
                "cf_version": cf_version,
                "train_row_count": stats.train_row_count,
                "validation_row_count": stats.validation_row_count,
                "test_row_count": stats.test_row_count,
                "row_count_delta_vs_cf": stats.quality.row_count_delta_vs_cf if stats.quality else None,
                "cold_start_fraction": stats.quality.cold_start_fraction if stats.quality else None,
                "join_dropped_candidates": stats.quality.join_dropped_candidates if stats.quality else None,
                "manifest_key": manifest_key,
            },
        )

        return stats

    finally:
        # Always close DuckDB, even if the job fails halfway through.
        conn.close()

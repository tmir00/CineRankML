"""DuckDB pipeline that builds hybrid ranker feature datasets from frozen artifacts."""

from __future__ import annotations

import logging
import tempfile
import time

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
from common.features.similarity import WeightedEmbeddingAccumulator
from create_features.manifest import build_complete_hybrid_manifest, utc_now
from common.schemas.hybrid_ranker_dataset_manifest import HybridRankerPartEntry
from common.storage.duckdb_s3 import configure_duckdb_s3, create_duckdb_connection
from common.schemas.cf_dataset_manifest import CfDatasetManifest, CfDatasetPartEntry
from common.storage.snapshot_reader import resolve_snapshot_id, snapshot_table_glob_uri
from common.storage.cf_artifact_reader import load_cf_artifact_manifest, resolve_cf_version
from common.storage.cf_dataset_reader import load_cf_dataset_manifest, resolve_cf_dataset_version
from common.features.behavior import UserBehaviorAccumulator
from common.features.normalization import MetadataNormalizationStats, normalize_candidate_metadata
from common.features.schema import CF_EMBEDDING_DIM, CONTENT_EMBEDDING_DIM, INPUT_DIM
from common.metrics.hybrid_feature_stats import HybridFeatureQualityStats, compute_row_count_delta


logger = logging.getLogger(__name__)
SplitName = Literal["train", "validation", "test"]

_STREAMED_RATINGS_BATCH_SQL = """
SELECT
    r.user_id,
    r.movie_id,
    r.rating,
    r.rating_timestamp,
    r.id,
    c.split
FROM ratings r
INNER JOIN _user_batch ub
    ON r.user_id = ub.user_id
LEFT JOIN candidates c
    ON c.user_id = r.user_id
   AND c.movie_id = r.movie_id
   AND c.rating = r.rating
   AND c.rating_timestamp = r.rating_timestamp
   AND c.id = r.id
ORDER BY r.user_id, r.rating_timestamp, r.id
"""

_DUCKDB_LOOKUP_TABLES = (
    "catalog",
    "content_embeddings",
    "cf_embeddings",
    "user_id_map",
    "movie_id_map",
    "cf_candidates",
)

_SPLIT_BUFFER_INITIAL_CAPACITY = 10_000


@dataclass
class _SplitPartBuffer:
    """
    Hold one split's output rows in compact NumPy columns before Parquet upload.

    Do this by:
    1. Storing user_id, movie_id, rating, and features in parallel arrays.
    2. Growing the arrays in chunks when the buffer fills up.
    3. Exposing the filled row count through len(buffer).
    """

    max_rows: int
    capacity: int = field(init=False)
    user_ids: np.ndarray = field(init=False)
    movie_ids: np.ndarray = field(init=False)
    ratings: np.ndarray = field(init=False)
    features: np.ndarray = field(init=False)
    count: int = 0

    def __post_init__(self) -> None:
        # Start small and grow toward max_rows so we do not allocate the full part upfront.
        self.capacity = min(_SPLIT_BUFFER_INITIAL_CAPACITY, self.max_rows)
        self._allocate_arrays(self.capacity)

    def _allocate_arrays(self, capacity: int) -> None:
        """Create empty NumPy columns with room for capacity rows."""
        self.user_ids = np.zeros(capacity, dtype=np.int64)
        self.movie_ids = np.zeros(capacity, dtype=np.int64)
        self.ratings = np.zeros(capacity, dtype=np.float32)
        self.features = np.zeros((capacity, INPUT_DIM), dtype=np.float32)

    def _grow(self) -> None:
        """
        Double buffer capacity up to max_rows.

        Do this by:
        1. Computing the next capacity size.
        2. Copying existing rows into larger arrays.
        """
        new_capacity = min(self.capacity * 2, self.max_rows)
        if new_capacity <= self.capacity:
            raise RuntimeError(
                f"split part buffer exceeded max_rows={self.max_rows} before flush"
            )

        new_user_ids = np.zeros(new_capacity, dtype=np.int64)
        new_movie_ids = np.zeros(new_capacity, dtype=np.int64)
        new_ratings = np.zeros(new_capacity, dtype=np.float32)
        new_features = np.zeros((new_capacity, INPUT_DIM), dtype=np.float32)

        new_user_ids[: self.count] = self.user_ids[: self.count]
        new_movie_ids[: self.count] = self.movie_ids[: self.count]
        new_ratings[: self.count] = self.ratings[: self.count]
        new_features[: self.count] = self.features[: self.count]

        self.user_ids = new_user_ids
        self.movie_ids = new_movie_ids
        self.ratings = new_ratings
        self.features = new_features
        self.capacity = new_capacity

    def append(
        self,
        user_id: int,
        movie_id: int,
        rating: float,
        feature_vector: np.ndarray,
    ) -> None:
        """
        Add one feature row to the buffer.

        ============================ Arguments ============================
        user_id: The user this training example belongs to.
        movie_id: The candidate movie id.
        rating: The label rating value.
        feature_vector: The 1356-dim hybrid feature vector as float32 NumPy array.
        """
        if self.count >= self.capacity:
            self._grow()

        row_index = self.count
        self.user_ids[row_index] = user_id
        self.movie_ids[row_index] = movie_id
        self.ratings[row_index] = float(rating)
        self.features[row_index] = np.asarray(feature_vector, dtype=np.float32)
        self.count += 1

    def __len__(self) -> int:
        """Return how many rows are currently stored in the buffer."""
        return self.count


@dataclass
class _UserWalkAccumulators:
    """Running profile and behavior state for one user during the timeline walk."""

    content_profile_acc: WeightedEmbeddingAccumulator
    cf_profile_acc: WeightedEmbeddingAccumulator
    behavior_acc: UserBehaviorAccumulator


@dataclass
class _FeatureGenerationWalkContext:
    """Shared mutable state used while streaming rating rows into feature parts."""

    client: BaseClient
    settings: CreateFeaturesSettings
    dataset_version: str
    content_embeddings: dict[int, np.ndarray]
    cf_embeddings: dict[int, np.ndarray]
    catalog: dict[int, CatalogRow]
    metadata_stats: MetadataNormalizationStats
    split_row_counts: dict[SplitName, int]
    split_parts: dict[SplitName, list[HybridRankerPartEntry]]
    split_buffers: dict[SplitName, _SplitPartBuffer]
    split_part_indexes: dict[SplitName, int]
    part_key_builders: dict[SplitName, Callable[[str, int], str]]
    cold_start_rows: int = 0
    missing_content_embedding_rows: int = 0
    missing_cf_embedding_rows: int = 0


def _new_user_walk_accumulators() -> _UserWalkAccumulators:
    """Create fresh accumulators when the stream moves to a new user."""
    return _UserWalkAccumulators(
        content_profile_acc=WeightedEmbeddingAccumulator(CONTENT_EMBEDDING_DIM),
        cf_profile_acc=WeightedEmbeddingAccumulator(CF_EMBEDDING_DIM),
        behavior_acc=UserBehaviorAccumulator(),
    )


def _log_hybrid_feature_progress(
    *,
    users_processed: int,
    total_users: int,
    split_row_counts: dict[SplitName, int],
    split_parts: dict[SplitName, list[HybridRankerPartEntry]],
    feature_phase_start: float,
) -> None:
    """Log one progress snapshot during hybrid feature generation."""
    elapsed_seconds = time.perf_counter() - feature_phase_start
    logger.info(
        "Hybrid feature generation progress users=%s/%s train_rows=%s "
        "validation_rows=%s test_rows=%s train_parts=%s validation_parts=%s "
        "test_parts=%s elapsed_seconds=%.1f",
        users_processed,
        total_users,
        split_row_counts["train"],
        split_row_counts["validation"],
        split_row_counts["test"],
        len(split_parts["train"]),
        len(split_parts["validation"]),
        len(split_parts["test"]),
        elapsed_seconds,
    )


def _process_rating_row(
    *,
    user_id: int,
    movie_id: int,
    rating: float,
    split_name: SplitName | None,
    accumulators: _UserWalkAccumulators,
    walk_context: _FeatureGenerationWalkContext,
) -> None:
    """
    Apply one streamed rating row to the current user's walk state.

    Do this by:
    1. Looking up candidate movie embeddings.
    2. Updating history only when the row is not a prediction target.
    3. Building and buffering one feature row when the row is a prediction target.
    4. Observing the rating into history after a feature row is written.

    ============================ Arguments ============================
    user_id: The user whose timeline is currently being walked.
    movie_id: The movie in this rating event.
    rating: The rating value for this event.
    split_name: Train/validation/test split when this row is a candidate, else None.
    accumulators: Current user's running profile and behavior accumulators.
    walk_context: Shared output buffers and quality counters for the full walk.
    """
    # Look up embeddings once for this movie.
    # If missing, use zero vectors so feature generation can continue.
    missing_content = movie_id not in walk_context.content_embeddings
    missing_cf = movie_id not in walk_context.cf_embeddings
    movie_content = walk_context.content_embeddings.get(
        movie_id,
        _zero_embedding(CONTENT_EMBEDDING_DIM),
    )
    movie_cf = walk_context.cf_embeddings.get(movie_id, _zero_embedding(CF_EMBEDDING_DIM))

    if split_name is None:
        # This rating is not a prediction target.
        # It only becomes part of the user's history for future rows.
        accumulators.behavior_acc.observe(rating)
        accumulators.content_profile_acc.observe(movie_content, rating)
        accumulators.cf_profile_acc.observe(movie_cf, rating)
        return

    # This rating is a prediction target.
    # Build user profiles from history BEFORE the current rating.
    #
    # Important:
    # The current movie has not been added to history yet.
    # That is what prevents leakage.
    user_content_profile = accumulators.content_profile_acc.profile()
    user_cf_profile = accumulators.cf_profile_acc.profile()

    # If the user has no valid history yet, use zero profile vectors.
    if user_content_profile.shape != (CONTENT_EMBEDDING_DIM,):
        user_content_profile = _zero_embedding(CONTENT_EMBEDDING_DIM)

    if user_cf_profile.shape != (CF_EMBEDDING_DIM,):
        user_cf_profile = _zero_embedding(CF_EMBEDDING_DIM)

    # Candidate movie metadata.
    # This describes the movie we are trying to predict the user's rating for.
    catalog_row = walk_context.catalog.get(movie_id, CatalogRow(None, None, None, None, None))

    # Normalize metadata using train-fit stats, so train/validation/test/inference
    # all use the same scaling.
    metadata_values = normalize_candidate_metadata(
        year=catalog_row.year,
        runtime=catalog_row.runtime,
        tmdb_popularity=catalog_row.tmdb_popularity,
        tmdb_vote_average=catalog_row.tmdb_vote_average,
        tmdb_vote_count=catalog_row.tmdb_vote_count,
        stats=walk_context.metadata_stats,
    )

    # Point-in-time user behavior stats.
    # The snapshot uses only ratings observed before this row.
    user_behavior = accumulators.behavior_acc.snapshot()

    if float(user_behavior[0]) == 0.0:
        walk_context.cold_start_rows += 1
    if missing_content:
        walk_context.missing_content_embedding_rows += 1
    if missing_cf:
        walk_context.missing_cf_embedding_rows += 1

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
    walk_context.split_buffers[split_name].append(
        user_id,
        movie_id,
        rating,
        feature_vector,
    )
    walk_context.split_row_counts[split_name] += 1

    # Flush this split if the buffer has reached the configured part size.
    if len(walk_context.split_buffers[split_name]) >= walk_context.settings.hybrid_part_row_limit:
        part_entry = _flush_output_part(
            walk_context.client,
            walk_context.settings.s3_bucket,
            walk_context.dataset_version,
            split_name,
            walk_context.split_part_indexes[split_name],
            walk_context.split_buffers[split_name],
            walk_context.part_key_builders[split_name],
        )
        walk_context.split_parts[split_name].append(part_entry)
        walk_context.split_part_indexes[split_name] += 1
        walk_context.split_buffers[split_name] = _SplitPartBuffer(
            walk_context.settings.hybrid_part_row_limit
        )

    # Add the current rating to history AFTER writing the feature row.
    #
    # This lets the current movie help predict future ratings,
    # but not itself.
    accumulators.behavior_acc.observe(rating)
    accumulators.content_profile_acc.observe(movie_content, rating)
    accumulators.cf_profile_acc.observe(movie_cf, rating)


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


def _release_duckdb_lookup_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Drop DuckDB tables that were copied into Python before feature generation.

    Keeping ratings and candidates is enough for the streamed walk. Releasing the
    lookup tables lowers peak memory before the heavy sort/join phase starts.
    """
    for table_name in _DUCKDB_LOOKUP_TABLES:
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")


def _create_user_batch_table(conn: duckdb.DuckDBPyConnection, user_ids: list[int]) -> None:
    """Materialize one batch of user ids for a scoped ratings stream query."""
    conn.execute(
        "CREATE OR REPLACE TEMP TABLE _user_batch AS SELECT unnest(?) AS user_id",
        [user_ids],
    )


def _drop_user_batch_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Remove the temporary user batch table after one scoped stream finishes."""
    conn.execute("DROP TABLE IF EXISTS _user_batch")


def _iter_streamed_rating_batches(
    conn: duckdb.DuckDBPyConnection,
    user_ids: list[int],
    *,
    user_batch_size: int,
    fetch_batch_size: int,
):
    """
    Yield rating-row batches for candidate users in bounded DuckDB chunks.

    Do this by:
    1. Walking distinct user ids in fixed-size batches.
    2. Running one sorted ratings query per user batch.
    3. Streaming rows from that query with fetchmany().
    """
    for batch_start in range(0, len(user_ids), user_batch_size):
        user_batch = user_ids[batch_start : batch_start + user_batch_size]
        _create_user_batch_table(conn, user_batch)
        try:
            rating_stream = conn.execute(_STREAMED_RATINGS_BATCH_SQL)
            while True:
                rating_batch = rating_stream.fetchmany(fetch_batch_size)
                if not rating_batch:
                    break
                yield rating_batch
        finally:
            _drop_user_batch_table(conn)


def _process_rating_batches(
    rating_batches,
    *,
    walk_context: _FeatureGenerationWalkContext,
    total_users: int,
    feature_phase_start: float,
) -> int:
    """
    Walk streamed rating batches and update shared feature-generation state.

    ============================ Returns ============================
    Number of distinct users processed.
    """
    users_processed = 0
    current_user_id: int | None = None
    accumulators = _new_user_walk_accumulators()

    for rating_batch in rating_batches:
        for (
            row_user_id,
            movie_id,
            rating,
            _rating_timestamp,
            _event_id,
            split,
        ) in rating_batch:
            row_user_id = int(row_user_id)

            if row_user_id != current_user_id:
                current_user_id = row_user_id
                users_processed += 1
                if (
                    walk_context.settings.hybrid_progress_log_every_n > 0
                    and users_processed % walk_context.settings.hybrid_progress_log_every_n == 0
                ):
                    _log_hybrid_feature_progress(
                        users_processed=users_processed,
                        total_users=total_users,
                        split_row_counts=walk_context.split_row_counts,
                        split_parts=walk_context.split_parts,
                        feature_phase_start=feature_phase_start,
                    )
                accumulators = _new_user_walk_accumulators()

            movie_id = int(movie_id)
            rating = float(rating)
            split_name: SplitName | None = (
                str(split) if split in ("train", "validation", "test") else None
            )

            _process_rating_row(
                user_id=row_user_id,
                movie_id=movie_id,
                rating=rating,
                split_name=split_name,
                accumulators=accumulators,
                walk_context=walk_context,
            )

    return users_processed


def _split_part_buffer_to_arrow_table(buffer: _SplitPartBuffer) -> pa.Table:
    """
    Build one PyArrow table from a filled split part buffer.

    Do this by:
    1. Flatten the stacked feature matrix into one float32 values array.
    2. Wrapping those values as a fixed-size list column with length 1356.
    3. Combining user_id, movie_id, rating, and features into one table.

    ============================ Arguments ============================
    buffer: Filled split part buffer ready for Parquet upload.

    ============================ Returns ============================
    PyArrow table with user_id, movie_id, rating, and features columns.
    """
    row_count = buffer.count
    features_flat = pa.array(
        buffer.features[:row_count].reshape(-1),
        type=pa.float32(),
    )
    features_column = pa.FixedSizeListArray.from_arrays(
        features_flat,
        list_size=INPUT_DIM,
    )
    return pa.table(
        {
            "user_id": buffer.user_ids[:row_count],
            "movie_id": buffer.movie_ids[:row_count],
            "rating": buffer.ratings[:row_count],
            "features": features_column,
        }
    )


def _flush_output_part(
    client: BaseClient,
    bucket: str,
    dataset_version: str,
    split: SplitName,
    part_index: int,
    buffer: _SplitPartBuffer,
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
    buffer: Buffered output rows with user_id, movie_id, rating, and NumPy features.
    part_key_builder: Function that builds the object key for a split part.

    ============================ Returns ============================
    HybridRankerPartEntry metadata for the uploaded part.
    """
    object_key = part_key_builder(dataset_version, part_index)
    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = Path(temp_dir) / f"{split}-part-{part_index:05d}.parquet"
        table = _split_part_buffer_to_arrow_table(buffer)
        pq.write_table(table, local_path)
        upload_file(client, bucket, object_key, local_path)

    logger.info(
        "Uploaded hybrid %s part index=%s rows=%s object_key=%s",
        split,
        part_index,
        len(buffer),
        object_key,
    )

    return HybridRankerPartEntry(object_key=object_key, row_count=len(buffer))


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
    join_dropped_candidates: int,
) -> tuple[dict[SplitName, int], dict[SplitName, list[HybridRankerPartEntry]], HybridFeatureQualityStats]:
    """
    Build hybrid ranker feature rows and upload them as train/validation/test Parquet parts.

    Big idea:
    For each user, walk through their ratings in time order. The current rating is the
    prediction target only if it has a non-null split from the candidates join. When it
    is a candidate row, build the feature vector from the user's history BEFORE this
    rating, then use the current rating as the label.

    This prevents leakage:
    - The model can use movies the user rated before the candidate.
    - The model cannot use the candidate movie/rating inside the user profile.
    - After the row is written, the current rating is added to history for future rows.

    Do this by:
    1. Streaming candidate-user ratings in bounded user batches.
    2. Writing feature rows only for candidate rows.
    3. Buffering rows and flushing each split to Parquet when the buffer is full.

    ============================ Arguments ============================
    conn: DuckDB connection with ratings and candidates tables loaded for one streamed join query.
    client: boto3 S3 client for uploads.
    settings: Hybrid feature generation configuration.
    dataset_version: Output hybrid dataset version.
    content_embeddings: movie_id -> content embedding lookup.
    cf_embeddings: movie_id -> CF embedding lookup.
    catalog: movie_id -> raw candidate metadata lookup.
    metadata_stats: Train-fit metadata normalization stats.
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
    # Rows are stored as compact NumPy columns instead of Python dicts.
    # This avoids writing one tiny file per training example.
    split_buffers: dict[SplitName, _SplitPartBuffer] = {
        "train": _SplitPartBuffer(settings.hybrid_part_row_limit),
        "validation": _SplitPartBuffer(settings.hybrid_part_row_limit),
        "test": _SplitPartBuffer(settings.hybrid_part_row_limit),
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

    # Count how many users have at least one candidate row.
    # This is only used for progress logging denominators.
    user_ids = [
        int(row[0])
        for row in conn.execute(
            """
            SELECT DISTINCT user_id
            FROM candidates
            ORDER BY user_id
            """
        ).fetchall()
    ]
    total_users = len(user_ids)
    feature_phase_start = time.perf_counter()
    logger.info(
        "Starting streamed hybrid feature generation users=%s user_batch_size=%s "
        "part_row_limit=%s fetch_batch_size=%s",
        total_users,
        settings.hybrid_user_batch_size,
        settings.hybrid_part_row_limit,
        settings.hybrid_rating_fetch_batch_size,
    )

    walk_context = _FeatureGenerationWalkContext(
        client=client,
        settings=settings,
        dataset_version=dataset_version,
        content_embeddings=content_embeddings,
        cf_embeddings=cf_embeddings,
        catalog=catalog,
        metadata_stats=metadata_stats,
        split_row_counts=split_row_counts,
        split_parts=split_parts,
        split_buffers=split_buffers,
        split_part_indexes=split_part_indexes,
        part_key_builders=part_key_builders,
    )

    users_processed = _process_rating_batches(
        _iter_streamed_rating_batches(
            conn,
            user_ids,
            user_batch_size=settings.hybrid_user_batch_size,
            fetch_batch_size=settings.hybrid_rating_fetch_batch_size,
        ),
        walk_context=walk_context,
        total_users=total_users,
        feature_phase_start=feature_phase_start,
    )

    cold_start_rows = walk_context.cold_start_rows
    missing_content_embedding_rows = walk_context.missing_content_embedding_rows
    missing_cf_embedding_rows = walk_context.missing_cf_embedding_rows

    # Flush any leftover rows that did not fill a complete part.
    for split_name in ("train", "validation", "test"):
        if len(split_buffers[split_name]) > 0:
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

    feature_phase_elapsed = time.perf_counter() - feature_phase_start
    logger.info(
        "Finished streamed hybrid feature generation users=%s/%s train_rows=%s "
        "validation_rows=%s test_rows=%s train_parts=%s validation_parts=%s "
        "test_parts=%s elapsed_seconds=%.1f",
        users_processed,
        total_users,
        split_row_counts["train"],
        split_row_counts["validation"],
        split_row_counts["test"],
        len(split_parts["train"]),
        len(split_parts["validation"]),
        len(split_parts["test"]),
        feature_phase_elapsed,
    )

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
    5. Computing user behavior stats during the per-user timeline walk using only ratings before each candidate row.
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

    logger.info(
        "Resolved hybrid feature generation inputs snapshot_id=%s cf_dataset_version=%s "
        "cf_version=%s dataset_version=%s",
        snapshot_id,
        cf_dataset_version,
        cf_version,
        dataset_version,
    )

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
    duckdb_phase_start = time.perf_counter()

    try:
        logger.info("Loading snapshot and CF inputs into DuckDB")
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

        ratings_count, catalog_count, content_count, cf_count = conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM ratings) AS ratings_count,
                (SELECT COUNT(*) FROM catalog) AS catalog_count,
                (SELECT COUNT(*) FROM content_embeddings) AS content_count,
                (SELECT COUNT(*) FROM cf_embeddings) AS cf_count
            """
        ).fetchone()
        logger.info(
            "Loaded DuckDB source tables ratings=%s catalog=%s content_embeddings=%s cf_embeddings=%s",
            ratings_count,
            catalog_count,
            content_count,
            cf_count,
        )

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
                "Dropped CF candidate rows that could not join back to ratings_events dropped_rows=%s",
                join_dropped_candidates,
            )

        logger.info(
            "Built candidate tables cf_candidates=%s joined_candidates=%s",
            cf_rows,
            joined_rows,
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

        _release_duckdb_lookup_tables(conn)
        logger.info("Released DuckDB lookup tables before hybrid feature generation")

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
            "Hybrid feature generation complete dataset_version=%s snapshot_id=%s "
            "cf_dataset_version=%s cf_version=%s train_row_count=%s "
            "validation_row_count=%s test_row_count=%s row_count_delta_vs_cf=%s "
            "cold_start_fraction=%s join_dropped_candidates=%s manifest_key=%s",
            dataset_version,
            snapshot_id,
            cf_dataset_version,
            cf_version,
            stats.train_row_count,
            stats.validation_row_count,
            stats.test_row_count,
            stats.quality.row_count_delta_vs_cf if stats.quality else None,
            stats.quality.cold_start_fraction if stats.quality else None,
            stats.quality.join_dropped_candidates if stats.quality else None,
            manifest_key,
        )

        return stats

    finally:
        # Always close DuckDB, even if the job fails halfway through.
        conn.close()

"""Streaming IterableDataset over hybrid ranker Parquet parts in MinIO."""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from botocore.client import BaseClient
from torch.utils.data import IterableDataset

from common.features.schema import INPUT_DIM
from common.schemas.hybrid_ranker_dataset_manifest import HybridRankerPartEntry
from common.storage.s3 import download_file


logger = logging.getLogger(__name__)


def _features_and_ratings_from_table(table: pa.Table) -> tuple[np.ndarray, np.ndarray]:
    """
    Load feature vectors and ratings from one hybrid ranker Parquet table.

    Do this by:
    1. Reading the fixed-size list feature column into one float32 values buffer.
    2. Reshaping that buffer to (row_count, INPUT_DIM).
    3. Reading ratings as a float32 NumPy array.
    """
    features_column = table.column("features").combine_chunks()
    row_count = len(features_column)
    if row_count == 0:
        return (
            np.empty((0, INPUT_DIM), dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )

    if pa.types.is_fixed_size_list(features_column.type):
        values = features_column.values.to_numpy(zero_copy_only=False)
        features = np.asarray(values, dtype=np.float32).reshape(row_count, INPUT_DIM)
    else:
        features = np.asarray(
            [np.asarray(row, dtype=np.float32) for row in features_column.to_pylist()],
            dtype=np.float32,
        )

    ratings = table.column("rating").to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    return features, ratings


def _optional_id_columns_from_table(table: pa.Table) -> tuple[np.ndarray, np.ndarray] | None:
    """Return user_id and movie_id arrays when both columns are present."""
    if "user_id" not in table.column_names or "movie_id" not in table.column_names:
        return None
    user_ids = table.column("user_id").to_numpy(zero_copy_only=False)
    movie_ids = table.column("movie_id").to_numpy(zero_copy_only=False)
    return user_ids, movie_ids


def _row_order(row_count: int, *, shuffle_within_part: bool, seed: int) -> np.ndarray:
    """Return row indices in traversal order, optionally shuffled."""
    if not shuffle_within_part or row_count == 0:
        return np.arange(row_count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return rng.permutation(row_count)


def _yield_part_rows(
    table: pa.Table,
    *,
    shuffle_within_part: bool,
    seed: int,
    include_ids: bool,
) -> Iterator[tuple]:
    """Yield dataset rows from one in-memory Parquet table."""
    features, ratings = _features_and_ratings_from_table(table)
    row_count = len(ratings)
    row_indices = _row_order(row_count, shuffle_within_part=shuffle_within_part, seed=seed)
    id_columns = _optional_id_columns_from_table(table) if include_ids else None

    for row_index in row_indices:
        if id_columns is None:
            yield features[row_index], float(ratings[row_index])
        else:
            user_ids, movie_ids = id_columns
            yield features[row_index], float(ratings[row_index]), int(user_ids[row_index]), int(movie_ids[row_index])


class HybridParquetIterableDataset(IterableDataset):
    """
    Stream one hybrid ranker Parquet part at a time from MinIO.

    Do this by:
    1. Iterating part object keys in the order provided by the caller.
    2. Downloading each part to a temp file and reading it with PyArrow.
    3. Yielding (features, rating, user_id, movie_id) row tuples.
    """

    def __init__(self, client: BaseClient, bucket: str, parts: Sequence[HybridRankerPartEntry], *, \
                    shuffle_within_part: bool = False, seed: int = 0, include_ids: bool = False) -> None:
        """
        Configure one streaming dataset over hybrid ranker Parquet parts.

        ============================ Arguments ============================
        client: The boto3 S3 client.
        bucket: Source MinIO/S3 bucket.
        parts: Train, validation, or test part metadata from the dataset manifest.
        shuffle_within_part: When True, shuffle rows inside each downloaded part.
        seed: RNG seed for within-part shuffling.
        include_ids: When True, yield user_id and movie_id alongside features and rating.
        """
        super().__init__()
        self._client = client
        self._bucket = bucket
        self._parts = list(parts)
        self._shuffle_within_part = shuffle_within_part
        self._seed = seed
        self._include_ids = include_ids

    def __iter__(self) -> Iterator[tuple]:
        """
        Yield feature rows from each part in order.

        Do this by:
        1. Getting the worker info.
        2. If the worker info is not None and the number of workers is greater than 1, raise an error.
        3. Create a list of columns to read from the parquet file.
        4. If the include_ids flag is True, add the user_id and movie_id columns to the list of columns.
        5. Iterate over the parts in order.
        6. Download the part to a temporary file.
        7. Read the part into a PyArrow table.
        8. Convert feature and rating columns into NumPy arrays.
        9. If the shuffle_within_part flag is True, shuffle row indices inside the part.
        10. Iterate over the rows and yield the features and rating.
        11. If the include_ids flag is True, yield the user_id and movie_id alongside the features and rating.

        ============================ Returns ============================
        Tuples of (features, rating) or (features, rating, user_id, movie_id).
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            raise RuntimeError("HybridParquetIterableDataset supports only num_workers=0")

        columns = ["features", "rating"]
        if self._include_ids:
            columns.extend(["user_id", "movie_id"])

        total_parts = len(self._parts)
        for part_index, part in enumerate(self._parts):
            with tempfile.TemporaryDirectory() as temp_dir:
                local_path = Path(temp_dir) / f"part-{part_index:05d}.parquet"
                logger.info(
                    "Loading parquet part %d/%d object_key=%s",
                    part_index + 1,
                    total_parts,
                    part.object_key,
                )
                download_file(self._client, self._bucket, part.object_key, local_path)
                table = pq.read_table(local_path, columns=columns)
                logger.info(
                    "Loaded parquet part %d/%d rows=%d",
                    part_index + 1,
                    total_parts,
                    len(table),
                )
                yield from _yield_part_rows(
                    table,
                    shuffle_within_part=self._shuffle_within_part,
                    seed=self._seed + part_index,
                    include_ids=self._include_ids,
                )


class HybridParquetLocalIterableDataset(IterableDataset):
    """
    Stream one local hybrid ranker Parquet part at a time.

    Used by unit tests and local development without MinIO.
    """

    def __init__(self, part_paths: Sequence[Path], *, shuffle_within_part: bool = False, seed: int = 0, include_ids: bool = False,) -> None:
        super().__init__()
        self._part_paths = list(part_paths)
        self._shuffle_within_part = shuffle_within_part
        self._seed = seed
        self._include_ids = include_ids

    def __iter__(self) -> Iterator[tuple]:
        """
        Yield feature rows from each part in order. Used for unit tests and local development without MinIO.
        """
        for part_index, part_path in enumerate(self._part_paths):
            columns = ["features", "rating"]
            if self._include_ids:
                columns.extend(["user_id", "movie_id"])
            table = pq.read_table(part_path, columns=columns)
            yield from _yield_part_rows(
                table,
                shuffle_within_part=self._shuffle_within_part,
                seed=self._seed + part_index,
                include_ids=self._include_ids,
            )

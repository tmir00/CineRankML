"""Streaming IterableDataset over hybrid ranker Parquet parts in MinIO."""

from __future__ import annotations

import random
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path

import pyarrow.parquet as pq
import torch
from botocore.client import BaseClient
from torch.utils.data import IterableDataset

from common.schemas.hybrid_ranker_dataset_manifest import HybridRankerPartEntry
from common.storage.s3 import download_file


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
        3. Create a random number generator with the seed.
        4. Create a list of columns to read from the parquet file.
        5. If the include_ids flag is True, add the user_id and movie_id columns to the list of columns.
        6. Iterate over the parts in order.
        7. Download the part to a temporary file.
        8. Read the part into a PyArrow table.
        9. Convert the PyArrow table to a list of rows.
        10. If the shuffle_within_part flag is True, shuffle the rows.
        11. Iterate over the rows and yield the features and rating.
        12. If the include_ids flag is True, yield the user_id and movie_id alongside the features and rating.

        ============================ Returns ============================
        Tuples of (features, rating) or (features, rating, user_id, movie_id).
        """
        # Get the worker info like number of workers.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            # Raise an error if the number of workers is greater than 1.
            raise RuntimeError("HybridParquetIterableDataset supports only num_workers=0")

        # Create a random number generator with the seed.
        rng = random.Random(self._seed)
        # Create a list of columns to read from the parquet file.
        columns = ["features", "rating"]
        if self._include_ids:
            columns.extend(["user_id", "movie_id"])

        # Iterate over the parts in order.
        for part_index, part in enumerate(self._parts):
            # Download the part to a temporary file.
            with tempfile.TemporaryDirectory() as temp_dir:
                local_path = Path(temp_dir) / f"part-{part_index:05d}.parquet"
                download_file(self._client, self._bucket, part.object_key, local_path)

                # Read the part into a PyArrow table.
                table = pq.read_table(local_path, columns=columns)
                # Convert the PyArrow table to a list of rows.
                rows = list(zip(*(table.column(name).to_pylist() for name in columns)))

                # If the shuffle_within_part flag is True, shuffle the rows.
                if self._shuffle_within_part:
                    rng.shuffle(rows)

                # Iterate over the rows and yield the features and rating.
                for row in rows:
                    features = [float(value) for value in row[0]]
                    rating = float(row[1])

                    # If the include_ids flag is True, yield the user_id and movie_id alongside the features and rating.
                    if self._include_ids:
                        yield features, rating, int(row[2]), int(row[3])
                    # Otherwise, yield the features and rating.
                    else:
                        yield features, rating


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
        rng = random.Random(self._seed)
        columns = ["features", "rating"]
        if self._include_ids:
            columns.extend(["user_id", "movie_id"])

        for part_path in self._part_paths:
            table = pq.read_table(part_path, columns=columns)
            rows = list(zip(*(table.column(name).to_pylist() for name in columns)))

            if self._shuffle_within_part:
                rng.shuffle(rows)

            for row in rows:
                features = [float(value) for value in row[0]]
                rating = float(row[1])
                if self._include_ids:
                    yield features, rating, int(row[2]), int(row[3])
                else:
                    yield features, rating

"""Streaming IterableDataset over CF dataset Parquet parts in MinIO."""

from __future__ import annotations

import random
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path

import pyarrow.parquet as pq
import torch
from botocore.client import BaseClient
from torch.utils.data import IterableDataset

from common.schemas.cf_dataset_manifest import CfDatasetPartEntry
from common.storage.s3 import download_file


class CfParquetIterableDataset(IterableDataset):
    """
    Stream one CF dataset Parquet part at a time from MinIO.

    Do this by:
    1. Iterating part object keys in the order provided by the caller.
    2. Downloading each part to a temp file and reading it with PyArrow.
    3. Yielding (user_idx, movie_idx, rating) row tuples.
    """

    def __init__(self, client: BaseClient, bucket: str, parts: Sequence[CfDatasetPartEntry], *,
                    shuffle_within_part: bool = False, seed: int = 0) -> None:
        """
        Configure one streaming dataset over CF Parquet parts.

        ============================ Arguments ============================
        client: The boto3 S3 client.
        bucket: Source MinIO/S3 bucket.
        parts: Train or validation part metadata from the CF dataset manifest.
        shuffle_within_part: When True, shuffle rows inside each downloaded part.
        seed: RNG seed for within-part shuffling.
        """
        super().__init__()
        self._client = client
        self._bucket = bucket
        self._parts = list(parts)
        self._shuffle_within_part = shuffle_within_part
        self._seed = seed

    def __iter__(self) -> Iterator[tuple[int, int, float]]:
        """
        Yield mapped rating rows from each part in order.

        ============================ Returns ============================
        Tuples of (user_idx, movie_idx, rating).
        """
        # Check if the dataset is being used in a multi-worker environment.
        worker_info = torch.utils.data.get_worker_info()
        # If the dataset is being used in a multi-worker environment, raise an error.
        if worker_info is not None and worker_info.num_workers > 1:
            raise RuntimeError("CfParquetIterableDataset supports only num_workers=0")

        # Create a random number generator for within-part shuffling.
        rng = random.Random(self._seed)
        # Iterate over the parts in order.
        for part_index, part in enumerate(self._parts):
            # Create a temporary directory to download the part to.
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download the part to the temporary directory.
                local_path = Path(temp_dir) / f"part-{part_index:05d}.parquet"
                download_file(self._client, self._bucket, part.object_key, local_path)
                # Read the part into a PyArrow table.
                table = pq.read_table(local_path, columns=["user_idx", "movie_idx", "rating"])
                # Convert the PyArrow table to a list of tuples.
                rows = list(
                    zip(
                        table.column("user_idx").to_pylist(),
                        table.column("movie_idx").to_pylist(),
                        table.column("rating").to_pylist(),
                    )
                )

                # Shuffle the rows within the part if requested.
                if self._shuffle_within_part:
                    rng.shuffle(rows)

                # Yield the rows.
                for user_idx, movie_idx, rating in rows:
                    yield int(user_idx), int(movie_idx), float(rating)


class CfParquetLocalIterableDataset(IterableDataset):
    """
    Stream one local CF dataset Parquet part at a time.

    Used by unit tests and local development without MinIO.
    """

    def __init__(self, part_paths: Sequence[Path], *, shuffle_within_part: bool = False, seed: int = 0) -> None:
        super().__init__()
        self._part_paths = list(part_paths)
        self._shuffle_within_part = shuffle_within_part
        self._seed = seed

    def __iter__(self) -> Iterator[tuple[int, int, float]]:
        # Create a random number generator for within-part shuffling.
        rng = random.Random(self._seed)
        # Iterate over the parts in order.
        for part_index, part_path in enumerate(self._part_paths):
            # Read the part into a PyArrow table.
            table = pq.read_table(part_path, columns=["user_idx", "movie_idx", "rating"])
            # Convert the PyArrow table to a list of tuples.
            rows = list(
                zip(
                    table.column("user_idx").to_pylist(),
                    table.column("movie_idx").to_pylist(),
                    table.column("rating").to_pylist(),
                )
            )
            # Shuffle the rows within the part if requested.
            if self._shuffle_within_part:
                rng.shuffle(rows)
            # Yield the rows.
            for user_idx, movie_idx, rating in rows:
                yield int(user_idx), int(movie_idx), float(rating)

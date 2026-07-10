"""Tests for local CF Parquet dataset iteration."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from train_cf.dataset import CfParquetLocalIterableDataset


def test_cf_parquet_local_iterable_dataset_yields_rows(tmp_path: Path) -> None:
    """A local Parquet part should stream user_idx, movie_idx, and rating rows."""
    part_path = tmp_path / "part-00000.parquet"
    table = pa.table(
        {
            "user_idx": [0, 1],
            "movie_idx": [2, 3],
            "rating": [4.0, 5.0],
        }
    )
    pq.write_table(table, part_path)

    dataset = CfParquetLocalIterableDataset([part_path])
    rows = list(iter(dataset))

    assert rows == [(0, 2, 4.0), (1, 3, 5.0)]

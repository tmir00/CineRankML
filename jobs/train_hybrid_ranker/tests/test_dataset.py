"""Unit tests for HybridParquetLocalIterableDataset."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from pathlib import Path

from train_hybrid_ranker.dataset import HybridParquetLocalIterableDataset


def test_local_dataset_yields_feature_rows(tmp_path: Path) -> None:
    part_path = tmp_path / "part-00000.parquet"
    table = pa.table(
        {
            "user_id": [1, 1],
            "movie_id": [10, 20],
            "rating": [4.0, 3.0],
            "features": [[1.0, 2.0], [3.0, 4.0]],
        }
    )
    pq.write_table(table, part_path)

    dataset = HybridParquetLocalIterableDataset([part_path])
    rows = list(iter(dataset))

    assert len(rows) == 2
    assert rows[0][0] == [1.0, 2.0]
    assert rows[0][1] == 4.0

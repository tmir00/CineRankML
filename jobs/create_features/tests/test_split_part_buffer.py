"""Tests for compact NumPy split part buffers and Parquet round-trips."""

from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq

from pathlib import Path

from common.features.schema import INPUT_DIM
from create_features.prep import _SplitPartBuffer, _split_part_buffer_to_arrow_table


def test_split_part_buffer_append_and_len() -> None:
    """Appending rows should update len(buffer) and store feature vectors exactly."""
    buffer = _SplitPartBuffer(max_rows=100)
    feature_a = np.arange(INPUT_DIM, dtype=np.float32)
    feature_b = np.arange(INPUT_DIM, dtype=np.float32) + 1.0

    buffer.append(1, 10, 4.0, feature_a)
    buffer.append(2, 20, 3.5, feature_b)

    assert len(buffer) == 2
    np.testing.assert_array_equal(buffer.features[0], feature_a)
    np.testing.assert_array_equal(buffer.features[1], feature_b)
    assert buffer.user_ids[0] == 1
    assert buffer.movie_ids[1] == 20
    assert buffer.ratings[1] == 3.5


def test_split_part_buffer_grows_past_initial_capacity() -> None:
    """The buffer should grow beyond its initial capacity without losing rows."""
    buffer = _SplitPartBuffer(max_rows=50_000)
    initial_capacity = buffer.capacity
    assert initial_capacity == 10_000

    rows_to_append = initial_capacity + 1
    for row_index in range(rows_to_append):
        feature_vector = np.full(INPUT_DIM, float(row_index), dtype=np.float32)
        buffer.append(1, row_index, 4.0, feature_vector)

    assert len(buffer) == rows_to_append
    assert buffer.capacity > initial_capacity
    np.testing.assert_array_equal(
        buffer.features[rows_to_append - 1],
        np.full(INPUT_DIM, float(rows_to_append - 1), dtype=np.float32),
    )


def test_split_part_buffer_round_trip_parquet(tmp_path: Path) -> None:
    """Parquet written from the buffer should read back the same feature vectors."""
    buffer = _SplitPartBuffer(max_rows=10)
    expected_features: list[list[float]] = []

    for row_index in range(3):
        feature_vector = np.linspace(row_index, row_index + 1, INPUT_DIM, dtype=np.float32)
        buffer.append(100 + row_index, 200 + row_index, 2.0 + row_index, feature_vector)
        expected_features.append(feature_vector.astype(float).tolist())

    table = _split_part_buffer_to_arrow_table(buffer)
    parquet_path = tmp_path / "part-00000.parquet"
    pq.write_table(table, parquet_path)

    read_table = pq.read_table(parquet_path)
    read_features = read_table.column("features").to_pylist()
    read_ratings = read_table.column("rating").to_pylist()

    assert read_features == expected_features
    assert read_ratings == [2.0, 3.0, 4.0]

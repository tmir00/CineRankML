"""Tests for streamed rating walks in hybrid feature generation."""

from __future__ import annotations

import duckdb

from create_features.prep import (
    _create_user_batch_table,
    _drop_user_batch_table,
    _iter_streamed_rating_batches,
    _new_user_walk_accumulators,
)


def _walk_streamed_rows(conn: duckdb.DuckDBPyConnection, *, batch_size: int = 2) -> list[tuple[int, str | None, float]]:
    """
    Walk the streamed ratings query and record candidate snapshots.

    Returns tuples of (user_id, split_name, prior_rating_count).
    """
    current_user_id: int | None = None
    accumulators = _new_user_walk_accumulators()
    candidate_snapshots: list[tuple[int, str | None, float]] = []

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

    for rating_batch in _iter_streamed_rating_batches(
        conn,
        user_ids,
        user_batch_size=len(user_ids),
        fetch_batch_size=batch_size,
    ):
        for row_user_id, _movie_id, rating, _rating_timestamp, _event_id, split in rating_batch:
            row_user_id = int(row_user_id)
            if row_user_id != current_user_id:
                current_user_id = row_user_id
                accumulators = _new_user_walk_accumulators()

            split_name = str(split) if split in ("train", "validation", "test") else None
            if split_name is None:
                accumulators.behavior_acc.observe(float(rating))
                continue

            prior_rating_count = float(accumulators.behavior_acc.snapshot()[0])
            candidate_snapshots.append((row_user_id, split_name, prior_rating_count))
            accumulators.behavior_acc.observe(float(rating))

    return candidate_snapshots


def test_streamed_query_routes_splits_and_resets_user_state() -> None:
    """The sorted stream should attach splits and reset history on user boundaries."""
    conn = duckdb.connect()
    conn.execute(
        """
        CREATE TABLE ratings (
            user_id BIGINT,
            movie_id BIGINT,
            rating DOUBLE,
            rating_timestamp BIGINT,
            id BIGINT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE candidates (
            user_id BIGINT,
            movie_id BIGINT,
            rating DOUBLE,
            rating_timestamp BIGINT,
            id BIGINT,
            split VARCHAR
        )
        """
    )

    # User 1: one history row, then one train candidate.
    # User 2: only one rating row and it is the candidate (cold start).
    conn.execute(
        """
        INSERT INTO ratings VALUES
            (1, 101, 3.0, 100, 1),
            (1, 102, 4.5, 200, 2),
            (2, 202, 2.0, 250, 4)
        """
    )
    conn.execute(
        """
        INSERT INTO candidates VALUES
            (1, 102, 4.5, 200, 2, 'train'),
            (2, 202, 2.0, 250, 4, 'test')
        """
    )

    snapshots = _walk_streamed_rows(conn)

    assert snapshots == [
        (1, "train", 1.0),
        (2, "test", 0.0),
    ]


def test_user_batch_table_scopes_streamed_rows() -> None:
    """A user batch table should limit the streamed query to selected users."""
    conn = duckdb.connect()
    conn.execute(
        """
        CREATE TABLE ratings (
            user_id BIGINT,
            movie_id BIGINT,
            rating DOUBLE,
            rating_timestamp BIGINT,
            id BIGINT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE candidates (
            user_id BIGINT,
            movie_id BIGINT,
            rating DOUBLE,
            rating_timestamp BIGINT,
            id BIGINT,
            split VARCHAR
        )
        """
    )
    conn.execute(
        """
        INSERT INTO ratings VALUES
            (1, 101, 3.0, 100, 1),
            (2, 202, 2.0, 250, 4)
        """
    )
    conn.execute(
        """
        INSERT INTO candidates VALUES
            (1, 101, 3.0, 100, 1, 'train'),
            (2, 202, 2.0, 250, 4, 'test')
        """
    )

    _create_user_batch_table(conn, [1])
    try:
        rows = conn.execute(
            """
            SELECT r.user_id
            FROM ratings r
            INNER JOIN _user_batch ub ON r.user_id = ub.user_id
            ORDER BY r.user_id
            """
        ).fetchall()
    finally:
        _drop_user_batch_table(conn)

    assert rows == [(1,)]

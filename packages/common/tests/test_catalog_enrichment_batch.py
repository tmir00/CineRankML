"""Tests for catalog enrichment batch selection."""

from __future__ import annotations

from unittest.mock import MagicMock

from common.db.repositories.catalog import (
    count_enrichable_movies,
    fetch_enrichment_batch,
)


def _compiled_where(session: MagicMock) -> str:
    if session.scalars.called:
        stmt = session.scalars.call_args[0][0]
    else:
        stmt = session.execute.call_args[0][0]
    return str(stmt.compile(compile_kwargs={"literal_binds": True}))


def test_fetch_enrichment_batch_pending_filters_null_status_only() -> None:
    session = MagicMock()
    session.scalars.return_value.all.return_value = []

    fetch_enrichment_batch(session, batch_size=10, remaining_limit=None, enrich_all=False)

    where_clause = _compiled_where(session)
    assert "catalog_movies.enrichment_status IS NULL" in where_clause
    assert "catalog_movies.tmdb_id IS NOT NULL" in where_clause
    assert "catalog_movies.movie_id >" not in where_clause


def test_fetch_enrichment_batch_enrich_all_uses_tmdb_id_and_cursor() -> None:
    session = MagicMock()
    session.scalars.return_value.all.return_value = []

    fetch_enrichment_batch(
        session,
        batch_size=25,
        remaining_limit=None,
        enrich_all=True,
        after_movie_id=42,
    )

    where_clause = _compiled_where(session)
    assert "catalog_movies.tmdb_id IS NOT NULL" in where_clause
    assert "catalog_movies.movie_id > 42" in where_clause
    assert "catalog_movies.enrichment_status IS NULL" not in where_clause


def test_fetch_enrichment_batch_respects_remaining_limit() -> None:
    session = MagicMock()
    session.scalars.return_value.all.return_value = []

    fetch_enrichment_batch(session, batch_size=50, remaining_limit=3, enrich_all=False)

    stmt = session.scalars.call_args[0][0]
    assert stmt._limit_clause.value == 3  # type: ignore[attr-defined]


def test_count_enrichable_movies_pending_mode() -> None:
    session = MagicMock()
    session.execute.return_value.scalar_one.return_value = 7

    assert count_enrichable_movies(session, enrich_all=False) == 7

    where_clause = _compiled_where(session)
    assert "catalog_movies.enrichment_status IS NULL" in where_clause


def test_count_enrichable_movies_enrich_all_mode() -> None:
    session = MagicMock()
    session.execute.return_value.scalar_one.return_value = 99

    assert count_enrichable_movies(session, enrich_all=True) == 99

    where_clause = _compiled_where(session)
    assert "catalog_movies.tmdb_id IS NOT NULL" in where_clause
    assert "catalog_movies.enrichment_status IS NULL" not in where_clause

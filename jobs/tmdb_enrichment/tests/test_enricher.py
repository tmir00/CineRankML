"""Tests for TMDB enrichment loop behavior."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from common.config.settings import EnrichmentSettings
from common.tmdb.client import TmdbMovieDetails
from tmdb_enrichment.enricher import EnrichmentStats, run_enrichment_loop


@dataclass
class _Movie:
    movie_id: int
    tmdb_id: int
    title: str = "Test Movie"


def _details() -> TmdbMovieDetails:
    return TmdbMovieDetails(
        overview="Overview",
        tagline=None,
        original_language="en",
        runtime=120,
        tmdb_popularity=1.0,
        tmdb_vote_average=7.0,
        tmdb_vote_count=100,
        tmdb_keywords=["sci-fi"],
        poster_path="/poster.jpg",
    )


@patch("tmdb_enrichment.enricher.mark_catalog_movie_dirty")
@patch("tmdb_enrichment.enricher.apply_enrichment")
@patch("tmdb_enrichment.enricher.fetch_enrichment_batch")
@patch("tmdb_enrichment.enricher.mark_movies_without_tmdb_skipped", return_value=0)
def test_run_enrichment_loop_marks_dirty_on_success(
    _mark_skipped: MagicMock,
    fetch_batch: MagicMock,
    apply_enrichment: MagicMock,
    mark_dirty: MagicMock,
) -> None:
    session = MagicMock()
    client = MagicMock()
    client.fetch_movie.return_value = (_details(), None)
    fetch_batch.side_effect = [[_Movie(movie_id=1, tmdb_id=10)], []]

    stats = EnrichmentStats()
    settings = EnrichmentSettings(enrichment_batch_size=10, enrichment_log_every_n=1000)

    run_enrichment_loop(session, client, settings, stats, enrich_all=False)

    apply_enrichment.assert_called_once()
    mark_dirty.assert_called_once_with(session, 1)
    assert stats.processed == 1
    assert stats.failed == 0
    fetch_batch.assert_any_call(
        session,
        batch_size=10,
        remaining_limit=None,
        enrich_all=False,
        after_movie_id=0,
    )


@patch("tmdb_enrichment.enricher.mark_catalog_movie_dirty")
@patch("tmdb_enrichment.enricher.apply_enrichment")
@patch("tmdb_enrichment.enricher.fetch_enrichment_batch")
@patch("tmdb_enrichment.enricher.mark_movies_without_tmdb_skipped", return_value=0)
def test_run_enrichment_loop_enrich_all_advances_cursor(
    _mark_skipped: MagicMock,
    fetch_batch: MagicMock,
    _apply_enrichment: MagicMock,
    _mark_dirty: MagicMock,
) -> None:
    session = MagicMock()
    client = MagicMock()
    client.fetch_movie.return_value = (_details(), None)
    fetch_batch.side_effect = [
        [_Movie(movie_id=5, tmdb_id=50)],
        [],
    ]

    stats = EnrichmentStats()
    settings = EnrichmentSettings(enrichment_batch_size=10, enrichment_log_every_n=1000)

    run_enrichment_loop(session, client, settings, stats, enrich_all=True)

    assert fetch_batch.call_args_list[0].kwargs["enrich_all"] is True
    assert fetch_batch.call_args_list[0].kwargs["after_movie_id"] == 0
    assert fetch_batch.call_args_list[1].kwargs["after_movie_id"] == 5

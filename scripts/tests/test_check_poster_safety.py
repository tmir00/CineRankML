"""Tests for the offline poster safety script."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import check_poster_safety


def test_check_poster_safety_dry_run_prints_counts(monkeypatch, capsys) -> None:
    session = MagicMock()
    session_factory = MagicMock(return_value=session)

    monkeypatch.setattr(check_poster_safety, "get_session_factory", lambda: session_factory)
    monkeypatch.setattr(
        check_poster_safety,
        "count_poster_safety_stats",
        lambda _session: SimpleNamespace(
            total_catalog=100,
            total_with_poster=80,
            total_questionable=12,
            total_already_checked=5,
        ),
    )
    monkeypatch.setattr(
        check_poster_safety,
        "fetch_questionable_poster_candidates",
        lambda *_args, **_kwargs: [SimpleNamespace(movie_id=1)],
    )
    monkeypatch.setattr(
        check_poster_safety,
        "_build_parser",
        lambda: SimpleNamespace(
            parse_args=lambda: SimpleNamespace(
                limit=100,
                force=False,
                dry_run=True,
                only_certification=None,
                sexual_score_threshold=0.35,
                model="omni-moderation-latest",
                request_delay=0.5,
                commit_every=100,
            )
        ),
    )

    check_poster_safety.main()

    output = capsys.readouterr().out
    assert "\"total_catalog_movies\": 100" in output
    assert "\"total_to_check_in_run\": 1" in output
    assert "\"checked_count\": 0" in output

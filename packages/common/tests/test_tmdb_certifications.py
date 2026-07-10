"""Tests for TMDB release_dates certification parsing."""

from __future__ import annotations

from common.tmdb.certifications import extract_us_certification


def test_extract_us_certification_returns_first_non_empty_us_value() -> None:
    payload = {
        "results": [
            {"iso_3166_1": "CA", "release_dates": [{"certification": "14A"}]},
            {
                "iso_3166_1": "US",
                "release_dates": [
                    {"certification": ""},
                    {"certification": "PG-13"},
                    {"certification": "R"},
                ],
            },
        ]
    }

    assert extract_us_certification(payload) == "PG-13"


def test_extract_us_certification_returns_none_when_missing() -> None:
    assert extract_us_certification({"results": [{"iso_3166_1": "GB"}]}) is None

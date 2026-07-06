"""Tests for ratings-topic event parsing."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from common.schemas.events import RatingCreatedEvent, RatingDeletedEvent, parse_rating_stream_event


def _created_payload() -> dict:
    now = datetime.now(tz=UTC)
    return {
        "event_id": str(uuid4()),
        "event_type": "rating_created",
        "stream_pipeline_version": "ratings-v1",
        "source": "api",
        "occurred_at": now.isoformat(),
        "produced_at": now.isoformat(),
        "user_id": 1,
        "movie_id": 50,
        "rating": 4.5,
        "rating_timestamp": now.isoformat(),
    }


def _deleted_payload() -> dict:
    now = datetime.now(tz=UTC)
    return {
        "event_id": str(uuid4()),
        "event_type": "rating_deleted",
        "stream_pipeline_version": "ratings-v1",
        "source": "api",
        "occurred_at": now.isoformat(),
        "produced_at": now.isoformat(),
        "user_id": 1,
        "movie_id": 50,
        "rating_timestamp": now.isoformat(),
    }


def test_parse_rating_stream_event_created() -> None:
    event = parse_rating_stream_event(_created_payload())
    assert isinstance(event, RatingCreatedEvent)
    assert event.rating == 4.5


def test_parse_rating_stream_event_deleted() -> None:
    event = parse_rating_stream_event(_deleted_payload())
    assert isinstance(event, RatingDeletedEvent)
    assert event.movie_id == 50


def test_parse_rating_stream_event_unknown_type_raises() -> None:
    payload = _created_payload()
    payload["event_type"] = "rating_updated"
    with pytest.raises(ValueError, match="Unknown ratings event_type"):
        parse_rating_stream_event(payload)

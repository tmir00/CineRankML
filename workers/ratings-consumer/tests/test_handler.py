"""Tests for ratings-topic consumer dispatch."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

from common.schemas.events import RatingCreatedEvent, RatingDeletedEvent
from ratings_consumer.handler import process_rating_stream_event


def _created_event() -> RatingCreatedEvent:
    now = datetime.now(tz=UTC)
    return RatingCreatedEvent(
        event_id=uuid4(),
        event_type="rating_created",
        stream_pipeline_version="ratings-v1",
        source="api",
        occurred_at=now,
        produced_at=now,
        user_id=1,
        movie_id=50,
        rating=4.0,
        rating_timestamp=now,
    )


def _deleted_event() -> RatingDeletedEvent:
    now = datetime.now(tz=UTC)
    return RatingDeletedEvent(
        event_id=uuid4(),
        event_type="rating_deleted",
        stream_pipeline_version="ratings-v1",
        source="api",
        occurred_at=now,
        produced_at=now,
        user_id=1,
        movie_id=50,
        rating_timestamp=now,
    )


@patch("ratings_consumer.handler.insert_rating_event", return_value=True)
def test_process_rating_stream_event_routes_created(mock_insert_created: MagicMock) -> None:
    with patch("ratings_consumer.handler.get_session_factory") as mock_factory:
        session = MagicMock()
        mock_factory.return_value = MagicMock(return_value=session)
        status = process_rating_stream_event(_created_event())

    assert status == "success"
    mock_insert_created.assert_called_once()
    session.commit.assert_called_once()


@patch("ratings_consumer.handler.insert_rating_deleted_event", return_value=True)
def test_process_rating_stream_event_routes_deleted(mock_insert_deleted: MagicMock) -> None:
    with patch("ratings_consumer.handler.get_session_factory") as mock_factory:
        session = MagicMock()
        mock_factory.return_value = MagicMock(return_value=session)
        status = process_rating_stream_event(_deleted_event())

    assert status == "success"
    mock_insert_deleted.assert_called_once()
    session.commit.assert_called_once()

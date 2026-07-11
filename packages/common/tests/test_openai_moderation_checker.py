"""Tests for the OpenAI moderation poster safety wrapper."""

from __future__ import annotations

import sys
import types

import pytest

from common.poster_safety.openai_moderation_checker import (
    _evaluate_moderation_result,
    check_poster_with_openai_moderation,
)


def _make_moderation_response(
    *,
    sexual: bool = False,
    sexual_minors: bool = False,
    sexual_score: float = 0.0,
    flagged: bool = False,
) -> dict[str, object]:
    return {
        "flagged": flagged or sexual or sexual_minors,
        "categories": {
            "sexual": sexual,
            "sexual/minors": sexual_minors,
        },
        "category_scores": {
            "sexual": sexual_score,
            "sexual/minors": 0.0,
        },
        "category_applied_input_types": {
            "sexual": ["image"],
        },
    }


def test_evaluate_moderation_result_keeps_safe_poster() -> None:
    result = _evaluate_moderation_result(
        _make_moderation_response(sexual=False, sexual_minors=False, sexual_score=0.10),
        sexual_score_threshold=0.35,
    )

    assert result.poster_safe is True
    assert result.score == 0.10
    assert result.reason == "no_sexual_flags"


def test_evaluate_moderation_result_hides_when_sexual_flagged() -> None:
    result = _evaluate_moderation_result(
        _make_moderation_response(sexual=True, sexual_score=0.12),
        sexual_score_threshold=0.35,
    )

    assert result.poster_safe is False
    assert result.score == 0.12
    assert result.reason == "sexual=true"


def test_evaluate_moderation_result_hides_when_sexual_minors_flagged() -> None:
    result = _evaluate_moderation_result(
        _make_moderation_response(sexual_minors=True, sexual_score=0.05),
        sexual_score_threshold=0.35,
    )

    assert result.poster_safe is False
    assert result.reason == "sexual/minors=true"


def test_evaluate_moderation_result_hides_when_sexual_score_exceeds_threshold() -> None:
    result = _evaluate_moderation_result(
        _make_moderation_response(sexual_score=0.42),
        sexual_score_threshold=0.35,
    )

    assert result.poster_safe is False
    assert result.score == 0.42
    assert result.reason == "sexual_score=0.420>0.350"


def test_check_poster_with_openai_moderation_calls_api(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class _FakeModerations:
        def create(self, *, model: str, input: list[dict[str, object]]):
            assert model == "omni-moderation-latest"
            assert input[0]["type"] == "image_url"
            return types.SimpleNamespace(
                model_dump=lambda: {
                    "results": [_make_moderation_response(sexual_score=0.05)],
                }
            )

    class _FakeOpenAI:
        def __init__(self, *, max_retries: int = 0) -> None:
            assert max_retries == 0
            self.moderations = _FakeModerations()

    fake_openai_module = types.SimpleNamespace(OpenAI=_FakeOpenAI)
    monkeypatch.setitem(sys.modules, "openai", fake_openai_module)

    result = check_poster_with_openai_moderation(
        image_url="https://image.tmdb.org/t/p/w342/example.jpg",
        sexual_score_threshold=0.35,
        client=_FakeOpenAI(max_retries=0),
    )

    assert result.poster_safe is True
    assert result.score == 0.05


def test_check_poster_with_openai_moderation_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "common.poster_safety.openai_moderation_checker._load_openai_api_key_from_env_file",
        lambda: None,
    )

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        check_poster_with_openai_moderation(
            image_url="https://image.tmdb.org/t/p/w342/example.jpg",
        )

"""Tests for CF dataset split fraction validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from common.config.settings import CfDatasetSettings


def test_cf_dataset_settings_rejects_fractions_not_summing_to_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Train, validation, and test fractions must sum to 1.0."""
    monkeypatch.setenv("CF_TRAIN_FRACTION", "0.8")
    monkeypatch.setenv("CF_VALIDATION_FRACTION", "0.1")
    monkeypatch.setenv("CF_TEST_FRACTION", "0.05")

    with pytest.raises(ValidationError, match="must sum to 1.0"):
        CfDatasetSettings(_env_file=None)


def test_cf_dataset_settings_accepts_default_80_10_10_fractions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default fractions should be 0.8 / 0.1 / 0.1."""
    monkeypatch.delenv("CF_TRAIN_FRACTION", raising=False)
    monkeypatch.delenv("CF_VALIDATION_FRACTION", raising=False)
    monkeypatch.delenv("CF_TEST_FRACTION", raising=False)
    monkeypatch.delenv("TRAIN_FRACTION", raising=False)

    settings = CfDatasetSettings(_env_file=None)

    assert settings.train_fraction == 0.8
    assert settings.validation_fraction == 0.1
    assert settings.test_fraction == 0.1

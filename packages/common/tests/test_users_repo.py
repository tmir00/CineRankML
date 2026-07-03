"""Tests for app user repository helpers."""

from __future__ import annotations

from common.db.repositories.users import normalize_username


def test_normalize_username_lowercases_and_trims() -> None:
    assert normalize_username("  Alice  ") == "alice"

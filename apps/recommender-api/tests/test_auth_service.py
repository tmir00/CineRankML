"""Tests for recommender-api auth helpers."""

from __future__ import annotations

from recommender_api.services.auth import hash_password, verify_password


def test_password_hash_roundtrip() -> None:
    password_hash = hash_password("secret-password")
    assert verify_password("secret-password", password_hash)
    assert not verify_password("wrong-password", password_hash)

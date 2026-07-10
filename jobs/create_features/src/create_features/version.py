"""Resolve hybrid ranker dataset version identifiers."""

from __future__ import annotations

from datetime import UTC, datetime

from common.config.settings import CreateFeaturesSettings


def resolve_dataset_version(settings: CreateFeaturesSettings) -> str:
    """
    Return the hybrid dataset version from settings or generate a UTC timestamp id.

    ============================ Arguments ============================
    settings: Hybrid feature generation configuration.

    ============================ Returns ============================
    Version string like 2026-06-25T121000Z.
    """
    if settings.dataset_version:
        return settings.dataset_version
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H%M%SZ")

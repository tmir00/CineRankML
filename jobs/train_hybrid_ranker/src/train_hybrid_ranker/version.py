"""Generate hybrid ranker model version identifiers."""

from __future__ import annotations

from datetime import UTC, datetime

from common.config.settings import HybridTrainingSettings


def resolve_model_version(settings: HybridTrainingSettings) -> str:
    """
    Return the hybrid model version from settings or generate a UTC timestamp id.

    ============================ Arguments ============================
    settings: Hybrid training configuration.

    ============================ Returns ============================
    Version string like hybrid-v1-2026-06-25T123000Z.
    """
    if settings.model_version:
        return settings.model_version
    timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H%M%SZ")
    return f"hybrid-v1-{timestamp}"

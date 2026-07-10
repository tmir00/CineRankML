"""Generate CF artifact version identifiers."""

from __future__ import annotations

from datetime import UTC, datetime

from common.config.settings import CfTrainingSettings


def resolve_cf_version(settings: CfTrainingSettings) -> str:
    """
    Return the CF artifact version from settings or generate a UTC timestamp id.

    ============================ Arguments ============================
    settings: CF training configuration.

    ============================ Returns ============================
    Version string like cf-v1-2026-06-25T122000Z.
    """
    if settings.cf_version:
        return settings.cf_version
    timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H%M%SZ")
    return f"cf-v1-{timestamp}"

"""Shared logging setup for long-running workers."""

from __future__ import annotations

import logging
import sys


def configure_worker_logging(level: str = "INFO") -> None:
    """
    Configure root logging for a worker process.

    Do this by:
    1. Parsing the requested level name (e.g. DEBUG, INFO).
    2. Falling back to INFO when the level name is invalid.
    3. Applying a stdout basicConfig with force=True so reruns in tests stay predictable.

    ============================ Arguments ============================
    level: Logging level name from LOG_LEVEL env (default INFO).
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
        force=True,
    )

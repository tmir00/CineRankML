"""OpenAI omni-moderation poster safety classification."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODERATION_MODEL = "omni-moderation-latest"
DEFAULT_SEXUAL_SCORE_THRESHOLD = 0.35
MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 5.0


@dataclass(frozen=True)
class PosterSafetyResult:
    """Outcome of one offline poster safety check."""

    poster_safe: bool
    score: float
    reason: str | None
    flagged: bool
    categories: dict[str, bool]
    category_scores: dict[str, float]
    category_applied_input_types: dict[str, list[str]]


def _load_openai_api_key_from_env_file() -> None:
    """Load repository .env into os.environ when OPENAI_API_KEY is not already set."""
    if os.getenv("OPENAI_API_KEY"):
        return
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        return


def _require_openai_api_key() -> None:
    _load_openai_api_key_from_env_file()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is required")


def create_moderation_client() -> Any:
    """Create a reusable OpenAI client for a batch poster safety run."""
    from openai import OpenAI  # lazy import — not needed in API containers

    _require_openai_api_key()
    return OpenAI(max_retries=0)


def _is_retryable_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {429, 500, 502, 503, 504}:
        return True
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) in {
        429,
        500,
        502,
        503,
        504,
    }:
        return True
    return False


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    retry_after = headers.get("retry-after")
    if retry_after is None:
        return None
    try:
        return float(retry_after)
    except (TypeError, ValueError):
        return None


def _moderate_image(
    *,
    client: Any,
    image_url: str,
    model: str,
) -> dict[str, object]:
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.moderations.create(
                model=model,
                input=[
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    }
                ],
            )
            results = response.model_dump().get("results", [])
            if not results:
                raise ValueError("OpenAI moderation response contained no results")
            first_result = results[0]
            if not isinstance(first_result, dict):
                raise ValueError("OpenAI moderation result was not a dictionary")
            return first_result
        except Exception as exc:
            last_error = exc
            if attempt >= MAX_RETRIES or not _is_retryable_error(exc):
                raise
            retry_after = _retry_after_seconds(exc)
            sleep_seconds = retry_after if retry_after is not None else RETRY_BACKOFF_SECONDS * attempt
            logger.warning(
                "OpenAI moderation retry %s/%s in %.1fs after error: %s",
                attempt,
                MAX_RETRIES,
                sleep_seconds,
                exc,
            )
            time.sleep(sleep_seconds)

    if last_error is not None:
        raise last_error
    raise RuntimeError("OpenAI moderation failed without an error")


def _normalize_categories(raw: object) -> dict[str, bool]:
    if not isinstance(raw, dict):
        return {}
    return {str(key): bool(value) for key, value in raw.items()}


def _normalize_category_scores(raw: object) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    return {str(key): float(value) for key, value in raw.items()}


def _normalize_applied_input_types(raw: object) -> dict[str, list[str]]:
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, list[str]] = {}
    for key, value in raw.items():
        if isinstance(value, list):
            normalized[str(key)] = [str(item) for item in value]
        else:
            normalized[str(key)] = []
    return normalized


def _evaluate_moderation_result(
    result: dict[str, object],
    *,
    sexual_score_threshold: float,
) -> PosterSafetyResult:
    categories = _normalize_categories(result.get("categories", {}))
    category_scores = _normalize_category_scores(result.get("category_scores", {}))
    category_applied_input_types = _normalize_applied_input_types(
        result.get("category_applied_input_types", {})
    )
    flagged = bool(result.get("flagged", False))

    sexual_flagged = bool(categories.get("sexual", False))
    sexual_minors_flagged = bool(categories.get("sexual/minors", False))
    sexual_score = float(category_scores.get("sexual", 0.0))

    reasons: list[str] = []
    if sexual_flagged:
        reasons.append("sexual=true")
    if sexual_minors_flagged:
        reasons.append("sexual/minors=true")
    if sexual_score > sexual_score_threshold:
        reasons.append(f"sexual_score={sexual_score:.3f}>{sexual_score_threshold:.3f}")

    poster_safe = not (
        sexual_flagged or sexual_minors_flagged or sexual_score > sexual_score_threshold
    )
    reason = "; ".join(reasons) if reasons else "no_sexual_flags"
    return PosterSafetyResult(
        poster_safe=poster_safe,
        score=sexual_score,
        reason=reason,
        flagged=flagged,
        categories=categories,
        category_scores=category_scores,
        category_applied_input_types=category_applied_input_types,
    )


def check_poster_with_openai_moderation(
    *,
    image_url: str,
    sexual_score_threshold: float = DEFAULT_SEXUAL_SCORE_THRESHOLD,
    model: str = DEFAULT_MODERATION_MODEL,
    client: Any | None = None,
) -> PosterSafetyResult:
    """
    Classify a poster image with OpenAI omni-moderation and return a safety verdict.

    Do this by:
    1. Sending one poster image URL to the OpenAI Moderations API.
    2. Waiting for the response before returning (no parallel requests).
    3. Marking poster_safe=false when sexual content is flagged or the score is high.

    ============================ Arguments ============================
    image_url: Public HTTPS URL for the poster image (e.g. TMDB w342 URL).
    sexual_score_threshold: Hide when category_scores.sexual exceeds this value.
    model: OpenAI moderation model name (default omni-moderation-latest).
    client: Optional reusable OpenAI client for batch runs.

    ============================ Returns ============================
    PosterSafetyResult with poster_safe, sexual score, reason, and raw moderation fields.

    Raises:
        ValueError: When OPENAI_API_KEY is missing or the API response is invalid.
        Exception: On API/runtime errors (caller should leave poster_checked=false).
    """
    moderation_client = client if client is not None else create_moderation_client()
    result = _moderate_image(
        client=moderation_client,
        image_url=image_url,
        model=model,
    )
    return _evaluate_moderation_result(
        result,
        sexual_score_threshold=sexual_score_threshold,
    )

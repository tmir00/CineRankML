"""Offline poster safety checker for questionable catalog movies."""

from __future__ import annotations

import argparse
import json
import logging
import time

from datetime import UTC, datetime

from common.db.repositories.catalog import (
    PosterSafetyCandidate,
    PosterSafetyUpdate,
    count_poster_safety_stats,
    fetch_questionable_poster_candidates,
    mark_catalog_movie_dirty,
    update_poster_safety_batch,
)
from common.db.session import get_session_factory
from common.poster_safety.openai_moderation_checker import (
    DEFAULT_MODERATION_MODEL,
    DEFAULT_SEXUAL_SCORE_THRESHOLD,
    PosterSafetyResult,
    check_poster_with_openai_moderation,
    create_moderation_client,
)
from common.poster_safety.tmdb_poster import build_tmdb_poster_url

logger = logging.getLogger(__name__)

LOG_SEPARATOR = "=" * 80
DEFAULT_REQUEST_DELAY_SECONDS = 0.0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an offline poster safety check for questionable movies."
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-certification", type=str, default=None)
    parser.add_argument(
        "--sexual-score-threshold",
        type=float,
        default=DEFAULT_SEXUAL_SCORE_THRESHOLD,
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODERATION_MODEL)
    parser.add_argument(
        "--request-delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY_SECONDS,
        help="Seconds to wait after each OpenAI moderation response before the next movie.",
    )
    parser.add_argument("--commit-every", type=int, default=100)
    return parser


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def _format_categories(categories: dict[str, bool]) -> list[str]:
    return [name for name, flagged in sorted(categories.items()) if flagged]


def _format_category_scores(category_scores: dict[str, float]) -> dict[str, float]:
    return {
        name: round(score, 4)
        for name, score in sorted(
            category_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if score > 0.0
    }


def _print_check_report(
    *,
    candidate: PosterSafetyCandidate,
    poster_url: str | None,
    provider: str,
    poster_safe: bool,
    score: float,
    reason: str | None,
    result: PosterSafetyResult | None = None,
    error: str | None = None,
) -> None:
    print(LOG_SEPARATOR)
    print(f"Title: {candidate.title}")
    print(f"Movie ID: {candidate.movie_id}")
    print(f"Certification: {candidate.certification_us or 'n/a'}")
    print(f"Adult: {candidate.adult}")
    if poster_url:
        print(f"Poster URL: {poster_url}")
    print(f"Provider: {provider}")
    print(f"Poster safe: {poster_safe}")
    print(f"Sexual score: {score:.4f}")
    print(f"Verdict reason: {reason}")

    if result is not None:
        print("OpenAI moderation response:")
        print(f"  flagged: {result.flagged}")
        true_categories = _format_categories(result.categories)
        print(f"  categories_true: {true_categories if true_categories else '[]'}")
        print("  category_scores:")
        for name, value in _format_category_scores(result.category_scores).items():
            print(f"    {name}: {value:.4f}")
        if result.category_applied_input_types:
            print("  category_applied_input_types:")
            for name, input_types in sorted(result.category_applied_input_types.items()):
                if input_types:
                    print(f"    {name}: {input_types}")
    if error is not None:
        print(f"Error: {error}")
    print(LOG_SEPARATOR)
    print()


def main() -> None:
    _configure_logging()
    args = _build_parser().parse_args()
    session_factory = get_session_factory()
    session = session_factory()

    summary = {
        "checked_count": 0,
        "hidden_count": 0,
        "safe_count": 0,
        "failed_count": 0,
        "skipped_count": 0,
    }

    try:
        stats = count_poster_safety_stats(session)
        candidates = fetch_questionable_poster_candidates(
            session,
            limit=args.limit,
            force=args.force,
            only_certification=args.only_certification,
        )
        preamble = {
            "total_catalog_movies": stats.total_catalog,
            "total_movies_with_poster_path": stats.total_with_poster,
            "total_questionable_risky_movies": stats.total_questionable,
            "total_already_checked": stats.total_already_checked,
            "total_to_check_in_run": len(candidates),
        }
        print(json.dumps(preamble, indent=2))
        print()

        if args.dry_run:
            print(json.dumps(summary, indent=2))
            return

        updates: list[PosterSafetyUpdate] = []
        dirty_movie_ids: list[int] = []
        moderation_client = create_moderation_client()

        for candidate in candidates:
            checked_at = datetime.now(tz=UTC)

            if candidate.adult:
                _print_check_report(
                    candidate=candidate,
                    poster_url=build_tmdb_poster_url(candidate.poster_path, size="w342"),
                    provider="adult_flag",
                    poster_safe=False,
                    score=1.0,
                    reason="adult=true",
                )
                updates.append(
                    PosterSafetyUpdate(
                        movie_id=candidate.movie_id,
                        poster_checked=True,
                        poster_safe=False,
                        poster_safety_provider="adult_flag",
                        poster_safety_score=1.0,
                        poster_safety_reason="adult=true",
                        poster_checked_at=checked_at,
                    )
                )
                dirty_movie_ids.append(candidate.movie_id)
                summary["checked_count"] += 1
                summary["hidden_count"] += 1
            else:
                poster_url = build_tmdb_poster_url(candidate.poster_path, size="w342")
                try:
                    result = check_poster_with_openai_moderation(
                        image_url=poster_url,
                        sexual_score_threshold=args.sexual_score_threshold,
                        model=args.model,
                        client=moderation_client,
                    )
                except Exception as exc:  # noqa: BLE001 - retry later by leaving row untouched
                    _print_check_report(
                        candidate=candidate,
                        poster_url=poster_url,
                        provider="openai_omni_moderation",
                        poster_safe=False,
                        score=0.0,
                        reason="check_failed",
                        error=str(exc),
                    )
                    summary["failed_count"] += 1
                    if args.request_delay > 0:
                        time.sleep(args.request_delay)
                    continue

                _print_check_report(
                    candidate=candidate,
                    poster_url=poster_url,
                    provider="openai_omni_moderation",
                    poster_safe=result.poster_safe,
                    score=result.score,
                    reason=result.reason,
                    result=result,
                )
                updates.append(
                    PosterSafetyUpdate(
                        movie_id=candidate.movie_id,
                        poster_checked=True,
                        poster_safe=result.poster_safe,
                        poster_safety_provider="openai_omni_moderation",
                        poster_safety_score=result.score,
                        poster_safety_reason=result.reason,
                        poster_checked_at=checked_at,
                    )
                )
                dirty_movie_ids.append(candidate.movie_id)
                summary["checked_count"] += 1
                if result.poster_safe:
                    summary["safe_count"] += 1
                else:
                    summary["hidden_count"] += 1
                if args.request_delay > 0:
                    time.sleep(args.request_delay)

            if len(updates) >= args.commit_every:
                update_poster_safety_batch(session, updates)
                for movie_id in dirty_movie_ids:
                    mark_catalog_movie_dirty(session, movie_id)
                session.commit()
                updates.clear()
                dirty_movie_ids.clear()

        if updates:
            update_poster_safety_batch(session, updates)
            for movie_id in dirty_movie_ids:
                mark_catalog_movie_dirty(session, movie_id)
            session.commit()

        print(json.dumps(summary, indent=2))
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()

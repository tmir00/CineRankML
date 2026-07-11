"""Merge ranked recommendations from main and candidate models by split fractions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RankedCandidate:
    """One scored movie before final top-K selection."""

    movie_id: int
    predicted_score: float
    model_role: str
    model_version: str
    candidate_index: int
    title: str
    year: int | None
    genres: list[str]
    poster_path: str | None
    poster_safe: bool
    show_poster: bool
    certification_us: str | None
    retrieval_source: str


def merge_ranked_recommendations(*, main_ranked: list[RankedCandidate], candidate_ranked: list[RankedCandidate], \
                                    main_slots: int, candidate_slots: int) -> list[RankedCandidate]:
    """
    Build the final recommendation list from two model-ranked pools.

    Do this by:
    1. Taking the top main_slots from the main ranked list.
    2. Taking the top candidate_slots from the candidate ranked list.
    3. Removing duplicate movie_ids (keeping the higher predicted score).

    ============================ Arguments ============================
    main_ranked: Movies ranked by the main model (best first).
    candidate_ranked: Movies ranked by the candidate model (best first).
    main_slots: How many slots to allocate to main.
    candidate_slots: How many slots to allocate to candidate.

    ============================ Returns ============================
    Combined list before rank_position assignment (order not final).
    """
    selected: list[RankedCandidate] = []
    seen: dict[int, RankedCandidate] = {}

    for item in main_ranked[:main_slots] + candidate_ranked[:candidate_slots]:
        existing = seen.get(item.movie_id)
        if existing is None or item.predicted_score > existing.predicted_score:
            seen[item.movie_id] = item

    selected.extend(seen.values())
    selected.sort(key=lambda row: row.predicted_score, reverse=True)
    return selected

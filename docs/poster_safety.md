# Poster safety

Poster safety is precomputed offline. The API and frontend only read stored
catalog fields such as `poster_safe`, `poster_checked`, and `show_poster`.
There is no request-time moderation call.

## What gets checked

- `adult=true` is treated as automatically unsafe, so the poster is hidden
  without running the classifier.
- R-rated and NC-17 movies are considered questionable and stay hidden until
  the offline script finishes checking them.
- US certification comes from TMDB `release_dates`.
- Non-adult questionable movies are checked with OpenAI `omni-moderation-latest`.
  A poster is hidden when any of these are true:
  - `categories.sexual`
  - `categories["sexual/minors"]`
  - `category_scores.sexual > sexual_score_threshold` (default `0.35`)

## Prerequisites

Set `OPENAI_API_KEY` in the repository root `.env` before running the checker.
The script runs locally and calls the OpenAI Moderations API.

```bash
# .env
OPENAI_API_KEY=sk-...
```

## Run the checker

```bash
uv run --package poster-safety check-poster-safety --limit 100 --dry-run
uv run --package poster-safety check-poster-safety --limit 1000 --sexual-score-threshold 0.35
uv run --package poster-safety check-poster-safety --only-certification R --limit 5000
uv run --package poster-safety check-poster-safety --model omni-moderation-latest --request-delay 0.5
```

The checker sends **one OpenAI request at a time** using a reused client for the
whole run. Optional `--request-delay` (default `0`) sleeps between movies when
you need to avoid rate limits.

To replace prior `nudenet` results with OpenAI moderation, re-run with `--force`:

```bash
uv run --package poster-safety check-poster-safety --force --sexual-score-threshold 0.35
```

Run commands from the repository root so `.env` and shared workspace imports
resolve correctly.

## After the checker finishes

Propagate updated poster fields to search and the UI:

```bash
docker compose --profile search run --rm opensearch-sync
```

Restart or rebuild the API/frontend if they are already running.

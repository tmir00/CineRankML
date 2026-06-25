# Project Plan

This project is a production-style movie recommender MLE platform.

The goal is to build the system in small vertical phases. Each phase should be implemented, tested, and committed before moving to the next phase.

## Python tooling (`uv`)

This project uses [uv](https://docs.astral.sh/uv/) for Python package management.

### Layout

* Root `pyproject.toml` defines a **uv workspace**.
* `uv.lock` at the repo root pins exact dependency versions for the whole monorepo. **Commit `uv.lock`.**
* `.python-version` pins the Python version (use **3.12+**).
* `packages/common/` is the first workspace member and holds shared code (DB, config, logging, schemas).
* Each deployable (`workers/*`, `jobs/*`, `apps/*`, `services/*`) gets its own `pyproject.toml` when that container is built. That file is the service's dependency list (the `uv` equivalent of a per-service `requirements.txt`).

### How per-service dependencies work

* `packages/common/pyproject.toml` — shared deps used across the repo (for example `sqlalchemy`, `pydantic`, `alembic`).
* `workers/ratings-consumer/pyproject.toml` — only what that worker needs (for example `common` + Kafka client).
* `jobs/train_cf/pyproject.toml` — only what that job needs (for example `common` + `torch` + `pandas`).

Local development (install everything):

```bash
uv sync
```

Docker / CI (install one service only):

```bash
uv sync --frozen --package ratings-consumer --no-dev
```

Do **not** add a separate `requirements.txt` per service. Use workspace `pyproject.toml` files and the shared `uv.lock` instead.

### When to add workspace members

Add a new workspace member when a phase introduces a new deployable container. Do not create all `pyproject.toml` files upfront.

| Phase | Workspace members to add |
|-------|--------------------------|
| 1 | `packages/common` |
| 2 | Kafka workers |
| 3 | `jobs/seed_catalog`, `jobs/tmdb_enrichment` |
| 4 | `services/embedder-api`, `jobs/opensearch_sync` |
| 5 | `jobs/snapshot_to_s3` |
| 6+ | `jobs/train_cf`, remaining jobs, API as each phase ships |

## Phase 1: Docker, Postgres, and Schemas

Goal:
Set up local infrastructure and database schema.

Build:

* `uv` workspace bootstrap:
  * Root `pyproject.toml` with `[tool.uv.workspace]`.
  * `packages/common/pyproject.toml` as the shared installable package.
  * `uv.lock` and `.python-version`.
  * Phase 1 Python deps in `packages/common` (for example `sqlalchemy`, `alembic`, `psycopg`, `pydantic`).
* Docker Compose for Postgres, Prometheus and Grafana.
* Alembic setup under `packages/common/db`.
* Initial database tables:

  * `catalog_movies`
  * `ratings_events`
  * `tag_events`
  * `movie_tag_counts`
  * `catalog_dirty_movies`
  * `movie_content_embeddings`
  * `embedding_versions`
  * `pipeline_runs`
  * `recommendation_impressions`
  * `recommendation_ratings`

Acceptance criteria:

* `uv sync` installs dependencies from the committed `uv.lock`.
* `docker compose up` starts core infrastructure for db and monitoring.
* Alembic migrations run successfully.
* Tables are created with indexes and timestamps.

## Phase 2: Kafka Ingestion

Goal:
Stream MovieLens ratings and tags into Postgres.

Build:

* Kafka in root `docker-compose.yml` (KRaft broker; `ratings` and `tags` topics only).
* `workers/ratings-producer`
* `workers/tags-producer`
* `workers/ratings-consumer`
* `workers/tags-consumer`
* A `pyproject.toml` per worker (workspace members with Kafka-specific deps).
* Shared Kafka utilities in `packages/common/kafka`
* Shared event schemas in `packages/common/schemas` (Pydantic models used by producers and consumers).
* Prometheus scrape configs for worker metrics; Grafana dashboard panels for ingestion health and table counts.

### Event schemas (`packages/common/schemas`)

Use one shared base shape for ratings and tags events. Producers and consumers must share the same Pydantic models.

**Base fields (every event):**

```json
{
  "event_id": "018f5c2e-7e4a-7b3d-9e2f-9a2e7c8f4b11",
  "event_type": "rating_created",
  "stream_pipeline_version": "ratings-v1",
  "source": "movielens",
  "occurred_at": "2026-06-22T14:30:00Z",
  "produced_at": "2026-06-22T14:30:02Z"
}
```

| Field | Meaning |
|-------|---------|
| `event_id` | Unique ID for this event. Used for deduplication/idempotency. |
| `event_type` | What happened, e.g. `rating_created`, `tag_created`. |
| `stream_pipeline_version` | Version of the Kafka ingestion schema/pipeline. |
| `source` | Where the event came from, e.g. `movielens`, `api`, `recommendation`. |
| `occurred_at` | When the user action originally happened. |
| `produced_at` | When the producer sent it to Kafka. |

**Rating event example:**

```json
{
  "event_id": "018f5c2e-7e4a-7b3d-9e2f-9a2e7c8f4b11",
  "event_type": "rating_created",
  "stream_pipeline_version": "ratings-v1",
  "source": "api",
  "occurred_at": "2026-06-22T14:30:00Z",
  "produced_at": "2026-06-22T14:30:02Z",
  "user_id": 123,
  "movie_id": 50,
  "rating": 4.5,
  "rating_timestamp": "2026-06-22T14:30:00Z",
  "request_id": "rec-request-abc123",
  "model_version": "hybrid-ranker-v2",
  "experiment_id": "exp-main-vs-v2"
}
```

**Tag event example:**

```json
{
  "event_id": "018f5c2e-7e4a-7b3d-9e2f-9a2e7c8f4b12",
  "event_type": "tag_created",
  "stream_pipeline_version": "tags-v1",
  "source": "movielens",
  "occurred_at": "2026-06-22T14:31:00Z",
  "produced_at": "2026-06-22T14:31:02Z",
  "user_id": 123,
  "movie_id": 50,
  "tag": "funny",
  "tag_timestamp": "2026-06-22T14:31:00Z"
}
```

**Pydantic models:**

```python
class BaseKafkaEvent(BaseModel):
    event_id: UUID
    event_type: str
    stream_pipeline_version: str
    source: str
    occurred_at: datetime
    produced_at: datetime

class RatingCreatedEvent(BaseKafkaEvent):
    event_type: Literal["rating_created"]
    user_id: int
    movie_id: int
    rating: float
    rating_timestamp: datetime
    request_id: str | None = None
    model_version: str | None = None
    experiment_id: str | None = None

class TagCreatedEvent(BaseKafkaEvent):
    event_type: Literal["tag_created"]
    user_id: int
    movie_id: int
    tag: str
    tag_timestamp: datetime
```

Validate every event with Pydantic before produce and before consume.

### DLQ (dead-letter queue)

Use a single Postgres table **`dead_letter_events`**, not Kafka DLQ topics. When validation fails or a consumer cannot write to the event tables, the consumer saves the raw payload and error details to `dead_letter_events`, then commits the Kafka offset so the poison message does not block the pipeline.

Monitor failed messages via the `dead_letter_events` row count (postgres_exporter) and the `dlq_events_total` Prometheus counter.

### Prometheus / Grafana (Phase 2)

Track ingestion health in Prometheus/Grafana alongside existing Postgres exporter panels.

| Metric | Source |
|--------|--------|
| Query latency | Worker/app code (histogram), or `pg_stat_statements` later |
| Failed writes total | Worker/app code (counter) |
| `ratings_events` row count | Custom SQL via postgres_exporter, or Grafana Postgres datasource |
| `tag_events` row count | Custom SQL via postgres_exporter, or Grafana Postgres datasource |
| `catalog_dirty_movies` row count | Custom SQL via postgres_exporter, or Grafana Postgres datasource |
| `dead_letter_events` row count | Custom SQL via postgres_exporter |

Also expose standard worker metrics: `events_consumed_total`, `consumer_lag`, `dlq_events_total`, `db_write_failures_total`, `db_write_latency_seconds`.

Acceptance criteria:

* Producers publish valid JSON events matching the shared Pydantic schemas.
* Consumers validate events with Pydantic before writing to Postgres.
* Consumers write to Postgres idempotently using `event_id`.
* Kafka offsets are committed after a successful DB write, or after the message is saved to `dead_letter_events`.
* Invalid or unprocessable messages are saved to `dead_letter_events` (not silently dropped).
* Consumer and write-failure metrics are exposed to Prometheus.
* Grafana shows ingestion metrics and table row counts for `ratings_events`, `tag_events`, `catalog_dirty_movies`, and `dead_letter_events`.

## Phase 3: Catalog Seed and TMDB Enrichment

Goal:
Load MovieLens movie metadata and enrich it with TMDB data.

Build:

* `jobs/seed_catalog`
* `jobs/tmdb_enrichment`
* A `pyproject.toml` per job (workspace members).

Acceptance criteria:

* MovieLens `movies.csv` and `links.csv` load into `catalog_movies`.
* TMDB fields are added where available.
* Enriched movies are marked dirty for OpenSearch sync.

## Phase 4: OpenSearch Indexing

Goal:
Build and maintain the movie retrieval index.

Build:

* OpenSearch index mapping.
* Initial indexing flow and rebuild-from-Postgres support.
* `services/embedder-api` (MiniLM-L6-v2 content embeddings over HTTP).
* `jobs/opensearch_sync` (batch sync of dirty movies to OpenSearch).

Acceptance criteria:

* Movies can be indexed into OpenSearch.
* Dirty movies are synced from Postgres to OpenSearch.
* Content embeddings are written to `movie_content_embeddings` via embedder-api.
* OpenSearch index can be rebuilt from Postgres (`REBUILD_INDEX=true`).
* Sync metrics are exposed:
  * `dirty_movies_count` and `oldest_dirty_movie_age_seconds` (postgres_exporter).
  * Embedder health (`up{job="embedder-api"}`) and request rate (`embed_requests_total`).
  * OpenSearch index document count (`elasticsearch_indices_docs` via opensearch-exporter sidecar).

## Phase 5: MinIO Snapshots

Goal:
Export immutable training snapshots from Postgres to MinIO (S3-compatible object storage).

Build:

* MinIO in root `docker-compose.yml` (local bucket `cinerankml`).
* `jobs/snapshot_to_s3` (writes via the S3-compatible API to `s3://cinerankml/...`).

### MinIO bucket layout

Local MinIO exposes bucket `cinerankml` with top-level prefixes `raw/`, `snapshots/`, `features/`, `artifacts/`, and `models/`:

```text
s3://cinerankml/
  raw/
    movielens/
      movies.csv
      ratings.csv
      tags.csv
      links.csv

  snapshots/
    snapshot_id=2026-06-25T120000Z/
      ratings_events/
        part-00000.parquet
        part-00001.parquet
      tag_events/
        part-00000.parquet
      catalog_movies/
        part-00000.parquet
      movie_content_embeddings/
        part-00000.parquet
      manifest.json

  features/
    hybrid_ranker/
      dataset_version=2026-06-25T121000Z/
        train/
          part-00000.parquet
          part-00001.parquet
        validation/
          part-00000.parquet
        test/
          part-00000.parquet
        manifest.json

  artifacts/
    collaborative_filtering/
      cf_version=cf-v1-2026-06-25T122000Z/
        movie_cf_embeddings.parquet
        cf_model.pt
        config.json
        metrics.json
        manifest.json

  models/
    hybrid_ranker/
      model_version=hybrid-v1-2026-06-25T123000Z/
        hybrid_ranker_model.pt
        model_config.json
        training_metrics.json
        test_metrics.json
        manifest.json
```

Phase 5 implements `raw/` (optional seed copies) and `snapshots/`; later phases write to `features/`, `artifacts/`, and `models/`.

Acceptance criteria:

* MinIO is running locally with bucket `cinerankml`.
* Ratings, tag events, catalog movies, and content embeddings are exported as partitioned Parquet parts under `snapshots/snapshot_id=.../{table}/part-*.parquet`.
* Snapshot path includes `snapshot_id`.
* `manifest.json` is written last.
* Snapshot is usable only if manifest status is `complete`.

## Phase 6: Collaborative Filtering Training

Goal:
Train a small dot-product collaborative filtering model and save learned **movie behavior embeddings** for the hybrid ranker.

The CF model is **not the final recommender**. It is a **feature generator**: it learns from rating behavior and produces `movie_cf_embeddings.parquet`, which Phase 7 turns into CF features for the hybrid ranker.

Build:

* `jobs/train_cf`
* MLflow as a local Docker service (tracking URI for CF experiments)
* Shared snapshot reader utilities (load latest complete snapshot from MinIO unless `SNAPSHOT_ID` is set)

### Why CF embeddings?

Content embeddings (from movie text/metadata) answer:

> "Are these movies similar by title, genres, and overview?"

CF embeddings (from ratings) answer:

> "Do users who liked similar movies also tend to like this movie?"

That behavior signal is useful because content alone can miss patterns like: Movie A and Movie B have very different descriptions, but users who like A often like B. CF learns those patterns from `ratings_events`.

### Pipeline role

```text
ratings_events (snapshot)
  → CF model learns movie behavior embeddings
  → movie_cf_embeddings.parquet
  → hybrid feature engineering (Phase 7) builds CF features
  → hybrid ranker (Phase 8+) uses those features
```

### CF model

This is a **collaborative filtering training job**, not a feature-engineering job.

Given `user_id` and `movie_id`, the model learns embeddings so the dot product predicts the user's rating.

Training input is one row per rating event from the frozen snapshot `ratings_events` table:

```text
user_id, movie_id, rating
```

Example:

```text
user_id = 10
movie_id = 50
rating = 4.5
```

Internally:

* `user_id` → user embedding lookup → e.g. `[0.23, 0.12, 0.44, 0.51, ...]`
* `movie_id` → movie embedding lookup → e.g. `[0.14, 0.21, 0.10, 0.31, ...]`

Prediction:

```text
predicted_rating = dot(user_embedding, movie_embedding)
```

Example:

```text
predicted_rating = 3.4
actual rating = 4.5
loss = error between predicted and actual rating
```

Training updates user and movie embeddings to reduce that error.

Embedding dimension is **64** (required by the hybrid ranker feature schema).

### Train / holdout split

Use a **temporal split** on snapshot ratings ordered by rating timestamp:

* **Train:** oldest 80% of ratings
* **Holdout:** newest 20% of ratings (used for CF validation metrics; not used for gradient updates)

Do not use holdout ratings during training.

### Snapshot input selection

* Default: train on the **latest complete snapshot** (`manifest.json` with `status=complete`).
* Override: set `SNAPSHOT_ID` to a specific snapshot id.

### Training runtime

* Support **CPU and GPU** training (auto-detect or env-controlled device selection).
* Log train loss and holdout RMSE/MAE to MLflow.

### CF artifact layout

```text
s3://cinerankml/artifacts/collaborative_filtering/
  cf_version=cf-v1-2026-06-25T122000Z/
    movie_cf_embeddings.parquet
    cf_model.pt
    config.json
    metrics.json
    manifest.json
```

`movie_cf_embeddings.parquet` is the primary downstream artifact:

```text
movie_id
cf_embedding
```

Example:

```text
movie_id = 50
cf_embedding = [0.14, 0.21, 0.10, 0.31, ...]
```

`cf_model.pt` stores the trained embedding model for reproducibility and re-export.

Write data files first and `manifest.json` last with `status=complete`.

### CF features (built in Phase 7, not Phase 6)

Phase 6 saves **movie-level** embeddings only. Phase 7 (`create_features`) combines them with content embeddings and metadata to build per-example hybrid features.

For each recommendation row (user profile + candidate movie):

| Feature | Dim | How it is built |
|---------|-----|-----------------|
| `user_cf_profile` | 64 | Weighted average of `cf_embedding` for movies the user rated |
| `candidate_cf_embedding` | 64 | Lookup `cf_embedding` for the candidate movie |
| `cf_cosine_similarity` | 1 | Cosine similarity between `user_cf_profile` and `candidate_cf_embedding` |
| `user_cf_profile * candidate_cf_embedding` | 64 | Elementwise product |

Example:

```text
User rated:
  Movie 5 = 5 stars
  Movie 6 = 4 stars

Look up movie_cf_embedding[5] and movie_cf_embedding[6]
  → user_cf_profile = weighted average of those embeddings

Candidate movie 50:
  candidate_cf_embedding = movie_cf_embedding[50]

  cf_cosine_similarity = similarity(user_cf_profile, candidate_cf_embedding)
  elementwise_product = user_cf_profile * candidate_cf_embedding
```

These four CF feature groups are slots 5–8 in the hybrid ranker input vector (total `1356` dims; see Phase 7 / ML training rules).

### Inference (Phase 9+)

Do **not** call the CF model directly at online inference. Training already produced `movie_cf_embeddings.parquet`. At request time:

1. Load/fetch saved movie CF embeddings for movies the user rated → build `user_cf_profile`
2. Load/fetch `candidate_cf_embedding` for each candidate movie
3. Compute `cf_cosine_similarity` and elementwise product
4. Pass those CF features into the hybrid ranker

Acceptance criteria:

* CF model trains on the oldest 80% of snapshot ratings (temporal split).
* Holdout RMSE/MAE on the newest 20% are logged.
* Versioned `movie_cf_embeddings.parquet` and `cf_model.pt` are saved to MinIO (`s3://cinerankml/artifacts/...`).
* `config.json`, `metrics.json`, and `manifest.json` are saved.
* CF run is logged to MLflow (local Docker tracking server).

## Phase 7: Feature Generation

Goal:
Create hybrid ranker train/validation/test datasets by combining content embeddings, CF embeddings, and metadata into fixed-size feature vectors.

Build:

* `jobs/create_features`
* Shared feature utilities in `packages/common/features`

### CF features from Phase 6 artifacts

Phase 7 reads frozen `movie_cf_embeddings.parquet` (not the live CF model) and builds these behavior-based features per row:

* `user_cf_profile` `[64]`
* `candidate_cf_embedding` `[64]`
* `cf_cosine_similarity` `[1]`
* `user_cf_profile * candidate_cf_embedding` `[64]`

Together with content-based features (user/candidate content embeddings, cosine similarity, elementwise product), behavior features, and metadata features, each example becomes a **1356-dimensional** input vector for the hybrid ranker.

Acceptance criteria:

* Uses frozen snapshot, content embeddings, and CF embeddings.
* Outputs train/validation/test datasets to MinIO (`s3://cinerankml/features/...`).
* Each feature vector has input dimension `1356`.
* Dataset manifest records `snapshot_id`, `cf_version`, `content_embedding_version`, and `feature_schema_version`.

## Phase 8: Hybrid Model Training and Evaluation

Goal:
Train and evaluate the hybrid neural reranker.

Build:

* `jobs/train_hybrid_ranker`
* `jobs/evaluate_model`

Acceptance criteria:

* Hybrid model trains using feature datasets.
* Validation RMSE is used for early stopping.
* Test metrics are calculated after training.
* Fixed benchmark metrics are calculated for fair model comparison.
* Model artifacts are saved to MinIO (`s3://cinerankml/models/...`).
* Metrics and artifacts are logged to MLflow.

## Phase 9: FastAPI Inference

Goal:
Serve online recommendations.

Build:

* `apps/recommender-api`

Acceptance criteria:

* `/recommend` accepts user ratings or `user_id`.
* API retrieves candidates from OpenSearch.
* API builds feature matrix with shape `[num_candidates, 1356]` (including CF features from saved `movie_cf_embeddings`, not a live CF model forward pass).
* API scores candidates with main model.
* API returns top-K recommendations.
* Recommendation impressions are logged to Postgres.
* API metrics are exposed.

## Phase 10: MLflow and Monitoring Polish

Goal:
Make the project production-style and observable.

Build:

* MLflow model aliases: `main` and `candidate` (basic MLflow tracking starts in Phase 6).
* Prometheus scrape configs.
* Grafana dashboards.
* Basic online experiment monitoring.

Acceptance criteria:

* Main/candidate model versions are tracked.
* Prometheus collects API, worker, and job metrics.
* Grafana shows service health, Kafka health, OpenSearch sync health, and online recommendation metrics.
* Postgres stores online experiment logs.

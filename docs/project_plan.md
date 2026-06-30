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
| 6 | `jobs/prepare_cf_dataset` |
| 7 | `jobs/train_cf` |
| 8+ | `jobs/create_features`, remaining jobs, API as each phase ships |

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
    cf_dataset/
      cf_dataset_version=2026-06-25T121500Z/
        user_id_map.parquet
        movie_id_map.parquet
        train/
          part-00000.parquet
          part-00001.parquet
        validation/
          part-00000.parquet
        test/
          part-00000.parquet
        manifest.json

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
        cf_config.json
        cf_metrics.json
        manifest.json
        training_curve.png

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

### Snapshot export order (required for Phase 6)

* Export `ratings_events` **ordered by `rating_timestamp`, `id`** (tie-break on `id`).
* Add a Postgres index on **`(rating_timestamp, id)`** on `ratings_events`.
* Other tables may keep existing keyset export order (`id`).
* Time-ordered snapshot parts make DuckDB temporal split cheaper and deterministic in Phase 6.

Acceptance criteria:

* MinIO is running locally with bucket `cinerankml`.
* Ratings, tag events, catalog movies, and content embeddings are exported as partitioned Parquet parts under `snapshots/snapshot_id=.../{table}/part-*.parquet`.
* `ratings_events` parts are time-ordered by `rating_timestamp`, `id`.
* Postgres has an index on `ratings_events (rating_timestamp, id)`.
* Snapshot path includes `snapshot_id`.
* `manifest.json` is written last.
* Snapshot is usable only if manifest status is `complete`.

## Phase 6: Prepare CF Dataset

Goal:
Use DuckDB over MinIO snapshot Parquet to produce a versioned, mapped, shuffled CF training dataset for PyTorch training in Phase 7.

Build:

* `jobs/prepare_cf_dataset`
* Shared DuckDB + MinIO utilities in `packages/common/storage` (S3 Parquet reads, MinIO connection settings)
* Shared snapshot reader utilities (resolve latest complete snapshot from MinIO unless `SNAPSHOT_ID` is set)
* Env: `CF_SHUFFLE_SEED` (default `42`) for deterministic train shuffle

### Why a separate prep phase?

Raw snapshot ratings are too large to load entirely into Python memory for sorting, mapping, and shuffling. DuckDB scans Parquet parts on MinIO directly and writes a compact, training-ready dataset that Phase 7 streams batch-by-batch.

### Pipeline role

```text
ratings_events (snapshot, time-ordered)
  → DuckDB prep (Phase 6)
  → features/cf_dataset/
  → CF PyTorch training (Phase 7)
  → movie_cf_embeddings.parquet
  → hybrid feature engineering (Phase 8)
  → hybrid ranker (Phase 9+) uses those features
```

### Snapshot input selection

* Default: use the **latest complete snapshot** (`manifest.json` with `status=complete`).
* Override: set `SNAPSHOT_ID` to a specific snapshot id.

### DuckDB prep pipeline (in-job stages — not persisted)

These steps run inside `prepare_cf_dataset` using DuckDB over `read_parquet('s3://...')`. Intermediate results are **not** written to MinIO.

1. **Split** — temporal 80/10/10 on `rating_timestamp` (oldest → train, middle → validation, newest → test). Use `ROW_NUMBER() OVER (ORDER BY rating_timestamp, id)` or equivalent; leverage time-ordered snapshot export from Phase 5.
2. **Build maps** — `user_id_map` from train-split users only; `movie_id_map` from `catalog_movies` (all catalog movies get an index).
3. **Join maps** — produce mapped columns: `user_idx`, `movie_idx`, `rating`.
4. **Shuffle train once (deterministic)** — DuckDB shuffles mapped train rows **once** during prep using a fixed seed (not per training epoch). Write shuffled train parts to MinIO. The same snapshot + seed must produce the same `train/` part order and row order.

Validation and test are mapped but **not shuffled**.

### Deterministic train shuffle

Train shuffle must be **reproducible** for the same snapshot. Use a configured seed via env var:

```text
CF_SHUFFLE_SEED=42
```

Default `CF_SHUFFLE_SEED` is `42` unless overridden. Record the seed used in `manifest.json` as `shuffle_seed`.

Conceptually, prefer a deterministic ordering such as:

```sql
ORDER BY hash(user_id, movie_id, rating_timestamp, {shuffle_seed})
```

Alternatively, set DuckDB's random seed before a shuffle step so the same snapshot and seed yield identical `train/` output.

Do **not** use non-deterministic shuffle without recording a seed — Phase 7 training depends on stable prep output for reproducibility and debugging.

### Train / validation / test split

Use a **temporal 80/10/10 split** on snapshot ratings ordered by rating timestamp:

* **Train:** oldest 80% of ratings
* **Validation:** next 10% (CF early stopping and per-epoch metrics in Phase 7)
* **Test:** newest 10% (locked for hybrid final evaluation in Phase 8+; never used in CF training or validation)

Do not use validation or test ratings during CF training.

### Persisted MinIO layout (final outputs only)

Path: `s3://cinerankml/features/cf_dataset/cf_dataset_version={cf_dataset_version}/`

```text
features/cf_dataset/
  cf_dataset_version=2026-06-25T121500Z/
    user_id_map.parquet          # columns: user_id, user_idx
    movie_id_map.parquet         # columns: movie_id, movie_idx
    train/
      part-00000.parquet         # columns: user_idx, movie_idx, rating (shuffled)
      part-00001.parquet
    validation/
      part-00000.parquet         # columns: user_idx, movie_idx, rating (time-ordered)
    test/
      part-00000.parquet         # columns: user_idx, movie_idx, rating (time-ordered, locked)
    manifest.json                # written last; status=complete
```

Do **not** persist intermediate `train_raw`, `validation_raw`, `test_raw`, or unshuffled `train_mapped` objects.

### CF dataset manifest fields

The `manifest.json` written last must include these fields (row counts and `shuffle_seed` are required for lineage and reproducibility):

| Field | Description |
|-------|-------------|
| `snapshot_id` | Source snapshot this dataset was built from |
| `cf_dataset_version` | Version id for this CF dataset |
| `train_row_count` | Number of rows in `train/` |
| `validation_row_count` | Number of rows in `validation/` |
| `test_row_count` | Number of rows in `test/` (locked for hybrid eval) |
| `num_users` | Size of `user_id_map` (train-split users) |
| `num_movies` | Size of `movie_id_map` (catalog movies) |
| `train_fraction` | Temporal train split ratio (default `0.8`) |
| `validation_fraction` | Temporal validation split ratio (default `0.1`) |
| `test_fraction` | Temporal test split ratio (default `0.1`) |
| `shuffle_seed` | Seed used for deterministic train shuffle (e.g. `42` from `CF_SHUFFLE_SEED`) |
| `created_at` | UTC timestamp when prep started or finished |
| `status` | `complete` or `failed` |

Also recommended for operational traceability:

* `finished_at`, `pipeline_run_id`
* Object keys and per-part row counts for maps and `train/` / `validation/` / `test/` parts

**Especially important:** `shuffle_seed` — the train shuffle must be reproducible when re-running prep on the same snapshot with the same seed.

Acceptance criteria:

* Reads ratings and catalog Parquet from MinIO via DuckDB.
* Produces temporal 80/10/10 split consistent with this plan.
* Shuffles train rows deterministically using `CF_SHUFFLE_SEED` (default `42`) and records `shuffle_seed` in `manifest.json`.
* Manifest includes `snapshot_id`, `cf_dataset_version`, `train_row_count`, `validation_row_count`, `test_row_count`, `num_users`, `num_movies`, `train_fraction`, `validation_fraction`, `test_fraction`, `shuffle_seed`, `created_at`, and `status`.
* Writes only final objects listed above; `manifest.json` last with `status=complete`.
* Records lineage back to `snapshot_id`.
* Job metrics are exposed via Prometheus; run is tracked in `pipeline_runs` (no MLflow required in this phase).

## Phase 7: CF PyTorch Training

Goal:
Train a small dot-product collaborative filtering model on the prepared CF dataset and save learned **movie behavior embeddings** for the hybrid ranker.

The CF model is **not the final recommender**. It is a **feature generator**: it learns from rating behavior and produces `movie_cf_embeddings.parquet`, which Phase 8 turns into CF features for the hybrid ranker.

Build:

* `jobs/train_cf`
* MLflow as a local Docker service (tracking URI for CF experiments)

### Why CF embeddings?

Content embeddings (from movie text/metadata) answer:

> "Are these movies similar by title, genres, and overview?"

CF embeddings (from ratings) answer:

> "Do users who liked similar movies also tend to like this movie?"

That behavior signal is useful because content alone can miss patterns like: Movie A and Movie B have very different descriptions, but users who like A often like B. CF learns those patterns from `ratings_events`.

### Pipeline role

```text
features/cf_dataset/ (Phase 6)
  → CF model learns movie behavior embeddings
  → movie_cf_embeddings.parquet
  → hybrid feature engineering (Phase 8) builds CF features
  → hybrid ranker (Phase 9+) uses those features
```

### CF model

Given `user_idx` and `movie_idx`, the model learns embeddings so the dot product predicts the user's rating.

Training input is one row per mapped rating from `features/cf_dataset/train/`:

```text
user_idx, movie_idx, rating
```

Internally:

* `user_idx` → user embedding lookup → e.g. `[0.23, 0.12, 0.44, 0.51, ...]`
* `movie_idx` → movie embedding lookup → e.g. `[0.14, 0.21, 0.10, 0.31, ...]`

Prediction:

```text
predicted_rating = dot(user_embedding, movie_embedding)
```

Training updates user and movie embeddings to reduce that error.

Embedding dimension is **64** (required by the hybrid ranker feature schema).

### CF dataset input selection

* Default: use the **latest complete CF dataset** (`manifest.json` with `status=complete` under `features/cf_dataset/`).
* Override: set `CF_DATASET_VERSION` to a specific version id.
* Lineage: record `cf_dataset_version` and `snapshot_id` in CF artifacts and MLflow.

### Training data access (streaming)

* Do **not** materialize all train rows in a giant `TensorDataset`.
* Use **`IterableDataset` + `DataLoader`** over `train/part-*.parquet`.
* Memory holds roughly: one Parquet part + one mini-batch + model weights + optimizer state.

### Shuffle strategy

* Do **not** run `ORDER BY random()` every epoch.
* Phase 6 already shuffled train rows once when writing `train/`.
* Each training epoch: shuffle **part order** and optionally shuffle **rows within each loaded part**.

### Validation evaluation (streaming)

* Stream `validation/part-*.parquet` in batches.
* Compute running RMSE/MAE; do **not** load the entire validation split into RAM or accumulate all predictions.
* Do **not** stream `test/` during CF training (test is locked for hybrid evaluation).

### Training runtime

* Support **CPU and GPU** training (auto-detect or env-controlled device selection).
* Log training config, per-epoch metrics, final metrics, artifacts, and lineage to MLflow.

### MLflow tracking (CF training runs)

Use experiment `collaborative_filtering` (or `MLFLOW_EXPERIMENT_NAME`). Each `train_cf` run must log **tags**, **params**, **metrics**, and **artifacts** as below.

**Tags** (run search / lineage; set once at run start):

| Tag | Example |
|-----|---------|
| `cf_version` | `cf-v1-2026-06-25T122000Z` |
| `cf_dataset_version` | `2026-06-29T205141Z` |
| `snapshot_id` | `2026-06-25T120000Z` |
| `train_fraction` | `0.8` |
| `validation_fraction` | `0.1` |
| `test_fraction` | `0.1` |
| `embedding_dim` | `64` |
| `learning_rate` | `0.01` |
| `batch_size` | `4096` |
| `num_epochs` | `20` |
| `optimizer` | `Adam` |
| `loss_function` | `MSELoss` |
| `shuffle_seed` | `42` |
| `model_type` | `dot_product_cf` |

**Per-epoch metrics** (log each epoch):

| Metric | Description |
|--------|-------------|
| `train_loss` | Average training MSE loss for the epoch |
| `validation_rmse` | Validation RMSE for the epoch |
| `validation_mae` | Validation MAE for the epoch |
| `epoch_duration_seconds` | Wall-clock time for the epoch |

**Final metrics** (log once after training completes):

| Metric | Description |
|--------|-------------|
| `best_epoch` | Epoch with lowest validation RMSE |
| `best_validation_rmse` | Best validation RMSE across epochs |
| `best_validation_mae` | Validation MAE at the best epoch |
| `num_train_rows` | Train split row count (from CF dataset manifest) |
| `num_validation_rows` | Validation split row count (from CF dataset manifest) |
| `num_users` | User embedding table size |
| `num_movies` | Movie embedding table size |
| `movie_embedding_coverage` | Fraction of catalog movies with at least one train rating |
| `default_embedding_count` | Movie embeddings still near init (optional quality check) |
| `nan_embedding_count` | Movie embedding rows containing NaN (must be 0) |
| `embedding_norm_mean` | Mean L2 norm of exported movie embeddings |
| `embedding_norm_std` | Std dev of L2 norms of exported movie embeddings |

**MLflow artifacts** (log to the run artifact store):

| Artifact | Description |
|----------|-------------|
| `cf_model.pt` | Trained PyTorch checkpoint |
| `movie_cf_embeddings.parquet` | Exported movie behavior embeddings |
| `cf_config.json` | Full run config and lineage |
| `cf_metrics.json` | Final metric summary JSON |
| `manifest.json` | Copy of the MinIO commit manifest |
| `training_curve.png` | Plot of train loss and validation RMSE/MAE vs epoch |

Also write the same files to MinIO under `artifacts/collaborative_filtering/` and log them to the MLflow run artifact store (including `training_curve.png`).

### CF artifact layout (MinIO)

```text
s3://cinerankml/artifacts/collaborative_filtering/
  cf_version=cf-v1-2026-06-25T122000Z/
    movie_cf_embeddings.parquet
    cf_model.pt
    cf_config.json
    cf_metrics.json
    manifest.json
    training_curve.png
```

`movie_cf_embeddings.parquet` is the primary downstream artifact:

```text
movie_id
cf_embedding
```

`cf_model.pt` stores the trained embedding model for reproducibility and re-export.

Write data files first and `manifest.json` last with `status=complete`.

Acceptance criteria:

* Trains on prepared `features/cf_dataset/` train parts via streaming `IterableDataset`.
* Validation RMSE/MAE are computed by streaming `validation/` parts; `test/` is never used in CF training.
* Versioned `movie_cf_embeddings.parquet` and `cf_model.pt` are saved to MinIO (`s3://cinerankml/artifacts/...`).
* `cf_config.json`, `cf_metrics.json`, and `manifest.json` are saved.
* MLflow run logs all tags, per-epoch metrics, final metrics, and artifacts listed above (including `training_curve.png`).
* `cf_metrics.json` and MinIO `manifest.json` mirror final metric and lineage fields for non-MLflow consumers.

## Phase 8: Feature Generation

Goal:
Create hybrid ranker train/validation/test datasets by combining content embeddings, CF embeddings, and metadata into fixed-size feature vectors.

Build:

* `jobs/create_features`
* Shared feature utilities in `packages/common/features`
* DuckDB over MinIO snapshot Parquet for user-level aggregations (e.g. building `user_cf_profile` inputs)
* Reuse the locked `test/` split from the CF dataset manifest (`features/cf_dataset/.../test/`) for hybrid final evaluation

### CF features from Phase 7 artifacts

Phase 8 reads frozen `movie_cf_embeddings.parquet` (not the live CF model) and builds these behavior-based features per row:

* `user_cf_profile` `[64]`
* `candidate_cf_embedding` `[64]`
* `cf_cosine_similarity` `[1]`
* `user_cf_profile * candidate_cf_embedding` `[64]`

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

These four CF feature groups are slots 5–8 in the hybrid ranker input vector (total `1356` dims; see ML training rules).

Together with content-based features (user/candidate content embeddings, cosine similarity, elementwise product), behavior features, and metadata features, each example becomes a **1356-dimensional** input vector for the hybrid ranker.

Acceptance criteria:

* Uses frozen snapshot, content embeddings, and CF embeddings.
* Outputs train/validation/test datasets to MinIO (`s3://cinerankml/features/hybrid_ranker/...`).
* Each feature vector has input dimension `1356`.
* Dataset manifest records `snapshot_id`, `cf_version`, `content_embedding_version`, and `feature_schema_version`.

## Phase 9: Hybrid Model Training and Evaluation

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

## Phase 10: FastAPI Inference

Goal:
Serve online recommendations.

Build:

* `apps/recommender-api`

### Inference (Phase 10+)

Do **not** call the CF model directly at online inference. Training already produced `movie_cf_embeddings.parquet`. At request time:

1. Load/fetch saved movie CF embeddings for movies the user rated → build `user_cf_profile`
2. Load/fetch `candidate_cf_embedding` for each candidate movie
3. Compute `cf_cosine_similarity` and elementwise product
4. Pass those CF features into the hybrid ranker

Acceptance criteria:

* `/recommend` accepts user ratings or `user_id`.
* API retrieves candidates from OpenSearch.
* API builds feature matrix with shape `[num_candidates, 1356]` (including CF features from saved `movie_cf_embeddings`, not a live CF model forward pass).
* API scores candidates with main model.
* API returns top-K recommendations.
* Recommendation impressions are logged to Postgres.
* API metrics are exposed.

## Phase 11: MLflow and Monitoring Polish

Goal:
Make the project production-style and observable.

Build:

* MLflow model aliases: `main` and `candidate` (basic MLflow tracking starts in Phase 7).
* Prometheus scrape configs.
* Grafana dashboards.
* Basic online experiment monitoring.

Acceptance criteria:

* Main/candidate model versions are tracked.
* Prometheus collects API, worker, and job metrics.
* Grafana shows service health, Kafka health, OpenSearch sync health, and online recommendation metrics.
* Postgres stores online experiment logs.

# Project Plan

This project is a production-style movie recommender MLE platform.

The goal is to build the system in small vertical phases. Each phase should be implemented, tested, and committed before moving to the next phase.

## Phase 1: Docker, Postgres, and Schemas

Goal:
Set up local infrastructure and database schema.

Build:

* Docker Compose for Postgres, Prometheus and Grafana.
* Alembic setup.
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

* `docker compose up` starts core infrastructure for db and monitoring.
* Alembic migrations run successfully.
* Tables are created with indexes and timestamps.

## Phase 2: Kafka Ingestion

Goal:
Stream MovieLens ratings and tags into Postgres.

Build:

* `workers/ratings-producer`
* `workers/tags-producer`
* `workers/ratings-consumer`
* `workers/tags-consumer`
* Shared Kafka utilities in `packages/common/kafka`

Acceptance criteria:

* Producers publish valid JSON events.
* Consumers validate events.
* Consumers write to Postgres idempotently using `event_id`.
* Kafka offsets are committed only after DB transaction success.
* Consumer metrics are exposed.

## Phase 3: Catalog Seed and TMDB Enrichment

Goal:
Load MovieLens movie metadata and enrich it with TMDB data.

Build:

* `jobs/seed_catalog`
* `jobs/tmdb_enrichment`

Acceptance criteria:

* MovieLens `movies.csv` and `links.csv` load into `catalog_movies`.
* TMDB fields are added where available.
* Enriched movies are marked dirty for OpenSearch sync.

## Phase 4: OpenSearch Indexing

Goal:
Build and maintain the movie retrieval index.

Build:

* OpenSearch index mapping.
* Initial indexing flow.
* `jobs/opensearch-sync`

Acceptance criteria:

* Movies can be indexed into OpenSearch.
* Dirty movies are synced from Postgres to OpenSearch.
* OpenSearch index can be rebuilt from Postgres.
* Query latency and sync metrics are exposed.

## Phase 5: S3 Snapshots

Goal:
Export immutable training snapshots from Postgres to S3.

Build:

* `jobs/snapshot_to_s3`

Acceptance criteria:

* Ratings, catalog movies, and content embeddings are exported as Parquet.
* Snapshot path includes `snapshot_id`.
* `manifest.json` is written last.
* Snapshot is usable only if manifest status is `complete`.

## Phase 6: Collaborative Filtering Embeddings

Goal:
Train collaborative filtering movie embeddings.

Build:

* `jobs/train_cf`

Acceptance criteria:

* CF model trains on training ratings.
* Validation RMSE/MAE are logged.
* Versioned `movie_cf_embeddings.parquet` is saved to S3.
* CF metrics and manifest are saved.
* CF run is logged to MLflow.

## Phase 7: Feature Generation

Goal:
Create hybrid ranker train/validation/test datasets.

Build:

* `jobs/create_features`
* Shared feature utilities in `packages/common/features`

Acceptance criteria:

* Uses frozen snapshot, content embeddings, and CF embeddings.
* Outputs train/validation/test datasets to S3.
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
* Model artifacts are saved to S3.
* Metrics and artifacts are logged to MLflow.

## Phase 9: FastAPI Inference

Goal:
Serve online recommendations.

Build:

* `apps/recommender-api`

Acceptance criteria:

* `/recommend` accepts user ratings or `user_id`.
* API retrieves candidates from OpenSearch.
* API builds feature matrix with shape `[num_candidates, 1356]`.
* API scores candidates with main model.
* API returns top-K recommendations.
* Recommendation impressions are logged to Postgres.
* API metrics are exposed.

## Phase 10: MLflow and Monitoring Polish

Goal:
Make the project production-style and observable.

Build:

* MLflow model aliases: `main` and `candidate`.
* Prometheus scrape configs.
* Grafana dashboards.
* Basic online experiment monitoring.

Acceptance criteria:

* Main/candidate model versions are tracked.
* Prometheus collects API, worker, and job metrics.
* Grafana shows service health, Kafka health, OpenSearch sync health, and online recommendation metrics.
* Postgres stores online experiment logs.

"""Core OpenSearch sync loop for dirty catalog movies."""

from __future__ import annotations

import logging

from dataclasses import dataclass
from sqlalchemy.orm import Session
from opensearchpy import OpenSearch

from common.opensearch.mapping import (
    ensure_movies_index,
    finalize_rebuild_index,
    prepare_rebuild_index,
)

from common.db.repositories.catalog import (
    DirtyMovieRow,
    clear_dirty_movie,
    count_dirty_movies,
    fetch_dirty_movie_batch,
    mark_all_catalog_movies_dirty,
    record_dirty_sync_failure,
)

from common.db.repositories.embeddings import (
    StoredMovieEmbedding,
    ensure_embedding_version,
    get_movie_embeddings,
    upsert_movie_content_embedding,
)

from common.embeddings.client import EmbedderClient
from common.opensearch.bulk import bulk_index_movies
from common.opensearch.documents import build_movie_document
from common.db.repositories.tags import fetch_top_tags_for_movies
from common.embeddings.text import build_embedding_text, embedding_text_hash
from common.config.settings import EmbedderSettings, OpenSearchSettings, SyncSettings


logger = logging.getLogger(__name__)


@dataclass
class SyncStats:
    """Counters collected during one sync job run."""
    processed: int = 0
    failed: int = 0


def _needs_embedding(movie: DirtyMovieRow, tags: list[str], stored: StoredMovieEmbedding | None,
                        embedder_settings: EmbedderSettings) -> tuple[bool, str, str]:
    """
    Decide whether one movie needs a fresh embedding.

    Do this by:
    1. Building the embedding text.
    2. Hashing the embedding text.
    3. Checking if the stored embedding is None, has a different hash, or has a different dimension.

    ============================ Returns ============================
    A tuple of (needs_embed, canonical_text, text_hash).
    """
    # Build the embedding text.
    text = build_embedding_text(movie, tags)
    
    # Hash the embedding text.
    text_hash = embedding_text_hash(text)
    
    # Run the checks.
    if stored is None:
        return True, text, text_hash
    if stored.embedding_text_hash != text_hash:
        return True, text, text_hash
    if len(stored.embedding) != embedder_settings.embedding_dimension:
        return True, text, text_hash
    
    # Return that the movie does not need a fresh embedding.
    return False, text, text_hash


def _process_batch(session: Session, client: OpenSearch, embedder: EmbedderClient, index_name: str, batch: list[DirtyMovieRow], \
                        opensearch_settings: OpenSearchSettings, embedder_settings: EmbedderSettings, \
                            stats: SyncStats) -> None:
    """
    Sync one batch of dirty movies through embedding and OpenSearch indexing.

    Do this by:
    1. Loading tags and existing embeddings for the batch.
    2. Calling embedder-api for movies with missing or stale embeddings.
    3. Bulk indexing documents into OpenSearch.
    4. Clearing dirty rows for successful movies and recording failures.

    ============================ Arguments ============================
    session: SQLAlchemy session.
    client: OpenSearch client.
    embedder: Embedder client.
    index_name: Name of the OpenSearch index.
    batch: List of dirty movie rows to process.
    opensearch_settings: OpenSearch connection settings.
    embedder_settings: Embedding settings.
    """
    # Fetch the movie ids for each movie in the batch.
    movie_ids = [movie.movie_id for movie in batch]
    # Fetch the top tags for each movie.
    tags_by_movie = fetch_top_tags_for_movies(session, movie_ids)
    # Fetch the stored embeddings for each movie.
    stored_embeddings = get_movie_embeddings(session, movie_ids, embedder_settings.embedding_version)

    texts_to_embed: list[str] = []
    embed_targets: list[tuple[DirtyMovieRow, str, str]] = []

    # Iterate over the batch and check if each movie needs a fresh embedding.
    for movie in batch:
        tags = tags_by_movie.get(movie.movie_id, [])
        stored = stored_embeddings.get(movie.movie_id)
        needs_embed, text, text_hash = _needs_embedding(movie, tags, stored, embedder_settings)
        if needs_embed:
            texts_to_embed.append(text)
            embed_targets.append((movie, text_hash, text))

    # If there are movies that need a fresh embedding, embed them.
    if texts_to_embed:
        # Embed the texts.
        embeddings = embedder.embed_texts(texts_to_embed)
        # Iterate over the embeddings and upsert them into the embedding_versions table in postgres.
        for (movie, text_hash, _), embedding in zip(embed_targets, embeddings, strict=True):
            upsert_movie_content_embedding(
                session,
                movie.movie_id,
                embedder_settings.embedding_version,
                embedding,
                text_hash,
            )
            # Update the stored embeddings dictionary with the new embedding.
            stored_embeddings[movie.movie_id] = StoredMovieEmbedding(
                movie_id=movie.movie_id,
                embedding=embedding,
                embedding_text_hash=text_hash,
            )

    # Build the list of documents to index into OpenSearch.
    documents: list[tuple[int, dict]] = []
    # Iterate over the batch and build the documents.
    for movie in batch:
        tags = tags_by_movie.get(movie.movie_id, [])
        stored = stored_embeddings.get(movie.movie_id)
        # If we were unable to embed the movie, record the failure and continue.        
        if stored is None:
            stats.failed += 1
            record_dirty_sync_failure(session, movie.movie_id, "missing_embedding")
            continue

        # Build the document for the movie.
        body = build_movie_document(
            movie,
            tags,
            stored.embedding,
            embedder_settings.embedding_version,
        )
        # Add the document to the list of documents to index.
        documents.append((movie.movie_id, body))

    # If there are no documents to index, commit the session and return.
    if not documents:
        session.commit()
        return

    # Bulk index the documents into OpenSearch.
    success_count, errors = bulk_index_movies(
        client,
        index_name,
        documents,
    )

    # Build the set of failed movie ids. 
    # This will be used to clear the dirty movie rows for failed movies.
    failed_ids: set[int] = set()
    # Iterate over the errors and record the error type.
    for error in errors:
        item = error.get("index", {})
        movie_id_raw = item.get("_id")
        if movie_id_raw is not None:
            failed_ids.add(int(movie_id_raw))

    # Iterate over the documents and unmark the successful movies as dirty.
    for movie_id, _ in documents:
        # If the movie failed to index, record the failure and continue.
        if movie_id in failed_ids:
            stats.failed += 1
            record_dirty_sync_failure(session, movie_id, "bulk_index_failed")
            continue

        # Unmark the movie as dirty.
        clear_dirty_movie(session, movie_id)
        # Increment the processed counter.
        stats.processed += 1

    # If the bulk index was not successful, log a warning.
    if success_count < len(documents):
        # Log the warning.
        logger.warning(
            "Bulk index completed with partial failures",
            extra={
                "requested": len(documents),
                "success_count": success_count,
                "error_count": len(errors),
            },
        )

    session.commit()


def run_sync_loop(session_factory, client: OpenSearch, embedder: EmbedderClient, sync_settings: SyncSettings,
                        opensearch_settings: OpenSearchSettings, embedder_settings: EmbedderSettings) -> SyncStats:
    """
    Run the dirty-movie sync loop until the queue is empty or a limit is hit.

    Do this by:
    1. Optionally preparing a rebuild index and marking all movies dirty.
    2. Processing dirty batches until none remain.
    3. Swapping the alias when a rebuild index was used.

    ============================ Arguments ============================
    session_factory: SQLAlchemy session factory.
    client: OpenSearch client.
    embedder: HTTP client for embedder-api.
    sync_settings: Batch size, limit, and rebuild flags.
    opensearch_settings: OpenSearch connection settings.
    embedder_settings: Embedding version and model settings.

    ============================ Returns ============================
    Counters for processed and failed movies.
    """
    stats = SyncStats()
    rebuild_index_name: str | None = None

    # Create a session to bootstrap the database.
    bootstrap_session = session_factory()
    try:
        # Ensure the embedding version is stored in the database.
        ensure_embedding_version(
            bootstrap_session,
            embedder_settings.embedding_version,
            embedder_settings.embedding_model_name,
            embedder_settings.embedding_dimension,
            embedder_settings.embedding_text_template_version,
        )

        # If we are rebuilding the index, prepare the index and mark all movies dirty.
        if sync_settings.rebuild_index:
            # Prepare the index.
            rebuild_index_name = prepare_rebuild_index(client, opensearch_settings)
            # Mark all movies dirty.
            marked = mark_all_catalog_movies_dirty(bootstrap_session)
            # Commit the session.
            bootstrap_session.commit()
            logger.info(
                "Prepared rebuild index and marked catalog movies dirty",
                extra={
                    "index_name": rebuild_index_name,
                    "marked_dirty": marked,
                },
            )
        else:
            # If we are not rebuilding the index, set the rebuild index name to None.
            rebuild_index_name = None

        # Commit the session.
        bootstrap_session.commit()
    except Exception:
        bootstrap_session.rollback()
        raise
    finally:
        bootstrap_session.close()

    # If we are rebuilding the index, set the index name to the rebuild index name.
    if rebuild_index_name is not None:
        index_name = rebuild_index_name
    else:
        # If we are not rebuilding the index, ensure the movies index exists.
        index_name = ensure_movies_index(client, opensearch_settings)

    # Set the remaining limit to the sync limit.
    remaining = sync_settings.sync_limit

    # Loop until the remaining limit is 0 or the sync limit is hit.
    while True:
        # Set the batch limit to the sync batch size.
        batch_limit = sync_settings.sync_batch_size
        if remaining is not None:
            if remaining <= 0:
                break
            batch_limit = min(batch_limit, remaining)

        # Create a session to process the batch.
        session = session_factory()
        try:
            # Fetch the next batch of dirty movies.
            batch = fetch_dirty_movie_batch(session, batch_limit)
            # If there are no dirty movies, break the loop.
            if not batch:
                break

            # Process the batch (embed and index the movie batch into OpenSearch and Postgres).
            _process_batch(
                session,
                client,
                embedder,
                index_name,
                batch,
                opensearch_settings,
                embedder_settings,
                stats,
            )

        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        # Decrement the remaining limit by the number of movies processed.
        if remaining is not None:
            remaining -= len(batch)

        # If we have processed enough movies, log the progress.
        if stats.processed and stats.processed % sync_settings.sync_log_every_n == 0:
            # Create a session to count the remaining dirty movies.
            count_session = session_factory()
            try:
                # Count the remaining dirty movies.
                dirty_remaining = count_dirty_movies(count_session)

            finally:
                count_session.close()
            logger.info(
                "OpenSearch sync progress",
                extra={
                    "processed": stats.processed,
                    "failed": stats.failed,
                    "dirty_remaining": dirty_remaining,
                },
            )

    # If we are rebuilding the index, finalize the rebuild alias swap.
    if rebuild_index_name is not None:
        # Finalize the rebuild alias swap.
        finalize_rebuild_index(client, opensearch_settings, rebuild_index_name)
        logger.info(
            "Finalized rebuild alias swap",
            extra={"index_name": rebuild_index_name},
        )

    return stats

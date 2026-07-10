"""Read and write into embedding version and movie content embedding tables."""

from __future__ import annotations

from sqlalchemy import select
from dataclasses import dataclass
from datetime import UTC, datetime
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert
from common.db.models.embeddings import EmbeddingVersion, MovieContentEmbedding


@dataclass(frozen=True)
class StoredMovieEmbedding:
    """One stored content embedding row."""

    movie_id: int
    embedding: list[float]
    embedding_text_hash: str


def ensure_embedding_version(session: Session, version: str, model_name: str, dimension: int, \
                                template_version: str) -> None:
    """
    Insert the active embedding version row when it does not exist yet.

    ============================ Arguments ============================
    session: An open SQLAlchemy session inside a transaction.
    version: Embedding version label such as content-v1.
    model_name: Sentence-transformer model name.
    dimension: Vector dimension for this version.
    template_version: Text template version used to build inputs.
    """
    stmt = (
        insert(EmbeddingVersion)
        .values(
            embedding_version=version,
            model_name=model_name,
            dimension=dimension,
            text_template_version=template_version,
        )
        .on_conflict_do_nothing(index_elements=["embedding_version"])
    )
    session.execute(stmt)


def upsert_movie_content_embedding(session: Session, movie_id: int, version: str, embedding: list[float], \
                                    text_hash: str) -> None:
    """
    Insert or update one movie_content_embeddings row.

    ============================ Arguments ============================
    session: An open SQLAlchemy session inside a transaction.
    movie_id: Catalog movie id.
    version: Active embedding version label.
    embedding: Content vector returned by embedder-api.
    text_hash: Hash of the canonical embedding text.
    """
    now = datetime.now(tz=UTC)
    stmt = (
        insert(MovieContentEmbedding)
        .values(
            movie_id=movie_id,
            embedding_version=version,
            embedding=embedding,
            embedding_text_hash=text_hash,
            generated_at=now,
        )
        .on_conflict_do_update(
            index_elements=["movie_id"],
            set_={
                "embedding_version": version,
                "embedding": embedding,
                "embedding_text_hash": text_hash,
                "generated_at": now,
            },
        )
    )
    session.execute(stmt)


def get_movie_embeddings(session: Session, movie_ids: list[int], version: str) -> dict[int, StoredMovieEmbedding]:
    """
    Load stored embeddings for a batch of movies.

    ============================ Arguments ============================
    session: An open SQLAlchemy session.
    movie_ids: Movies to look up.
    version: Expected embedding version label.

    ============================ Returns ============================
    Mapping of movie_id to stored embedding metadata.
    """
    # If there are no movie ids to look up, return an empty dictionary.
    if not movie_ids:
        return {}

    # Build the SQLAlchemy select statement.
    stmt = select(MovieContentEmbedding).where(
        MovieContentEmbedding.movie_id.in_(movie_ids),
        MovieContentEmbedding.embedding_version == version,
    )

    # Execute the select statement and return the results.
    rows = session.scalars(stmt).all()
    
    # Build the mapping of movie_id to stored embedding metadata.
    return {
        row.movie_id: StoredMovieEmbedding(
            movie_id=row.movie_id,
            embedding=list(row.embedding),
            embedding_text_hash=row.embedding_text_hash,
        )
        for row in rows
    }

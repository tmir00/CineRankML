from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, Text, func
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from common.db.base import Base


class EmbeddingVersion(Base):
    __tablename__ = "embedding_versions"

    embedding_version: Mapped[str] = mapped_column(Text, primary_key=True)
    model_name: Mapped[str] = mapped_column(Text, nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    text_template_version: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class MovieContentEmbedding(Base):
    __tablename__ = "movie_content_embeddings"

    movie_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("catalog_movies.movie_id"), primary_key=True
    )
    embedding_version: Mapped[str] = mapped_column(
        Text, ForeignKey("embedding_versions.embedding_version"), nullable=False, index=True
    )
    embedding: Mapped[list[float]] = mapped_column(ARRAY(Float), nullable=False)
    embedding_text_hash: Mapped[str] = mapped_column(Text, nullable=False)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

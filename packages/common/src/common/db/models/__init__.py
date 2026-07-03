"""SQLAlchemy ORM models — import all models so Alembic sees full metadata."""

from common.db.models.pipeline import PipelineRun
from common.db.models.dead_letter import DeadLetterEvent
from common.db.models.events import RatingsEvent, TagEvent
from common.db.models.checkpoints import CsvIngestionCheckpoint
from common.db.models.embeddings import EmbeddingVersion, MovieContentEmbedding
from common.db.models.catalog import CatalogDirtyMovie, CatalogMovie, MovieTagCount
from common.db.models.recommendations import RecommendationImpression, RecommendationRating
from common.db.models.users import AppUser, UserSession

__all__ = [
    "AppUser",
    "CatalogDirtyMovie",
    "CatalogMovie",
    "CsvIngestionCheckpoint",
    "DeadLetterEvent",
    "EmbeddingVersion",
    "MovieContentEmbedding",
    "MovieTagCount",
    "PipelineRun",
    "RatingsEvent",
    "RecommendationImpression",
    "RecommendationRating",
    "TagEvent",
    "UserSession",
]

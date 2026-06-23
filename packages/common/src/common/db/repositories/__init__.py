"""Database write helpers for Kafka consumers."""

from common.db.repositories.dead_letter import insert_dead_letter_event
from common.db.repositories.events import insert_rating_event, insert_tag_event
from common.db.repositories.checkpoints import CsvCheckpoint, get_checkpoint, upsert_checkpoint
from common.db.repositories.tags import mark_movie_dirty_if_catalog_exists, upsert_movie_tag_count

__all__ = [
    "CsvCheckpoint",
    "get_checkpoint",
    "insert_dead_letter_event",
    "insert_rating_event",
    "insert_tag_event",
    "mark_movie_dirty_if_catalog_exists",
    "upsert_checkpoint",
    "upsert_movie_tag_count",
]

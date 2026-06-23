"""ORM model for CSV producer ingestion progress stored in Postgres."""

from datetime import datetime
from common.db.base import Base
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import DateTime, Integer, Text, func


class CsvIngestionCheckpoint(Base):
    """Tracks how far a CSV producer has read and published for one source file."""

    __tablename__ = "csv_ingestion_checkpoints"

    source_file: Mapped[str] = mapped_column(Text, primary_key=True)
    last_row_number: Mapped[int] = mapped_column(Integer, nullable=False)
    last_event_id: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

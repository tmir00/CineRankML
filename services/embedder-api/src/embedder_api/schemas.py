"""Request and response schemas for embedder-api."""

from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    """Request body for the /v1/embed endpoint."""

    texts: list[str] = Field(min_length=1)


class EmbedResponse(BaseModel):
    """Response body containing embedding vectors."""

    embeddings: list[list[float]]
    model_name: str
    dimension: int

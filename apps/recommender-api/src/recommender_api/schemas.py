"""Pydantic request and response schemas for recommender-api."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    """Request body for user registration."""

    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    """Request body for user login."""

    username: str
    password: str


class AuthUserResponse(BaseModel):
    """Authenticated user summary returned after login or register."""

    user_id: int
    username: str
    rating_count: int
    can_recommend: bool


class UserStatusResponse(BaseModel):
    """Public username lookup response for onboarding checks."""

    exists: bool
    user_id: int | None = None
    rating_count: int = 0
    can_recommend: bool = False


class RatingInput(BaseModel):
    """One inline rating sent with a recommend request."""

    movie_id: int
    rating: float = Field(ge=0.5, le=5.0)


class RecommendRequest(BaseModel):
    """Request body for POST /v1/recommend."""

    ratings: list[RatingInput] = Field(default_factory=list)
    top_k: int = Field(default=20, ge=1, le=100)


class RecommendationItem(BaseModel):
    """One ranked recommendation returned to the client."""

    movie_id: int
    title: str
    year: int | None
    genres: list[str]
    poster_path: str | None = None
    predicted_score: float
    rank_position: int


class RecommendResponse(BaseModel):
    """Response body for POST /v1/recommend."""

    request_id: str
    model_version: str
    recommendations: list[RecommendationItem]


class SubmitRatingRequest(BaseModel):
    """Request body for POST /v1/ratings."""

    movie_id: int
    rating: float = Field(ge=0.5, le=5.0)
    request_id: str | None = None
    model_version: str | None = None
    experiment_id: str | None = None


class SubmitRatingResponse(BaseModel):
    """Response body after publishing one rating event."""

    status: str = "queued"


class DeleteRatingResponse(BaseModel):
    """Response body after publishing one rating_deleted event."""

    status: str = "queued"


class UserRatingItem(BaseModel):
    """One active user rating returned to the frontend."""

    movie_id: int
    title: str
    year: int | None
    genres: list[str]
    poster_path: str | None = None
    rating: float
    rated_at: datetime


class UserRatingsResponse(BaseModel):
    """Response body for GET /v1/ratings."""

    ratings: list[UserRatingItem]


class MovieSearchItem(BaseModel):
    """One movie returned from title search."""

    movie_id: int
    title: str
    year: int | None
    genres: list[str]
    poster_path: str | None = None


class MovieSearchResponse(BaseModel):
    """Response body for GET /v1/movies/search."""

    movies: list[MovieSearchItem]

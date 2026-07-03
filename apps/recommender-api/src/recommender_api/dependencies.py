"""FastAPI dependencies for database sessions and auth."""

from __future__ import annotations

from collections.abc import Generator

from fastapi import HTTPException, Request, status
from sqlalchemy.orm import Session

from common.db.models.users import AppUser
from common.db.repositories.sessions import get_session as get_user_session
from common.db.repositories.users import get_user_by_id
from recommender_api.runtime import InferenceRuntime
from recommender_api.settings import RecommenderApiSettings


def get_db_session_factory(runtime: InferenceRuntime) -> Generator[Session, None, None]:
    """
    Yield one SQLAlchemy session per request.

    ============================ Arguments ============================
    runtime: Startup-loaded inference runtime with the session factory.
    """
    session = runtime.session_factory()
    try:
        yield session
    finally:
        session.close()


def get_current_user(
    request: Request,
    runtime: InferenceRuntime,
    settings: RecommenderApiSettings,
) -> AppUser:
    """
    Resolve the authenticated user from the session cookie.

    Do this by:
    1. Reading the session_id cookie from the request.
    2. Loading the active session row from Postgres.
    3. Returning the linked app user or raising HTTP 401.

    ============================ Arguments ============================
    request: Incoming FastAPI request.
    runtime: Startup-loaded inference runtime.
    settings: Recommender API settings containing the cookie name.

    ============================ Returns ============================
    The authenticated AppUser row.
    """
    session_id = request.cookies.get(settings.session_cookie_name)
    if not session_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    db = runtime.session_factory()
    try:
        user_session = get_user_session(db, session_id)
        if user_session is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session")

        user = get_user_by_id(db, user_session.user_id)
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        return user
    finally:
        db.close()

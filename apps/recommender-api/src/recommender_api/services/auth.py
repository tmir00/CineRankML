"""Password hashing and session helpers for recommender-api auth."""

from __future__ import annotations

import bcrypt
import secrets

from sqlalchemy.orm import Session
from common.db.models.users import AppUser
from datetime import UTC, datetime, timedelta
from recommender_api.settings import RecommenderApiSettings
from common.db.repositories.sessions import create_session, delete_session


def hash_password(password: str) -> str:
    """Hash a plaintext password with bcrypt."""
    password_bytes = password.encode("utf-8")
    return bcrypt.hashpw(password_bytes, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Return True when the password matches the stored bcrypt hash."""
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def generate_session_id() -> str:
    """Create a cryptographically secure session token for the cookie."""
    return secrets.token_urlsafe(32)


def create_user_session(session: Session, user: AppUser, settings: RecommenderApiSettings) -> str:
    """
    Persist one login session and return the cookie token.

    Do this by:
    1. Generating a random session id.
    2. Computing the expiry timestamp from the configured TTL (time to live).
    3. Inserting the row into user_sessions.

    ============================ Arguments ============================
    session: An open SQLAlchemy session inside a transaction.
    user: Authenticated app user.
    settings: Recommender API runtime settings.

    ============================ Returns ============================
    Session token to store in the browser cookie.
    """
    session_id = generate_session_id()
    expires_at = datetime.now(tz=UTC) + timedelta(days=settings.session_ttl_days)
    create_session(session, session_id=session_id, user_id=user.user_id, expires_at=expires_at)
    return session_id


def logout_session(session: Session, session_id: str) -> None:
    """
    Delete one session row during logout.

    ============================ Arguments ============================
    session: An open SQLAlchemy session inside a transaction.
    session_id: Cookie token to invalidate.
    """
    delete_session(session, session_id)

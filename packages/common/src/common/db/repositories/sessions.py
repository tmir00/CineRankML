"""Postgres-backed login session storage."""

from __future__ import annotations

from sqlalchemy import delete
from sqlalchemy.orm import Session
from datetime import UTC, datetime
from common.db.models.users import UserSession


def create_session(session: Session, session_id: str, user_id: int, expires_at: datetime) -> UserSession:
    """
    Store one new login session row.

    ============================ Arguments ============================
    session: An open SQLAlchemy session inside a transaction.
    session_id: Random token stored in the browser cookie.
    user_id: Authenticated app user id.
    expires_at: UTC timestamp when the session should stop working.

    ============================ Returns ============================
    The persisted UserSession row.
    """
    row = UserSession(session_id=session_id, user_id=user_id, expires_at=expires_at)
    session.add(row)
    session.flush()
    return row


def get_session(session: Session, session_id: str) -> UserSession | None:
    """
    Load one session row when it exists and has not expired.

    ============================ Arguments ============================
    session: An open SQLAlchemy session.
    session_id: Cookie token from the client.

    ============================ Returns ============================
    The active UserSession, or None when missing or expired.
    """
    # Load the session row from the database. 
    row = session.get(UserSession, session_id)
    # If the session row is not found, return None.
    if row is None:
        return None
    # If the session row has expired, return None.
    if row.expires_at <= datetime.now(tz=UTC):
        return None
    # Return the session row.
    return row


def delete_session(session: Session, session_id: str) -> None:
    """
    Remove one session row during logout.

    ============================ Arguments ============================
    session: An open SQLAlchemy session inside a transaction.
    session_id: Cookie token to invalidate.
    """
    session.execute(delete(UserSession).where(UserSession.session_id == session_id))


def delete_expired_sessions(session: Session) -> int:
    """
    Delete session rows whose expiry time has passed.

    ============================ Arguments ============================
    session: An open SQLAlchemy session inside a transaction.

    ============================ Returns ============================
    Number of deleted session rows.
    """
    now = datetime.now(tz=UTC)
    result = session.execute(delete(UserSession).where(UserSession.expires_at <= now))
    return int(result.rowcount or 0)

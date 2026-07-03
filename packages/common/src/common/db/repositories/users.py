"""Read and write app user accounts."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session
from common.db.models.users import AppUser


def normalize_username(username: str) -> str:
    """Normalize usernames to lowercase trimmed form for storage and lookup."""
    return username.strip().lower()


def create_user(session: Session, username: str, password_hash: str) -> AppUser:
    """
    Insert one new app user row.

    Do this by:
    1. Normalizing the username to lowercase.
    2. Inserting the user with the bcrypt password hash.
    3. Refreshing the row so the allocated user_id is available.

    ============================ Arguments ============================
    session: An open SQLAlchemy session inside a transaction.
    username: Unique login name chosen by the user.
    password_hash: Bcrypt hash of the user's password.

    ============================ Returns ============================
    The newly created AppUser row.
    """
    user = AppUser(username=normalize_username(username), password_hash=password_hash)
    session.add(user)
    session.flush()
    session.refresh(user)
    return user


def get_user_by_username(session: Session, username: str) -> AppUser | None:
    """
    Look up one app user by normalized username.

    ============================ Arguments ============================
    session: An open SQLAlchemy session.
    username: Login name to search for.

    ============================ Returns ============================
    The matching AppUser, or None when the username does not exist.
    """
    stmt = select(AppUser).where(AppUser.username == normalize_username(username))
    return session.scalars(stmt).first()


def get_user_by_id(session: Session, user_id: int) -> AppUser | None:
    """
    Look up one app user by primary key.

    ============================ Arguments ============================
    session: An open SQLAlchemy session.
    user_id: App user id (starts at 1_000_000).

    ============================ Returns ============================
    The matching AppUser, or None when the id does not exist.
    """
    return session.get(AppUser, user_id)

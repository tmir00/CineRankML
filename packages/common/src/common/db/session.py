from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from collections.abc import Generator
from sqlalchemy.orm import Session, sessionmaker
from common.config.settings import get_database_settings

_engine = None
_SessionLocal = None


def get_engine() -> Engine:
    """
    Get the SQLAlchemy engine for the database. 
    The engine is what knows how to connect to the database.
    We cache the engine so that we don't have to create it multiple times.
    """
    global _engine
    # If the engine is not already created, create it.
    if _engine is None:
        
        # Get the database settings.
        settings = get_database_settings()

        # Create the engine using the database settings.
        _engine = create_engine(
            settings.database_url,
            pool_size=settings.pool_size,
            max_overflow=settings.max_overflow,
        )
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """
    A session is what we use to interact with the database.
    Whenever a new session is needed, we use the session factory to create it or return the cached session.
    """
    global _SessionLocal

    # If the session factory is not already created, create it.
    if _SessionLocal is None:
        # Create the session factory using the engine and the sessionmaker class.
        # autocommit=False means changes are committed only when session.commit() is called.
        # autoflush=False means pending changes are not automatically flushed before queries.
        _SessionLocal = sessionmaker(bind=get_engine(), autocommit=False, autoflush=False)
    return _SessionLocal


def get_db_session() -> Generator[Session, None, None]:
    """
    Get a database session so that we can use it to interact with the database.
    """
    # Get the session factory.
    session_factory = get_session_factory()
    # Create a new session.
    session = session_factory()
    # Yield the session so that it can be used in the caller to interact with the database.
    try:
        yield session
    finally:
        # Close the session when the caller is done with it.
        session.close()

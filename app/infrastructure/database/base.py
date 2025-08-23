"""
Database base classes and connection management.
"""

from typing import Optional, Generator
from sqlalchemy import create_engine, Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from ...core.config import get_settings

# SQLAlchemy base class for all models
Base = declarative_base()

# Global engine and session factory
_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


class DatabaseManager:
    """Database connection manager."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self._engine = None
        self._session_factory = None
    
    def get_engine(self) -> Engine:
        """Get database engine."""
        if self._engine is None:
            self._engine = create_engine(
                self.database_url,
                # Add common engine options
                pool_pre_ping=True,
                pool_recycle=300,
            )
        return self._engine
    
    def get_session_factory(self) -> sessionmaker:
        """Get session factory."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.get_engine()
            )
        return self._session_factory
    
    def create_all_tables(self):
        """Create all tables in the database."""
        Base.metadata.create_all(bind=self.get_engine())
    
    def drop_all_tables(self):
        """Drop all tables in the database."""
        Base.metadata.drop_all(bind=self.get_engine())


def get_database_url() -> str:
    """Get database URL from settings."""
    settings = get_settings()
    return settings.database.url


def create_database_engine() -> Engine:
    """Create database engine from settings."""
    settings = get_settings()
    return create_engine(
        settings.database.url,
        echo=settings.database.echo,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        pool_timeout=settings.database.pool_timeout,
        pool_recycle=settings.database.pool_recycle,
        pool_pre_ping=True,
    )


def get_database_session() -> Generator[Session, None, None]:
    """
    Dependency for getting database sessions.
    This is typically used with FastAPI's Depends().
    """
    global _engine, _SessionLocal
    
    if _engine is None:
        _engine = create_database_engine()
    
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=_engine
        )
    
    session = _SessionLocal()
    try:
        yield session
    finally:
        session.close()


# Alias for compatibility
get_db = get_database_session
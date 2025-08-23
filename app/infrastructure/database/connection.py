"""
Database connection management and session handling.
"""

import logging
from typing import Generator, AsyncGenerator
from contextlib import contextmanager, asynccontextmanager

from sqlalchemy import create_engine, pool
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from ...core.config import DatabaseSettings
from ...core.exceptions import DatabaseError

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and sessions."""
    
    def __init__(self, settings: DatabaseSettings):
        self.settings = settings
        self._engine = None
        self._async_engine = None
        self._session_factory = None
        self._async_session_factory = None
    
    def get_engine(self):
        """Get synchronous database engine."""
        if self._engine is None:
            logger.info("Creating synchronous database engine")
            self._engine = create_engine(
                self.settings.url,
                echo=self.settings.echo,
                pool_size=self.settings.pool_size,
                max_overflow=self.settings.max_overflow,
                pool_timeout=self.settings.pool_timeout,
                pool_recycle=self.settings.pool_recycle,
                poolclass=pool.NullPool  # For compatibility with existing code
            )
            logger.info("Synchronous database engine created successfully")
        
        return self._engine
    
    def get_async_engine(self):
        """Get asynchronous database engine."""
        if self._async_engine is None:
            logger.info("Creating asynchronous database engine")
            # Convert sync URL to async URL
            async_url = self.settings.url.replace("mssql+pyodbc://", "mssql+aioodbc://")
            
            self._async_engine = create_async_engine(
                async_url,
                echo=self.settings.echo,
                pool_size=self.settings.pool_size,
                max_overflow=self.settings.max_overflow,
                pool_timeout=self.settings.pool_timeout,
                pool_recycle=self.settings.pool_recycle,
            )
            logger.info("Asynchronous database engine created successfully")
        
        return self._async_engine
    
    def get_session_factory(self):
        """Get synchronous session factory."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.get_engine(),
                autocommit=False,
                autoflush=False
            )
        
        return self._session_factory
    
    def get_async_session_factory(self):
        """Get asynchronous session factory."""
        if self._async_session_factory is None:
            self._async_session_factory = async_sessionmaker(
                bind=self.get_async_engine(),
                class_=AsyncSession,
                autocommit=False,
                autoflush=False
            )
        
        return self._async_session_factory
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get a database session with proper error handling and cleanup.
        
        Yields:
            Database session
            
        Raises:
            DatabaseError: If session creation or operation fails
        """
        session = self.get_session_factory()()
        try:
            logger.debug("Database session created")
            yield session
        except SQLAlchemyError as e:
            logger.error(f"Database error during session: {e}", exc_info=True)
            session.rollback()
            raise DatabaseError(f"Database operation failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during database session: {e}", exc_info=True)
            session.rollback()
            raise DatabaseError(f"Unexpected database error: {e}")
        finally:
            session.close()
            logger.debug("Database session closed")
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get an async database session with proper error handling and cleanup.
        
        Yields:
            Async database session
            
        Raises:
            DatabaseError: If session creation or operation fails
        """
        session = self.get_async_session_factory()()
        try:
            logger.debug("Async database session created")
            yield session
        except SQLAlchemyError as e:
            logger.error(f"Database error during async session: {e}", exc_info=True)
            await session.rollback()
            raise DatabaseError(f"Database operation failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during async database session: {e}", exc_info=True)
            await session.rollback()
            raise DatabaseError(f"Unexpected database error: {e}")
        finally:
            await session.close()
            logger.debug("Async database session closed")
    
    def close_connections(self) -> None:
        """Close all database connections."""
        logger.info("Closing database connections")
        
        if self._engine:
            self._engine.dispose()
            self._engine = None
        
        if self._async_engine:
            # Note: async engine disposal should be awaited, but we can't do that here
            # This is mainly for cleanup during shutdown
            self._async_engine = None
        
        logger.info("Database connections closed")


# Global database manager instance
_db_manager: DatabaseManager = None


def initialize_database(settings: DatabaseSettings) -> DatabaseManager:
    """
    Initialize the global database manager.
    
    Args:
        settings: Database configuration settings
        
    Returns:
        Database manager instance
    """
    global _db_manager
    _db_manager = DatabaseManager(settings)
    logger.info("Database manager initialized")
    return _db_manager


def get_database_manager() -> DatabaseManager:
    """
    Get the global database manager instance.
    
    Returns:
        Database manager instance
        
    Raises:
        DatabaseError: If database manager is not initialized
    """
    if _db_manager is None:
        raise DatabaseError("Database manager not initialized. Call initialize_database() first.")
    
    return _db_manager


# Dependency functions for FastAPI
def get_db() -> Generator[Session, None, None]:
    """
    Database dependency for FastAPI endpoints.
    Provides a database session for each request.
    
    Yields:
        Database session
    """
    db_manager = get_database_manager()
    with db_manager.get_session() as session:
        yield session


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Async database dependency for FastAPI endpoints.
    Provides an async database session for each request.
    
    Yields:
        Async database session
    """
    db_manager = get_database_manager()
    async with db_manager.get_async_session() as session:
        yield session


# Legacy compatibility functions
@contextmanager
def database_session() -> Generator[Session, None, None]:
    """
    Legacy function for compatibility with existing code.
    
    Yields:
        Database session
    """
    db_manager = get_database_manager()
    with db_manager.get_session() as session:
        yield session
# database.py
import logging
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError

from .config import get_config

config = get_config()

logging.info("Initializing database connection...")
logging.debug(f"Database URL: {config.DATABASE_URL.split('@')[0]}@[HIDDEN]")

try:
    engine = create_engine(
        config.DATABASE_URL,
        poolclass=QueuePool,
        pool_size=config.DB_POOL_SIZE,
        max_overflow=config.DB_MAX_OVERFLOW,
        pool_pre_ping=True,  # Validate connections before use
        echo=config.DB_ECHO,
        connect_args={
            "timeout": 30,  # Connection timeout in seconds
        }
    )
    logging.info("Database engine created successfully")
    logging.info(f"Pool configuration: size={config.DB_POOL_SIZE}, max_overflow={config.DB_MAX_OVERFLOW}")
except Exception as e:
    logging.error(f"Failed to create database engine: {str(e)}", exc_info=True)
    raise

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """
    Database dependency that provides a database session for FastAPI endpoints.
    Includes proper error handling and logging.
    """
    db = SessionLocal()
    try:
        logging.debug("Database session created")
        yield db
    except SQLAlchemyError as e:
        logging.error(f"Database error during session: {str(e)}", exc_info=True)
        db.rollback()
        raise
    except Exception as e:
        logging.error(f"Unexpected error during database session: {str(e)}", exc_info=True)
        db.rollback()
        raise
    finally:
        db.close()
        logging.debug("Database session closed")


@contextmanager
def database_session():
    """
    Context manager for database sessions with proper error handling and cleanup.
    Used for Celery tasks and other non-FastAPI contexts.
    """
    db = SessionLocal()
    try:
        logging.debug("Database session created")
        yield db
    except SQLAlchemyError as e:
        logging.error(f"Database error occurred: {e}", exc_info=True)
        db.rollback()
        raise
    except Exception as e:
        logging.error(f"Unexpected error during database operation: {e}", exc_info=True)
        db.rollback()
        raise
    finally:
        db.close()
        logging.debug("Database session closed")

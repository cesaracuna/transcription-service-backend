"""
Database utilities for diagnostics and maintenance.
"""

import logging
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from .database import SessionLocal
from .models import JobStatus

logger = logging.getLogger(__name__)

def check_enum_values():
    """
    Check current enum values in the database for diagnostic purposes.
    """
    logger.info("Checking enum values in database...")
    
    db = SessionLocal()
    try:
        # Check current status values in database
        check_query = text("""
            SELECT DISTINCT status, COUNT(*) as count
            FROM jobs 
            WHERE status IS NOT NULL
            GROUP BY status
            ORDER BY status
        """)
        
        result = db.execute(check_query)
        status_counts = [(row[0], row[1]) for row in result.fetchall()]
        logger.info(f"Status values and counts in database: {status_counts}")
        
        # Verify enum values match
        python_enum_values = [status.value for status in JobStatus]
        logger.info(f"Python enum values: {python_enum_values}")
        
        return status_counts
        
    except SQLAlchemyError as e:
        logger.error(f"Error checking enum values: {e}")
        return []
    finally:
        db.close()

def validate_database_consistency():
    """
    Validate that database enum values are consistent with Python enum.
    """
    logger.info("Validating database consistency...")
    
    try:
        status_counts = check_enum_values()
        python_enum_values = set(status.value for status in JobStatus)
        database_values = set(status for status, count in status_counts)
        
        # Check for any values in database that aren't in Python enum
        unexpected_values = database_values - python_enum_values
        if unexpected_values:
            logger.warning(f"Found unexpected status values in database: {unexpected_values}")
            return False
        
        logger.info("Database consistency validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Error during consistency validation: {e}")
        return False

if __name__ == "__main__":
    # Run diagnostics
    status_counts = check_enum_values()
    print(f"Status counts in database: {status_counts}")
    
    is_consistent = validate_database_consistency()
    print(f"Database consistency: {'VALID' if is_consistent else 'INVALID'}")
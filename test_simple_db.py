#!/usr/bin/env python3
"""
Simple database connection test using the same setup as other working scripts.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.database import database_session
from src.models import Job
from src.config import get_config

def test_database():
    """Test database connection using existing database session."""

    config = get_config()

    print("=== DATABASE CONNECTION TEST ===")
    print(f"Database URL: {config.DATABASE_URL}")

    try:
        with database_session() as db:
            # Try to query jobs table
            job_count = db.query(Job).count()
            print("SUCCESS: Database connection working!")
            print(f"   Found {job_count} jobs in database")
            return True

    except Exception as e:
        print(f"FAILED: {e}")
        return False

if __name__ == "__main__":
    test_database()
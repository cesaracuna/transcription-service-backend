#!/usr/bin/env python3
"""
Test database connectivity with different connection strings.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fix Windows console encoding
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

try:
    from sqlalchemy import create_engine, text
    from src.config import get_config
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the virtual environment")
    sys.exit(1)

def test_connection_string(connection_string, description):
    """Test a specific connection string."""
    print(f"\n=== Testing {description} ===")
    print(f"Connection string: {connection_string}")

    try:
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 as test"))
            print(f"✅ SUCCESS: {result.fetchone()}")
            return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def main():
    """Test various connection strings."""
    config = get_config()

    print("=== DATABASE CONNECTION TESTS ===")

    # Original connection string
    original = config.DATABASE_URL
    test_connection_string(original, "Original config")

    # Try without port
    without_port = r"mssql+pyodbc://transapp:transapp@(localdb)\MSSQLLocalDB/Pluralia?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes"
    test_connection_string(without_port, "Without port")

    # Try with different authentication
    windows_auth = r"mssql+pyodbc://@(localdb)\MSSQLLocalDB/Pluralia?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes&Trusted_Connection=yes"
    test_connection_string(windows_auth, "Windows Authentication")

    # Try simplified LocalDB connection
    simple_localdb = r"mssql+pyodbc://@(localdb)\MSSQLLocalDB/Pluralia?driver=ODBC+Driver+18+for+SQL+Server&Trusted_Connection=yes&TrustServerCertificate=yes&Encrypt=no"
    test_connection_string(simple_localdb, "Simple LocalDB")

if __name__ == "__main__":
    main()
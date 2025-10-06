#!/usr/bin/env python3
"""
Startup script for the Pluralia Transcription Backend.
This script provides an easy way to start the application with the new structure.
"""

import sys
import os
import uvicorn

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.config import get_config

def main():
    """Main function to start the application."""
    config = get_config()
    
    print(f"Starting {config.APP_NAME} v{config.APP_VERSION}")
    print(f"Server will run on {config.HOST}:{config.PORT}")
    print(f"Debug mode: {config.DEBUG}")
    
    uvicorn.run(
        "app.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        workers=1 if config.DEBUG else config.WORKERS,
        log_level=config.LOG_LEVEL.lower()
    )

if __name__ == "__main__":
    main()
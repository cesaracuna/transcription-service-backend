#!/usr/bin/env python3
"""
Migration script for transitioning from old structure to new production structure.
This script helps migrate from the src/ based structure to the new app/ structure.
"""

import os
import shutil
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories for the new structure."""
    directories = [
        "storage/audio",
        "storage/logs", 
        "storage/temp",
        "logs",  # Keep logs in root for compatibility
        "tests/unit",
        "tests/integration",
        "tests/e2e"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def update_environment_variables():
    """Create .env.example with updated paths."""
    env_content = """# Application Settings
APP_NAME=Pluralia Transcription Backend
APP_VERSION=2.0.0
DEBUG=false
LOG_LEVEL=INFO

# Server Settings
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Security Settings
MAX_FILE_SIZE_MB=500
MAX_FILENAME_LENGTH=255

# Database Settings
DATABASE_URL=mssql+pyodbc://transcribeapp:transcribeapp@localhost,51392/Transcribe?driver=ODBC+Driver+17+for+SQL+Server&TrustServerCertificate=yes
DB_ECHO=false
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20

# Redis/Celery Settings
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# AI Model Settings
WHISPER_MODEL_PATH=openai/whisper-large-v3
DIARIZATION_MODEL_PATH=pyannote/speaker-diarization-3.1
HF_TOKEN=your_hugging_face_token_here
DEVICE=cuda:0

# File Storage Settings (updated paths)
AUDIO_STORAGE_PATH=./storage/audio
LOG_STORAGE_PATH=./storage/logs
TEMP_STORAGE_PATH=./storage/temp

# Audio Processing Settings
SAMPLE_RATE=16000
MIN_SEGMENT_LENGTH=160
MAX_SEGMENT_DURATION=30.0
MIN_CHUNK_DURATION=5.0
CHUNK_OVERLAP=1.0

# Whisper Generation Settings
WHISPER_BEAM_SIZE=1
WHISPER_BATCH_SIZE=16
WHISPER_TEMPERATURE=0.0
WHISPER_NO_REPEAT_NGRAM_SIZE=3

# Diarization Settings
DIARIZATION_ONSET=0.005
DIARIZATION_OFFSET=0.005
DIARIZATION_MIN_DURATION_ON=0.015
DIARIZATION_MIN_DURATION_OFF=0.04

# Hallucination Detection Settings
SILENCE_RMS_THRESHOLD=0.005
REPETITION_COMPRESSION_THRESHOLD=0.45
LONE_UTTERANCE_MAX_SEGMENTS=2
LONE_UTTERANCE_MAX_DURATION=3.0

# CORS Origins (comma-separated)
CORS_ORIGINS=http://localhost,http://localhost:8080,https://localhost:7003,http://localhost:5149,http://servicesdev.oas.org,https://servicesdev.oas.org
"""
    
    with open(".env.example", "w", encoding="utf-8") as f:
        f.write(env_content)
    logger.info("Created .env.example with updated configuration")

def create_pyproject_toml():
    """Create pyproject.toml for modern Python project configuration."""
    pyproject_content = """[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "pluralia-backend"
version = "2.0.0"
description = "Advanced audio transcription and diarization API with security features"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "fastapi[all]>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "sqlalchemy>=2.0.0",
    "celery>=5.3.0",
    "transformers>=4.36.0",
    "pyannote.audio==3.1.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
    "mypy>=1.6.0",
]

[tool.black]
line-length = 120
target-version = ['py310']

[tool.isort]
profile = "black"
line_length = 120

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
"""
    
    with open("pyproject.toml", "w", encoding="utf-8") as f:
        f.write(pyproject_content)
    logger.info("Created pyproject.toml")

def update_gitignore():
    """Update .gitignore for new structure."""
    gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# celery beat schedule file
celerybeat-schedule

# SageMath parsed files
*.sage.py

# Environments
.env
.env.*
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# Application specific
/storage/
/logs/
/audio_files/
*.wav
*.mp3
*.flac
*.aac
*.ogg
*.m4a
*.wma

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Old structure (keep for transition)
/src.rar
/migrate_to_production.py
"""
    
    with open(".gitignore", "w", encoding="utf-8") as f:
        f.write(gitignore_content)
    logger.info("Updated .gitignore")

def main():
    """Main migration function."""
    logger.info("Starting migration to production structure...")
    
    # Create directories
    create_directories()
    
    # Update configuration files
    update_environment_variables()
    create_pyproject_toml()
    update_gitignore()
    
    logger.info("Migration completed successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Copy your .env.example to .env and update with your actual values")
    logger.info("2. Install dependencies: pip install -r requirements/production.txt")
    logger.info("3. Start the application: python start.py")
    logger.info("4. Test the new structure: python -m pytest tests/ (once tests are created)")
    logger.info("\nThe old src/ folder is preserved for reference.")

if __name__ == "__main__":
    main()
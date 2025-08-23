"""
Database infrastructure layer.

This package contains all database-related infrastructure:
- SQLAlchemy ORM models and table definitions
- Database connection management and session handling
- Repository implementations for domain interfaces
- Database migrations and schema management
- Query optimization and database utilities
"""

from .base import (
    Base,
    DatabaseManager,
    get_database_session,
    create_database_engine,
    get_database_url
)

from .models import (
    UserModel,
    TranscriptionJobModel,
    TranscriptionSegmentModel,
    DiarizationSegmentModel,
    HallucinationModel,
    SpeakerModel
)

from .repositories import (
    SQLUserRepository,
    SQLTranscriptionJobRepository,
    SQLTranscriptionSegmentRepository,
    SQLDiarizationSegmentRepository,
    SQLHallucinationRepository,
    SQLSpeakerRepository
)

__all__ = [
    # Base database components
    "Base",
    "DatabaseManager",
    "get_database_session",
    "create_database_engine", 
    "get_database_url",
    
    # ORM models
    "UserModel",
    "TranscriptionJobModel",
    "TranscriptionSegmentModel",
    "DiarizationSegmentModel",
    "HallucinationModel",
    "SpeakerModel",
    
    # Repository implementations
    "SQLUserRepository",
    "SQLTranscriptionJobRepository",
    "SQLTranscriptionSegmentRepository",
    "SQLDiarizationSegmentRepository",
    "SQLHallucinationRepository",
    "SQLSpeakerRepository"
]
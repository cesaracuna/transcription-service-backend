"""
Database infrastructure layer.

This package contains all database-related infrastructure:
- SQLAlchemy ORM models and table definitions
- Database connection management and session handling
- Repository implementations for domain interfaces
- Database migrations and schema management
- Query optimization and database utilities
"""

# Import base components
try:
    from .base import (
        Base,
        DatabaseManager,
        get_database_session,
        create_database_engine,
        get_database_url
    )
except ImportError:
    Base = None
    DatabaseManager = None
    get_database_session = None
    create_database_engine = None
    get_database_url = None

# Import connection components
try:
    from .connection import (
        get_db,
        get_async_db,
        initialize_database,
        get_database_manager,
        database_session
    )
except ImportError:
    get_db = None
    get_async_db = None
    initialize_database = None
    get_database_manager = None
    database_session = None

# Import models
try:
    from .models import (
        UserModel,
        TranscriptionJobModel,
        TranscriptionSegmentModel,
        DiarizationSegmentModel,
        HallucinationModel,
        SpeakerModel
    )
except ImportError:
    UserModel = None
    TranscriptionJobModel = None
    TranscriptionSegmentModel = None
    DiarizationSegmentModel = None
    HallucinationModel = None
    SpeakerModel = None

# Import repositories
try:
    from .repositories import (
        SQLUserRepository,
        SQLTranscriptionJobRepository,
        SQLTranscriptionSegmentRepository,
        SQLDiarizationSegmentRepository,
        SQLHallucinationRepository,
        SQLSpeakerRepository
    )
except ImportError:
    SQLUserRepository = None
    SQLTranscriptionJobRepository = None
    SQLTranscriptionSegmentRepository = None
    SQLDiarizationSegmentRepository = None
    SQLHallucinationRepository = None
    SQLSpeakerRepository = None

__all__ = [
    # Base database components
    "Base",
    "DatabaseManager",
    "get_database_session",
    "create_database_engine", 
    "get_database_url",
    
    # Connection components
    "get_db",
    "get_async_db",
    "initialize_database",
    "get_database_manager",
    "database_session",
    
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
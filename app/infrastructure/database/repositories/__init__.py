"""
Repository implementations for data access.

This package contains concrete implementations of repository interfaces
defined in the domain layer. These implementations handle:
- CRUD operations using SQLAlchemy ORM
- Query optimization and database-specific operations
- Data mapping between domain models and ORM models
- Transaction management and error handling
- Pagination and filtering
"""

try:
    from .base import SQLAlchemyRepository, SyncSQLAlchemyRepository
except ImportError:
    SQLAlchemyRepository = None
    SyncSQLAlchemyRepository = None

try:
    from .users import SQLUserRepository
except ImportError:
    SQLUserRepository = None

try:
    from .jobs import SQLTranscriptionJobRepository
except ImportError:
    SQLTranscriptionJobRepository = None

# Set defaults for missing repositories  
SQLTranscriptionSegmentRepository = None
SQLDiarizationSegmentRepository = None
SQLHallucinationRepository = None
SQLSpeakerRepository = None

# Set defaults for missing dependency functions
get_user_repository = None
get_transcription_job_repository = None
get_transcription_segment_repository = None
get_diarization_segment_repository = None
get_hallucination_repository = None
get_speaker_repository = None

__all__ = [
    # Base repository
    "SQLAlchemyRepository",
    "SyncSQLAlchemyRepository",
    
    # Repository implementations  
    "SQLUserRepository",
    "SQLTranscriptionJobRepository",
    "SQLTranscriptionSegmentRepository", 
    "SQLDiarizationSegmentRepository",
    "SQLHallucinationRepository",
    "SQLSpeakerRepository",
    
    # Dependency injection
    "get_user_repository",
    "get_transcription_job_repository",
    "get_transcription_segment_repository",
    "get_diarization_segment_repository", 
    "get_hallucination_repository",
    "get_speaker_repository"
]
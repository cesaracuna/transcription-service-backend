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

from .base import SQLBaseRepository
from .user_repository import SQLUserRepository
from .transcription_job_repository import SQLTranscriptionJobRepository
from .transcription_segment_repository import SQLTranscriptionSegmentRepository
from .diarization_segment_repository import SQLDiarizationSegmentRepository
from .hallucination_repository import SQLHallucinationRepository
from .speaker_repository import SQLSpeakerRepository

# Dependency injection functions
from .dependencies import (
    get_user_repository,
    get_transcription_job_repository,
    get_transcription_segment_repository,
    get_diarization_segment_repository,
    get_hallucination_repository,
    get_speaker_repository
)

__all__ = [
    # Base repository
    "SQLBaseRepository",
    
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
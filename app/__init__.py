"""
Transcription Service Backend

AI-powered audio transcription and speaker diarization service.
Built with FastAPI, Celery, PostgreSQL, and Redis.
"""

__version__ = "1.0.0"
__author__ = "Cesar Acuna"
__email__ = "CAcuna@oas.org"

# Application metadata
__title__ = "Transcription Service Backend"
__description__ = "AI-powered audio transcription and speaker diarization service"
__license__ = "MIT"
__copyright__ = "2025, Organization of American States"

# Export main application components
from .core.config import get_settings
from .core.exceptions import (
    ValidationError,
    ResourceNotFoundError,
    TranscriptionServiceError
)

# Import domain models and API components if they exist
try:
    from .domain.shared.enums import JobStatus, SegmentType, ConfidenceLevel
except ImportError:
    # Domain enums not fully implemented yet
    JobStatus = SegmentType = ConfidenceLevel = None

try:
    from .domain.transcription.models import User, TranscriptionJob, TranscriptionSegment
except ImportError:
    # Domain models not implemented yet
    User = TranscriptionJob = TranscriptionSegment = None

try:
    from .api.v1 import api_router as api_v1_router
except ImportError:
    # API router not implemented yet
    api_v1_router = None

__all__ = [
    # Metadata
    "__version__",
    "__title__",
    "__description__",
    "__author__",
    "__email__",
    "__license__",
    "__copyright__",
    
    # Core components
    "get_settings",
    
    # Exceptions
    "ValidationError",
    "ResourceNotFoundError", 
    "TranscriptionServiceError",
    
    # Domain models
    "User",
    "TranscriptionJob",
    "TranscriptionSegment",
    
    # Enums
    "JobStatus",
    "SegmentType",
    "ConfidenceLevel",
    
    # API
    "api_v1_router"
]
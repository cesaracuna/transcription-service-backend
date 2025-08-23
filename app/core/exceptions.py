"""
Custom exception classes for the transcription service.
Provides specific error types for different failure scenarios.
"""

from typing import Optional, Dict, Any


class TranscriptionServiceError(Exception):
    """Base exception for all transcription service errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ConfigurationError(TranscriptionServiceError):
    """Raised when there's a configuration error."""
    pass


class DatabaseError(TranscriptionServiceError):
    """Raised when there's a database-related error."""
    pass


class AudioProcessingError(TranscriptionServiceError):
    """Raised when audio processing fails."""
    pass


class ModelLoadingError(TranscriptionServiceError):
    """Raised when AI model loading fails."""
    pass


class TranscriptionError(TranscriptionServiceError):
    """Raised when transcription process fails."""
    pass


class DiarizationError(TranscriptionServiceError):
    """Raised when speaker diarization fails."""
    pass


class HallucinationDetectionError(TranscriptionServiceError):
    """Raised when hallucination detection fails."""
    pass


class StorageError(TranscriptionServiceError):
    """Raised when file storage operations fail."""
    pass


class ValidationError(TranscriptionServiceError):
    """Raised when input validation fails."""
    pass


class AuthenticationError(TranscriptionServiceError):
    """Raised when authentication fails."""
    pass


class AuthorizationError(TranscriptionServiceError):
    """Raised when authorization fails."""
    pass


class JobNotFoundError(TranscriptionServiceError):
    """Raised when a job is not found."""
    pass


class UserNotFoundError(TranscriptionServiceError):
    """Raised when a user is not found."""
    pass


class ResourceNotFoundError(TranscriptionServiceError):
    """Raised when a requested resource is not found."""
    pass


class RateLimitError(TranscriptionServiceError):
    """Raised when rate limiting is triggered."""
    pass


class ServiceUnavailableError(TranscriptionServiceError):
    """Raised when a service is temporarily unavailable."""
    pass
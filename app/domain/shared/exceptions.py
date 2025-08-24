"""
Domain-specific exception classes.

These exceptions represent business rule violations and domain-specific errors
that occur within the domain layer of the application.
"""

from typing import Optional, Dict, Any, List


class DomainException(Exception):
    """Base exception for all domain-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class DomainValidationError(DomainException):
    """
    Raised when domain validation rules are violated.
    
    This represents business rule violations, invariant failures,
    and other domain-specific validation errors.
    """
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        validation_errors: Optional[List[str]] = None,
        **kwargs
    ):
        self.field = field
        self.validation_errors = validation_errors or []
        super().__init__(message, **kwargs)


class BusinessRuleViolationError(DomainException):
    """
    Raised when business rules are violated.
    
    This represents violations of complex business logic that
    cannot be expressed through simple validation.
    """
    
    def __init__(
        self,
        message: str,
        rule_name: Optional[str] = None,
        **kwargs
    ):
        self.rule_name = rule_name
        super().__init__(message, **kwargs)


class ResourceNotFoundError(DomainException):
    """
    Raised when a domain resource cannot be found.
    
    This is used for aggregate roots and domain entities
    that are expected to exist but cannot be located.
    """
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        **kwargs
    ):
        self.resource_type = resource_type
        self.resource_id = resource_id
        super().__init__(message, **kwargs)


class ConcurrencyConflictError(DomainException):
    """
    Raised when concurrent operations conflict.
    
    This represents optimistic concurrency control failures
    and other concurrency-related domain issues.
    """
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        expected_version: Optional[int] = None,
        actual_version: Optional[int] = None,
        **kwargs
    ):
        self.resource_type = resource_type
        self.expected_version = expected_version
        self.actual_version = actual_version
        super().__init__(message, **kwargs)


class DomainStateError(DomainException):
    """
    Raised when domain objects are in an invalid state.
    
    This represents state transitions that violate business rules
    or attempts to perform operations on objects in invalid states.
    """
    
    def __init__(
        self,
        message: str,
        current_state: Optional[str] = None,
        expected_state: Optional[str] = None,
        **kwargs
    ):
        self.current_state = current_state
        self.expected_state = expected_state
        super().__init__(message, **kwargs)


class InvariantViolationError(DomainException):
    """
    Raised when domain invariants are violated.
    
    This represents violations of fundamental business rules
    that must always hold true for domain objects.
    """
    
    def __init__(
        self,
        message: str,
        invariant_name: Optional[str] = None,
        **kwargs
    ):
        self.invariant_name = invariant_name
        super().__init__(message, **kwargs)


# Transcription-specific domain exceptions
class TranscriptionJobError(DomainException):
    """Base exception for transcription job related errors."""
    pass


class InvalidJobStateError(DomainStateError):
    """Raised when a job is in an invalid state for the requested operation."""
    pass


class AudioFileError(DomainValidationError):
    """Raised when audio file validation fails."""
    pass


class SpeakerDiarizationError(DomainException):
    """Raised when speaker diarization fails."""
    pass


class HallucinationDetectionError(DomainException):
    """Raised when hallucination detection encounters errors."""
    pass


class UserError(DomainException):
    """Base exception for user-related domain errors."""
    pass


class UserValidationError(DomainValidationError):
    """Raised when user validation fails."""
    pass
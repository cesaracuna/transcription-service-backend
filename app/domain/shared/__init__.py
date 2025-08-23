"""
Shared domain logic and utilities.

This package contains domain components that are shared across different
bounded contexts within the transcription service:
- Common enumerations and constants
- Value objects used by multiple aggregates
- Domain exceptions and error handling
- Domain events and event handling
- Base classes and interfaces
- Domain utilities and helpers
"""

# Import enumerations
from .enums import (
    JobStatus,
    SegmentType,
    ConfidenceLevel,
    DeviceType,
    ModelType,
    AudioFormat,
    LanguageCode
)

# Import value objects
from .value_objects import (
    AudioMetadata,
    TranscriptionResult,
    ModelConfiguration,
    TimeRange,
    SpeakerInfo,
    ProcessingStats
)

# Import exceptions
from .exceptions import (
    DomainException,
    DomainValidationError,
    BusinessRuleViolationError,
    ResourceNotFoundError,
    InvalidOperationError,
    ConcurrencyError
)

# Import domain events
from .events import (
    DomainEvent,
    DomainEventHandler,
    EventBus,
    JobCreatedEvent,
    JobStatusChangedEvent,
    JobCompletedEvent,
    JobFailedEvent,
    UserRegisteredEvent,
    UserUpdatedEvent
)

# Import base classes
from .base import (
    Entity,
    ValueObject,
    AggregateRoot,
    DomainService,
    Repository
)

# Import specifications
from .specifications import (
    Specification,
    AndSpecification,
    OrSpecification,
    NotSpecification,
    JobCanBeProcessedSpecification,
    UserCanCreateJobSpecification,
    AudioFileIsValidSpecification
)

__all__ = [
    # Enumerations
    "JobStatus",
    "SegmentType", 
    "ConfidenceLevel",
    "DeviceType",
    "ModelType",
    "AudioFormat",
    "LanguageCode",
    
    # Value objects
    "AudioMetadata",
    "TranscriptionResult",
    "ModelConfiguration",
    "TimeRange",
    "SpeakerInfo", 
    "ProcessingStats",
    
    # Exceptions
    "DomainException",
    "DomainValidationError",
    "BusinessRuleViolationError",
    "ResourceNotFoundError",
    "InvalidOperationError",
    "ConcurrencyError",
    
    # Domain events
    "DomainEvent",
    "DomainEventHandler",
    "EventBus",
    "JobCreatedEvent",
    "JobStatusChangedEvent", 
    "JobCompletedEvent",
    "JobFailedEvent",
    "UserRegisteredEvent",
    "UserUpdatedEvent",
    
    # Base classes
    "Entity",
    "ValueObject",
    "AggregateRoot",
    "DomainService",
    "Repository",
    
    # Specifications
    "Specification",
    "AndSpecification",
    "OrSpecification", 
    "NotSpecification",
    "JobCanBeProcessedSpecification",
    "UserCanCreateJobSpecification",
    "AudioFileIsValidSpecification"
]
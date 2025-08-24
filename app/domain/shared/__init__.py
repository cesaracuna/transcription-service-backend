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
try:
    from .enums import (
        JobStatus,
        SegmentType,
        ConfidenceLevel,
        DeviceType,
        ModelType,
        AudioFormat,
        LanguageCode
    )
except ImportError:
    JobStatus = SegmentType = ConfidenceLevel = None
    DeviceType = ModelType = AudioFormat = LanguageCode = None

# Import value objects
try:
    from .value_objects import (
        AudioMetadata,
        TranscriptionResult,
        ModelConfiguration,
        TimeRange,
        SpeakerInfo,
        ProcessingStats
    )
except ImportError:
    AudioMetadata = TranscriptionResult = ModelConfiguration = None
    TimeRange = SpeakerInfo = ProcessingStats = None

# Import exceptions
try:
    from .exceptions import (
        DomainException,
        DomainValidationError,
        BusinessRuleViolationError,
        ResourceNotFoundError,
        ConcurrencyConflictError,
        DomainStateError,
        InvariantViolationError,
        TranscriptionJobError,
        InvalidJobStateError,
        AudioFileError,
        SpeakerDiarizationError,
        HallucinationDetectionError,
        UserError,
        UserValidationError
    )
except ImportError:
    DomainException = DomainValidationError = BusinessRuleViolationError = None
    ResourceNotFoundError = ConcurrencyConflictError = DomainStateError = None
    InvariantViolationError = TranscriptionJobError = InvalidJobStateError = None
    AudioFileError = SpeakerDiarizationError = HallucinationDetectionError = None
    UserError = UserValidationError = None

# Import domain events
try:
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
except ImportError:
    DomainEvent = DomainEventHandler = EventBus = None
    JobCreatedEvent = JobStatusChangedEvent = JobCompletedEvent = None
    JobFailedEvent = UserRegisteredEvent = UserUpdatedEvent = None

# Import base classes
try:
    from .base import (
        Entity,
        ValueObject,
        AggregateRoot,
        DomainService,
        Repository
    )
except ImportError:
    Entity = ValueObject = AggregateRoot = None
    DomainService = Repository = None

# Import specifications
try:
    from .specifications import (
        Specification,
        AndSpecification,
        OrSpecification,
        NotSpecification,
        JobCanBeProcessedSpecification,
        UserCanCreateJobSpecification,
        AudioFileIsValidSpecification
    )
except ImportError:
    Specification = AndSpecification = OrSpecification = None
    NotSpecification = JobCanBeProcessedSpecification = None
    UserCanCreateJobSpecification = AudioFileIsValidSpecification = None

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
    "ConcurrencyConflictError",
    "DomainStateError",
    "InvariantViolationError",
    "TranscriptionJobError",
    "InvalidJobStateError",
    "AudioFileError",
    "SpeakerDiarizationError",
    "HallucinationDetectionError",
    "UserError",
    "UserValidationError",
    
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
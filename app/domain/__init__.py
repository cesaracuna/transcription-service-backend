"""
Domain layer containing business logic and entities.

This package implements the Domain-Driven Design (DDD) patterns and contains:
- Domain models and entities with business logic
- Value objects for data integrity
- Domain services for complex business operations
- Repository interfaces for data persistence abstraction
- Domain events for decoupled communication
- Business rules and invariants

The domain layer is independent of infrastructure concerns and contains
the core business logic of the transcription service.
"""

# Import domain models
try:
    from .transcription.models import (
        User,
        TranscriptionJob,
        TranscriptionSegment,
        DiarizationSegment,
        Hallucination
    )
except ImportError:
    User = TranscriptionJob = TranscriptionSegment = None
    DiarizationSegment = Hallucination = None

# Import domain services
try:
    from .transcription.services import (
        TranscriptionService,
        UserService,
        AudioProcessingService,
        HallucinationDetectionService
    )
except ImportError:
    TranscriptionService = UserService = AudioProcessingService = None
    HallucinationDetectionService = None

# Import repository interfaces
try:
    from .transcription.repositories import (
        UserRepository,
        TranscriptionJobRepository,
        TranscriptionSegmentRepository,
        DiarizationSegmentRepository,
        HallucinationRepository
    )
except ImportError:
    UserRepository = TranscriptionJobRepository = TranscriptionSegmentRepository = None
    DiarizationSegmentRepository = HallucinationRepository = None

# Import shared domain components
try:
    from .shared.enums import (
        JobStatus,
        SegmentType,
        ConfidenceLevel,
        DeviceType,
        ModelType
    )
except ImportError:
    JobStatus = SegmentType = ConfidenceLevel = None
    DeviceType = ModelType = None

try:
    from .shared.value_objects import (
        AudioMetadata,
        TranscriptionResult,
        ModelConfiguration
    )
except ImportError:
    AudioMetadata = TranscriptionResult = ModelConfiguration = None

try:
    from .shared.exceptions import (
        DomainValidationError,
        BusinessRuleViolationError,
        ResourceNotFoundError
    )
except ImportError:
    DomainValidationError = BusinessRuleViolationError = ResourceNotFoundError = None

try:
    from .shared.events import (
        DomainEvent,
        JobCreatedEvent,
        JobCompletedEvent,
        JobFailedEvent,
        UserRegisteredEvent
    )
except ImportError:
    DomainEvent = JobCreatedEvent = JobCompletedEvent = None
    JobFailedEvent = UserRegisteredEvent = None

__all__ = [
    # Domain models
    "User",
    "TranscriptionJob",
    "TranscriptionSegment", 
    "DiarizationSegment",
    "Hallucination",
    
    # Domain services  
    "TranscriptionService",
    "UserService",
    "AudioProcessingService",
    "HallucinationDetectionService",
    
    # Repository interfaces
    "UserRepository",
    "TranscriptionJobRepository",
    "TranscriptionSegmentRepository",
    "DiarizationSegmentRepository", 
    "HallucinationRepository",
    
    # Shared enums
    "JobStatus",
    "SegmentType",
    "ConfidenceLevel",
    "DeviceType",
    "ModelType",
    
    # Value objects
    "AudioMetadata",
    "TranscriptionResult",
    "ModelConfiguration",
    
    # Domain exceptions
    "DomainValidationError",
    "BusinessRuleViolationError", 
    "ResourceNotFoundError",
    
    # Domain events
    "DomainEvent",
    "JobCreatedEvent",
    "JobCompletedEvent",
    "JobFailedEvent",
    "UserRegisteredEvent"
]
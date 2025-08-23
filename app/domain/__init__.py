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
from .transcription.models import (
    User,
    TranscriptionJob,
    TranscriptionSegment,
    DiarizationSegment,
    Hallucination
)

# Import domain services
from .transcription.services import (
    TranscriptionService,
    UserService,
    AudioProcessingService,
    HallucinationDetectionService
)

# Import repository interfaces
from .transcription.repositories import (
    UserRepository,
    TranscriptionJobRepository,
    TranscriptionSegmentRepository,
    DiarizationSegmentRepository,
    HallucinationRepository
)

# Import shared domain components
from .shared.enums import (
    JobStatus,
    SegmentType,
    ConfidenceLevel,
    DeviceType,
    ModelType
)

from .shared.value_objects import (
    AudioMetadata,
    TranscriptionResult,
    ModelConfiguration
)

from .shared.exceptions import (
    DomainValidationError,
    BusinessRuleViolationError,
    ResourceNotFoundError
)

from .shared.events import (
    DomainEvent,
    JobCreatedEvent,
    JobCompletedEvent,
    JobFailedEvent,
    UserRegisteredEvent
)

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
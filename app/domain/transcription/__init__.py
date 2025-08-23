"""
Transcription domain logic.

This package contains the core business logic for the transcription bounded context:
- Domain models and aggregates (User, TranscriptionJob, etc.)
- Domain services for complex business operations
- Repository interfaces for data access abstraction
- Business rules and invariants specific to transcription
- Domain events related to transcription workflow
"""

# Import domain models (aggregates and entities)
from .models import (
    User,
    TranscriptionJob,
    TranscriptionSegment,
    DiarizationSegment,
    Hallucination,
    Speaker
)

# Import domain services
from .services import (
    TranscriptionService,
    UserService,
    AudioProcessingService,
    HallucinationDetectionService,
    SpeakerDiarizationService,
    TranscriptionValidationService
)

# Import repository interfaces
from .repositories import (
    UserRepository,
    TranscriptionJobRepository,
    TranscriptionSegmentRepository,
    DiarizationSegmentRepository,
    HallucinationRepository,
    SpeakerRepository
)

# Import domain-specific value objects
from .value_objects import (
    JobMetadata,
    ProcessingConfiguration,
    QualityMetrics,
    AudioFeatures,
    SpeakerProfile
)

# Import domain-specific events
from .events import (
    JobCreatedEvent,
    JobProcessingStartedEvent,
    JobProcessingCompletedEvent,
    JobProcessingFailedEvent,
    TranscriptionGeneratedEvent,
    DiarizationCompletedEvent,
    HallucinationDetectedEvent,
    UserCreatedEvent,
    UserActivatedEvent,
    UserDeactivatedEvent
)

# Import specifications
from .specifications import (
    JobCanBeProcessedSpecification,
    JobCanBeCancelledSpecification,
    UserCanCreateJobSpecification,
    AudioFileIsValidSpecification,
    TranscriptionIsCompleteSpecification
)

__all__ = [
    # Domain models
    "User",
    "TranscriptionJob",
    "TranscriptionSegment",
    "DiarizationSegment",
    "Hallucination",
    "Speaker",
    
    # Domain services
    "TranscriptionService",
    "UserService", 
    "AudioProcessingService",
    "HallucinationDetectionService",
    "SpeakerDiarizationService",
    "TranscriptionValidationService",
    
    # Repository interfaces
    "UserRepository",
    "TranscriptionJobRepository",
    "TranscriptionSegmentRepository",
    "DiarizationSegmentRepository",
    "HallucinationRepository",
    "SpeakerRepository",
    
    # Value objects
    "JobMetadata",
    "ProcessingConfiguration",
    "QualityMetrics",
    "AudioFeatures",
    "SpeakerProfile",
    
    # Domain events
    "JobCreatedEvent",
    "JobProcessingStartedEvent",
    "JobProcessingCompletedEvent",
    "JobProcessingFailedEvent",
    "TranscriptionGeneratedEvent", 
    "DiarizationCompletedEvent",
    "HallucinationDetectedEvent",
    "UserCreatedEvent",
    "UserActivatedEvent",
    "UserDeactivatedEvent",
    
    # Specifications
    "JobCanBeProcessedSpecification",
    "JobCanBeCancelledSpecification", 
    "UserCanCreateJobSpecification",
    "AudioFileIsValidSpecification",
    "TranscriptionIsCompleteSpecification"
]
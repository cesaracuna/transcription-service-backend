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
try:
    from .models import (
        User,
        TranscriptionJob,
        TranscriptionSegment,
        DiarizationSegment,
        Hallucination,
        Speaker
    )
except ImportError:
    User = TranscriptionJob = TranscriptionSegment = None
    DiarizationSegment = Hallucination = Speaker = None

# Import domain services
try:
    from .services import (
        TranscriptionService,
        UserService,
        AudioProcessingService,
        HallucinationDetectionService,
        SpeakerDiarizationService,
        TranscriptionValidationService
    )
except ImportError:
    TranscriptionService = UserService = AudioProcessingService = None
    HallucinationDetectionService = SpeakerDiarizationService = None
    TranscriptionValidationService = None

# Import repository interfaces
try:
    from .repositories import (
        UserRepository,
        TranscriptionJobRepository,
        TranscriptionSegmentRepository,
        DiarizationSegmentRepository,
        HallucinationRepository,
        SpeakerRepository
    )
except ImportError:
    UserRepository = TranscriptionJobRepository = TranscriptionSegmentRepository = None
    DiarizationSegmentRepository = HallucinationRepository = SpeakerRepository = None

# Import domain-specific value objects
try:
    from .value_objects import (
        JobMetadata,
        ProcessingConfiguration,
        QualityMetrics,
        AudioFeatures,
        SpeakerProfile
    )
except ImportError:
    JobMetadata = ProcessingConfiguration = QualityMetrics = None
    AudioFeatures = SpeakerProfile = None

# Import domain-specific events
try:
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
except ImportError:
    JobCreatedEvent = JobProcessingStartedEvent = JobProcessingCompletedEvent = None
    JobProcessingFailedEvent = TranscriptionGeneratedEvent = DiarizationCompletedEvent = None
    HallucinationDetectedEvent = UserCreatedEvent = UserActivatedEvent = None
    UserDeactivatedEvent = None

# Import specifications
try:
    from .specifications import (
        JobCanBeProcessedSpecification,
        JobCanBeCancelledSpecification,
        UserCanCreateJobSpecification,
        AudioFileIsValidSpecification,
        TranscriptionIsCompleteSpecification
    )
except ImportError:
    JobCanBeProcessedSpecification = JobCanBeCancelledSpecification = None
    UserCanCreateJobSpecification = AudioFileIsValidSpecification = None
    TranscriptionIsCompleteSpecification = None

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
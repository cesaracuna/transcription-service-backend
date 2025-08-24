"""
Infrastructure layer handling external dependencies and data persistence.

This package contains all infrastructure concerns including:
- Database implementations and ORM models
- External service integrations (Redis, AI models, etc.)
- File storage and management
- Message queues and background workers
- Third-party API clients
- Configuration and dependency injection

The infrastructure layer implements interfaces defined in the domain layer
and provides concrete implementations for data persistence and external services.
"""

# Import database components
from .database import (
    Base,
    DatabaseManager,
    get_database_session,
    create_database_engine
)

# Import repository implementations
from .database.repositories import (
    SQLUserRepository,
    SQLTranscriptionJobRepository,
    SQLTranscriptionSegmentRepository,
    SQLDiarizationSegmentRepository,
    SQLHallucinationRepository
)

# Import external services
from .external import (
    RedisClient,
    get_redis_client,
    close_redis_client
)

# Import AI model components from external module (with graceful handling)
from .external import (
    ModelRegistry,
    AIModelRegistry,
    WhisperModelManager,
    DiarizationModelManager,
    get_model_registry,
    initialize_models,
    cleanup_models
)

# Import storage components
try:
    from .storage import (
        FileManager,
        AudioFileManager,
        LocalFileStorage,
        S3FileStorage,
        get_file_manager
    )
except ImportError:
    FileManager = None
    AudioFileManager = None
    LocalFileStorage = None
    S3FileStorage = None
    get_file_manager = None

# Import worker components
try:
    from .workers import (
        celery_app,
        transcription_task,
        diarization_task,
        hallucination_detection_task,
        cleanup_task
    )
except ImportError:
    celery_app = None
    transcription_task = None
    diarization_task = None
    hallucination_detection_task = None
    cleanup_task = None

# Import event handlers
try:
    from .events import (
        DomainEventDispatcher,
        JobCreatedEventHandler,
        JobCompletedEventHandler,
        UserRegisteredEventHandler
    )
except ImportError:
    DomainEventDispatcher = None
    JobCreatedEventHandler = None
    JobCompletedEventHandler = None
    UserRegisteredEventHandler = None

__all__ = [
    # Database
    "Base",
    "DatabaseManager",
    "get_database_session",
    "create_database_engine",
    
    # Repository implementations
    "SQLUserRepository",
    "SQLTranscriptionJobRepository",
    "SQLTranscriptionSegmentRepository",
    "SQLDiarizationSegmentRepository",
    "SQLHallucinationRepository",
    
    # External services
    "RedisClient", 
    "get_redis_client",
    "close_redis_client",
    
    # AI models
    "ModelRegistry",
    "AIModelRegistry", 
    "WhisperModelManager",
    "DiarizationModelManager",
    "initialize_models",
    "cleanup_models",
    "get_model_registry",
    
    # Storage
    "FileManager",
    "AudioFileManager", 
    "LocalFileStorage",
    "S3FileStorage",
    "get_file_manager",
    
    # Workers
    "celery_app",
    "transcription_task",
    "diarization_task",
    "hallucination_detection_task",
    "cleanup_task",
    
    # Event handlers
    "DomainEventDispatcher",
    "JobCreatedEventHandler",
    "JobCompletedEventHandler",
    "UserRegisteredEventHandler"
]
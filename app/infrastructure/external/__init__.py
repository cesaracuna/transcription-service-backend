"""
External service integrations and AI model management.

This package handles integrations with external services and systems:
- Redis client for caching and message brokering
- AI model management (Whisper, PyAnnote, etc.)
- Third-party API clients and integrations
- External file storage services (S3, GCS, etc.)
- Monitoring and telemetry services
- Email and notification services
"""

from .redis_client import (
    RedisClient,
    get_redis_client,
    close_redis_client
)

try:
    from .ai_models import (
        AIModelRegistry,
        WhisperModelManager,
        DiarizationModelManager,
        HallucinationDetectionModelManager,
        get_model_registry,
        initialize_models,
        cleanup_models
    )
    # Alias for compatibility
    ModelRegistry = AIModelRegistry
except ImportError:
    # Fallback when AI models are not available (missing dependencies)
    AIModelRegistry = None
    ModelRegistry = None
    WhisperModelManager = None
    DiarizationModelManager = None
    HallucinationDetectionModelManager = None
    get_model_registry = None
    initialize_models = None
    cleanup_models = None

try:
    from .http_client import (
        HTTPClient,
        get_http_client,
        close_http_client
    )
except ImportError:
    HTTPClient = None
    get_http_client = None
    close_http_client = None

try:
    from .storage_services import (
        S3Client,
        GCSClient,
        get_cloud_storage_client
    )
except ImportError:
    S3Client = None
    GCSClient = None
    get_cloud_storage_client = None

try:
    from .monitoring import (
        PrometheusMetrics,
        TelemetryClient,
        get_telemetry_client
    )
except ImportError:
    PrometheusMetrics = None
    TelemetryClient = None
    get_telemetry_client = None

try:
    from .notification import (
        EmailService,
        SlackNotifier,
        WebhookNotifier,
        get_notification_service
    )
except ImportError:
    EmailService = None
    SlackNotifier = None
    WebhookNotifier = None
    get_notification_service = None

__all__ = [
    # Redis
    "RedisClient",
    "get_redis_client", 
    "close_redis_client",
    
    # AI Models
    "AIModelRegistry",
    "ModelRegistry",
    "WhisperModelManager",
    "DiarizationModelManager",
    "HallucinationDetectionModelManager",
    "get_model_registry",
    "initialize_models",
    "cleanup_models",
    
    # HTTP Client
    "HTTPClient",
    "get_http_client",
    "close_http_client",
    
    # Storage services
    "S3Client",
    "GCSClient", 
    "get_cloud_storage_client",
    
    # Monitoring
    "PrometheusMetrics",
    "TelemetryClient",
    "get_telemetry_client",
    
    # Notifications
    "EmailService",
    "SlackNotifier",
    "WebhookNotifier",
    "get_notification_service"
]
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

from .ai_models import (
    ModelRegistry,
    WhisperModelManager,
    DiarizationModelManager,
    HallucinationDetectionModelManager,
    get_model_registry,
    initialize_models,
    cleanup_models
)

from .http_client import (
    HTTPClient,
    get_http_client,
    close_http_client
)

from .storage_services import (
    S3Client,
    GCSClient,
    get_cloud_storage_client
)

from .monitoring import (
    PrometheusMetrics,
    TelemetryClient,
    get_telemetry_client
)

from .notification import (
    EmailService,
    SlackNotifier,
    WebhookNotifier,
    get_notification_service
)

__all__ = [
    # Redis
    "RedisClient",
    "get_redis_client", 
    "close_redis_client",
    
    # AI Models
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
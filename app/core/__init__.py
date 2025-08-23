"""
Core application configuration and utilities.

This package contains the fundamental components of the application:
- Configuration management (settings, environment variables)
- Dependency injection and IoC container
- Application exceptions and error handling
- Security utilities and authentication
- Logging configuration
- Application lifecycle management
"""

from .config import (
    Settings,
    DatabaseSettings,
    RedisSettings,
    CelerySettings,
    AIModelSettings,
    AudioProcessingSettings,
    HallucinationDetectionSettings,
    DiarizationSettings,
    SecuritySettings,
    LoggingSettings,
    get_settings
)

from .exceptions import (
    TranscriptionServiceError,
    ConfigurationError,
    DatabaseError,
    AudioProcessingError,
    ModelLoadingError,
    TranscriptionError,
    DiarizationError,
    HallucinationDetectionError,
    StorageError,
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    JobNotFoundError,
    UserNotFoundError,
    ResourceNotFoundError,
    RateLimitError,
    ServiceUnavailableError
)

# Import modules that exist
try:
    from .dependencies import *
except ImportError:
    pass
    
try:
    from .security import *
except ImportError:
    pass
    
try:
    from .logging import (
        setup_logging,
        get_logger
    )
except ImportError:
    # Create fallback functions if logging module fails to import
    def setup_logging(settings):
        import logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def get_logger(name):
        import logging
        return logging.getLogger(name)

__all__ = [
    # Configuration
    "Settings",
    "DatabaseSettings", 
    "RedisSettings",
    "CelerySettings",
    "AIModelSettings",
    "AudioProcessingSettings",
    "HallucinationDetectionSettings",
    "DiarizationSettings",
    "SecuritySettings",
    "LoggingSettings",
    "get_settings",
    
    # Logging
    "setup_logging",
    "get_logger",
    
    # Exceptions
    "TranscriptionServiceError",
    "ConfigurationError",
    "DatabaseError",
    "AudioProcessingError",
    "ModelLoadingError",
    "TranscriptionError",
    "DiarizationError",
    "HallucinationDetectionError",
    "StorageError",
    "ValidationError",
    "AuthenticationError",
    "AuthorizationError",
    "JobNotFoundError",
    "UserNotFoundError",
    "ResourceNotFoundError",
    "RateLimitError",
    "ServiceUnavailableError"
]
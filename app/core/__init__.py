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
    AISettings,
    get_settings
)

from .dependencies import (
    get_current_user,
    get_transcription_service,
    get_user_service,
    get_file_manager,
    get_redis_client,
    get_database
)

from .exceptions import (
    BaseAppException,
    DomainValidationError,
    ResourceNotFoundError,
    BusinessRuleViolationError,
    DatabaseError,
    ExternalServiceError,
    AuthenticationError,
    AuthorizationError
)

from .security import (
    verify_password,
    get_password_hash,
    create_access_token,
    decode_access_token,
    get_current_user_from_token
)

from .logging import (
    setup_logging,
    get_logger,
    log_performance,
    log_audit_event
)

__all__ = [
    # Configuration
    "Settings",
    "DatabaseSettings", 
    "RedisSettings",
    "CelerySettings",
    "AISettings",
    "get_settings",
    
    # Dependencies
    "get_current_user",
    "get_transcription_service",
    "get_user_service",
    "get_file_manager",
    "get_redis_client", 
    "get_database",
    
    # Exceptions
    "BaseAppException",
    "DomainValidationError",
    "ResourceNotFoundError",
    "BusinessRuleViolationError",
    "DatabaseError",
    "ExternalServiceError",
    "AuthenticationError",
    "AuthorizationError",
    
    # Security
    "verify_password",
    "get_password_hash",
    "create_access_token",
    "decode_access_token",
    "get_current_user_from_token",
    
    # Logging
    "setup_logging",
    "get_logger",
    "log_performance",
    "log_audit_event"
]
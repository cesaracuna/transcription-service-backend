"""
Centralized logging configuration using structlog for better observability.
"""

import logging
import logging.handlers
import sys
from typing import Optional
from pathlib import Path

try:
    import structlog
    from structlog.types import FilteringBoundLogger
    STRUCTLOG_AVAILABLE = True
except ImportError:
    # Fallback when structlog is not available
    structlog = None
    FilteringBoundLogger = object
    STRUCTLOG_AVAILABLE = False

from .config import LoggingSettings


def configure_structlog(
    log_level: str = "INFO",
    use_json: bool = False,
    service_name: str = "transcription-service",
) -> None:
    """
    Configure structlog for structured logging.
    
    Args:
        log_level: Logging level
        use_json: Whether to use JSON output format
        service_name: Service name for logging context
    """
    if not STRUCTLOG_AVAILABLE:
        # Fallback to standard logging
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            stream=sys.stdout,
            level=getattr(logging, log_level.upper()),
        )
        return
    
    timestamper = structlog.processors.TimeStamper(fmt="ISO")
    
    shared_processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if use_json:
        # JSON output for production
        structlog.configure(
            processors=shared_processors + [
                structlog.processors.dict_tracebacks,
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    else:
        # Human-readable output for development
        structlog.configure(
            processors=shared_processors + [
                structlog.dev.ConsoleRenderer(colors=True),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )


def setup_logging(
    settings: LoggingSettings,
    process_name: Optional[str] = None,
) -> Optional[FilteringBoundLogger]:
    """
    Set up comprehensive logging configuration.
    
    Args:
        settings: Logging configuration settings
        process_name: Optional process name for context
        
    Returns:
        Configured structlog logger or standard logger
    """
    # Configure structlog or fallback
    configure_structlog(
        log_level=settings.level,
        use_json=settings.use_json,
        service_name="transcription-service",
    )
    
    # Set up file logging if configured
    if settings.file_path:
        file_path = Path(settings.file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            filename=file_path,
            maxBytes=settings.max_file_size,
            backupCount=settings.backup_count,
            encoding="utf-8",
        )
        
        file_formatter = logging.Formatter(settings.format)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(getattr(logging, settings.level.upper()))
        
        # Add file handler to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
    
    # Create logger with context
    if STRUCTLOG_AVAILABLE:
        logger = structlog.get_logger()
        if process_name:
            logger = logger.bind(process=process_name)
    else:
        # Fallback to standard logging
        logger = logging.getLogger("transcription-service")
        if process_name:
            logger = logger.getChild(process_name)
    
    # Silence noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    
    return logger


def get_logger(name: str, **context):
    """
    Get a logger with optional context.
    
    Args:
        name: Logger name (usually __name__)
        **context: Additional context to bind to logger
        
    Returns:
        Configured logger with context
    """
    if STRUCTLOG_AVAILABLE:
        logger = structlog.get_logger(name)
        if context:
            logger = logger.bind(**context)
    else:
        # Fallback to standard logging
        logger = logging.getLogger(name)
        # Note: Standard logger doesn't support binding context
    
    return logger
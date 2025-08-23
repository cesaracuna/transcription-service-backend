"""
Worker infrastructure for background task processing.

This package contains Celery-based background task processing:
- Celery application configuration and setup
- Background task definitions for transcription workflow
- Task routing and queue management
- Error handling and retry mechanisms
- Task monitoring and progress tracking
- Periodic tasks and scheduled jobs
"""

from .celery_app import (
    celery_app,
    create_celery_app,
    configure_celery,
    celery_config
)

from .tasks import (
    transcription_task,
    diarization_task,
    audio_preprocessing_task,
    hallucination_detection_task,
    post_processing_task,
    cleanup_old_files_task,
    generate_reports_task
)

from .monitoring import (
    TaskMonitor,
    TaskProgressTracker,
    get_task_monitor,
    update_task_progress,
    get_task_progress
)

from .utils import (
    get_task_signature,
    create_task_chain,
    create_task_group,
    retry_with_exponential_backoff
)

# Task decorators and utilities
from .decorators import (
    track_performance,
    handle_task_errors,
    log_task_execution,
    require_models_loaded
)

__all__ = [
    # Celery app
    "celery_app",
    "create_celery_app",
    "configure_celery",
    "celery_config",
    
    # Task definitions
    "transcription_task",
    "diarization_task",
    "audio_preprocessing_task", 
    "hallucination_detection_task",
    "post_processing_task",
    "cleanup_old_files_task",
    "generate_reports_task",
    
    # Monitoring
    "TaskMonitor",
    "TaskProgressTracker",
    "get_task_monitor",
    "update_task_progress",
    "get_task_progress",
    
    # Utils
    "get_task_signature",
    "create_task_chain",
    "create_task_group",
    "retry_with_exponential_backoff",
    
    # Decorators
    "track_performance",
    "handle_task_errors",
    "log_task_execution",
    "require_models_loaded"
]
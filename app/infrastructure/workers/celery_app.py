"""
Celery application configuration and setup.
Modern Celery configuration with proper error handling and monitoring.
"""

import logging
from typing import Any, Dict, Optional

from celery import Celery
from celery.signals import (
    setup_logging, worker_ready, worker_shutdown,
    task_prerun, task_postrun, task_failure, task_success
)

from ...core.config import settings
from ...core.logging import setup_logging as setup_app_logging, get_logger

logger = get_logger(__name__)


def create_celery_app() -> Celery:
    """
    Create and configure Celery application.
    
    Returns:
        Configured Celery application
    """
    logger.info("Creating Celery application")
    
    # Create Celery app
    celery_app = Celery(
        "transcription-service",
        broker=settings.celery.broker_url,
        backend=settings.celery.result_backend
    )
    
    # Configure Celery
    celery_app.conf.update(
        # Task settings
        task_track_started=settings.celery.task_track_started,
        task_serializer=settings.celery.task_serializer,
        result_serializer=settings.celery.result_serializer,
        accept_content=settings.celery.accept_content,
        
        # Timezone settings
        timezone=settings.celery.timezone,
        enable_utc=settings.celery.enable_utc,
        
        # Worker settings
        worker_cancel_long_running_tasks_on_connection_loss=settings.celery.worker_cancel_long_running_tasks_on_connection_loss,
        worker_prefetch_multiplier=1,  # Important for long-running tasks
        worker_max_tasks_per_child=10,  # Restart workers after 10 tasks to prevent memory leaks
        
        # Task routing
        task_routes={
            'transcription.*': {'queue': 'transcription'},
            'post_processing.*': {'queue': 'post_processing'},
            'health.*': {'queue': 'health'},
        },
        
        # Result backend settings
        result_expires=3600,  # Results expire after 1 hour
        result_backend_transport_options={
            'master_name': 'mymaster',
            'visibility_timeout': 3600,
        },
        
        # Monitoring
        worker_send_task_events=True,
        task_send_sent_event=True,
        
        # Error handling
        task_reject_on_worker_lost=True,
        task_acks_late=True,  # Acknowledge task after completion
        
        # Beat schedule (if using periodic tasks)
        beat_schedule={
            'health-check': {
                'task': 'health.check_system_health',
                'schedule': 300.0,  # Every 5 minutes
            },
            'cleanup-old-files': {
                'task': 'maintenance.cleanup_old_files',
                'schedule': 3600.0,  # Every hour
            },
        },
    )
    
    # Auto-discover tasks
    celery_app.autodiscover_tasks([
        'app.domain.transcription',
        'app.infrastructure.workers',
    ])
    
    logger.info("Celery application created and configured")
    return celery_app


# Create global Celery app instance
celery_app = create_celery_app()


# Signal handlers for monitoring and logging
@setup_logging.connect
def config_loggers(*args, **kwargs):
    """Configure logging when Celery starts."""
    setup_app_logging(
        settings.logging,
        process_name="CeleryWorker"
    )


@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Handle worker ready signal."""
    logger.info(f"Celery worker ready: {sender}")


@worker_shutdown.connect
def worker_shutdown_handler(sender=None, **kwargs):
    """Handle worker shutdown signal."""
    logger.info(f"Celery worker shutting down: {sender}")


@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds):
    """Handle task prerun signal."""
    logger.info(f"Task starting: {task.name} [{task_id}]")


@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, state=None, **kwds):
    """Handle task postrun signal."""
    logger.info(f"Task completed: {task.name} [{task_id}] - State: {state}")


@task_success.connect
def task_success_handler(sender=None, result=None, **kwargs):
    """Handle task success signal."""
    logger.info(f"Task succeeded: {sender.name}")


@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwargs):
    """Handle task failure signal."""
    logger.error(f"Task failed: {sender.name} [{task_id}] - Error: {exception}")
    logger.error(f"Traceback: {traceback}")


# Export the app
__all__ = ["celery_app"]
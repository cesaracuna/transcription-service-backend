"""
Celery application configuration.
Centralized Celery instance used by both workers and API clients.
"""
import logging
from celery import Celery

from .config import get_config

logger = logging.getLogger(__name__)

# Get configuration
config = get_config()

# Create single Celery application instance
celery_app = Celery(
    "pluralia",
    broker=config.CELERY_BROKER_URL,
    backend=config.CELERY_RESULT_BACKEND
)

# Configure Celery
celery_app.conf.update(
    # Task tracking and monitoring
    task_track_started=True,
    worker_cancel_long_running_tasks_on_connection_loss=True,

    # Serialization
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',

    # Timezone
    timezone='UTC',
    enable_utc=True,

    # Task execution
    task_acks_late=True,  # Acknowledge task after completion (fault tolerance)
    task_reject_on_worker_lost=True,  # Reject task if worker is lost
    worker_prefetch_multiplier=1,  # Only prefetch 1 task at a time (good for long-running tasks)

    # Result backend
    result_expires=3600,  # Results expire after 1 hour
    result_extended=True,  # Store additional task metadata

    # Task routing - all tasks go to celery queue (default Celery queue)
    task_default_queue='celery',
    task_default_exchange='celery',
    task_default_routing_key='celery',

    # Queue configuration
    task_queues={
        'celery': {
            'exchange': 'celery',
            'routing_key': 'celery',
        }
    },

    # Worker configuration
    worker_max_tasks_per_child=50,  # Restart worker after 50 tasks (prevent memory leaks)
    worker_disable_rate_limits=True,  # Disable rate limits for simplicity
    worker_concurrency=1,  # Only 1 task at a time (ensures sequential processing)

    # Time limits (global defaults, can be overridden per task)
    task_time_limit=7200,  # 2 hours hard limit
    task_soft_time_limit=6600,  # 1 hour 50 minutes soft limit

    # Task result backend configuration
    result_backend_transport_options={
        'master_name': 'pluralia',
        'visibility_timeout': 43200,  # 12 hours
    },

    # Broker configuration
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=10,
)

logger.info("Celery application configured successfully")
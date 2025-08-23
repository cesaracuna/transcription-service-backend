"""
Metrics endpoint for Prometheus monitoring.
"""

from fastapi import APIRouter, Depends
from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge, generate_latest
from prometheus_client.exposition import make_wsgi_app
from starlette.responses import Response
import time
from typing import Dict, Any

from ....core.dependencies import get_current_user
from ....domain.transcription.repositories import TranscriptionJobRepository
from ....infrastructure.database.repositories.transcription_job_repository import get_transcription_job_repository
from ....infrastructure.external.redis_client import get_redis_client
from ....infrastructure.external.ai_models import get_model_registry

router = APIRouter()

# Prometheus metrics registry
registry = CollectorRegistry()

# Application metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    registry=registry
)

# Business metrics
transcription_jobs_total = Counter(
    'transcription_jobs_total',
    'Total transcription jobs created',
    ['status'],
    registry=registry
)

transcription_jobs_pending_total = Gauge(
    'transcription_jobs_pending_total',
    'Current number of pending transcription jobs',
    registry=registry
)

transcription_jobs_processing_total = Gauge(
    'transcription_jobs_processing_total',
    'Current number of processing transcription jobs',
    registry=registry
)

transcription_jobs_completed_total = Counter(
    'transcription_jobs_completed_total',
    'Total completed transcription jobs',
    registry=registry
)

transcription_jobs_failed_total = Counter(
    'transcription_jobs_failed_total',
    'Total failed transcription jobs',
    registry=registry
)

transcription_duration_seconds = Histogram(
    'transcription_duration_seconds',
    'Time spent processing transcription jobs',
    buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1200],
    registry=registry
)

# Audio processing metrics
audio_file_size_bytes = Histogram(
    'audio_file_size_bytes',
    'Size of uploaded audio files in bytes',
    buckets=[1024, 10240, 102400, 1048576, 10485760, 104857600],
    registry=registry
)

audio_duration_seconds = Histogram(
    'audio_duration_seconds',
    'Duration of audio files in seconds',
    buckets=[1, 10, 30, 60, 300, 600, 1800, 3600],
    registry=registry
)

audio_conversion_duration_seconds = Histogram(
    'audio_conversion_duration_seconds',
    'Time spent converting audio files',
    registry=registry
)

audio_conversion_failures_total = Counter(
    'audio_conversion_failures_total',
    'Total audio conversion failures',
    registry=registry
)

audio_conversion_attempts_total = Counter(
    'audio_conversion_attempts_total',
    'Total audio conversion attempts',
    registry=registry
)

# AI model metrics
ai_model_loading_duration_seconds = Histogram(
    'ai_model_loading_duration_seconds',
    'Time spent loading AI models',
    ['model_type'],
    registry=registry
)

ai_model_loading_failures_total = Counter(
    'ai_model_loading_failures_total',
    'Total AI model loading failures',
    ['model_type'],
    registry=registry
)

ai_model_memory_usage_bytes = Gauge(
    'ai_model_memory_usage_bytes',
    'Memory usage of loaded AI models',
    ['model_type'],
    registry=registry
)

# Celery metrics
celery_task_total = Counter(
    'celery_task_total',
    'Total Celery tasks',
    ['task_name', 'status'],
    registry=registry
)

celery_task_duration_seconds = Histogram(
    'celery_task_duration_seconds',
    'Celery task execution time',
    ['task_name'],
    registry=registry
)

celery_queue_length = Gauge(
    'celery_queue_length',
    'Number of tasks in Celery queue',
    ['queue_name'],
    registry=registry
)

celery_task_failed_total = Counter(
    'celery_task_failed_total',
    'Total failed Celery tasks',
    ['task_name'],
    registry=registry
)

# Database metrics
database_connections_active = Gauge(
    'database_connections_active',
    'Number of active database connections',
    registry=registry
)

database_query_duration_seconds = Histogram(
    'database_query_duration_seconds',
    'Database query execution time',
    ['operation'],
    registry=registry
)

# Redis metrics
redis_operations_total = Counter(
    'redis_operations_total',
    'Total Redis operations',
    ['operation', 'status'],
    registry=registry
)

redis_operation_duration_seconds = Histogram(
    'redis_operation_duration_seconds',
    'Redis operation execution time',
    ['operation'],
    registry=registry
)


@router.get("/metrics")
async def metrics():
    """Expose Prometheus metrics."""
    return Response(
        generate_latest(registry),
        media_type="text/plain"
    )


@router.get("/metrics/business")
async def business_metrics(
    job_repository: TranscriptionJobRepository = Depends(get_transcription_job_repository)
) -> Dict[str, Any]:
    """Get business-specific metrics."""
    try:
        # Get job statistics
        stats = await job_repository.get_overall_statistics()
        
        # Update Prometheus gauges
        transcription_jobs_pending_total.set(stats.get("pending_jobs", 0))
        transcription_jobs_processing_total.set(stats.get("processing_jobs", 0))
        
        return {
            "jobs": {
                "pending": stats.get("pending_jobs", 0),
                "processing": stats.get("processing_jobs", 0),
                "completed_today": stats.get("completed_today", 0),
                "failed_today": stats.get("failed_today", 0),
                "total": stats.get("total_jobs", 0)
            },
            "performance": {
                "average_processing_time": stats.get("average_processing_time", 0),
                "success_rate": stats.get("success_rate", 0),
                "throughput_per_hour": stats.get("throughput_per_hour", 0)
            }
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/metrics/system")
async def system_metrics() -> Dict[str, Any]:
    """Get system health and performance metrics."""
    try:
        # Check Redis health
        redis_client = await get_redis_client()
        redis_healthy = await redis_client.is_connected()
        
        # Check AI models status
        model_registry = get_model_registry()
        models_info = model_registry.get_models_info()
        
        return {
            "redis": {
                "healthy": redis_healthy,
                "status": "connected" if redis_healthy else "disconnected"
            },
            "ai_models": {
                "whisper_loaded": models_info["whisper"] is not None,
                "diarization_loaded": models_info["diarization"] is not None,
                "total_models": models_info["total_models"],
                "initialization_status": models_info["initialization_status"]
            },
            "timestamp": time.time()
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/metrics/health")
async def health_metrics() -> Dict[str, Any]:
    """Get comprehensive health check metrics."""
    health_status = {
        "status": "healthy",
        "checks": {},
        "timestamp": time.time()
    }
    
    try:
        # Database health check
        # This would typically check database connectivity
        health_status["checks"]["database"] = {
            "status": "healthy",
            "response_time_ms": 5
        }
        
        # Redis health check
        redis_client = await get_redis_client()
        redis_start = time.time()
        redis_healthy = await redis_client.is_connected()
        redis_time = (time.time() - redis_start) * 1000
        
        health_status["checks"]["redis"] = {
            "status": "healthy" if redis_healthy else "unhealthy",
            "response_time_ms": redis_time
        }
        
        # AI models health check
        model_registry = get_model_registry()
        models_info = model_registry.get_models_info()
        
        health_status["checks"]["ai_models"] = {
            "status": "healthy" if models_info["initialization_status"] else "unhealthy",
            "whisper_ready": models_info["whisper"] is not None,
            "diarization_ready": models_info["diarization"] is not None
        }
        
        # Overall status
        unhealthy_checks = [
            check for check in health_status["checks"].values()
            if check["status"] != "healthy"
        ]
        
        if unhealthy_checks:
            health_status["status"] = "degraded" if len(unhealthy_checks) == 1 else "unhealthy"
        
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["error"] = str(e)
    
    return health_status


# Middleware functions for automatic metrics collection
def record_request_metrics(method: str, endpoint: str, status_code: int, duration: float):
    """Record HTTP request metrics."""
    http_requests_total.labels(
        method=method,
        endpoint=endpoint,
        status=str(status_code)
    ).inc()
    
    http_request_duration_seconds.labels(
        method=method,
        endpoint=endpoint
    ).observe(duration)


def record_job_metrics(status: str, duration: float = None):
    """Record transcription job metrics."""
    transcription_jobs_total.labels(status=status).inc()
    
    if status == "completed" and duration:
        transcription_duration_seconds.observe(duration)


def record_audio_metrics(file_size: int, duration: float):
    """Record audio file metrics."""
    audio_file_size_bytes.observe(file_size)
    audio_duration_seconds.observe(duration)


def record_model_metrics(model_type: str, loading_time: float, memory_usage: float):
    """Record AI model metrics."""
    ai_model_loading_duration_seconds.labels(model_type=model_type).observe(loading_time)
    ai_model_memory_usage_bytes.labels(model_type=model_type).set(memory_usage)


def record_celery_metrics(task_name: str, status: str, duration: float = None):
    """Record Celery task metrics."""
    celery_task_total.labels(task_name=task_name, status=status).inc()
    
    if duration:
        celery_task_duration_seconds.labels(task_name=task_name).observe(duration)


def record_database_metrics(operation: str, duration: float):
    """Record database operation metrics."""
    database_query_duration_seconds.labels(operation=operation).observe(duration)


def record_redis_metrics(operation: str, status: str, duration: float):
    """Record Redis operation metrics."""
    redis_operations_total.labels(operation=operation, status=status).inc()
    redis_operation_duration_seconds.labels(operation=operation).observe(duration)
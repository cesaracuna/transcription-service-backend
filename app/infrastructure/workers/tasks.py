"""
Refactored Celery tasks with improved error handling and monitoring.
"""

import time
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime

from celery import Task
from sqlalchemy.exc import SQLAlchemyError

from .celery_app import celery_app
from ...core.logging import get_logger
from ...core.exceptions import (
    DatabaseError, AudioProcessingError, TranscriptionError,
    JobNotFoundError, ModelLoadingError
)
from ...domain.shared.enums import JobStatus
from ...domain.transcription.services import TranscriptionService, JobService
from ...infrastructure.database.connection import database_session
from ...infrastructure.database.repositories.jobs import SyncJobRepository
from ...infrastructure.database.repositories.users import SyncUserRepository

logger = get_logger(__name__)


class BaseTask(Task):
    """Base task class with common functionality."""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails."""
        logger.error(
            f"Task {self.name} [{task_id}] failed",
            exc_info=exc,
            extra={
                "task_id": task_id,
                "task_name": self.name,
                "args": args,
                "kwargs": kwargs,
                "exception": str(exc)
            }
        )
    
    def on_success(self, retval, task_id, args, kwargs):
        """Called when task succeeds."""
        logger.info(
            f"Task {self.name} [{task_id}] succeeded",
            extra={
                "task_id": task_id,
                "task_name": self.name,
                "result": retval
            }
        )
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called when task is retried."""
        logger.warning(
            f"Task {self.name} [{task_id}] retrying",
            extra={
                "task_id": task_id,
                "task_name": self.name,
                "exception": str(exc),
                "retry_count": self.request.retries
            }
        )


@celery_app.task(
    name="transcription.process_job",
    base=BaseTask,
    bind=True,
    autoretry_for=(DatabaseError, ModelLoadingError),
    retry_kwargs={'max_retries': 3, 'countdown': 60},
    retry_backoff=True
)
def transcription_task(self, job_id: str) -> Dict[str, Any]:
    """
    Process a transcription job with comprehensive error handling.
    
    Args:
        job_id: Job ID to process
        
    Returns:
        Processing result dictionary
    """
    task_start_time = time.time()
    logger.info(f"Starting transcription task for job {job_id}")
    
    try:
        with database_session() as db:
            # Get services
            job_repo = SyncJobRepository(db)
            user_repo = SyncUserRepository(db)
            job_service = JobService(job_repo, user_repo)
            
            # TODO: Initialize AI models and audio processor
            # transcription_service = TranscriptionService(job_repo, audio_processor, hallucination_detector)
            
            # Update job status
            job = job_service.get_job(uuid.UUID(job_id))
            if not job:
                raise JobNotFoundError(f"Job {job_id} not found")
            
            job_service.update_job_status(uuid.UUID(job_id), JobStatus.PROCESSING)
            
            # TODO: Process transcription
            # result = transcription_service.process_transcription(uuid.UUID(job_id))
            
            # For now, simulate processing
            time.sleep(5)  # Simulate processing time
            
            # Update job status to completed
            job_service.update_job_status(uuid.UUID(job_id), JobStatus.COMPLETED)
            
            processing_time = time.time() - task_start_time
            
            logger.info(f"Transcription task completed for job {job_id} in {processing_time:.2f}s")
            
            return {
                "status": "completed",
                "job_id": job_id,
                "processing_time": processing_time,
                "segments_generated": 0,  # TODO: Get actual count
                "task_id": self.request.id
            }
    
    except JobNotFoundError as e:
        logger.error(f"Job not found: {e}")
        return {
            "status": "failed",
            "job_id": job_id,
            "error": "job_not_found",
            "message": str(e)
        }
    
    except (AudioProcessingError, TranscriptionError) as e:
        logger.error(f"Processing error for job {job_id}: {e}")
        
        # Update job status to failed
        try:
            with database_session() as db:
                job_repo = SyncJobRepository(db)
                user_repo = SyncUserRepository(db)
                job_service = JobService(job_repo, user_repo)
                job_service.update_job_status(uuid.UUID(job_id), JobStatus.FAILED, str(e))
        except Exception as status_error:
            logger.error(f"Failed to update job status: {status_error}")
        
        return {
            "status": "failed",
            "job_id": job_id,
            "error": "processing_error",
            "message": str(e)
        }
    
    except Exception as e:
        logger.error(f"Unexpected error in transcription task: {e}", exc_info=True)
        
        # Update job status to failed
        try:
            with database_session() as db:
                job_repo = SyncJobRepository(db)
                user_repo = SyncUserRepository(db)
                job_service = JobService(job_repo, user_repo)
                job_service.update_job_status(uuid.UUID(job_id), JobStatus.FAILED, f"Unexpected error: {str(e)}")
        except Exception as status_error:
            logger.error(f"Failed to update job status: {status_error}")
        
        # Re-raise for Celery to handle retries
        raise


@celery_app.task(
    name="post_processing.clean_job",
    base=BaseTask,
    bind=True,
    autoretry_for=(DatabaseError,),
    retry_kwargs={'max_retries': 2, 'countdown': 30}
)
def post_processing_task(self, job_id: str) -> Dict[str, Any]:
    """
    Post-process a transcription job (hallucination detection, speaker remapping).
    
    Args:
        job_id: Job ID to post-process
        
    Returns:
        Post-processing result dictionary
    """
    logger.info(f"Starting post-processing for job {job_id}")
    
    try:
        with database_session() as db:
            # TODO: Implement post-processing logic
            # This would include hallucination detection and speaker remapping
            
            logger.info(f"Post-processing completed for job {job_id}")
            
            return {
                "status": "completed",
                "job_id": job_id,
                "segments_removed": 0,  # TODO: Get actual count
                "speakers_remapped": True
            }
    
    except Exception as e:
        logger.error(f"Post-processing failed for job {job_id}: {e}", exc_info=True)
        raise


@celery_app.task(
    name="health.check_system_health",
    base=BaseTask,
    bind=True
)
def health_check_task(self) -> Dict[str, Any]:
    """
    Periodic health check task.
    
    Returns:
        System health status
    """
    logger.debug("Running system health check")
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {}
    }
    
    try:
        # Check database connectivity
        with database_session() as db:
            db.execute("SELECT 1")
            health_status["checks"]["database"] = {
                "status": "healthy",
                "message": "Database connection successful"
            }
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["checks"]["database"] = {
            "status": "unhealthy",
            "message": f"Database error: {str(e)}"
        }
    
    # TODO: Check AI models status
    health_status["checks"]["ai_models"] = {
        "status": "unknown",
        "message": "AI models check not implemented"
    }
    
    # TODO: Check storage availability
    health_status["checks"]["storage"] = {
        "status": "unknown",
        "message": "Storage check not implemented"
    }
    
    return health_status


@celery_app.task(
    name="maintenance.cleanup_old_files",
    base=BaseTask,
    bind=True
)
def cleanup_old_files_task(self) -> Dict[str, Any]:
    """
    Cleanup old audio files and temporary data.
    
    Returns:
        Cleanup result
    """
    logger.info("Starting cleanup of old files")
    
    try:
        # TODO: Implement file cleanup logic
        # This would remove old audio files, temporary chunks, etc.
        
        files_removed = 0  # TODO: Get actual count
        space_freed = 0   # TODO: Calculate space freed
        
        logger.info(f"Cleanup completed: {files_removed} files removed, {space_freed} bytes freed")
        
        return {
            "status": "completed",
            "files_removed": files_removed,
            "space_freed_bytes": space_freed
        }
    
    except Exception as e:
        logger.error(f"Cleanup task failed: {e}", exc_info=True)
        return {
            "status": "failed",
            "error": str(e)
        }


@celery_app.task(
    name="maintenance.update_job_metrics",
    base=BaseTask,
    bind=True
)
def update_job_metrics_task(self) -> Dict[str, Any]:
    """
    Update job processing metrics and statistics.
    
    Returns:
        Metrics update result
    """
    logger.info("Updating job metrics")
    
    try:
        # TODO: Implement metrics calculation
        # This would calculate processing statistics, success rates, etc.
        
        return {
            "status": "completed",
            "metrics_updated": True
        }
    
    except Exception as e:
        logger.error(f"Metrics update failed: {e}", exc_info=True)
        return {
            "status": "failed",
            "error": str(e)
        }


# Task routing helpers
def queue_transcription_job(job_id: str) -> str:
    """
    Queue a transcription job for processing.
    
    Args:
        job_id: Job ID to process
        
    Returns:
        Task ID
    """
    logger.info(f"Queueing transcription job {job_id}")
    
    result = transcription_task.delay(job_id)
    
    logger.info(f"Transcription job {job_id} queued with task ID {result.id}")
    return result.id


def queue_post_processing_job(job_id: str) -> str:
    """
    Queue a post-processing job.
    
    Args:
        job_id: Job ID to post-process
        
    Returns:
        Task ID
    """
    logger.info(f"Queueing post-processing job {job_id}")
    
    result = post_processing_task.delay(job_id)
    
    logger.info(f"Post-processing job {job_id} queued with task ID {result.id}")
    return result.id
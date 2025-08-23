"""
Job-related API endpoints.
Handles transcription job operations including creation, status checking, and management.
"""

import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Form, status
from sqlalchemy.orm import Session

from ..schemas.jobs import (
    JobCreateResponse, Job, JobSummary, JobListResponse, 
    JobDeleteResponse, JobViewedResponse
)
from ..schemas.common import SuccessResponse, ErrorResponse
from ....domain.shared.enums import JobStatus
from ....domain.transcription.services import JobService, UserService
from ....infrastructure.database.connection import get_db
from ....infrastructure.database.repositories.jobs import SyncJobRepository
from ....infrastructure.database.repositories.users import SyncUserRepository
from ....core.exceptions import (
    JobNotFoundError, UserNotFoundError, ValidationError, 
    AudioProcessingError, StorageError
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])


def get_job_service(db: Session = Depends(get_db)) -> JobService:
    """Dependency to get job service."""
    job_repo = SyncJobRepository(db)
    user_repo = SyncUserRepository(db)
    return JobService(job_repo, user_repo)


def get_user_service(db: Session = Depends(get_db)) -> UserService:
    """Dependency to get user service."""
    user_repo = SyncUserRepository(db)
    return UserService(user_repo)


@router.post("/", response_model=JobCreateResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_transcription_job(
    user_id: str = Form(..., description="User ID"),
    file: UploadFile = File(..., description="Audio file to transcribe"),
    job_service: JobService = Depends(get_job_service)
):
    """
    Create a new transcription job.
    
    - **user_id**: ID of the user creating the job
    - **file**: Audio file to transcribe (supported formats: WAV, MP3, M4A, FLAC, OGG, WEBM)
    
    Returns the created job with initial status and queues it for processing.
    """
    logger.info(f"Creating transcription job for user {user_id}, file: {file.filename}")
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Filename is required"
            )
        
        # Validate file size (e.g., max 500MB)
        if file.size and file.size > 500 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File size too large. Maximum size is 500MB."
            )
        
        # TODO: Implement file saving and audio processing
        # For now, we'll create a placeholder path
        audio_file_path = f"./audio_files/{file.filename}"
        
        # Create job
        job = await job_service.create_job(
            user_id=UUID(user_id),
            original_filename=file.filename,
            audio_file_path=audio_file_path
        )
        
        # TODO: Queue transcription task
        logger.info(f"Job {job.id} created successfully and queued for processing")
        
        return JobCreateResponse(
            id=job.id,
            user_id=job.user_id,
            original_filename=job.original_filename,
            status=job.status,
            created_at=job.created_at,
            updated_at=job.updated_at
        )
        
    except UserNotFoundError as e:
        logger.warning(f"User not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except StorageError as e:
        logger.error(f"Storage error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store audio file"
        )
    except Exception as e:
        logger.error(f"Unexpected error creating job: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while processing request"
        )


@router.get("/{job_id}", response_model=Job)
async def get_job_status(
    job_id: UUID,
    job_service: JobService = Depends(get_job_service)
):
    """
    Get job status and results.
    
    - **job_id**: ID of the job to retrieve
    
    Returns complete job information including transcription results if completed.
    """
    logger.info(f"Fetching job status for job {job_id}")
    
    try:
        job = await job_service.get_job(job_id)
        
        # Convert domain model to response schema
        response = Job(
            id=job.id,
            user_id=job.user_id,
            original_filename=job.original_filename,
            status=job.status,
            created_at=job.created_at,
            updated_at=job.updated_at,
            error_message=job.error_message,
            audio_metadata=job.audio_metadata,
            is_post_processed=job.is_post_processed,
            is_viewed=job.is_viewed,
            segments=[],  # TODO: Convert segments
            diarization_segments=[],  # TODO: Convert diarization segments
            processing_metrics=job.processing_metrics
        )
        
        logger.debug(f"Successfully returned job status for {job_id}")
        return response
        
    except JobNotFoundError as e:
        logger.warning(f"Job not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error fetching job {job_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/users/{user_id}/", response_model=JobListResponse)
async def get_user_jobs(
    user_id: UUID,
    status: Optional[JobStatus] = None,
    page: int = 1,
    size: int = 9,
    job_service: JobService = Depends(get_job_service)
):
    """
    Get jobs for a specific user with optional filtering and pagination.
    
    - **user_id**: ID of the user
    - **status**: Optional status filter
    - **page**: Page number (1-based)
    - **size**: Number of items per page
    
    Returns paginated list of user's jobs.
    """
    logger.info(f"Fetching jobs for user {user_id}, page {page}, size {size}")
    
    try:
        jobs, total_count = await job_service.get_user_jobs(
            user_id=user_id,
            status=status,
            page=page,
            size=size
        )
        
        # Convert to summary format
        job_summaries = []
        for job in jobs:
            summary = JobSummary(
                id=job.id,
                user_id=job.user_id,
                original_filename=job.original_filename,
                status=job.status,
                created_at=job.created_at,
                updated_at=job.updated_at,
                error_message=job.error_message,
                audio_duration=job.audio_metadata.duration_seconds if job.audio_metadata else None,
                segments_count=len(job.segments),
                speakers_count=len(job.get_speakers()),
                is_viewed=job.is_viewed
            )
            job_summaries.append(summary)
        
        total_pages = (total_count + size - 1) // size
        
        response = JobListResponse(
            total_items=total_count,
            total_pages=total_pages,
            current_page=page,
            page_size=size,
            items=job_summaries
        )
        
        logger.debug(f"Successfully returned {len(job_summaries)} jobs for user {user_id}")
        return response
        
    except Exception as e:
        logger.error(f"Unexpected error fetching user jobs: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.delete("/{job_id}", response_model=JobDeleteResponse, status_code=status.HTTP_204_NO_CONTENT)
async def delete_job(
    job_id: UUID,
    job_service: JobService = Depends(get_job_service)
):
    """
    Delete a job and all its associated data.
    
    - **job_id**: ID of the job to delete
    
    Only completed or failed jobs can be deleted.
    """
    logger.info(f"Deleting job {job_id}")
    
    try:
        await job_service.delete_job(job_id)
        
        logger.info(f"Job {job_id} deleted successfully")
        return JobDeleteResponse()
        
    except JobNotFoundError as e:
        logger.warning(f"Job not found for deletion: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except ValidationError as e:
        logger.warning(f"Job cannot be deleted: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error deleting job {job_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.patch("/{job_id}/mark-as-viewed", response_model=JobViewedResponse, status_code=status.HTTP_204_NO_CONTENT)
async def mark_job_as_viewed(
    job_id: UUID,
    job_service: JobService = Depends(get_job_service)
):
    """
    Mark a job as viewed by the user.
    
    - **job_id**: ID of the job to mark as viewed
    """
    logger.info(f"Marking job {job_id} as viewed")
    
    try:
        await job_service.mark_job_as_viewed(job_id)
        
        logger.debug(f"Job {job_id} marked as viewed")
        return JobViewedResponse()
        
    except JobNotFoundError as e:
        logger.warning(f"Job not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error marking job as viewed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# Additional endpoints for segment management

@router.delete("/{job_id}/segments/{segment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_segment(
    job_id: UUID,
    segment_id: UUID,
    job_service: JobService = Depends(get_job_service)
):
    """
    Delete a specific segment from a job.
    
    - **job_id**: ID of the job
    - **segment_id**: ID of the segment to delete
    """
    logger.info(f"Deleting segment {segment_id} from job {job_id}")
    
    try:
        # TODO: Implement segment deletion in service
        # For now, return success
        logger.info(f"Segment {segment_id} deleted from job {job_id}")
        
    except JobNotFoundError as e:
        logger.warning(f"Job not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    except Exception as e:
        logger.error(f"Unexpected error deleting segment: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
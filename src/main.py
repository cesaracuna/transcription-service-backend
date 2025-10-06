import uuid
import os
import logging
import math

from .config import initialize_config, get_config
from .logging_config import setup_logging
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query
from sqlalchemy.orm import Session
from .security import (
    comprehensive_file_validation, generate_secure_filename, 
    create_secure_storage_path, SecurityError
)
from pydub import AudioSegment
from pydub.effects import normalize

# Initialize configuration
config = get_config()
initialize_config()

from celery import chain as celery_chain
from .celery_app import celery_app
from .database import get_db
from . import models
from . import schemas
from typing import Optional

app = FastAPI(
    title=config.APP_NAME,
    version=config.APP_VERSION,
    debug=config.DEBUG
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure storage paths
AUDIO_STORAGE_PATH = str(config.AUDIO_STORAGE_PATH)
logging.info(f"Audio storage path: {AUDIO_STORAGE_PATH}")

# Configure logging for the API process
setup_logging(process_name="FastAPI", log_level=config.LOG_LEVEL)

# create_transcription_job
@app.post("/jobs/", response_model=schemas.JobCreateResponse, status_code=202)
async def create_transcription_job(
        user_id: str = Form(...),
        file: UploadFile = File(...),
        db: Session = Depends(get_db)
):
    logging.info(f"Received transcription job request from user {user_id}, "
                f"filename: {file.filename}, content_type: {file.content_type}")
    
    try:
        # 1. Initial security validations
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        # Validate and sanitize user_id format
        try:
            # Sanitize the user_id (remove spaces, convert to lowercase)
            sanitized_user_id = user_id.strip().lower()
            logging.debug(f"Original user_id: '{user_id}', sanitized: '{sanitized_user_id}'")
            
            user_uuid = uuid.UUID(sanitized_user_id)
            logging.debug(f"Valid UUID received: {user_uuid}")
        except ValueError as e:
            logging.error(f"Invalid UUID format for user_id '{user_id}' (length: {len(user_id)}): {e}")
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid user ID: '{user_id}' is not a valid UUID. Expected format: 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx'"
            )
        
        # Verify file size before processing
        if file.size and file.size > config.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413, 
                detail=f"File exceeds maximum allowed size ({config.MAX_FILE_SIZE_MB}MB)"
            )
        
        # 2. Generate secure filename
        secure_filename = generate_secure_filename(file.filename)
        original_file_path = create_secure_storage_path(AUDIO_STORAGE_PATH, secure_filename)
        
        logging.info(f"Saving uploaded file securely: {file.filename} -> {secure_filename}")
        logging.debug(f"Target path: {original_file_path}")
        
        # 3. Save file to disk with streaming for large files
        bytes_written = 0
        try:
            with open(original_file_path, "wb") as buffer:
                while True:
                    chunk = await file.read(8192)  # 8KB chunks
                    if not chunk:
                        break
                    buffer.write(chunk)
                    bytes_written += len(chunk)
                    
                    # Verify limit during writing
                    if bytes_written > config.MAX_FILE_SIZE_MB * 1024 * 1024:
                        # Clean up partial file
                        try:
                            os.remove(original_file_path)
                        except:
                            pass
                        raise HTTPException(
                            status_code=413, 
                            detail=f"File exceeds maximum allowed size ({config.MAX_FILE_SIZE_MB}MB)"
                        )
                        
        except Exception as e:
            # Clean up partial file in case of error
            try:
                if os.path.exists(original_file_path):
                    os.remove(original_file_path)
            except:
                pass
            raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
        
        file_size = os.path.getsize(original_file_path)
        logging.info(f"File saved successfully: {file_size} bytes ({file_size/(1024*1024):.2f} MB)")

        # 4. Create database record immediately after file is saved
        logging.debug(f"Creating job record in database for user {user_uuid}")
        new_job = models.Job(
            user_id=user_uuid,
            original_filename=file.filename,  # Keep original filename for user reference
            audio_file_path=original_file_path,  # Store original file path initially
            status=models.JobStatus.PENDING
        )
        db.add(new_job)
        db.commit()
        db.refresh(new_job)

        logging.info(f"Job created successfully with ID: {new_job.id}")

        # 5. Queue audio validation and preprocessing task first
        logging.info(f"Queueing audio processing pipeline chain for job {new_job.id}")
        workflow = celery_chain(
            celery_app.signature("tasks.audio_preprocessing", args=[str(new_job.id)], immutable=True),
            celery_app.signature("tasks.speaker_segmentation", args=[str(new_job.id)], immutable=True),
            celery_app.signature("tasks.generate_transcript", args=[str(new_job.id)], immutable=True),
            celery_app.signature("tasks.complete_processing", args=[str(new_job.id)], immutable=True)
        )
        result = workflow.apply_async()
        
        logging.info(f"Transcription job {new_job.id} created and queued successfully for user {user_uuid}")
        return new_job
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except SecurityError as e:
        # user_uuid might not be defined if UUID validation failed
        user_ref = user_uuid if 'user_uuid' in locals() else user_id
        logging.error(f"Security error for user {user_ref}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Security error: {str(e)}")
    except Exception as e:
        # user_uuid might not be defined if UUID validation failed
        user_ref = user_uuid if 'user_uuid' in locals() else user_id
        logging.error(f"Unexpected error creating transcription job for user {user_ref}: {str(e)}", exc_info=True)
        # Clean up any partially created files
        try:
            if 'original_file_path' in locals() and os.path.exists(original_file_path):
                os.remove(original_file_path)
                logging.info(f"Cleaned up file after error: {original_file_path}")
        except:
            pass
        raise HTTPException(status_code=500, detail="Internal server error processing request")


@app.get("/jobs/{job_id}", response_model=schemas.Job)
def get_job_status(job_id: uuid.UUID, db: Session = Depends(get_db)):
    logging.info(f"Fetching job status for job ID: {job_id}")
    
    try:
        job = db.query(models.Job).filter(models.Job.id == job_id).first()
        if not job:
            logging.warning(f"Job not found: {job_id}")
            raise HTTPException(status_code=404, detail="Job not found")

        logging.debug(f"Job {job_id} found. Status: {job.status}, Segments: {len(job.segments)}")
        response_data = schemas.Job.from_orm(job)

        if job.status == models.JobStatus.COMPLETED:
            logging.debug(f"Building full text for completed job {job_id}")
            full_text_parts = []
            last_speaker = None
            sorted_segments = sorted(job.segments, key=lambda s: s.start_timestamp)
            
            for segment in sorted_segments:
                if last_speaker != segment.speaker:
                    if last_speaker is not None: 
                        full_text_parts.append("\n\n")
                    full_text_parts.append(f"{segment.speaker}:\n{segment.text}")
                    last_speaker = segment.speaker
                else:
                    full_text_parts.append(f" {segment.text}")
            
            response_data.full_text = "".join(full_text_parts)
            logging.debug(f"Full text built for job {job_id}: {len(response_data.full_text)} characters")
            
        logging.debug(f"Successfully returning job status for {job_id}")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Unexpected error fetching job status for {job_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/users/{user_id}/jobs/", response_model=schemas.PaginatedJobResponse)
def get_user_jobs(
    user_id: uuid.UUID,
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(9, ge=1, le=100, description="Page size"),
    status: Optional[models.JobStatus] = Query(None, description="Filter by job status")
):
    """
    Returns a paginated list of jobs for a specific user.
    Can be filtered by status. If no status is provided, returns all jobs.
    """
    logging.info(f"Fetching page {page} of completed jobs for user_id: {user_id}")

    # Base query - exclude DELETED jobs by default
    base_query = db.query(models.Job).filter(
        models.Job.user_id == user_id,
        models.Job.status != models.JobStatus.DELETED
    )

    # Apply filter if we get a status
    if status:
        base_query = base_query.filter(models.Job.status == status)

    # Count total items that meet the criteria
    total_items = base_query.count()

    jobs = base_query.order_by(models.Job.created_at.desc()).offset((page - 1) * size).limit(size).all()

    return {
        "total_items": total_items,
        "total_pages": math.ceil(total_items / size),
        "current_page": page,
        "items": jobs
    }


@app.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: uuid.UUID, db: Session = Depends(get_db)):
    """
    Deletes a job and all its associated segments and audio file.
    """
    logging.info(f"Received request to delete job ID: {job_id}")
    
    try:
        job = db.query(models.Job).filter(models.Job.id == job_id).first()
        if not job:
            logging.warning(f"Job ID {job_id} not found for deletion.")
            raise HTTPException(status_code=404, detail="Job not found")

        # Delete audio file from disk
        try:
            if os.path.exists(job.audio_file_path):
                os.remove(job.audio_file_path)
                logging.info(f"Deleted audio file: {job.audio_file_path}")
        except OSError as e:
            logging.error(f"Error deleting audio file {job.audio_file_path}: {e}")

        # Delete job from database (segments will be deleted automatically via cascade)
        db.delete(job)
        db.commit()
        logging.info(f"Successfully deleted job ID: {job_id} from database")
        return
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error deleting job {job_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.delete("/jobs/{job_id}/segments/{segment_id}", status_code=204)
def delete_segment(job_id: uuid.UUID, segment_id: uuid.UUID, db: Session = Depends(get_db)):
    """
    Deletes a single segment belonging to a specific job.
    """
    logging.info(f"Received request to delete Segment ID: {segment_id} from Job ID: {job_id}")

    # Make a safer query, ensuring that the segment
    # actually belongs to the specified job.
    segment = db.query(models.Segment).filter(
        models.Segment.id == segment_id,
        models.Segment.job_id == job_id
    ).first()

    if not segment:
        logging.warning(f"Segment ID {segment_id} not found or does not belong to Job {job_id}.")
        raise HTTPException(status_code=404, detail="Segment not found within the specified job")

    db.delete(segment)
    db.commit()
    logging.info(f"Successfully deleted Segment ID: {segment_id} from database.")

    return


@app.patch("/jobs/{job_id}/mark-as-viewed", status_code=204)
def mark_job_as_viewed(job_id: uuid.UUID, db: Session = Depends(get_db)):
    """
    Marks a specific job as viewed.
    """
    logging.info(f"Received request to mark Job ID as viewed: {job_id}")
    job = db.query(models.Job).filter(models.Job.id == job_id).first()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    job.is_viewed = True
    db.commit()

    # 204 Success, with no content to return!
    return


@app.get("/jobs/{job_id}/paragraphs", response_model=schemas.ParagraphsResponse)
def get_job_paragraphs(job_id: uuid.UUID, db: Session = Depends(get_db)):
    """
    Retrieves all formatted paragraphs for a specific transcription job.

    Returns paragraphs with original text, formatted text, formatting status,
    differences information, and speaker details ordered by timestamp.

    Args:
        job_id: UUID of the job to retrieve paragraphs for
        db: Database session dependency

    Returns:
        ParagraphsResponse: Job details with all paragraphs and formatting information

    Raises:
        HTTPException: 404 if job not found, 500 for internal errors
    """
    logging.info(f"Fetching paragraphs for job ID: {job_id}")

    try:
        # Verify job exists
        job = db.query(models.Job).filter(models.Job.id == job_id).first()
        if not job:
            logging.warning(f"Job not found: {job_id}")
            raise HTTPException(status_code=404, detail="Job not found")

        # Get all paragraphs for this job, ordered by timestamp
        paragraphs = db.query(models.Paragraph).filter(
            models.Paragraph.job_id == job_id
        ).order_by(models.Paragraph.start_timestamp).all()

        logging.info(f"Found {len(paragraphs)} paragraphs for job {job_id}")

        # Build response with job info and paragraphs
        response_data = {
            "job_id": job.id,
            "job_status": job.status,
            "original_filename": job.original_filename,
            "audio_duration": job.audio_duration,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "total_paragraphs": len(paragraphs),
            "paragraphs": paragraphs
        }

        logging.debug(f"Successfully returning {len(paragraphs)} paragraphs for job {job_id}")
        return response_data

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Unexpected error fetching paragraphs for job {job_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.patch("/jobs/{job_id}/mark-as-deleted", status_code=204)
def mark_job_as_deleted(job_id: uuid.UUID, db: Session = Depends(get_db)):
    """
    Marks a specific job as viewed.
    """
    logging.info(f"Received request to mark Job ID as viewed: {job_id}")
    job = db.query(models.Job).filter(models.Job.id == job_id).first()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    job.status = models.JobStatus.DELETED
    db.commit()

    # 204 Success, with no content to return!
    return

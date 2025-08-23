import uuid
import os
import time
import logging
import subprocess
import math

from .logging_config import setup_logging
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query
from sqlalchemy.orm import Session

from celery import Celery
from .database import get_db
from . import models
from . import schemas
from typing import Optional

client_celery_app = Celery(
    "client",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

app = FastAPI()

# 2. Define de dónde permites que vengan las peticiones
# origins = ["*"]
origins = [
    "http://localhost",
    "http://localhost:8080",
    "https://localhost:7003",
    "http://localhost:5149",
]

# 3. Añade el middleware a tu aplicación
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"], # Permite todas las cabeceras
)

AUDIO_STORAGE_PATH = "./audio_files"
os.makedirs(AUDIO_STORAGE_PATH, exist_ok=True)

# Configurar el logging para el proceso de la API
setup_logging(process_name="FastAPI")

def convert_to_wav_and_resample(input_audio_path, output_dir):
    """Converts audio to a standardized 16kHz mono WAV format."""
    logging.info(f"Starting audio format standardization...")
    
    try:
        # Input validation
        if not os.path.exists(input_audio_path):
            logging.error(f"Input audio file not found: {input_audio_path}")
            return None
            
        if not os.path.exists(output_dir):
            logging.info(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

        # Generate output path
        base_name = os.path.splitext(os.path.basename(input_audio_path))[0]
        output_wav_path = os.path.join(output_dir, f"{base_name}_16khz_mono.wav")
        
        logging.info(f"Converting audio file:")
        logging.info(f"  Input: {input_audio_path}")
        logging.info(f"  Output: {output_wav_path}")

        # Check if output file already exists
        if os.path.exists(output_wav_path):
            logging.info(f"Output file already exists, will be overwritten")

        # Build FFmpeg command
        command = [
            "ffmpeg",
            "-i", input_audio_path,
            "-map", "0:a",          # Map all audio tracks from input
            "-ac", "1",             # Mix all channels to mono
            "-ar", "16000",         # Set sample rate to 16kHz
            "-c:a", "pcm_s16le",    # Use standard WAV codec
            "-y", output_wav_path   # Overwrite output file
        ]

        logging.debug(f"Executing FFmpeg command: {' '.join(command)}")
        start_time = time.time()
        
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        
        conversion_time = time.time() - start_time
        logging.info(f"Audio conversion completed successfully in {conversion_time:.2f}s")
        logging.debug(f"FFmpeg stdout: {result.stdout}")
        
        # Verify output file was created
        if os.path.exists(output_wav_path):
            file_size = os.path.getsize(output_wav_path)
            logging.info(f"Output file created: {output_wav_path} ({file_size} bytes)")
            return output_wav_path
        else:
            logging.error("Output file was not created despite successful FFmpeg execution")
            return None
            
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg conversion failed with return code {e.returncode}")
        logging.error(f"FFmpeg stderr: {e.stderr}")
        logging.error(f"FFmpeg stdout: {e.stdout}")
        logging.error("Please ensure ffmpeg is installed and accessible in your PATH")
        return None
    except Exception as e:
        logging.error(f"Unexpected error during audio conversion: {str(e)}", exc_info=True)
        return None


@app.post("/jobs/", response_model=schemas.JobCreateResponse, status_code=202)
async def create_transcription_job(
        user_id: str = Form(...),
        file: UploadFile = File(...),
        db: Session = Depends(get_db)
):
    logging.info(f"Received transcription job request from user {user_id}, "
                f"filename: {file.filename}, size: {file.size} bytes")
    
    try:
        # 1. Save uploaded file
        unique_filename = f"{uuid.uuid4()}-{file.filename}"
        original_file_path = os.path.join(AUDIO_STORAGE_PATH, unique_filename)
        
        logging.debug(f"Saving uploaded file to: {original_file_path}")
        
        with open(original_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            
        file_size = os.path.getsize(original_file_path)
        logging.info(f"File saved successfully: {file_size} bytes")

        # 2. Convert to standardized format
        logging.info("Starting audio format conversion...")
        standardized_wav_path = convert_to_wav_and_resample(original_file_path, AUDIO_STORAGE_PATH)

        if not standardized_wav_path:
            logging.error(f"Audio conversion failed for file: {file.filename}")
            # Clean up original file
            try:
                os.remove(original_file_path)
                logging.debug("Cleaned up original file after conversion failure")
            except Exception as cleanup_error:
                logging.warning(f"Failed to clean up original file: {str(cleanup_error)}")
                
            raise HTTPException(
                status_code=400,
                detail="Failed to process the uploaded audio file. It might be corrupt or in an unsupported format.")

        # 3. Create database record
        logging.debug(f"Creating job record in database for user {user_id}")
        new_job = models.Job(
            user_id=uuid.UUID(user_id),
            original_filename=file.filename,
            audio_file_path=standardized_wav_path,
            status=models.JobStatus.PENDING
        )
        db.add(new_job)
        db.commit()
        db.refresh(new_job)
        
        logging.info(f"Job created successfully with ID: {new_job.id}")

        # 4. Queue transcription task
        logging.info(f"Queueing transcription task for job {new_job.id}")
        client_celery_app.send_task("jobs.transcribe", args=[str(new_job.id)])
        
        logging.info(f"Transcription job {new_job.id} queued successfully for user {user_id}")
        return new_job
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logging.error(f"Unexpected error creating transcription job for user {user_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error while processing the request")


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

    # consulta base
    base_query = db.query(models.Job).filter(models.Job.user_id == user_id)

    # aplicar filtro si nos dan un estado
    if status:
        base_query = base_query.filter(models.Job.status == status)

    # contamos el total de items que cumplen con el criterio
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

    # Hacemos una consulta más segura, asegurándonos de que el segmento
    # realmente pertenece al job especificado.
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


@app.get("/users/check/{user_name}")
def check_user_exists(user_name: str, db: Session = Depends(get_db)):
    """
    Verifies if a user exists by username.
    """
    logging.info(f"Checking if user exists: {user_name}")
    
    try:
        user = db.query(models.User).filter(models.User.username == user_name).first()
        
        if user:
            logging.info(f"User found: {user_name}")
            return {"exists": True, "user_id": str(user.id)}
        else:
            logging.info(f"User not found: {user_name}")
            return {"exists": False}
            
    except Exception as e:
        logging.error(f"Error checking user existence for {user_name}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

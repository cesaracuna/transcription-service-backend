import eventlet
eventlet.monkey_patch()

# Standard and application libraries
import uuid
import os
import time
import torch
import logging
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import selectinload, Session
from sqlalchemy.exc import SQLAlchemyError

# Application libraries (Celery, DB)
from celery import Celery
from .logging_config import setup_logging
from .database import database_session
from .models import Job, Segment, DiarizationSegment, JobStatus, Hallucination, JobHallucination

# Refactored modules
from .audio_processing import initialize_ai_models, process_audio_pipeline
from .hallucination_detection import detect_sandwiched_hallucinations, get_hallucination_texts, get_primary_languages
from .utils import parse_timestamp_to_seconds

# --- Configuration and Initialization ---
setup_logging(process_name="CeleryWorker")
logger = logging.getLogger(__name__)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

# Initialize AI models with error handling
try:
    logger.info("Initializing AI models...")
    start_time = time.time()
    whisper_processor, whisper_model, diarization_pipeline = initialize_ai_models(DEVICE)
    initialization_time = time.time() - start_time
    logger.info(f"AI models initialized successfully in {initialization_time:.2f} seconds")
except Exception as e:
    logger.critical(f"Failed to initialize AI models: {e}", exc_info=True)
    raise

# Celery application setup
celery_app = Celery("tasks", broker="redis://localhost:6379/0", backend="redis://localhost:6379/0")
celery_app.conf.update(
    task_track_started=True, 
    worker_cancel_long_running_tasks_on_connection_loss=True
)
logger.info("Celery application configured successfully")


def update_job_status(db: Session, job: Job, status: JobStatus, error_message: Optional[str] = None) -> None:
    """
    Updates job status with proper logging and error handling.
    
    Args:
        db: Database session
        job: Job instance to update
        status: New status to set
        error_message: Optional error message if status is FAILED
    """
    try:
        old_status = job.status
        job.status = status
        if error_message:
            job.error_message = error_message
        
        db.commit()
        logger.info(f"Job {job.id} status updated: {old_status} -> {status}")
        
        if error_message:
            logger.error(f"Job {job.id} failed with error: {error_message}")
            
    except SQLAlchemyError as e:
        logger.error(f"Failed to update job {job.id} status to {status}: {e}", exc_info=True)
        db.rollback()
        raise


def update_speaker_remapping_in_diarization(db: Session, job_id: uuid.UUID, speaker_mapping: Dict[str, str]) -> int:
    """
    Updates speaker labels in diarization_segments table after remapping.
    
    Args:
        db: Database session
        job_id: Job ID
        speaker_mapping: Dictionary mapping old speaker labels to new ones
        
    Returns:
        Number of diarization segments updated
    """
    if not speaker_mapping:
        logger.debug(f"No speaker mapping provided for diarization segments in job {job_id}")
        return 0
    
    logger.info(f"Updating speaker labels in diarization segments for job {job_id}")
    logger.debug(f"Speaker mapping: {speaker_mapping}")
    
    try:
        updates_made = 0
        
        # Get all diarization segments for this job
        diarization_segments = db.query(DiarizationSegment).filter(
            DiarizationSegment.job_id == job_id
        ).all()
        
        logger.debug(f"Found {len(diarization_segments)} diarization segments to potentially update")
        
        for segment in diarization_segments:
            old_speaker = segment.speaker_tag
            if old_speaker in speaker_mapping:
                new_speaker = speaker_mapping[old_speaker]
                logger.debug(f"Updating diarization segment: {old_speaker} -> {new_speaker}")
                segment.speaker_tag = new_speaker
                updates_made += 1
        
        db.commit()
        logger.info(f"Successfully updated {updates_made} diarization segments with new speaker labels")
        return updates_made
        
    except SQLAlchemyError as e:
        logger.error(f"Failed to update speaker remapping in diarization segments for job {job_id}: {e}", exc_info=True)
        db.rollback()
        raise


def save_diarization_segments_with_status(db: Session, job: Job, diarization_data: List[Dict[str, Any]], status: str = 'active') -> int:
    """
    Saves diarization segments to database with specified status.
    
    Args:
        db: Database session
        job: Job instance
        diarization_data: List of diarization segment data dictionaries
        status: Status to assign to segments (default: 'active')
        
    Returns:
        Number of diarization segments saved
    """
    if not diarization_data:
        logger.info(f"No diarization segments to save for job {job.id}")
        return 0
    
    logger.info(f"Saving {len(diarization_data)} diarization segments with status '{status}' for job {job.id}")
    
    try:
        segments_created = 0
        batch_size = 100  # Process in batches for better memory management
        
        for i in range(0, len(diarization_data), batch_size):
            batch = diarization_data[i:i + batch_size]
            logger.debug(f"Processing diarization batch {i//batch_size + 1}/{(len(diarization_data)-1)//batch_size + 1}")
            
            for segment_data in batch:
                diarization_segment = DiarizationSegment(
                    job_id=job.id,
                    speaker_tag=segment_data['speaker'],
                    start_seconds=segment_data['start'],
                    end_seconds=segment_data['end'],
                    status=status
                )
                db.add(diarization_segment)
                segments_created += 1
            
            # Commit each batch
            db.commit()
            logger.debug(f"Committed batch of {len(batch)} diarization segments with status '{status}'")
        
        logger.info(f"  - Successfully saved {segments_created} diarization segments with status '{status}' for job {job.id}")
        return segments_created
        
    except SQLAlchemyError as e:
        logger.error(f"Failed to save diarization segments for job {job.id}: {e}", exc_info=True)
        db.rollback()
        raise


def mark_diarization_segments_as_processed(db: Session, job_id: uuid.UUID) -> int:
    """
    Marks all diarization segments for a job as processed.
    
    Args:
        db: Database session
        job_id: Job ID
        
    Returns:
        Number of segments marked as processed
    """
    logger.info(f"Marking diarization segments as processed for job {job_id}")
    
    try:
        # Update all diarization segments for this job to 'processed' status
        segments_updated = db.query(DiarizationSegment).filter(
            DiarizationSegment.job_id == job_id
        ).update(
            {"status": "processed"},
            synchronize_session=False
        )
        
        db.commit()
        logger.info(f"Successfully marked {segments_updated} diarization segments as processed")
        return segments_updated
        
    except SQLAlchemyError as e:
        logger.error(f"Failed to mark diarization segments as processed for job {job_id}: {e}", exc_info=True)
        db.rollback()
        raise


def save_transcription_segments(db: Session, job: Job, segments_data: List[Dict[str, Any]]) -> int:
    """
    Saves transcription segments to database with batch processing and logging.
    
    Args:
        db: Database session
        job: Job instance
        segments_data: List of segment data dictionaries
        
    Returns:
        Number of segments saved
    """
    logger.info(f"Saving {len(segments_data)} transcription segments for job {job.id}")
    
    try:
        segments_created = 0
        batch_size = 100  # Process in batches for better memory management
        
        for i in range(0, len(segments_data), batch_size):
            batch = segments_data[i:i + batch_size]
            logger.debug(f"Processing segment batch {i//batch_size + 1}/{(len(segments_data)-1)//batch_size + 1}")
            
            for seg_data in batch:
                segment = Segment(
                    job_id=job.id,
                    start_timestamp=seg_data['start'],
                    end_timestamp=seg_data['end'],
                    speaker=seg_data['speaker'],
                    text=seg_data['text'],
                    language=seg_data['language']
                )
                db.add(segment)
                segments_created += 1
            
            # Commit each batch
            db.commit()
            logger.debug(f"Committed batch of {len(batch)} segments")
        
        logger.info(f"  - Successfully saved {segments_created} segments for job {job.id}")
        return segments_created
        
    except SQLAlchemyError as e:
        logger.error(f"Failed to save segments for job {job.id}: {e}", exc_info=True)
        db.rollback()
        raise


@celery_app.task(name="jobs.post_process", bind=True)
def post_process_job(self, job_id: str):
    """
    Cleans a completed transcription using contextual "sandwich" logic.
    Enhanced with comprehensive logging and better error handling.
    """
    task_start_time = time.time()
    logger.info(f"[TASK {self.request.id}] Starting post-processing for Job ID: {job_id}")
    
    with database_session() as db:
        try:
            # Fetch job with segments
            logger.debug(f"Fetching job {job_id} with segments")
            job = db.query(Job).options(selectinload(Job.segments)).filter(Job.id == uuid.UUID(job_id)).first()
            
            if not job:
                logger.warning(f"Job {job_id} not found in database")
                return {"status": "job_not_found", "job_id": job_id}
            
            logger.info(f"Job {job_id} found with {len(job.segments) if job.segments else 0} segments")
            
            if not job.segments:
                logger.info(f"Job {job_id} has no segments to process. Marking as completed.")
                update_job_status(db, job, JobStatus.COMPLETED)
                job.is_post_processed = True
                db.commit()
                return {"status": "no_segments", "job_id": job_id}

            # 1. Data preparation phase
            logger.info(f"Starting data preparation phase for job {job_id}")
            prep_start_time = time.time()
            
            hallucination_texts = get_hallucination_texts(db)
            logger.debug(f"Retrieved {len(hallucination_texts)} hallucination patterns")
            
            segments = sorted(job.segments, key=lambda s: s.start_timestamp)
            logger.debug(f"Sorted {len(segments)} segments by timestamp")
            
            primary_languages = get_primary_languages(segments)
            logger.info(f"Identified primary languages: {primary_languages}")
            
            prep_time = time.time() - prep_start_time
            logger.debug(f"Data preparation completed in {prep_time:.2f} seconds")

            # 2. Hallucination detection phase
            logger.info(f"Starting hallucination detection for job {job_id}")
            detection_start_time = time.time()
            
            segments_to_delete = detect_sandwiched_hallucinations(
                segments, hallucination_texts, primary_languages
            )
            
            detection_time = time.time() - detection_start_time
            logger.info(f"Hallucination detection completed in {detection_time:.2f} seconds. "
                       f"Found {len(segments_to_delete)} segments to delete")

            # 3. Cleanup phase
            if segments_to_delete:
                cleanup_start_time = time.time()
                logger.info(f"Removing {len(segments_to_delete)} hallucinated segments from job {job_id}")
                
                deleted_segments_info = []
                for segment in segments_to_delete:
                    deleted_segments_info.append({
                        "speaker": segment.speaker,
                        "start": segment.start_timestamp,
                        "end": segment.end_timestamp,
                        "text": segment.text[:50] + "..." if len(segment.text) > 50 else segment.text
                    })
                    db.delete(segment)
                
                cleanup_time = time.time() - cleanup_start_time
                logger.info(f"Segment cleanup completed in {cleanup_time:.2f} seconds")
                logger.debug(f"Deleted segments details: {deleted_segments_info}")
                
                # Commit deletion to database to get updated segment list
                db.commit()
                logger.debug("Committed segment deletions to database")
            else:
                logger.info(f"No hallucinated segments found in job {job_id}")

            # 4. Speaker remapping phase (after cleanup)
            logger.info(f"Starting speaker remapping phase for job {job_id}")
            remapping_start_time = time.time()
            
            # Get updated segments after deletion
            remaining_segments = db.query(Segment).filter(Segment.job_id == job.id).order_by(Segment.start_timestamp).all()
            logger.debug(f"Retrieved {len(remaining_segments)} remaining segments after cleanup")
            
            if remaining_segments:
                # Convert to dictionary format for remapping function
                segments_data = []
                for segment in remaining_segments:
                    segments_data.append({
                        'speaker': segment.speaker,
                        'start': segment.start_timestamp,
                        'end': segment.end_timestamp,
                        'text': segment.text,
                        'language': segment.language
                    })
                
                # Perform speaker remapping
                from .utils import remap_speakers_chronologically
                remapped_segments = remap_speakers_chronologically(segments_data)
                logger.debug(f"Remapped {len(remapped_segments)} segments")
                
                # Build speaker mapping dictionary for diarization segments
                speaker_mapping = {}
                
                # Update database with new speaker labels and build mapping
                for i, (db_segment, remapped_data) in enumerate(zip(remaining_segments, remapped_segments)):
                    old_speaker = db_segment.speaker
                    new_speaker = remapped_data['speaker']
                    
                    if old_speaker != new_speaker:
                        logger.debug(f"Updating segment {i}: {old_speaker} -> {new_speaker}")
                        db_segment.speaker = new_speaker
                        speaker_mapping[old_speaker] = new_speaker
                
                # Commit transcription segment changes
                db.commit()
                logger.debug("Committed speaker remapping changes to transcription segments")
                
                # Apply the same speaker remapping to diarization segments  
                logger.info("Applying speaker remapping to diarization segments")
                if speaker_mapping:
                    # Apply the specific mapping that was created during cleanup
                    diarization_updates = update_speaker_remapping_in_diarization(db, job.id, speaker_mapping)
                    logger.info(f"Updated {diarization_updates} diarization segments with new speaker labels")
                else:
                    # Even without cleanup, apply chronological remapping to diarization segments
                    # Get current diarization segments and create proper speaker mapping
                    diarization_segments = db.query(DiarizationSegment).filter(
                        DiarizationSegment.job_id == job.id
                    ).order_by(DiarizationSegment.start_seconds).all()
                    
                    if diarization_segments:
                        # Create speaker mapping from transcription segments (which were just remapped)
                        transcription_speakers = set(seg['speaker'] for seg in segments_data)
                        diarization_speakers = set(seg.speaker_tag for seg in diarization_segments)
                        
                        # Create mapping for diarization segments based on chronological appearance
                        diarization_speaker_mapping = {}
                        seen_speakers = set()
                        speaker_counter = 1
                        
                        for seg in diarization_segments:
                            if seg.speaker_tag not in seen_speakers:
                                new_label = f"SPEAKER_{speaker_counter:02d}"
                                diarization_speaker_mapping[seg.speaker_tag] = new_label
                                seen_speakers.add(seg.speaker_tag)
                                speaker_counter += 1
                        
                        if diarization_speaker_mapping:
                            diarization_updates = update_speaker_remapping_in_diarization(db, job.id, diarization_speaker_mapping)
                            logger.info(f"Applied chronological remapping to {diarization_updates} diarization segments")
            else:
                logger.warning("No segments remain after cleanup - skipping speaker remapping")
            
            remapping_time = time.time() - remapping_start_time
            logger.info(f"Speaker remapping completed in {remapping_time:.2f} seconds")

            # 5. Mark diarization segments as processed
            logger.info(f"Marking diarization segments as processed for job {job_id}")
            processed_segments = mark_diarization_segments_as_processed(db, job.id)
            
            # 6. Finalization
            logger.info(f"Finalizing post-processing for job {job_id}")
            job.is_post_processed = True
            update_job_status(db, job, JobStatus.COMPLETED)
            
            total_time = time.time() - task_start_time
            logger.info(f"[TASK {self.request.id}] Post-processing for Job {job_id} completed successfully in {total_time:.2f} seconds")
            
            # Calculate final segment count for reporting
            final_segments_count = len(remaining_segments) if segments_to_delete else len(segments)
            
            return {
                "status": "completed",
                "job_id": job_id,
                "segments_processed": len(segments),
                "segments_deleted": len(segments_to_delete),
                "final_segments_count": final_segments_count,
                "speakers_remapped": len(segments_to_delete) > 0,  # True if remapping was performed
                "diarization_segments_processed": processed_segments,
                "processing_time": total_time,
                "primary_languages": list(primary_languages)
            }

        except Exception as e:
            total_time = time.time() - task_start_time
            logger.error(f"[TASK {self.request.id}] CRITICAL ERROR during post-processing for Job {job_id} "
                        f"after {total_time:.2f} seconds: {e}", exc_info=True)
            
            # Try to update job status if possible
            try:
                if 'job' in locals() and job:
                    update_job_status(db, job, JobStatus.FAILED, f"Post-processing failed: {str(e)}")
            except Exception as status_error:
                logger.error(f"Failed to update job status after post-processing error: {status_error}")
            
            # Re-raise the exception for Celery to handle
            raise


@celery_app.task(name="jobs.transcribe", bind=True)
def transcribe_task(self, job_id: str):
    """
    Celery task that orchestrates the transcription process for a job.
    Enhanced with comprehensive logging, timing, and better error handling.
    """
    task_start_time = time.time()
    logger.info(f"[TASK {self.request.id}] Starting transcription for Job ID: {job_id}")
    
    with database_session() as db:
        try:
            # Fetch and validate job
            logger.debug(f"Fetching job {job_id} from database")
            job = db.query(Job).filter(Job.id == uuid.UUID(job_id)).first()
            
            if not job:
                logger.error(f"Job {job_id} not found in database")
                return {"status": "job_not_found", "job_id": job_id}
            
            logger.info(f"Job {job_id} found. Audio file: {job.audio_file_path}")
            
            # Validate audio file exists
            if not os.path.exists(job.audio_file_path):
                error_msg = f"Audio file not found: {job.audio_file_path}"
                logger.error(error_msg)
                update_job_status(db, job, JobStatus.FAILED, error_msg)
                return {"status": "audio_file_not_found", "job_id": job_id, "path": job.audio_file_path}

            # Update job status to processing
            update_job_status(db, job, JobStatus.PROCESSING)
            logger.info(f"Job {job_id} status updated to PROCESSING")

            # Run transcription logic
            logger.info(f"Starting AI transcription logic for job {job_id}")
            transcription_start_time = time.time()
            
            transcription_segments, audio_duration, diarization_turns = process_audio_pipeline(
                job.audio_file_path, whisper_processor, whisper_model, diarization_pipeline, DEVICE,
                db_session=db, job_instance=job
            )
            
            transcription_time = time.time() - transcription_start_time
            logger.info(f"Transcription logic completed in {transcription_time:.2f} seconds. "
                       f"Generated {len(transcription_segments)} segments for {audio_duration:.2f}s audio")

            # Update job with audio duration
            job.audio_duration = audio_duration
            db.commit()
            logger.debug(f"Updated job {job_id} with audio duration: {audio_duration:.2f}s")

            # Save transcription segments
            logger.info(f"Saving transcription results for job {job_id}")
            save_start_time = time.time()
            
            segments_saved = save_transcription_segments(db, job, transcription_segments)
            
            save_time = time.time() - save_start_time
            logger.info(f"Saved {segments_saved} transcription segments in {save_time:.2f} seconds")

            # Update status to post-processing and queue post-processing task
            update_job_status(db, job, JobStatus.POST_PROCESSING)
            logger.info(f"Job {job_id} moved to POST_PROCESSING status. Queueing post-processing task")
            
            # Queue post-processing task
            post_process_result = post_process_job.delay(job_id)
            logger.info(f"Post-processing task queued with ID: {post_process_result.id}")

            total_time = time.time() - task_start_time
            logger.info(f"[TASK {self.request.id}] Transcription for Job {job_id} completed successfully in {total_time:.2f} seconds")
            
            return {
                "status": "completed",
                "job_id": job_id,
                "segments_generated": len(transcription_segments),
                "segments_saved": segments_saved,
                "diarization_segments_generated": len(diarization_turns),
                "audio_duration": audio_duration,
                "transcription_time": transcription_time,
                "total_time": total_time,
                "post_process_task_id": post_process_result.id
            }

        except Exception as e:
            total_time = time.time() - task_start_time
            logger.error(f"[TASK {self.request.id}] ERROR processing Job {job_id} "
                        f"after {total_time:.2f} seconds: {e}", exc_info=True)
            
            # Try to update job status if possible
            try:
                if 'job' in locals() and job:
                    update_job_status(db, job, JobStatus.FAILED, str(e))
            except Exception as status_error:
                logger.error(f"Failed to update job status after transcription error: {status_error}")
            
            # Re-raise the exception for Celery to handle
            raise


# Health check task for monitoring
@celery_app.task(name="jobs.health_check")
def health_check():
    """
    Simple health check task to verify the worker is functioning properly.
    """
    logger.info("Health check task executed")
    
    try:
        # Test database connection
        with database_session() as db:
            db.execute("SELECT 1")
        
        # Test AI models are loaded
        if whisper_processor is None or whisper_model is None or diarization_pipeline is None:
            raise RuntimeError("AI models not properly loaded")
        
        return {
            "status": "healthy",
            "device": DEVICE,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }
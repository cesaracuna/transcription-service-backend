import eventlet
eventlet.monkey_patch(os=False, subprocess=False)

# Standard and application libraries
import uuid
import os
import time
import torch
import logging
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import selectinload, Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text

# from celery import chain
from celery.exceptions import SoftTimeLimitExceeded
from .celery_app import celery_app
from .logging_config import setup_logging
from .database import database_session
from .models import Job, Segment, DiarizationSegment, Paragraph, JobStatus
from .security import comprehensive_file_validation, SecurityError

# Refactored modules
from pydub import AudioSegment
from .audio_processing import initialize_ai_models, process_diarization, group_segments_for_transcription, transcribe_segment_group_from_db
from .audio_conversion import convert_to_wav_and_resample_ffmpeg
from .hallucination_detection import detect_sandwiched_hallucinations, get_hallucination_texts, get_primary_languages, detect_and_clean_question_mark_hallucinations

# --- Configuration and Initialization ---
from .config import get_config

config = get_config()
setup_logging(process_name="CeleryWorker", log_level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)

DEVICE = config.DEVICE if torch.cuda.is_available() else "cpu"
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


@celery_app.task(
    name="tasks.audio_preprocessing", 
    bind=True,
    max_retries=3,
    default_retry_delay=120,  # 2 minutes between retries
    time_limit=3600,  # 60 minutes hard timeout
    soft_time_limit=3300,  # 55 minutes soft timeout
    acks_late=True,
    reject_on_worker_lost=True)
def audio_preprocessing_task(self, job_id: str):
    """
    Validates and preprocesses audio file after job creation.
    Performs security validation and audio format conversion.
    If successful, queues diarization task. If failed, marks job as FAILED.
    """
    logger.info(f"Starting validation and preprocessing for job {job_id}")

    with database_session() as db:
        try:
            # Get job from database
            job = db.query(Job).filter(Job.id == uuid.UUID(job_id)).first()
            if not job:
                logger.error(f"Job {job_id} not found")
                return False

            logger.info(f"Processing job {job_id} - File: {job.original_filename}")

            # Update job status to indicate preprocessing has started
            update_job_status(db, job, JobStatus.PENDING)

            original_file_path = job.audio_file_path
            audio_storage_path = str(config.AUDIO_STORAGE_PATH)

            # 1. Security validation
            logger.info(f"Starting security validation for job {job_id}")
            validation_result = comprehensive_file_validation(original_file_path, job.original_filename)

            if not validation_result['is_valid']:
                # Clean up file and mark job as failed
                error_details = "; ".join(validation_result['errors'])
                logger.error(f"Security validation failed for job {job_id}: {error_details}")

                try:
                    os.remove(original_file_path)
                    logger.info(f"Removed invalid file: {original_file_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up invalid file: {cleanup_error}")

                update_job_status(db, job, JobStatus.FAILED, f"Security validation failed: {error_details}")
                return False

            # Log validation success
            logger.info(f"Security validation passed for job {job_id}:")
            logger.info(f"  - MIME type: {validation_result['detected_mime_type']}")
            logger.info(f"  - File size: {validation_result['file_size_mb']:.2f} MB")

            if validation_result['warnings']:
                logger.warning(f"Security warnings: {'; '.join(validation_result['warnings'])}")

            # 2. Audio format conversion
            logger.info(f"Starting audio format conversion for job {job_id}")
            standardized_wav_path = convert_to_wav_and_resample_ffmpeg(original_file_path, audio_storage_path)

            if not standardized_wav_path:
                logger.error(f"Audio conversion failed for job {job_id}")
                # Clean up original file
                try:
                    os.remove(original_file_path)
                    logger.debug("Cleaned up original file after conversion failure")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up original file: {str(cleanup_error)}")

                update_job_status(db, job, JobStatus.FAILED, "Audio conversion failed")
                return False

            # 3. Update job with standardized file path
            job.audio_file_path = standardized_wav_path
            db.commit()
            logger.info(f"Job {job_id} audio file path updated to: {standardized_wav_path}")
            logger.info(f"Successfully completed validation and preprocessing for job {job_id}")

            return True

        except SecurityError as e:
            logger.error(f"Security error in job {job_id}: {str(e)}", exc_info=True)
            update_job_status(db, job, JobStatus.FAILED, f"Security error: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in validation and preprocessing for job {job_id}: {str(e)}", exc_info=True)
            update_job_status(db, job, JobStatus.FAILED, f"Preprocessing failed: {str(e)}")
            return False


@celery_app.task(
    name="tasks.speaker_segmentation", 
    bind=True,
    max_retries=3,
    default_retry_delay=120,  # 2 minutes between retries
    time_limit=3600,  # 60 minutes hard timeout
    soft_time_limit=3300,  # 55 minutes soft timeout
    acks_late=True,
    reject_on_worker_lost=True)
def speaker_segmentation_task(self, job_id: str):
    """
    Processes diarization for a given job.
    Enhanced with comprehensive logging and better error handling.
    """
    task_start_time = time.time()
    logger.info(f"[TASK {self.request.id}] Starting diarization processing for Job ID: {job_id}")
    
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

            # Update job status to diarizing
            update_job_status(db, job, JobStatus.DIARIZING)

            # Run diarization logic using new separated function
            logger.info(f"Starting diarization logic for job {job_id}")
            diarization_start_time = time.time()
            diarization_segments, audio_duration = process_diarization(job.audio_file_path, diarization_pipeline)
            diarization_time = time.time() - diarization_start_time
            logger.info(f"  - Diarization logic completed in {diarization_time:.2f} seconds. ")

            # Update job with audio duration
            job.audio_duration = audio_duration
            db.commit()
            logger.debug(f"Updated job {job_id} with audio duration: {audio_duration:.2f}s")

            # Phase 4: Saving diarization results with PENDING status
            logger.info("Diarization Phase 4: Saving diarization results")
            segments_saved = save_diarization_segments_with_status(db, job, diarization_segments, status='PENDING')
            total_time = time.time() - task_start_time
            logger.debug(f"[TASK {self.request.id}] Diarization for Job {job_id} completed successfully in {total_time:.2f} seconds")
            logger.info(f"Diarization completed in {total_time:.2f} seconds. Chain will continue with transcription task.")

            # Keep job status as DIARIZING - next task will update to TRANSCRIBING when it starts
            # (No status update here - let next task handle it)

            return {
                "status": "completed",
                "job_id": job_id,
                "segments_generated": len(diarization_segments),
                "segments_saved": segments_saved,
                "audio_duration": audio_duration,
                "diarization_time": diarization_time,
                "total_time": total_time
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
                logger.error(f"Failed to update job status after diarization error: {status_error}")
            
            # Re-raise the exception for Celery to handle
            raise


@celery_app.task(
    name="tasks.generate_transcript",
    bind=True,
    max_retries=3,
    default_retry_delay=120,  # 2 minutes between retries
    time_limit=3600,  # 60 minutes hard timeout
    soft_time_limit=3300,  # 55 minutes soft timeout
    acks_late=True,
    reject_on_worker_lost=True
)
def generate_transcript_task(self, job_id: str):
    """
    Celery task that transcribes diarization segments from database one by one.
    Enhanced with comprehensive logging, timing, and better error handling.

    Retry policy:
    - Max retries: 3
    - Retry delay: 120 seconds
    - Time limit: 60 minutes (hard), 55 minutes (soft warning)
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

            # Update job status to TRANSCRIBING at the start of transcription
            update_job_status(db, job, JobStatus.TRANSCRIBING)

            # Validate audio file exists
            if not os.path.exists(job.audio_file_path):
                error_msg = f"Audio file not found: {job.audio_file_path}"
                logger.error(error_msg)
                update_job_status(db, job, JobStatus.FAILED, error_msg)
                return {"status": "audio_file_not_found", "job_id": job_id, "path": job.audio_file_path}

            # Phase 1: Grouping segments for optimized transcription
            logger.info("Transcription Phase 1: Grouping segments for optimized transcription")

            # Fetch pending diarization segments
            logger.info(f"  - Quering pending diarization segments...")
            pending_segments = db.query(DiarizationSegment).filter(
                DiarizationSegment.job_id == job.id,
                DiarizationSegment.status == 'PENDING'
            ).order_by(DiarizationSegment.start_seconds).all()
            
            if not pending_segments:
                logger.warning(f"  - No pending diarization segments found for job {job_id}")
                logger.info(f"  - No segments to transcribe. Chain will continue with post-processing.")
                return { "status": "no_segments", "job_id": job_id }

            # Group segments for more efficient transcription
            logger.info(f"  - Found {len(pending_segments)} pending segments for grouping")
            segment_groups = group_segments_for_transcription(pending_segments, max_group_duration=28.0)

            # # # temporal BORRAR DESPUES
            # # main_audio = AudioSegment.from_file(job.audio_file_path)
            # # logger.info(f"Successfully loaded audio file with duration: {len(main_audio)/1000:.2f}s")

            # # generated_files = []
            # # for i, segment_group in enumerate(segment_groups):
            # #     try:
            # #         # Get first and last segment from the group
            # #         first_segment = segment_group[0]
            # #         last_segment = segment_group[-1]

            # #         start_ms = int(first_segment.start_seconds * 1000)
            # #         end_ms = int(last_segment.end_seconds * 1000)
            # #         duration = (end_ms - start_ms) / 1000

            # #         logger.debug(f"Processing group {i+1}/{len(segment_groups)}: speaker {first_segment.speaker_tag}, {duration:.2f}s, {len(segment_group)} segments")

            # #         audio_segment = main_audio[start_ms:end_ms]
            # #         file_name = f"chunk_{i+1:03d}_{first_segment.speaker_tag.lower()}_{uuid.uuid4().hex[:6]}.wav"
            # #         file_path = os.path.join(config.AUDIO_STORAGE_PATH, file_name)

            # #         audio_segment.export(file_path, format="wav")
            # #         generated_files.append(file_path)

            # #         logger.debug(f"Successfully exported segment to: {file_name}")

            # #     except Exception as e:
            # #         logger.error(f"Failed to process group {i+1} (speaker {first_segment.speaker_tag}): {str(e)}")
            # #         continue
            # # # temporal BORRAR DESPUES



            logger.info(f"  - Starting group-based transcription for job {job_id} ({len(segment_groups)} groups)")
            transcription_start_time = time.time()
            
            transcribed_segments = []
            segments_processed = 0
            segments_skipped = 0
            groups_processed = 0
            
            for group_idx, segment_group in enumerate(segment_groups, 1):
                first_segment = segment_group[0]
                last_segment = segment_group[-1]
                group_duration = last_segment.end_seconds - first_segment.start_seconds
                
                logger.info(f"  - Processing group {group_idx:2}/{len(segment_groups)}: {first_segment.speaker_tag} "
                           f"({first_segment.start_seconds:.2f}s-{last_segment.end_seconds:.2f}s, {group_duration:.2f}s, {len(segment_group)} segments)")
                
                try:
                    # Transcribe this group
                    group_results = transcribe_segment_group_from_db(
                        segment_group=segment_group,
                        audio_path=job.audio_file_path,
                        whisper_processor=whisper_processor,
                        whisper_model=whisper_model,
                        device=DEVICE
                    )
                    
                    if group_results:
                        # # logger.info(f"  - Saving {len(group_results)} transcription results to database:")
                        
                        # Save each transcribed segment to database
                        for result_idx, segment_result in enumerate(group_results):
                            transcription_segment = Segment(
                                job_id=job.id,
                                start_timestamp=segment_result['start'],
                                end_timestamp=segment_result['end'],
                                speaker=segment_result['speaker'],
                                text=segment_result['text'],
                                language=segment_result['language']
                            )
                            db.add(transcription_segment)
                            transcribed_segments.append(segment_result)
                            segments_processed += 1
                            
                            # Log detailed save info
                            # # text_preview = segment_result['text'][:80] + ('...' if len(segment_result['text']) > 80 else '')
                            # # logger.info(f"  → Saved segment {result_idx + 1}/{len(group_results)}: [{segment_result['start']}-{segment_result['end']}] {segment_result['speaker']}: '{text_preview}' ({len(segment_result['text'])} chars)")
                        
                        # Mark all diarization segments in this group as completed
                        for diarization_segment in segment_group:
                            diarization_segment.status = 'COMPLETED'
                        
                        groups_processed += 1
                        
                        # Verify total text was saved
                        # # total_saved_text = ''.join([result['text'] for result in group_results])
                        # # logger.info(f"  → Group {group_idx} completed: {len(group_results)} segments, {len(total_saved_text)} total characters saved")
                    else:
                        # Mark all segments in group as skipped
                        for diarization_segment in segment_group:
                            diarization_segment.status = 'SKIPPED'
                        segments_skipped += len(segment_group)
                        logger.debug(f"  → Group {group_idx} skipped")
                    
                    # Commit progress every 5 groups
                    if group_idx % 10 == 0:
                        db.commit()
                        logger.info(f"  → Committed progress: {segments_processed} segments processed from {groups_processed} groups, {segments_skipped} skipped so far")
                        
                except Exception as group_error:
                    logger.error(f"Error processing group {group_idx}: {group_error}", exc_info=True)
                    # Mark all segments in group as failed
                    for diarization_segment in segment_group:
                        diarization_segment.status = 'FAILED'
                    segments_skipped += len(segment_group)
                    continue
            
            # Final commit
            db.commit()
            
            transcription_time = time.time() - transcription_start_time
            logger.info(f"Group-based transcription completed in {transcription_time:.2f} seconds. "
                       f"Groups processed: {groups_processed}/{len(segment_groups)}, "
                       f"Segments processed: {segments_processed}, Skipped: {segments_skipped}")

            # Apply speaker remapping to transcribed segments
            if transcribed_segments:
                logger.info("Applying speaker chronological remapping")
                from .utils import remap_speakers_chronologically
                remapped_segments = remap_speakers_chronologically(transcribed_segments)
                
                # Update database with remapped speakers
                transcription_segments_db = db.query(Segment).filter(Segment.job_id == job.id).order_by(Segment.start_timestamp).all()
                
                speaker_mapping = {}
                for db_segment, remapped_data in zip(transcription_segments_db, remapped_segments):
                    old_speaker = db_segment.speaker
                    new_speaker = remapped_data['speaker']
                    if old_speaker != new_speaker:
                        db_segment.speaker = new_speaker
                        speaker_mapping[old_speaker] = new_speaker
                
                if speaker_mapping:
                    # Update diarization segments with new speaker mapping
                    update_speaker_remapping_in_diarization(db, job.id, speaker_mapping)
                    logger.info(f"Applied speaker remapping: {speaker_mapping}")
                
                db.commit()

            # Keep job status as TRANSCRIBING - next task will update to FINALIZING when it starts
            # (No status update here - let next task handle it)
            logger.info(f"Transcription completed. Chain will continue with post-processing task.")

            total_time = time.time() - task_start_time
            logger.info(f"[TASK {self.request.id}] Transcription for Job {job_id} completed successfully in {total_time:.2f} seconds")

            return {
                "status": "completed",
                "job_id": job_id,
                "segments_processed": segments_processed,
                "segments_skipped": segments_skipped,
                "total_segments": len(pending_segments),
                "transcription_time": transcription_time,
                "total_time": total_time
            }

        except SoftTimeLimitExceeded:
            logger.warning(f"[TASK {self.request.id}] Soft time limit exceeded for Job {job_id}. Transcription is taking longer than expected.")
            try:
                if 'job' in locals() and job:
                    update_job_status(db, job, JobStatus.FAILED,
                                    "Transcription timeout - task took too long")
            except Exception as status_error:
                logger.error(f"Failed to update job status after timeout: {status_error}")
            raise

        except Exception as e:
            total_time = time.time() - task_start_time
            logger.error(f"[TASK {self.request.id}] ERROR processing Job {job_id} after {total_time:.2f} seconds: {e}", exc_info=True)

            # Try to update job status if possible
            try:
                if 'job' in locals() and job:
                    update_job_status(db, job, JobStatus.FAILED, str(e))
            except Exception as status_error:
                logger.error(f"Failed to update job status after transcription error: {status_error}")

            # Retry logic
            try:
                retry_count = self.request.retries
                logger.warning(f"[TASK {self.request.id}] Attempting retry {retry_count + 1}/3 for Job {job_id}")
                raise self.retry(exc=e, countdown=120)
            except self.MaxRetriesExceededError:
                logger.error(f"[TASK {self.request.id}] Max retries exceeded for Job {job_id}. Transcription permanently failed.")
                raise e


@celery_app.task(
    name="tasks.complete_processing",
    bind=True,
    max_retries=3,
    default_retry_delay=60,  # 60 seconds between retries
    time_limit=600,  # 10 minutes hard timeout
    soft_time_limit=540,  # 9 minutes soft timeout (warning)
    acks_late=True,  # Acknowledge task after completion (for fault tolerance)
    reject_on_worker_lost=True  # Reject task if worker is lost
)
def complete_processing_task(self, job_id: str):
    """
    Cleans a completed transcription using contextual "sandwich" logic.
    Enhanced with comprehensive logging and better error handling.

    Retry policy:
    - Max retries: 3
    - Retry delay: 60 seconds
    - Time limit: 10 minutes (hard), 9 minutes (soft warning)
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

            # Update job status to FINALIZING at the start of post-processing
            update_job_status(db, job, JobStatus.FINALIZING)

            logger.info(f"Job {job_id} found with {len(job.segments) if job.segments else 0} segments")
            
            if not job.segments:
                logger.info(f"Job {job_id} has no segments to process. Skipping to formatting task.")
                update_job_status(db, job, JobStatus.COMPLETED)
                job.is_post_processed = True
                db.commit()
                
                # # Still try to run formatting task (it will handle empty segments gracefully)
                # try:
                #     format_result = format_paragraphs_task.delay(job_id)
                #     logger.info(f"Paragraph formatting task queued for job with no segments: {format_result.id}")
                #     return {"status": "no_segments", "job_id": job_id, "format_task_id": format_result.id}
                # except Exception as format_error:
                #     logger.error(f"Failed to queue formatting task for job {job_id} with no segments: {format_error}")
                #     update_job_status(db, job, JobStatus.COMPLETED)
                #     db.commit()
                #     return {"status": "no_segments", "job_id": job_id, "format_error": str(format_error)}

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

            # 2b. Question mark hallucination detection phase (COMMENTED OUT - READY FOR ACTIVATION)
            # logger.info(f"Starting question mark hallucination detection for job {job_id}")
            # qmark_detection_start_time = time.time()
            # 
            # # Get current segments (after potential deletion from step 2)
            # current_segments = [s for s in segments if s not in segments_to_delete]
            # cleaned_segments, qmark_removed_count = detect_and_clean_question_mark_hallucinations(current_segments)
            # 
            # # Add question mark hallucinations to deletion list
            # qmark_segments_to_delete = [s for s in current_segments if s not in cleaned_segments]
            # segments_to_delete.extend(qmark_segments_to_delete)
            # 
            # qmark_detection_time = time.time() - qmark_detection_start_time
            # logger.info(f"Question mark hallucination detection completed in {qmark_detection_time:.2f} seconds. "
            #            f"Found {qmark_removed_count} additional segments to delete")

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
                        # transcription_speakers = set(seg['speaker'] for seg in segments_data)
                        # diarization_speakers = set(seg.speaker_tag for seg in diarization_segments)
                        
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
            
            # 6. Grouped turns and paragraph formatting phase
            logger.info(f"Starting grouped turns processing for job {job_id}")
            turns_start_time = time.time()
            
            try:
                # Ejecutar procedimiento almacenado para obtener turnos agrupados
                paragraphs_created = save_grouped_turns_to_paragraphs(db, job_id)

                turns_time = time.time() - turns_start_time
                logger.info(f"{paragraphs_created} paragraphs grouped in {turns_time:.2f} seconds")
                    
            except Exception as format_error:
                logger.error(f"Error during paragraph grouping for job {job_id}: {format_error}", exc_info=True)
                paragraphs_created = 0
            
            # 7. Finalization
            logger.info(f"Finalizing post-processing for job {job_id}")
            job.is_post_processed = True

            # Calculate final segment count and times for reporting
            final_segments_count = len(remaining_segments) if segments_to_delete else len(segments)
            total_time = time.time() - task_start_time

            logger.info(f"[TASK {self.request.id}] Post-processing for Job {job_id} completed successfully in {total_time:.2f} seconds")

            # Mark job as COMPLETED (only once, at the end)
            update_job_status(db, job, JobStatus.COMPLETED)
            db.commit()
            
            return {
                "status": "completed",
                "job_id": job_id,
                "segments_processed": len(segments),
                "segments_deleted": len(segments_to_delete),
                "final_segments_count": final_segments_count,
                "speakers_remapped": len(segments_to_delete) > 0,
                "diarization_segments_processed": processed_segments,
                "processing_time": total_time,
                "primary_languages": list(primary_languages),
                "format_task_id": "format_task_id"
            }

        except SoftTimeLimitExceeded:
            logger.warning(f"[TASK {self.request.id}] Soft time limit exceeded for Job {job_id}. "
                          f"Task is taking longer than expected.")
            # Try to update job status before hard timeout
            try:
                if 'job' in locals() and job:
                    update_job_status(db, job, JobStatus.FAILED, "Post-processing timeout - task took too long")
            except Exception as status_error:
                logger.error(f"Failed to update job status after timeout: {status_error}")
            raise

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

            # Retry logic
            try:
                retry_count = self.request.retries
                logger.warning(f"[TASK {self.request.id}] Attempting retry {retry_count + 1}/3 for Job {job_id}")
                raise self.retry(exc=e, countdown=60)
            except self.MaxRetriesExceededError:
                logger.error(f"[TASK {self.request.id}] Max retries exceeded for Job {job_id}. "
                           f"Task permanently failed.")
                # Re-raise the original exception
                raise e


@celery_app.task(name="tasks.check_services_health")
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
            "timestamp": time.time(),
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }


def save_grouped_turns_to_paragraphs(db: Session, job_id: str) -> int:
    """
    Gets grouped turns using GetGroupedTurnsByJob and inserts them
    into the Paragraphs table.

    Args:
        db: Database session
        job_id: Job ID

    Returns:
        Number of paragraphs/turns saved
    """
    logger.info(f"Saving grouped turns to paragraphs for job {job_id}")

    try:
        # 1. Get grouped turns using stored procedure
        grouped_turns = get_grouped_turns_by_job(db, job_id)

        if not grouped_turns:
            logger.info(f"No grouped turns found for job {job_id}")
            return 0

        logger.debug(f"Processing {len(grouped_turns)} grouped turns for job {job_id}")
        paragraphs_created = 0

        # 2. Insert each turn as a paragraph
        for turn in grouped_turns:
            paragraph = Paragraph(
                job_id=uuid.UUID(job_id),
                start_timestamp=str(turn['turn_start']),
                end_timestamp=str(turn['turn_end']),
                speaker=turn['speaker'],
                original_text=turn['combined_text'],
                formatted_text=turn['combined_text'],
                language=turn['primary_language']
            )
            db.add(paragraph)
            paragraphs_created += 1

        # 3. Commit all paragraphs
        db.commit()
        logger.info(f"Successfully saved {paragraphs_created} paragraphs for job {job_id}")
        return paragraphs_created

    except Exception as e:
        logger.error(f"Error saving grouped turns to paragraphs for job {job_id}: {e}", exc_info=True)
        db.rollback()
        raise


def get_grouped_turns_by_job(db: Session, job_id: str) -> List[Dict[str, Any]]:
    """
    Ejecuta el procedimiento almacenado GetGroupedTurnsByJob para obtener 
    participaciones agrupadas por usuario.
    
    Args:
        db: Sesión de base de datos
        job_id: ID del job
        
    Returns:
        Lista de turnos agrupados
    """
    logger.info(f"Executing GetGroupedTurnsByJob for job {job_id}")
    
    try:
        # Ejecutar procedimiento almacenado
        query = f"SELECT * FROM dbo.GetGroupedTurnsByJob('{job_id}') ORDER BY turn_id"
        result = db.execute(text(query))
        
        # Convertir resultado a lista de diccionarios
        turns = []
        for row in result:
            turns.append({
                'turn_id': row.turn_id,
                'job_id': row.job_id,
                'speaker': row.speaker,
                'turn_start': row.turn_start,
                'turn_end': row.turn_end,
                'turn_duration': row.turn_duration,
                'segment_count': row.segment_count,
                'combined_text': row.combined_text,
                'primary_language': row.primary_language
            })
        
        logger.info(f"Retrieved {len(turns)} grouped turns for job {job_id}")
        return turns
        
    except Exception as e:
        logger.error(f"Error executing GetGroupedTurnsByJob for job {job_id}: {e}", exc_info=True)
        raise


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


def save_diarization_segments_with_status(db: Session, job: Job, diarization_data: List[Dict[str, Any]], status: str = 'ACTIVE') -> int:
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
    
    # # logger.info(f"Saving {len(diarization_data)} diarization segments with status '{status}' for job {job.id}")
    
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
        
        logger.info(f"  - Successfully saved {segments_created} segments status '{status}' for job {job.id}")
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
        # Update all diarization segments for this job to 'PROCESSED' status
        segments_updated = db.query(DiarizationSegment).filter(
            DiarizationSegment.job_id == job_id
        ).update(
            {"status": "PROCESSED"},
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


# @celery_app.task(name="tasks.format_paragraphs", bind=True)
# def format_paragraphs_task(self, job_id: str):
#     """
#     Format transcription segments into clean, LLM-processed paragraphs.
    
#     This task:
#     1. Reads all segments for a job ordered by speaker and timestamp
#     2. Groups consecutive segments by speaker
#     3. Sends grouped text to LLM for formatting and cleaning
#     4. Splits LLM output into individual paragraphs
#     5. Saves formatted paragraphs to the paragraphs table
    
#     Args:
#         job_id (str): UUID of the job to process
    
#     Returns:
#         dict: Task execution result with status and statistics
#     """
#     logger.info(f"Starting paragraph formatting for job {job_id}")
    
#     try:
#         with database_session() as db:
#             # Get job and validate it exists
#             job = db.query(Job).filter(Job.id == job_id).first()
#             if not job:
#                 error_msg = f"Job {job_id} not found"
#                 logger.error(error_msg)
#                 return {"status": "error", "message": error_msg, "job_id": job_id}
            
#             # Update job status to FORMATTING
#             job.status = JobStatus.FORMATTING
#             db.commit()
#             logger.info(f"Job {job_id} status updated to FORMATTING")
            
#             # Get all segments ordered by speaker and timestamp
#             segments = db.query(Segment).filter(
#                 Segment.job_id == job_id
#             ).order_by(
#                 Segment.speaker,
#                 Segment.start_timestamp
#             ).all()
            
#             if not segments:
#                 logger.warning(f"No segments found for job {job_id}")
#                 job.status = JobStatus.COMPLETED
#                 db.commit()
#                 return {
#                     "status": "completed", 
#                     "message": "No segments to format",
#                     "job_id": job_id,
#                     "paragraphs_created": 0
#                 }
            
#             logger.info(f"Found {len(segments)} segments to process for job {job_id}")
            
#             # Convert segments to dictionaries for processing
#             segment_dicts = []
#             for segment in segments:
#                 segment_dicts.append({
#                     'start_timestamp': segment.start_timestamp,
#                     'end_timestamp': segment.end_timestamp,
#                     'speaker': segment.speaker,
#                     'text': segment.text,
#                     'language': segment.language
#                 })
            
#             # Group segments by speaker
#             grouped_segments = group_segments_by_speaker(segment_dicts)
#             logger.info(f"Grouped segments into {len(grouped_segments)} speaker sections")
            
#             # Process each group with LLM and save paragraphs
#             total_paragraphs_created = 0
#             groups_processed = 0
            
#             for i, group in enumerate(grouped_segments, 1):
#                 logger.info(f"Processing group {i}/{len(grouped_segments)}: {group['speaker']} "
#                            f"({group['segment_count']} segments, {len(group['text'])} chars)")
                
#                 try:
#                     # Format text with LLM
#                     formatted_paragraphs = format_text_with_llm(
#                         text=group['text'],
#                         speaker=group['speaker'],
#                         language=group['language']
#                     )
                    
#                     if not formatted_paragraphs:
#                         logger.warning(f"No paragraphs returned for group {i}, skipping")
#                         continue
                    
#                     logger.info(f"LLM formatted group {i} into {len(formatted_paragraphs)} paragraphs")
                    
#                     # Distribute timestamps across paragraphs
#                     paragraphs_with_timestamps = distribute_timestamps(
#                         paragraphs=formatted_paragraphs,
#                         start_timestamp=group['start_timestamp'],
#                         end_timestamp=group['end_timestamp']
#                     )
                    
#                     # Save each paragraph to database
#                     for j, paragraph_data in enumerate(paragraphs_with_timestamps, 1):
#                         paragraph = Paragraph(
#                             job_id=job.id,
#                             start_timestamp=paragraph_data['start_timestamp'],
#                             end_timestamp=paragraph_data['end_timestamp'],
#                             speaker=group['speaker'],
#                             text=paragraph_data['text'],
#                             language=group['language']
#                         )
#                         db.add(paragraph)
#                         total_paragraphs_created += 1
                        
#                         logger.debug(f"  Created paragraph {j}/{len(paragraphs_with_timestamps)}: "
#                                    f"'{paragraph_data['text'][:50]}{'...' if len(paragraph_data['text']) > 50 else ''}'")
                    
#                     groups_processed += 1
                    
#                     # Commit after each group to avoid losing work
#                     db.commit()
#                     logger.info(f"Group {i} processed successfully: {len(paragraphs_with_timestamps)} paragraphs saved")
                    
#                 except Exception as group_error:
#                     logger.error(f"Error processing group {i} for speaker {group['speaker']}: {group_error}", exc_info=True)
#                     # Continue with next group rather than failing entire task
#                     continue
            
#             # Update job status to COMPLETED
#             job.status = JobStatus.COMPLETED
#             db.commit()
            
#             logger.info(f"Paragraph formatting completed for job {job_id}: "
#                        f"{groups_processed}/{len(grouped_segments)} groups processed, "
#                        f"{total_paragraphs_created} paragraphs created")
            
#             return {
#                 "status": "completed",
#                 "job_id": job_id,
#                 "groups_total": len(grouped_segments),
#                 "groups_processed": groups_processed,
#                 "paragraphs_created": total_paragraphs_created,
#                 "segments_input": len(segments)
#             }
            
#     except SQLAlchemyError as db_error:
#         logger.error(f"Database error in paragraph formatting for job {job_id}: {db_error}", exc_info=True)
        
#         # Try to update job status to FAILED
#         try:
#             with database_session() as db:
#                 job = db.query(Job).filter(Job.id == job_id).first()
#                 if job:
#                     job.status = JobStatus.FAILED
#                     job.error_message = f"Database error during paragraph formatting: {str(db_error)}"
#                     db.commit()
#         except Exception as status_error:
#             logger.error(f"Failed to update job status after database error: {status_error}")
        
#         return {
#             "status": "error",
#             "job_id": job_id,
#             "error_type": "database_error",
#             "message": str(db_error)
#         }
        
#     except Exception as e:
#         logger.error(f"Unexpected error in paragraph formatting for job {job_id}: {e}", exc_info=True)
        
#         # Try to update job status to FAILED
#         try:
#             with database_session() as db:
#                 job = db.query(Job).filter(Job.id == job_id).first()
#                 if job:
#                     job.status = JobStatus.FAILED
#                     job.error_message = f"Unexpected error during paragraph formatting: {str(e)}"
#                     db.commit()
#         except Exception as status_error:
#             logger.error(f"Failed to update job status after unexpected error: {status_error}")
        
#         return {
#             "status": "error",
#             "job_id": job_id,
#             "error_type": "unexpected_error",
#             "message": str(e)
#         }

# @celery_app.task(name="tasks.diarization", bind=True)
# def diarization_task(self, job_id: str):
#     """
#     Processes diarization for a given job.
#     Enhanced with comprehensive logging and better error handling.
#     """
#     task_start_time = time.time()
#     logger.info(f"[TASK {self.request.id}] Starting diarization processing for Job ID: {job_id}")
    
#     with database_session() as db:
#         try:
#             # Fetch and validate job
#             logger.debug(f"Fetching job {job_id} from database")
#             job = db.query(Job).filter(Job.id == uuid.UUID(job_id)).first()
            
#             if not job:
#                 logger.error(f"Job {job_id} not found in database")
#                 return {"status": "job_not_found", "job_id": job_id}
            
#             logger.info(f"Job {job_id} found. Audio file: {job.audio_file_path}")
            
#             # Validate audio file exists
#             if not os.path.exists(job.audio_file_path):
#                 error_msg = f"Audio file not found: {job.audio_file_path}"
#                 logger.error(error_msg)
#                 update_job_status(db, job, JobStatus.FAILED, error_msg)
#                 return {"status": "audio_file_not_found", "job_id": job_id, "path": job.audio_file_path}

#             # Update job status to diarizing
#             update_job_status(db, job, JobStatus.DIARIZING)
#             logger.info(f"Job {job_id} status updated to DIARIZING")

#             # Run diarization logic using new separated function
#             logger.info(f"Starting diarization logic for job {job_id}")
#             diarization_start_time = time.time()
            
#             diarization_segments, audio_duration = process_diarization_only(
#                 job.audio_file_path, diarization_pipeline
#             )
            
#             diarization_time = time.time() - diarization_start_time
#             logger.info(f"Diarization logic completed in {diarization_time:.2f} seconds. "
#                        f"Generated {len(diarization_segments)} segments for {audio_duration:.2f}s audio")

#             # Update job with audio duration
#             job.audio_duration = audio_duration
#             db.commit()
#             logger.debug(f"Updated job {job_id} with audio duration: {audio_duration:.2f}s")

#             # Save diarization segments with PENDING status
#             logger.info(f"Saving diarization results for job {job_id}")
#             save_start_time = time.time()
            
#             segments_saved = save_diarization_segments_with_status(db, job, diarization_segments, status='PENDING')
            
#             save_time = time.time() - save_start_time
#             logger.info(f"Saved {segments_saved} diarization segments in {save_time:.2f} seconds")

#             # Update job status to ready for transcription
#             update_job_status(db, job, JobStatus.PROCESSING)
#             logger.info(f"Job {job_id} moved to PROCESSING status. Ready for transcription.")
            
#             # Queue transcription task
#             transcription_result = transcribe_task.delay(job_id)
#             logger.info(f"Transcription task queued with ID: {transcription_result.id}")

#             total_time = time.time() - task_start_time
#             logger.info(f"[TASK {self.request.id}] Diarization for Job {job_id} completed successfully in {total_time:.2f} seconds")
            
#             return {
#                 "status": "completed",
#                 "job_id": job_id,
#                 "segments_generated": len(diarization_segments),
#                 "segments_saved": segments_saved,
#                 "audio_duration": audio_duration,
#                 "diarization_time": diarization_time,
#                 "total_time": total_time,
#                 "transcription_task_id": transcription_result.id
#             }

#         except Exception as e:
#             total_time = time.time() - task_start_time
#             logger.error(f"[TASK {self.request.id}] ERROR processing Job {job_id} "
#                         f"after {total_time:.2f} seconds: {e}", exc_info=True)
            
#             # Try to update job status if possible
#             try:
#                 if 'job' in locals() and job:
#                     update_job_status(db, job, JobStatus.FAILED, str(e))
#             except Exception as status_error:
#                 logger.error(f"Failed to update job status after diarization error: {status_error}")
            
#             # Re-raise the exception for Celery to handle
#             raise

"""
Business logic services for transcription domain.
These services orchestrate domain operations and enforce business rules.
"""

import logging
from typing import List, Optional, Dict, Any, Set, Tuple
from uuid import UUID
from datetime import datetime

from ..shared.enums import JobStatus, Language
from ..shared.value_objects import (
    AudioMetadata,
    TimeInterval, 
    SpeakerSegment,
    TranscriptionSegment,
    ProcessingMetrics,
    HallucinationPattern
)
from .models import TranscriptionJob, User, HallucinationRule
from ...core.exceptions import (
    ValidationError,
    JobNotFoundError,
    UserNotFoundError,
    TranscriptionError
)

logger = logging.getLogger(__name__)


class JobService:
    """Service for managing transcription jobs."""
    
    def __init__(self, job_repository, user_repository):
        self.job_repository = job_repository
        self.user_repository = user_repository
    
    async def create_job(
        self, 
        user_id: UUID, 
        original_filename: str, 
        audio_file_path: str,
        audio_metadata: Optional[AudioMetadata] = None
    ) -> TranscriptionJob:
        """
        Create a new transcription job.
        
        Args:
            user_id: ID of the user creating the job
            original_filename: Original name of the audio file
            audio_file_path: Path to the processed audio file
            audio_metadata: Optional audio metadata
            
        Returns:
            Created transcription job
            
        Raises:
            UserNotFoundError: If user doesn't exist
            ValidationError: If job data is invalid
        """
        logger.info(f"Creating transcription job for user {user_id}")
        
        # Verify user exists
        user = await self.user_repository.get_by_id(user_id)
        if not user:
            raise UserNotFoundError(f"User {user_id} not found")
        
        # Create job
        job = TranscriptionJob(
            user_id=user_id,
            original_filename=original_filename,
            audio_file_path=audio_file_path,
            audio_metadata=audio_metadata,
            status=JobStatus.PENDING
        )
        
        # Validate job
        validation_errors = job.validate()
        if validation_errors:
            raise ValidationError(f"Job validation failed: {'; '.join(validation_errors)}")
        
        # Save job
        saved_job = await self.job_repository.create(job)
        
        logger.info(f"Created job {saved_job.id} for user {user_id}")
        return saved_job
    
    async def get_job(self, job_id: UUID) -> TranscriptionJob:
        """
        Get a job by ID.
        
        Args:
            job_id: Job ID
            
        Returns:
            Transcription job
            
        Raises:
            JobNotFoundError: If job doesn't exist
        """
        job = await self.job_repository.get_by_id(job_id)
        if not job:
            raise JobNotFoundError(f"Job {job_id} not found")
        return job
    
    async def update_job_status(
        self, 
        job_id: UUID, 
        status: JobStatus, 
        error_message: Optional[str] = None
    ) -> TranscriptionJob:
        """
        Update job status.
        
        Args:
            job_id: Job ID
            status: New status
            error_message: Optional error message for failed jobs
            
        Returns:
            Updated job
            
        Raises:
            JobNotFoundError: If job doesn't exist
        """
        job = await self.get_job(job_id)
        
        logger.info(f"Updating job {job_id} status: {job.status} -> {status}")
        
        job.update_status(status, error_message)
        
        return await self.job_repository.update(job)
    
    async def get_user_jobs(
        self, 
        user_id: UUID, 
        status: Optional[JobStatus] = None,
        page: int = 1,
        size: int = 10
    ) -> Tuple[List[TranscriptionJob], int]:
        """
        Get jobs for a user with optional filtering and pagination.
        
        Args:
            user_id: User ID
            status: Optional status filter
            page: Page number (1-based)
            size: Page size
            
        Returns:
            Tuple of (jobs, total_count)
        """
        return await self.job_repository.get_by_user(
            user_id=user_id,
            status=status,
            page=page,
            size=size
        )
    
    async def delete_job(self, job_id: UUID) -> None:
        """
        Delete a job and its associated data.
        
        Args:
            job_id: Job ID
            
        Raises:
            JobNotFoundError: If job doesn't exist
            ValidationError: If job cannot be deleted
        """
        job = await self.get_job(job_id)
        
        if not job.can_be_deleted():
            raise ValidationError(f"Job {job_id} cannot be deleted in status {job.status}")
        
        logger.info(f"Deleting job {job_id}")
        await self.job_repository.delete(job_id)
    
    async def mark_job_as_viewed(self, job_id: UUID) -> TranscriptionJob:
        """Mark a job as viewed by the user."""
        job = await self.get_job(job_id)
        job.is_viewed = True
        job.updated_at = datetime.utcnow()
        
        return await self.job_repository.update(job)


class UserService:
    """Service for managing users."""
    
    def __init__(self, user_repository):
        self.user_repository = user_repository
    
    async def create_user(self, username: str, email: str) -> User:
        """
        Create a new user.
        
        Args:
            username: User's username
            email: User's email address
            
        Returns:
            Created user
            
        Raises:
            ValidationError: If user data is invalid
        """
        user = User(username=username, email=email)
        
        # Validate user
        validation_errors = user.validate()
        if validation_errors:
            raise ValidationError(f"User validation failed: {'; '.join(validation_errors)}")
        
        # Check if username/email already exists
        existing_user = await self.user_repository.get_by_username(username)
        if existing_user:
            raise ValidationError(f"Username {username} already exists")
        
        existing_user = await self.user_repository.get_by_email(email)
        if existing_user:
            raise ValidationError(f"Email {email} already exists")
        
        return await self.user_repository.create(user)
    
    async def get_user(self, user_id: UUID) -> User:
        """Get a user by ID."""
        user = await self.user_repository.get_by_id(user_id)
        if not user:
            raise UserNotFoundError(f"User {user_id} not found")
        return user
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get a user by username."""
        return await self.user_repository.get_by_username(username)
    
    async def check_user_exists(self, username: str) -> bool:
        """Check if a user exists by username."""
        user = await self.get_user_by_username(username)
        return user is not None


class TranscriptionService:
    """Service for transcription business logic."""
    
    def __init__(self, job_repository, audio_processor, hallucination_detector):
        self.job_repository = job_repository
        self.audio_processor = audio_processor
        self.hallucination_detector = hallucination_detector
    
    async def process_transcription(self, job_id: UUID) -> TranscriptionJob:
        """
        Process a transcription job through the complete pipeline.
        
        Args:
            job_id: Job ID to process
            
        Returns:
            Processed job
            
        Raises:
            JobNotFoundError: If job doesn't exist
            TranscriptionError: If processing fails
        """
        job = await self.job_repository.get_by_id(job_id)
        if not job:
            raise JobNotFoundError(f"Job {job_id} not found")
        
        logger.info(f"Starting transcription processing for job {job_id}")
        
        try:
            # Update status to processing
            job.update_status(JobStatus.PROCESSING)
            await self.job_repository.update(job)
            
            # Process audio through pipeline
            segments, diarization_segments, metrics = await self.audio_processor.process_audio(
                job.audio_file_path
            )
            
            # Add segments to job
            for segment in segments:
                job.add_segment(segment)
            
            for diarization_segment in diarization_segments:
                job.add_diarization_segment(diarization_segment)
            
            job.processing_metrics = metrics
            
            # Update status to post-processing
            job.update_status(JobStatus.POST_PROCESSING)
            await self.job_repository.update(job)
            
            # Apply post-processing
            await self.apply_post_processing(job)
            
            # Mark as completed
            job.mark_as_completed()
            await self.job_repository.update(job)
            
            logger.info(f"Transcription processing completed for job {job_id}")
            return job
            
        except Exception as e:
            logger.error(f"Transcription processing failed for job {job_id}: {e}")
            job.mark_as_failed(str(e))
            await self.job_repository.update(job)
            raise TranscriptionError(f"Processing failed: {e}")
    
    async def apply_post_processing(self, job: TranscriptionJob) -> None:
        """
        Apply post-processing to transcription results.
        
        Args:
            job: Job to post-process
        """
        logger.info(f"Applying post-processing to job {job.id}")
        
        # Detect and remove hallucinations
        segments_to_remove = await self.hallucination_detector.detect_hallucinations(
            job.segments
        )
        
        if segments_to_remove:
            logger.info(f"Removing {len(segments_to_remove)} hallucinated segments")
            job.segments = [
                segment for segment in job.segments 
                if segment not in segments_to_remove
            ]
        
        # Remap speakers chronologically
        job.segments = self._remap_speakers_chronologically(job.segments)
        
        job.updated_at = datetime.utcnow()
    
    def _remap_speakers_chronologically(
        self, 
        segments: List[TranscriptionSegment]
    ) -> List[TranscriptionSegment]:
        """
        Remap speaker IDs to be chronological (SPEAKER_01, SPEAKER_02, etc.).
        
        Args:
            segments: List of transcription segments
            
        Returns:
            Segments with remapped speaker IDs
        """
        if not segments:
            return segments
        
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda s: s.interval.start)
        
        # Create speaker mapping
        speaker_mapping = {}
        speaker_counter = 1
        
        for segment in sorted_segments:
            if segment.speaker_id not in speaker_mapping:
                speaker_mapping[segment.speaker_id] = f"SPEAKER_{speaker_counter:02d}"
                speaker_counter += 1
        
        # Apply mapping
        remapped_segments = []
        for segment in segments:
            new_speaker_id = speaker_mapping[segment.speaker_id]
            remapped_segment = TranscriptionSegment(
                text=segment.text,
                speaker_id=new_speaker_id,
                interval=segment.interval,
                language=segment.language,
                confidence=segment.confidence
            )
            remapped_segments.append(remapped_segment)
        
        logger.info(f"Remapped {len(speaker_mapping)} speakers: {speaker_mapping}")
        return remapped_segments


class HallucinationDetectionService:
    """Service for hallucination detection and management."""
    
    def __init__(self, hallucination_repository):
        self.hallucination_repository = hallucination_repository
    
    async def detect_hallucinations(
        self, 
        segments: List[TranscriptionSegment]
    ) -> List[TranscriptionSegment]:
        """
        Detect hallucinated segments using contextual analysis.
        
        Args:
            segments: List of transcription segments
            
        Returns:
            List of segments identified as hallucinations
        """
        if len(segments) < 3:
            return []
        
        logger.info(f"Detecting hallucinations in {len(segments)} segments")
        
        # Get active hallucination patterns
        patterns = await self.hallucination_repository.get_active_patterns()
        primary_languages = self._get_primary_languages(segments)
        
        hallucinated_segments = []
        
        # Check each segment in context (sandwich detection)
        for i in range(1, len(segments) - 1):
            prev_segment = segments[i - 1]
            current_segment = segments[i]
            next_segment = segments[i + 1]
            
            if self._is_sandwiched_hallucination(
                prev_segment, current_segment, next_segment, patterns, primary_languages
            ):
                hallucinated_segments.append(current_segment)
        
        logger.info(f"Detected {len(hallucinated_segments)} hallucinated segments")
        return hallucinated_segments
    
    def _is_sandwiched_hallucination(
        self,
        prev_segment: TranscriptionSegment,
        current_segment: TranscriptionSegment,
        next_segment: TranscriptionSegment,
        patterns: List[HallucinationPattern],
        primary_languages: Set[str]
    ) -> bool:
        """Check if a segment is a sandwiched hallucination."""
        # Check if segment is sandwiched between same speaker
        if not (prev_segment.speaker_id == next_segment.speaker_id and 
                current_segment.speaker_id != prev_segment.speaker_id):
            return False
        
        # Check timing constraints
        if current_segment.duration > 4.0:  # Max duration for hallucination
            return False
        
        # Check for pattern matches
        for pattern in patterns:
            if pattern.matches(current_segment.text, current_segment.language):
                return True
        
        # Check for language outliers
        if (current_segment.language not in primary_languages and 
            current_segment.language != Language.UNKNOWN):
            return True
        
        return False
    
    def _get_primary_languages(self, segments: List[TranscriptionSegment]) -> Set[str]:
        """Get primary languages based on frequency."""
        language_counts = {}
        for segment in segments:
            if segment.language and segment.language != Language.UNKNOWN:
                language_counts[segment.language] = language_counts.get(segment.language, 0) + 1
        
        # Languages that appear more than twice are considered primary
        return {lang for lang, count in language_counts.items() if count > 2}
    
    async def create_hallucination_pattern(
        self, 
        text_pattern: str,
        language: Optional[str] = None,
        description: Optional[str] = None,
        is_regex: bool = False
    ) -> HallucinationRule:
        """Create a new hallucination detection pattern."""
        pattern = HallucinationPattern(
            text_pattern=text_pattern,
            language=language,
            description=description,
            is_regex=is_regex
        )
        
        rule = HallucinationRule(pattern=pattern)
        
        # Validate rule
        validation_errors = rule.validate()
        if validation_errors:
            raise ValidationError(f"Pattern validation failed: {'; '.join(validation_errors)}")
        
        return await self.hallucination_repository.create(rule)
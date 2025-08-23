"""
Domain models for transcription business logic.
These models contain business rules and behavior, separate from persistence concerns.
"""

from typing import List, Optional, Dict, Any, Set
from uuid import UUID, uuid4
from datetime import datetime
from dataclasses import dataclass, field

from ..shared.enums import JobStatus, ProcessingStage, Language
from ..shared.value_objects import (
    AudioMetadata, 
    TimeInterval, 
    SpeakerSegment, 
    TranscriptionSegment,
    ProcessingMetrics,
    JobProgress,
    HallucinationPattern
)


@dataclass
class TranscriptionJob:
    """
    Domain model for a transcription job.
    Contains business logic and rules for job processing.
    """
    id: UUID = field(default_factory=uuid4)
    user_id: UUID = field(default_factory=uuid4)
    original_filename: str = ""
    audio_file_path: str = ""
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None
    audio_metadata: Optional[AudioMetadata] = None
    is_post_processed: bool = False
    is_viewed: bool = False
    
    # Processing results
    segments: List[TranscriptionSegment] = field(default_factory=list)
    diarization_segments: List[SpeakerSegment] = field(default_factory=list)
    processing_metrics: Optional[ProcessingMetrics] = None
    
    def update_status(self, new_status: JobStatus, error_message: Optional[str] = None) -> None:
        """Update job status with timestamp and optional error."""
        self.status = new_status
        self.updated_at = datetime.utcnow()
        if error_message:
            self.error_message = error_message
    
    def mark_as_failed(self, error_message: str) -> None:
        """Mark job as failed with error message."""
        self.update_status(JobStatus.FAILED, error_message)
    
    def mark_as_completed(self) -> None:
        """Mark job as completed."""
        self.update_status(JobStatus.COMPLETED)
        self.is_post_processed = True
    
    def add_segment(self, segment: TranscriptionSegment) -> None:
        """Add a transcription segment to the job."""
        self.segments.append(segment)
        self.updated_at = datetime.utcnow()
    
    def add_diarization_segment(self, segment: SpeakerSegment) -> None:
        """Add a diarization segment to the job."""
        self.diarization_segments.append(segment)
    
    def get_speakers(self) -> Set[str]:
        """Get unique speaker IDs from segments."""
        return {segment.speaker_id for segment in self.segments}
    
    def get_total_duration(self) -> float:
        """Get total duration from audio metadata."""
        return self.audio_metadata.duration_seconds if self.audio_metadata else 0.0
    
    def get_speech_duration(self) -> float:
        """Get total duration of speech segments."""
        return sum(segment.duration for segment in self.segments)
    
    def get_speech_ratio(self) -> float:
        """Get ratio of speech to total audio duration."""
        total_duration = self.get_total_duration()
        if total_duration == 0:
            return 0.0
        return self.get_speech_duration() / total_duration
    
    def get_segments_by_speaker(self, speaker_id: str) -> List[TranscriptionSegment]:
        """Get all segments for a specific speaker."""
        return [segment for segment in self.segments if segment.speaker_id == speaker_id]
    
    def get_full_transcript(self) -> str:
        """Generate full transcript with speaker labels."""
        if not self.segments:
            return ""
        
        transcript_parts = []
        current_speaker = None
        
        # Sort segments by start time
        sorted_segments = sorted(self.segments, key=lambda s: s.interval.start)
        
        for segment in sorted_segments:
            if segment.speaker_id != current_speaker:
                if current_speaker is not None:
                    transcript_parts.append("\n\n")
                transcript_parts.append(f"{segment.speaker_id}:\n{segment.text}")
                current_speaker = segment.speaker_id
            else:
                transcript_parts.append(f" {segment.text}")
        
        return "".join(transcript_parts)
    
    def is_processing(self) -> bool:
        """Check if job is currently being processed."""
        return self.status in [
            JobStatus.DIARIZING,
            JobStatus.TRANSCRIBING,
            JobStatus.PROCESSING,
            JobStatus.POST_PROCESSING
        ]
    
    def can_be_deleted(self) -> bool:
        """Check if job can be safely deleted."""
        return self.status in [JobStatus.COMPLETED, JobStatus.FAILED]
    
    def validate(self) -> List[str]:
        """Validate job data and return list of validation errors."""
        errors = []
        
        if not self.original_filename:
            errors.append("Original filename is required")
        
        if not self.audio_file_path:
            errors.append("Audio file path is required")
        
        if self.status == JobStatus.FAILED and not self.error_message:
            errors.append("Failed jobs must have an error message")
        
        return errors


@dataclass
class User:
    """Domain model for users."""
    id: UUID = field(default_factory=uuid4)
    username: str = ""
    email: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True
    
    def validate(self) -> List[str]:
        """Validate user data and return list of validation errors."""
        errors = []
        
        if not self.username or len(self.username) < 3:
            errors.append("Username must be at least 3 characters long")
        
        if not self.email or "@" not in self.email:
            errors.append("Valid email address is required")
        
        return errors


@dataclass
class HallucinationRule:
    """Domain model for hallucination detection rules."""
    id: UUID = field(default_factory=uuid4)
    pattern: HallucinationPattern = field(default_factory=lambda: HallucinationPattern(""))
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def matches_segment(self, segment: TranscriptionSegment) -> bool:
        """Check if this rule matches a transcription segment."""
        if not self.is_active:
            return False
        
        return self.pattern.matches(segment.text, segment.language)
    
    def validate(self) -> List[str]:
        """Validate rule data and return list of validation errors."""
        errors = []
        
        if not self.pattern.text_pattern:
            errors.append("Pattern text is required")
        
        if self.pattern.is_regex:
            try:
                import re
                re.compile(self.pattern.text_pattern)
            except re.error as e:
                errors.append(f"Invalid regex pattern: {e}")
        
        return errors
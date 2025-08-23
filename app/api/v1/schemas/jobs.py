"""
Pydantic schemas for job-related API endpoints.
"""

from typing import Optional, List
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from .common import BaseSchema, TimestampMixin, PaginationResponse, TimeInterval, ProcessingMetrics
from ....domain.shared.enums import JobStatus, Language


class TranscriptionSegment(BaseSchema):
    """Schema for transcription segments."""
    text: str = Field(..., description="Transcribed text")
    speaker_id: str = Field(..., description="Speaker identifier")
    interval: TimeInterval = Field(..., description="Time interval")
    language: str = Field(..., description="Detected language")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")


class SpeakerSegment(BaseSchema):
    """Schema for speaker diarization segments."""
    speaker_id: str = Field(..., description="Speaker identifier")
    interval: TimeInterval = Field(..., description="Time interval")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")


class AudioMetadata(BaseSchema):
    """Schema for audio metadata."""
    duration_seconds: float = Field(..., ge=0, description="Audio duration in seconds")
    sample_rate: int = Field(..., gt=0, description="Sample rate in Hz")
    channels: int = Field(..., gt=0, description="Number of audio channels")
    format: str = Field(..., description="Audio format")
    file_size_bytes: int = Field(..., ge=0, description="File size in bytes")
    rms_level: float = Field(..., ge=0, description="RMS audio level")
    peak_level: float = Field(..., ge=0, description="Peak audio level")


class JobCreateRequest(BaseSchema):
    """Request schema for creating a new transcription job."""
    user_id: UUID = Field(..., description="User ID")
    # Note: file will be handled as UploadFile in the endpoint


class JobCreateResponse(BaseSchema, TimestampMixin):
    """Response schema for job creation."""
    id: UUID = Field(..., description="Job ID")
    user_id: UUID = Field(..., description="User ID")
    original_filename: str = Field(..., description="Original filename")
    status: JobStatus = Field(..., description="Job status")


class Job(BaseSchema, TimestampMixin):
    """Complete job schema with all details."""
    id: UUID = Field(..., description="Job ID")
    user_id: UUID = Field(..., description="User ID")
    original_filename: str = Field(..., description="Original filename")
    status: JobStatus = Field(..., description="Job status")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    audio_metadata: Optional[AudioMetadata] = Field(None, description="Audio metadata")
    is_post_processed: bool = Field(default=False, description="Post-processing status")
    is_viewed: bool = Field(default=False, description="User viewed status")
    
    # Transcription results
    segments: List[TranscriptionSegment] = Field(default_factory=list, description="Transcription segments")
    diarization_segments: List[SpeakerSegment] = Field(default_factory=list, description="Diarization segments")
    processing_metrics: Optional[ProcessingMetrics] = Field(None, description="Processing metrics")
    
    # Computed fields
    full_text: Optional[str] = Field(None, description="Complete transcript with speaker labels")
    
    @field_validator('full_text', mode='before')
    @classmethod
    def generate_full_text(cls, v, info):
        """Generate full text from segments if not provided."""
        if v is not None:
            return v
        
        # Get data from ValidationInfo context
        data = info.data if hasattr(info, 'data') else {}
        segments = data.get('segments', [])
        if not segments:
            return ""
        
        transcript_parts = []
        current_speaker = None
        
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda s: s.interval.start)
        
        for segment in sorted_segments:
            if segment.speaker_id != current_speaker:
                if current_speaker is not None:
                    transcript_parts.append("\n\n")
                transcript_parts.append(f"{segment.speaker_id}:\n{segment.text}")
                current_speaker = segment.speaker_id
            else:
                transcript_parts.append(f" {segment.text}")
        
        return "".join(transcript_parts)


class JobSummary(BaseSchema, TimestampMixin):
    """Summary job schema for lists."""
    id: UUID = Field(..., description="Job ID")
    user_id: UUID = Field(..., description="User ID")
    original_filename: str = Field(..., description="Original filename")
    status: JobStatus = Field(..., description="Job status")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    audio_duration: Optional[float] = Field(None, description="Audio duration in seconds")
    segments_count: int = Field(default=0, description="Number of segments")
    speakers_count: int = Field(default=0, description="Number of unique speakers")
    is_viewed: bool = Field(default=False, description="User viewed status")


class JobStatusUpdate(BaseSchema):
    """Schema for updating job status."""
    status: JobStatus = Field(..., description="New job status")
    error_message: Optional[str] = Field(None, description="Error message for failed jobs")


class JobListRequest(BaseSchema):
    """Request schema for listing user jobs."""
    user_id: UUID = Field(..., description="User ID")
    status: Optional[JobStatus] = Field(None, description="Filter by status")
    page: int = Field(default=1, ge=1, description="Page number")
    size: int = Field(default=9, ge=1, le=100, description="Page size")


class JobListResponse(PaginationResponse):
    """Response schema for job list with pagination."""
    items: List[JobSummary] = Field(..., description="Job items")


class JobDeleteResponse(BaseSchema):
    """Response schema for job deletion."""
    success: bool = Field(True, description="Deletion success")
    message: str = Field(default="Job deleted successfully", description="Success message")


class JobViewedResponse(BaseSchema):
    """Response schema for marking job as viewed."""
    success: bool = Field(True, description="Operation success")
    message: str = Field(default="Job marked as viewed", description="Success message")


class JobProgressResponse(BaseSchema):
    """Response schema for job progress updates."""
    job_id: UUID = Field(..., description="Job ID")
    status: JobStatus = Field(..., description="Current status")
    stage: str = Field(..., description="Current processing stage")
    progress_percentage: float = Field(..., ge=0, le=100, description="Progress percentage")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    current_operation: Optional[str] = Field(None, description="Current operation description")
"""
Value objects for the domain layer.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True)
class AudioMetadata:
    """Audio file metadata."""
    duration_seconds: float
    sample_rate: int
    channels: int
    format: str
    file_size_bytes: int
    rms_level: float
    peak_level: float


@dataclass(frozen=True)
class TimeInterval:
    """Represents a time interval with start and end."""
    start: float
    end: float
    
    @property
    def duration(self) -> float:
        """Get the duration of the interval."""
        return self.end - self.start
    
    def overlaps_with(self, other: "TimeInterval") -> bool:
        """Check if this interval overlaps with another."""
        return self.start < other.end and self.end > other.start
    
    def contains(self, timestamp: float) -> bool:
        """Check if a timestamp is within this interval."""
        return self.start <= timestamp <= self.end


@dataclass(frozen=True)
class SpeakerSegment:
    """Represents a speaker segment with timing and metadata."""
    speaker_id: str
    interval: TimeInterval
    confidence: Optional[float] = None
    
    @property
    def duration(self) -> float:
        """Get the duration of the segment."""
        return self.interval.duration


@dataclass(frozen=True)
class TranscriptionSegment:
    """Represents a transcribed segment."""
    text: str
    speaker_id: str
    interval: TimeInterval
    language: str
    confidence: Optional[float] = None
    
    @property
    def duration(self) -> float:
        """Get the duration of the segment."""
        return self.interval.duration
    
    @property
    def word_count(self) -> int:
        """Get the number of words in the text."""
        return len(self.text.split()) if self.text else 0


@dataclass(frozen=True)
class ProcessingMetrics:
    """Metrics for processing operations."""
    processing_time: float
    audio_duration: float
    segments_processed: int
    segments_skipped: int
    error_count: int
    memory_usage_mb: Optional[float] = None
    
    @property
    def real_time_factor(self) -> float:
        """Calculate real-time factor (processing_time / audio_duration)."""
        return self.processing_time / self.audio_duration if self.audio_duration > 0 else 0.0
    
    @property
    def segments_per_second(self) -> float:
        """Calculate segments processed per second."""
        return self.segments_processed / self.processing_time if self.processing_time > 0 else 0.0


@dataclass(frozen=True)
class JobProgress:
    """Represents job processing progress."""
    job_id: UUID
    stage: str
    progress_percentage: float
    estimated_completion: Optional[datetime] = None
    current_operation: Optional[str] = None
    
    def is_complete(self) -> bool:
        """Check if the job is complete."""
        return self.progress_percentage >= 100.0


@dataclass(frozen=True)
class HallucinationPattern:
    """Represents a pattern for hallucination detection."""
    text_pattern: str
    language: Optional[str] = None
    description: Optional[str] = None
    is_regex: bool = False
    
    def matches(self, text: str, language: Optional[str] = None) -> bool:
        """Check if this pattern matches the given text."""
        if self.language and language and self.language != language:
            return False
        
        if self.is_regex:
            import re
            return bool(re.search(self.text_pattern, text, re.IGNORECASE))
        else:
            return self.text_pattern.lower() in text.lower()


@dataclass(frozen=True)
class ModelConfiguration:
    """Configuration for AI models."""
    model_path: str
    device: str
    batch_size: int = 1
    additional_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_params is None:
            object.__setattr__(self, 'additional_params', {})


@dataclass(frozen=True)
class AudioChunk:
    """Represents a chunk of audio data."""
    data: bytes
    start_time: float
    end_time: float
    sample_rate: int
    
    @property
    def duration(self) -> float:
        """Get the duration of the chunk."""
        return self.end_time - self.start_time
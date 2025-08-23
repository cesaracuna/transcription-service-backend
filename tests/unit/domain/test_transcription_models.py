"""
Unit tests for transcription domain models.
"""

import pytest
import uuid
from datetime import datetime, timezone
from unittest.mock import patch

from app.domain.transcription.models import TranscriptionJob, User, TranscriptionSegment
from app.domain.shared.enums import JobStatus, SegmentType, ConfidenceLevel
from app.domain.shared.value_objects import AudioMetadata, TranscriptionResult
from app.domain.shared.exceptions import DomainValidationError


class TestUser:
    """Test cases for User domain model."""
    
    def test_user_creation(self):
        """Test user can be created with valid data."""
        user_id = uuid.uuid4()
        user = User(
            id=user_id,
            username="testuser",
            email="test@example.com"
        )
        
        assert user.id == user_id
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert isinstance(user.created_at, datetime)
    
    def test_user_equality(self):
        """Test user equality based on ID."""
        user_id = uuid.uuid4()
        user1 = User(id=user_id, username="test", email="test@example.com")
        user2 = User(id=user_id, username="different", email="different@example.com")
        user3 = User(id=uuid.uuid4(), username="test", email="test@example.com")
        
        assert user1 == user2  # Same ID
        assert user1 != user3  # Different ID
    
    def test_user_string_representation(self):
        """Test user string representation."""
        user = User(
            id=uuid.uuid4(),
            username="testuser",
            email="test@example.com"
        )
        
        assert str(user) == "testuser"


class TestTranscriptionJob:
    """Test cases for TranscriptionJob domain model."""
    
    @pytest.fixture
    def sample_user(self):
        """Create a sample user for testing."""
        return User(
            id=uuid.uuid4(),
            username="testuser",
            email="test@example.com"
        )
    
    @pytest.fixture
    def sample_job(self, sample_user):
        """Create a sample transcription job."""
        return TranscriptionJob(
            id=uuid.uuid4(),
            user_id=sample_user.id,
            original_filename="test_audio.wav",
            audio_file_path="/tmp/test_audio.wav",
            status=JobStatus.PENDING
        )
    
    def test_job_creation(self, sample_user):
        """Test job can be created with valid data."""
        job_id = uuid.uuid4()
        job = TranscriptionJob(
            id=job_id,
            user_id=sample_user.id,
            original_filename="test.wav",
            audio_file_path="/path/to/test.wav",
            status=JobStatus.PENDING
        )
        
        assert job.id == job_id
        assert job.user_id == sample_user.id
        assert job.original_filename == "test.wav"
        assert job.audio_file_path == "/path/to/test.wav"
        assert job.status == JobStatus.PENDING
        assert isinstance(job.created_at, datetime)
        assert job.started_at is None
        assert job.completed_at is None
    
    def test_update_status_to_processing(self, sample_job):
        """Test updating job status to processing."""
        with patch('app.domain.transcription.models.datetime') as mock_datetime:
            mock_now = datetime.now(timezone.utc)
            mock_datetime.now.return_value = mock_now
            
            sample_job.update_status(JobStatus.PROCESSING)
            
            assert sample_job.status == JobStatus.PROCESSING
            assert sample_job.started_at == mock_now
            assert sample_job.updated_at == mock_now
            assert sample_job.error_message is None
    
    def test_update_status_to_completed(self, sample_job):
        """Test updating job status to completed."""
        with patch('app.domain.transcription.models.datetime') as mock_datetime:
            mock_now = datetime.now(timezone.utc)
            mock_datetime.now.return_value = mock_now
            
            sample_job.update_status(JobStatus.COMPLETED)
            
            assert sample_job.status == JobStatus.COMPLETED
            assert sample_job.completed_at == mock_now
            assert sample_job.updated_at == mock_now
    
    def test_update_status_to_failed_with_error(self, sample_job):
        """Test updating job status to failed with error message."""
        error_msg = "Processing failed due to invalid audio format"
        
        with patch('app.domain.transcription.models.datetime') as mock_datetime:
            mock_now = datetime.now(timezone.utc)
            mock_datetime.now.return_value = mock_now
            
            sample_job.update_status(JobStatus.FAILED, error_msg)
            
            assert sample_job.status == JobStatus.FAILED
            assert sample_job.error_message == error_msg
            assert sample_job.completed_at == mock_now
            assert sample_job.updated_at == mock_now
    
    def test_set_audio_metadata(self, sample_job):
        """Test setting audio metadata."""
        metadata = AudioMetadata(
            duration_seconds=120.5,
            sample_rate=16000,
            channels=1,
            format="wav",
            file_size_bytes=1024000,
            rms_level=0.5,
            peak_level=0.8
        )
        
        sample_job.set_audio_metadata(metadata)
        
        assert sample_job.audio_metadata == metadata
        assert isinstance(sample_job.updated_at, datetime)
    
    def test_set_transcription_result(self, sample_job):
        """Test setting transcription result."""
        result = TranscriptionResult(
            text="Hello, this is a test transcription.",
            confidence=0.95,
            processing_time_seconds=45.2,
            model_version="whisper-large-v3",
            language="en",
            segments=[]
        )
        
        sample_job.set_transcription_result(result)
        
        assert sample_job.transcription_result == result
        assert isinstance(sample_job.updated_at, datetime)
    
    def test_is_terminal_status(self, sample_job):
        """Test checking if job has terminal status."""
        # Non-terminal statuses
        sample_job.status = JobStatus.PENDING
        assert not sample_job.is_terminal_status()
        
        sample_job.status = JobStatus.PROCESSING
        assert not sample_job.is_terminal_status()
        
        # Terminal statuses
        sample_job.status = JobStatus.COMPLETED
        assert sample_job.is_terminal_status()
        
        sample_job.status = JobStatus.FAILED
        assert sample_job.is_terminal_status()
        
        sample_job.status = JobStatus.CANCELLED
        assert sample_job.is_terminal_status()
    
    def test_get_duration_if_completed(self, sample_job):
        """Test getting job duration for completed jobs."""
        # Job not completed
        assert sample_job.get_duration() is None
        
        # Simulate job processing
        start_time = datetime.now(timezone.utc)
        end_time = start_time.replace(second=start_time.second + 30)  # 30 seconds later
        
        sample_job.started_at = start_time
        sample_job.completed_at = end_time
        
        duration = sample_job.get_duration()
        assert duration is not None
        assert duration.total_seconds() == 30.0
    
    def test_job_string_representation(self, sample_job):
        """Test job string representation."""
        expected = f"TranscriptionJob({sample_job.id}, {sample_job.original_filename}, {sample_job.status.value})"
        assert str(sample_job) == expected


class TestTranscriptionSegment:
    """Test cases for TranscriptionSegment domain model."""
    
    @pytest.fixture
    def sample_segment(self):
        """Create a sample transcription segment."""
        return TranscriptionSegment(
            id=uuid.uuid4(),
            job_id=uuid.uuid4(),
            start_time=0.0,
            end_time=5.0,
            text="Hello world",
            confidence=0.95,
            speaker_label="SPEAKER_00",
            segment_type=SegmentType.SPEECH
        )
    
    def test_segment_creation(self):
        """Test segment can be created with valid data."""
        segment_id = uuid.uuid4()
        job_id = uuid.uuid4()
        
        segment = TranscriptionSegment(
            id=segment_id,
            job_id=job_id,
            start_time=10.5,
            end_time=15.2,
            text="Test segment",
            confidence=0.88,
            speaker_label="SPEAKER_01",
            segment_type=SegmentType.SPEECH
        )
        
        assert segment.id == segment_id
        assert segment.job_id == job_id
        assert segment.start_time == 10.5
        assert segment.end_time == 15.2
        assert segment.text == "Test segment"
        assert segment.confidence == 0.88
        assert segment.speaker_label == "SPEAKER_01"
        assert segment.segment_type == SegmentType.SPEECH
    
    def test_segment_duration(self, sample_segment):
        """Test segment duration calculation."""
        assert sample_segment.duration == 5.0
    
    def test_segment_confidence_level(self, sample_segment):
        """Test segment confidence level classification."""
        # High confidence
        sample_segment.confidence = 0.95
        assert sample_segment.confidence_level == ConfidenceLevel.HIGH
        
        # Medium confidence
        sample_segment.confidence = 0.75
        assert sample_segment.confidence_level == ConfidenceLevel.MEDIUM
        
        # Low confidence
        sample_segment.confidence = 0.45
        assert sample_segment.confidence_level == ConfidenceLevel.LOW
    
    def test_segment_validation_invalid_time_range(self):
        """Test segment validation with invalid time range."""
        with pytest.raises(DomainValidationError, match="Start time must be less than end time"):
            TranscriptionSegment(
                id=uuid.uuid4(),
                job_id=uuid.uuid4(),
                start_time=10.0,
                end_time=5.0,  # Invalid: end before start
                text="Test",
                confidence=0.9,
                speaker_label="SPEAKER_00",
                segment_type=SegmentType.SPEECH
            )
    
    def test_segment_validation_negative_time(self):
        """Test segment validation with negative time."""
        with pytest.raises(DomainValidationError, match="Start time cannot be negative"):
            TranscriptionSegment(
                id=uuid.uuid4(),
                job_id=uuid.uuid4(),
                start_time=-1.0,  # Invalid: negative time
                end_time=5.0,
                text="Test",
                confidence=0.9,
                speaker_label="SPEAKER_00",
                segment_type=SegmentType.SPEECH
            )
    
    def test_segment_validation_invalid_confidence(self):
        """Test segment validation with invalid confidence."""
        with pytest.raises(DomainValidationError, match="Confidence must be between 0.0 and 1.0"):
            TranscriptionSegment(
                id=uuid.uuid4(),
                job_id=uuid.uuid4(),
                start_time=0.0,
                end_time=5.0,
                text="Test",
                confidence=1.5,  # Invalid: confidence > 1.0
                speaker_label="SPEAKER_00",
                segment_type=SegmentType.SPEECH
            )
    
    def test_segment_validation_empty_text(self):
        """Test segment validation with empty text."""
        with pytest.raises(DomainValidationError, match="Text cannot be empty"):
            TranscriptionSegment(
                id=uuid.uuid4(),
                job_id=uuid.uuid4(),
                start_time=0.0,
                end_time=5.0,
                text="",  # Invalid: empty text
                confidence=0.9,
                speaker_label="SPEAKER_00",
                segment_type=SegmentType.SPEECH
            )
    
    def test_segment_overlap_detection(self, sample_segment):
        """Test segment overlap detection."""
        # Overlapping segment
        overlapping = TranscriptionSegment(
            id=uuid.uuid4(),
            job_id=sample_segment.job_id,
            start_time=3.0,  # Overlaps with sample_segment (0.0-5.0)
            end_time=8.0,
            text="Overlapping text",
            confidence=0.9,
            speaker_label="SPEAKER_01",
            segment_type=SegmentType.SPEECH
        )
        
        assert sample_segment.overlaps_with(overlapping)
        assert overlapping.overlaps_with(sample_segment)
        
        # Non-overlapping segment
        non_overlapping = TranscriptionSegment(
            id=uuid.uuid4(),
            job_id=sample_segment.job_id,
            start_time=6.0,  # After sample_segment
            end_time=10.0,
            text="Non-overlapping text",
            confidence=0.9,
            speaker_label="SPEAKER_01",
            segment_type=SegmentType.SPEECH
        )
        
        assert not sample_segment.overlaps_with(non_overlapping)
        assert not non_overlapping.overlaps_with(sample_segment)
    
    def test_segment_string_representation(self, sample_segment):
        """Test segment string representation."""
        expected = f"Segment({sample_segment.start_time}-{sample_segment.end_time}s: {sample_segment.text[:50]})"
        assert str(sample_segment) == expected
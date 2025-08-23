"""
Unit tests for transcription service.
"""

import pytest
import uuid
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from app.domain.transcription.models import TranscriptionJob, User
from app.domain.transcription.services import TranscriptionService
from app.domain.transcription.repositories import TranscriptionJobRepository, UserRepository
from app.domain.shared.enums import JobStatus, SegmentType
from app.domain.shared.value_objects import AudioMetadata, TranscriptionResult
from app.domain.shared.exceptions import (
    DomainValidationError, 
    ResourceNotFoundError, 
    BusinessRuleViolationError
)
from app.infrastructure.storage.base import FileManager


class TestTranscriptionService:
    """Test cases for TranscriptionService."""
    
    @pytest.fixture
    def mock_job_repository(self):
        """Mock job repository."""
        return AsyncMock(spec=TranscriptionJobRepository)
    
    @pytest.fixture
    def mock_user_repository(self):
        """Mock user repository."""
        return AsyncMock(spec=UserRepository)
    
    @pytest.fixture
    def mock_file_manager(self):
        """Mock file manager."""
        return AsyncMock(spec=FileManager)
    
    @pytest.fixture
    def transcription_service(self, mock_job_repository, mock_user_repository, mock_file_manager):
        """Create transcription service with mocked dependencies."""
        return TranscriptionService(
            job_repository=mock_job_repository,
            user_repository=mock_user_repository,
            file_manager=mock_file_manager
        )
    
    @pytest.fixture
    def sample_user(self):
        """Create a sample user."""
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
    
    @pytest.fixture
    def sample_audio_metadata(self):
        """Create sample audio metadata."""
        return AudioMetadata(
            duration_seconds=120.5,
            sample_rate=16000,
            channels=1,
            format="wav",
            file_size_bytes=1024000,
            rms_level=0.5,
            peak_level=0.8
        )

    async def test_create_transcription_job_success(
        self, 
        transcription_service, 
        mock_user_repository,
        mock_job_repository,
        mock_file_manager,
        sample_user,
        sample_audio_metadata
    ):
        """Test successful transcription job creation."""
        # Arrange
        user_id = sample_user.id
        filename = "test_audio.wav"
        file_content = b"fake audio data"
        
        mock_user_repository.get_by_id.return_value = sample_user
        mock_file_manager.save_audio_file.return_value = "/storage/audio_files/test_audio.wav"
        mock_file_manager.get_audio_metadata.return_value = sample_audio_metadata
        mock_job_repository.save.return_value = None
        
        # Act
        job = await transcription_service.create_transcription_job(
            user_id=user_id,
            filename=filename,
            file_content=file_content
        )
        
        # Assert
        assert job.user_id == user_id
        assert job.original_filename == filename
        assert job.status == JobStatus.PENDING
        assert job.audio_metadata == sample_audio_metadata
        assert job.audio_file_path == "/storage/audio_files/test_audio.wav"
        
        mock_user_repository.get_by_id.assert_called_once_with(user_id)
        mock_file_manager.save_audio_file.assert_called_once()
        mock_file_manager.get_audio_metadata.assert_called_once()
        mock_job_repository.save.assert_called_once_with(job)
    
    async def test_create_transcription_job_user_not_found(
        self,
        transcription_service,
        mock_user_repository
    ):
        """Test job creation with non-existent user."""
        # Arrange
        user_id = uuid.uuid4()
        mock_user_repository.get_by_id.return_value = None
        
        # Act & Assert
        with pytest.raises(ResourceNotFoundError, match="User not found"):
            await transcription_service.create_transcription_job(
                user_id=user_id,
                filename="test.wav",
                file_content=b"data"
            )
    
    async def test_create_transcription_job_invalid_filename(
        self,
        transcription_service,
        mock_user_repository,
        sample_user
    ):
        """Test job creation with invalid filename."""
        # Arrange
        mock_user_repository.get_by_id.return_value = sample_user
        
        # Act & Assert
        with pytest.raises(DomainValidationError, match="Invalid audio file format"):
            await transcription_service.create_transcription_job(
                user_id=sample_user.id,
                filename="test.txt",  # Invalid format
                file_content=b"data"
            )
    
    async def test_get_transcription_job_success(
        self,
        transcription_service,
        mock_job_repository,
        sample_job
    ):
        """Test successful job retrieval."""
        # Arrange
        job_id = sample_job.id
        mock_job_repository.get_by_id.return_value = sample_job
        
        # Act
        result = await transcription_service.get_transcription_job(job_id)
        
        # Assert
        assert result == sample_job
        mock_job_repository.get_by_id.assert_called_once_with(job_id)
    
    async def test_get_transcription_job_not_found(
        self,
        transcription_service,
        mock_job_repository
    ):
        """Test job retrieval with non-existent job."""
        # Arrange
        job_id = uuid.uuid4()
        mock_job_repository.get_by_id.return_value = None
        
        # Act & Assert
        with pytest.raises(ResourceNotFoundError, match="Transcription job not found"):
            await transcription_service.get_transcription_job(job_id)
    
    async def test_update_job_status_success(
        self,
        transcription_service,
        mock_job_repository,
        sample_job
    ):
        """Test successful job status update."""
        # Arrange
        job_id = sample_job.id
        new_status = JobStatus.PROCESSING
        mock_job_repository.get_by_id.return_value = sample_job
        mock_job_repository.save.return_value = None
        
        # Act
        updated_job = await transcription_service.update_job_status(
            job_id=job_id,
            status=new_status
        )
        
        # Assert
        assert updated_job.status == new_status
        assert updated_job.error_message is None
        mock_job_repository.get_by_id.assert_called_once_with(job_id)
        mock_job_repository.save.assert_called_once_with(sample_job)
    
    async def test_update_job_status_with_error(
        self,
        transcription_service,
        mock_job_repository,
        sample_job
    ):
        """Test job status update with error message."""
        # Arrange
        job_id = sample_job.id
        new_status = JobStatus.FAILED
        error_message = "Processing failed due to invalid audio format"
        mock_job_repository.get_by_id.return_value = sample_job
        mock_job_repository.save.return_value = None
        
        # Act
        updated_job = await transcription_service.update_job_status(
            job_id=job_id,
            status=new_status,
            error_message=error_message
        )
        
        # Assert
        assert updated_job.status == new_status
        assert updated_job.error_message == error_message
    
    async def test_update_job_status_invalid_transition(
        self,
        transcription_service,
        mock_job_repository,
        sample_job
    ):
        """Test invalid job status transition."""
        # Arrange
        sample_job.status = JobStatus.COMPLETED  # Already completed
        mock_job_repository.get_by_id.return_value = sample_job
        
        # Act & Assert
        with pytest.raises(BusinessRuleViolationError, match="Cannot update status of completed job"):
            await transcription_service.update_job_status(
                job_id=sample_job.id,
                status=JobStatus.PROCESSING
            )
    
    async def test_set_transcription_result_success(
        self,
        transcription_service,
        mock_job_repository,
        sample_job
    ):
        """Test successful transcription result setting."""
        # Arrange
        job_id = sample_job.id
        result = TranscriptionResult(
            text="Hello, this is a test transcription.",
            confidence=0.95,
            processing_time_seconds=45.2,
            model_version="whisper-large-v3",
            language="en",
            segments=[]
        )
        
        mock_job_repository.get_by_id.return_value = sample_job
        mock_job_repository.save.return_value = None
        
        # Act
        updated_job = await transcription_service.set_transcription_result(
            job_id=job_id,
            result=result
        )
        
        # Assert
        assert updated_job.transcription_result == result
        assert updated_job.status == JobStatus.COMPLETED
        mock_job_repository.save.assert_called_once_with(sample_job)
    
    async def test_set_transcription_result_job_not_processing(
        self,
        transcription_service,
        mock_job_repository,
        sample_job
    ):
        """Test setting result on non-processing job."""
        # Arrange
        sample_job.status = JobStatus.PENDING  # Not processing
        mock_job_repository.get_by_id.return_value = sample_job
        
        result = TranscriptionResult(
            text="Test text",
            confidence=0.95,
            processing_time_seconds=10.0,
            model_version="whisper-base",
            language="en",
            segments=[]
        )
        
        # Act & Assert
        with pytest.raises(BusinessRuleViolationError, match="Can only set result for processing jobs"):
            await transcription_service.set_transcription_result(
                job_id=sample_job.id,
                result=result
            )
    
    async def test_get_user_jobs_success(
        self,
        transcription_service,
        mock_job_repository,
        sample_user
    ):
        """Test successful user jobs retrieval."""
        # Arrange
        user_id = sample_user.id
        jobs = [
            TranscriptionJob(
                id=uuid.uuid4(),
                user_id=user_id,
                original_filename="audio1.wav",
                audio_file_path="/tmp/audio1.wav",
                status=JobStatus.COMPLETED
            ),
            TranscriptionJob(
                id=uuid.uuid4(),
                user_id=user_id,
                original_filename="audio2.wav",
                audio_file_path="/tmp/audio2.wav",
                status=JobStatus.PROCESSING
            )
        ]
        
        mock_job_repository.get_by_user_id.return_value = jobs
        
        # Act
        result = await transcription_service.get_user_jobs(
            user_id=user_id,
            limit=10,
            offset=0
        )
        
        # Assert
        assert result == jobs
        mock_job_repository.get_by_user_id.assert_called_once_with(
            user_id=user_id,
            limit=10,
            offset=0
        )
    
    async def test_get_user_jobs_with_status_filter(
        self,
        transcription_service,
        mock_job_repository,
        sample_user
    ):
        """Test user jobs retrieval with status filter."""
        # Arrange
        user_id = sample_user.id
        status_filter = JobStatus.COMPLETED
        
        completed_jobs = [
            TranscriptionJob(
                id=uuid.uuid4(),
                user_id=user_id,
                original_filename="audio1.wav",
                audio_file_path="/tmp/audio1.wav",
                status=JobStatus.COMPLETED
            )
        ]
        
        mock_job_repository.get_by_user_id_and_status.return_value = completed_jobs
        
        # Act
        result = await transcription_service.get_user_jobs(
            user_id=user_id,
            status=status_filter,
            limit=10,
            offset=0
        )
        
        # Assert
        assert result == completed_jobs
        mock_job_repository.get_by_user_id_and_status.assert_called_once_with(
            user_id=user_id,
            status=status_filter,
            limit=10,
            offset=0
        )
    
    async def test_delete_transcription_job_success(
        self,
        transcription_service,
        mock_job_repository,
        mock_file_manager,
        sample_job
    ):
        """Test successful job deletion."""
        # Arrange
        job_id = sample_job.id
        sample_job.status = JobStatus.COMPLETED  # Only completed jobs can be deleted
        
        mock_job_repository.get_by_id.return_value = sample_job
        mock_file_manager.delete_audio_file.return_value = True
        mock_job_repository.delete.return_value = None
        
        # Act
        result = await transcription_service.delete_transcription_job(job_id)
        
        # Assert
        assert result is True
        mock_job_repository.get_by_id.assert_called_once_with(job_id)
        mock_file_manager.delete_audio_file.assert_called_once_with(sample_job.audio_file_path)
        mock_job_repository.delete.assert_called_once_with(job_id)
    
    async def test_delete_transcription_job_active_job(
        self,
        transcription_service,
        mock_job_repository,
        sample_job
    ):
        """Test deletion of active job."""
        # Arrange
        sample_job.status = JobStatus.PROCESSING  # Active job
        mock_job_repository.get_by_id.return_value = sample_job
        
        # Act & Assert
        with pytest.raises(BusinessRuleViolationError, match="Cannot delete active transcription job"):
            await transcription_service.delete_transcription_job(sample_job.id)
    
    async def test_cancel_transcription_job_success(
        self,
        transcription_service,
        mock_job_repository,
        sample_job
    ):
        """Test successful job cancellation."""
        # Arrange
        job_id = sample_job.id
        sample_job.status = JobStatus.PENDING  # Can be cancelled
        
        mock_job_repository.get_by_id.return_value = sample_job
        mock_job_repository.save.return_value = None
        
        # Act
        result = await transcription_service.cancel_transcription_job(job_id)
        
        # Assert
        assert result.status == JobStatus.CANCELLED
        mock_job_repository.save.assert_called_once_with(sample_job)
    
    async def test_cancel_transcription_job_already_completed(
        self,
        transcription_service,
        mock_job_repository,
        sample_job
    ):
        """Test cancellation of completed job."""
        # Arrange
        sample_job.status = JobStatus.COMPLETED
        mock_job_repository.get_by_id.return_value = sample_job
        
        # Act & Assert
        with pytest.raises(BusinessRuleViolationError, match="Cannot cancel completed or failed job"):
            await transcription_service.cancel_transcription_job(sample_job.id)
    
    async def test_validate_audio_file_valid_formats(self, transcription_service):
        """Test audio file validation with valid formats."""
        valid_formats = ["audio.wav", "audio.mp3", "audio.m4a", "audio.flac", "audio.ogg"]
        
        for filename in valid_formats:
            # Should not raise exception
            transcription_service._validate_audio_file(filename)
    
    async def test_validate_audio_file_invalid_formats(self, transcription_service):
        """Test audio file validation with invalid formats."""
        invalid_formats = ["file.txt", "file.pdf", "file.jpg", "file.mp4", "file"]
        
        for filename in invalid_formats:
            with pytest.raises(DomainValidationError, match="Invalid audio file format"):
                transcription_service._validate_audio_file(filename)
    
    async def test_job_statistics(
        self,
        transcription_service,
        mock_job_repository,
        sample_user
    ):
        """Test job statistics calculation."""
        # Arrange
        user_id = sample_user.id
        stats = {
            "total_jobs": 10,
            "completed_jobs": 7,
            "failed_jobs": 2,
            "pending_jobs": 1,
            "total_processing_time": 3600.0
        }
        
        mock_job_repository.get_user_statistics.return_value = stats
        
        # Act
        result = await transcription_service.get_user_statistics(user_id)
        
        # Assert
        assert result == stats
        mock_job_repository.get_user_statistics.assert_called_once_with(user_id)
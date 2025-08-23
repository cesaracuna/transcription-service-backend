"""
Integration tests for the complete transcription workflow.
"""

import pytest
import uuid
import asyncio
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from app.domain.transcription.models import User, TranscriptionJob
from app.domain.shared.enums import JobStatus
from app.domain.shared.value_objects import AudioMetadata


@pytest.mark.integration
class TestTranscriptionWorkflow:
    """Integration tests for complete transcription workflow."""
    
    async def test_complete_transcription_workflow(self, client, test_db, sample_user, sample_audio_file, mock_ai_models):
        """Test complete workflow from upload to result."""
        # Step 1: Create user in database
        test_db.add(sample_user)
        test_db.commit()
        
        # Step 2: Upload audio file
        with open(sample_audio_file, 'rb') as audio_file:
            response = client.post(
                f"/api/v1/users/{sample_user.id}/jobs",
                files={"audio_file": ("test_audio.wav", audio_file, "audio/wav")}
            )
        
        assert response.status_code == 201
        job_data = response.json()
        job_id = job_data["id"]
        
        # Verify job was created
        assert job_data["status"] == JobStatus.PENDING.value
        assert job_data["original_filename"] == "test_audio.wav"
        assert job_data["user_id"] == str(sample_user.id)
        
        # Step 3: Check job status
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        
        # Step 4: Mock Celery task processing
        with patch('app.infrastructure.workers.tasks.process_transcription_task.delay') as mock_task:
            mock_task.return_value.id = "mock-task-id"
            
            # Trigger processing (this would normally be done by Celery)
            response = client.post(f"/api/v1/jobs/{job_id}/process")
            assert response.status_code == 202
        
        # Step 5: Simulate processing completion
        # In real scenario, this would be handled by the Celery worker
        with patch('app.infrastructure.external.ai_models.get_model_registry') as mock_registry:
            mock_registry.return_value.get_whisper_models.return_value = (
                mock_ai_models["whisper_processor"],
                mock_ai_models["whisper_model"]
            )
            mock_registry.return_value.get_diarization_pipeline.return_value = (
                mock_ai_models["diarization_pipeline"]
            )
            
            # Update job status to completed
            response = client.put(
                f"/api/v1/jobs/{job_id}/status",
                json={
                    "status": JobStatus.COMPLETED.value,
                    "result": {
                        "text": "Mock transcription text",
                        "confidence": 0.95,
                        "processing_time_seconds": 10.5,
                        "model_version": "whisper-large-v3",
                        "language": "en",
                        "segments": []
                    }
                }
            )
            assert response.status_code == 200
        
        # Step 6: Verify final result
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        
        final_job = response.json()
        assert final_job["status"] == JobStatus.COMPLETED.value
        assert final_job["transcription_result"]["text"] == "Mock transcription text"
        assert final_job["transcription_result"]["confidence"] == 0.95
    
    async def test_job_failure_workflow(self, client, test_db, sample_user, sample_audio_file):
        """Test workflow when job fails."""
        # Setup user
        test_db.add(sample_user)
        test_db.commit()
        
        # Upload audio file
        with open(sample_audio_file, 'rb') as audio_file:
            response = client.post(
                f"/api/v1/users/{sample_user.id}/jobs",
                files={"audio_file": ("test_audio.wav", audio_file, "audio/wav")}
            )
        
        job_id = response.json()["id"]
        
        # Simulate processing failure
        response = client.put(
            f"/api/v1/jobs/{job_id}/status",
            json={
                "status": JobStatus.FAILED.value,
                "error_message": "Audio file is corrupted"
            }
        )
        assert response.status_code == 200
        
        # Verify failure status
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        
        job_data = response.json()
        assert job_data["status"] == JobStatus.FAILED.value
        assert job_data["error_message"] == "Audio file is corrupted"
    
    async def test_job_cancellation_workflow(self, client, test_db, sample_user, sample_audio_file):
        """Test job cancellation workflow."""
        # Setup user
        test_db.add(sample_user)
        test_db.commit()
        
        # Upload audio file
        with open(sample_audio_file, 'rb') as audio_file:
            response = client.post(
                f"/api/v1/users/{sample_user.id}/jobs",
                files={"audio_file": ("test_audio.wav", audio_file, "audio/wav")}
            )
        
        job_id = response.json()["id"]
        
        # Cancel job
        response = client.post(f"/api/v1/jobs/{job_id}/cancel")
        assert response.status_code == 200
        
        # Verify cancellation
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        
        job_data = response.json()
        assert job_data["status"] == JobStatus.CANCELLED.value
    
    async def test_multiple_jobs_workflow(self, client, test_db, sample_user, sample_audio_file):
        """Test handling multiple jobs for same user."""
        # Setup user
        test_db.add(sample_user)
        test_db.commit()
        
        job_ids = []
        
        # Create multiple jobs
        for i in range(3):
            with open(sample_audio_file, 'rb') as audio_file:
                response = client.post(
                    f"/api/v1/users/{sample_user.id}/jobs",
                    files={"audio_file": (f"test_audio_{i}.wav", audio_file, "audio/wav")}
                )
            
            assert response.status_code == 201
            job_ids.append(response.json()["id"])
        
        # Get all user jobs
        response = client.get(f"/api/v1/users/{sample_user.id}/jobs")
        assert response.status_code == 200
        
        jobs = response.json()
        assert len(jobs) == 3
        
        # Verify all jobs are for the correct user
        for job in jobs:
            assert job["user_id"] == str(sample_user.id)
            assert job["id"] in job_ids


@pytest.mark.integration  
class TestDatabaseOperations:
    """Integration tests for database operations."""
    
    async def test_user_crud_operations(self, test_db):
        """Test user CRUD operations."""
        from app.infrastructure.database.repositories.user_repository import SQLUserRepository
        
        repo = SQLUserRepository(test_db)
        
        # Create user
        user = User(
            id=uuid.uuid4(),
            username="testuser",
            email="test@example.com"
        )
        
        await repo.save(user)
        
        # Read user
        retrieved_user = await repo.get_by_id(user.id)
        assert retrieved_user is not None
        assert retrieved_user.username == "testuser"
        assert retrieved_user.email == "test@example.com"
        
        # Update user
        retrieved_user.email = "updated@example.com"
        await repo.save(retrieved_user)
        
        updated_user = await repo.get_by_id(user.id)
        assert updated_user.email == "updated@example.com"
        
        # Delete user
        await repo.delete(user.id)
        deleted_user = await repo.get_by_id(user.id)
        assert deleted_user is None
    
    async def test_job_crud_operations(self, test_db, sample_user):
        """Test transcription job CRUD operations."""
        from app.infrastructure.database.repositories.transcription_job_repository import SQLTranscriptionJobRepository
        from app.infrastructure.database.repositories.user_repository import SQLUserRepository
        
        user_repo = SQLUserRepository(test_db)
        job_repo = SQLTranscriptionJobRepository(test_db)
        
        # Create user first
        await user_repo.save(sample_user)
        
        # Create job
        job = TranscriptionJob(
            id=uuid.uuid4(),
            user_id=sample_user.id,
            original_filename="test.wav",
            audio_file_path="/tmp/test.wav",
            status=JobStatus.PENDING
        )
        
        await job_repo.save(job)
        
        # Read job
        retrieved_job = await job_repo.get_by_id(job.id)
        assert retrieved_job is not None
        assert retrieved_job.user_id == sample_user.id
        assert retrieved_job.status == JobStatus.PENDING
        
        # Update job status
        retrieved_job.update_status(JobStatus.PROCESSING)
        await job_repo.save(retrieved_job)
        
        updated_job = await job_repo.get_by_id(job.id)
        assert updated_job.status == JobStatus.PROCESSING
        
        # Get jobs by user
        user_jobs = await job_repo.get_by_user_id(sample_user.id)
        assert len(user_jobs) == 1
        assert user_jobs[0].id == job.id
    
    async def test_job_statistics(self, test_db, sample_user):
        """Test job statistics calculation."""
        from app.infrastructure.database.repositories.user_repository import SQLUserRepository
        from app.infrastructure.database.repositories.transcription_job_repository import SQLTranscriptionJobRepository
        
        user_repo = SQLUserRepository(test_db)
        job_repo = SQLTranscriptionJobRepository(test_db)
        
        # Create user
        await user_repo.save(sample_user)
        
        # Create jobs with different statuses
        jobs = [
            TranscriptionJob(
                id=uuid.uuid4(),
                user_id=sample_user.id,
                original_filename="test1.wav",
                audio_file_path="/tmp/test1.wav",
                status=JobStatus.COMPLETED
            ),
            TranscriptionJob(
                id=uuid.uuid4(),
                user_id=sample_user.id,
                original_filename="test2.wav",
                audio_file_path="/tmp/test2.wav",
                status=JobStatus.FAILED
            ),
            TranscriptionJob(
                id=uuid.uuid4(),
                user_id=sample_user.id,
                original_filename="test3.wav",
                audio_file_path="/tmp/test3.wav",
                status=JobStatus.PENDING
            )
        ]
        
        for job in jobs:
            await job_repo.save(job)
        
        # Get statistics
        stats = await job_repo.get_user_statistics(sample_user.id)
        
        assert stats["total_jobs"] == 3
        assert stats["completed_jobs"] == 1
        assert stats["failed_jobs"] == 1
        assert stats["pending_jobs"] == 1


@pytest.mark.integration
class TestFileStorageIntegration:
    """Integration tests for file storage operations."""
    
    async def test_audio_file_storage_workflow(self, file_manager, temp_storage_dir, sample_audio_file):
        """Test complete audio file storage workflow."""
        job_id = str(uuid.uuid4())
        
        # Save audio file
        with open(sample_audio_file, 'rb') as audio_file:
            file_path = await file_manager.save_audio_file(
                audio_file,
                "test_audio.wav",
                job_id
            )
        
        # Verify file exists
        assert await file_manager.file_exists(file_path)
        
        # Get file info
        file_info = await file_manager.get_file_info(file_path)
        assert file_info is not None
        assert file_info["size"] > 0
        
        # Get audio metadata (would normally use FFprobe)
        # For testing, we'll mock this
        with patch('app.infrastructure.storage.audio_storage.AudioFileManager._get_audio_metadata') as mock_metadata:
            mock_metadata.return_value = AudioMetadata(
                duration_seconds=1.0,
                sample_rate=16000,
                channels=1,
                format="wav",
                file_size_bytes=32000,
                rms_level=0.1,
                peak_level=0.3
            )
            
            metadata = await file_manager._get_audio_metadata(file_path)
            assert metadata.duration_seconds == 1.0
            assert metadata.sample_rate == 16000
        
        # Delete file
        delete_result = await file_manager.delete_audio_file(file_path)
        assert delete_result is True
        assert not await file_manager.file_exists(file_path)
    
    async def test_audio_conversion_workflow(self, temp_storage_dir, sample_audio_file):
        """Test audio conversion workflow."""
        from app.infrastructure.storage.audio_storage import AudioFileManager
        from app.infrastructure.storage.file_storage import LocalFileStorage
        
        storage = LocalFileStorage(str(temp_storage_dir))
        audio_manager = AudioFileManager(storage)
        job_id = str(uuid.uuid4())
        
        # Mock FFmpeg operations for testing
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            # Mock successful FFmpeg execution
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate.return_value = (b"", b"")
            mock_subprocess.return_value = mock_process
            
            with patch('app.infrastructure.storage.audio_storage.AudioFileManager._get_audio_metadata') as mock_metadata:
                mock_metadata.return_value = AudioMetadata(
                    duration_seconds=1.0,
                    sample_rate=16000,
                    channels=1,
                    format="wav",
                    file_size_bytes=32000,
                    rms_level=0.1,
                    peak_level=0.3
                )
                
                # Test conversion
                with open(sample_audio_file, 'rb') as audio_file:
                    converted_path, metadata = await audio_manager.save_and_convert_audio(
                        audio_file,
                        "test_audio.wav",
                        job_id,
                        convert_to_standard=True
                    )
                
                # Verify conversion was attempted
                assert mock_subprocess.called
                assert metadata.sample_rate == 16000
                assert metadata.channels == 1


@pytest.mark.integration
class TestRedisIntegration:
    """Integration tests for Redis operations."""
    
    async def test_job_progress_caching(self, async_client):
        """Test job progress caching workflow."""
        from app.infrastructure.external.redis_client import get_redis_client
        
        # This would connect to test Redis instance
        redis_client = await get_redis_client()
        
        job_id = str(uuid.uuid4())
        progress_data = {
            "status": "processing",
            "progress": 50,
            "current_step": "audio_conversion"
        }
        
        # Cache progress
        result = await redis_client.cache_job_progress(job_id, progress_data)
        assert result is True
        
        # Retrieve progress
        cached_progress = await redis_client.get_job_progress(job_id)
        assert cached_progress == progress_data
        
        # Update progress
        updated_progress = {
            "status": "processing",
            "progress": 75,
            "current_step": "transcription"
        }
        
        await redis_client.cache_job_progress(job_id, updated_progress)
        cached_progress = await redis_client.get_job_progress(job_id)
        assert cached_progress["progress"] == 75
        
        # Clear cache
        clear_result = await redis_client.clear_job_cache(job_id)
        assert clear_result is True
        
        # Verify cache is cleared
        cached_progress = await redis_client.get_job_progress(job_id)
        assert cached_progress is None


@pytest.mark.integration
@pytest.mark.slow
class TestConcurrentOperations:
    """Integration tests for concurrent operations."""
    
    async def test_concurrent_job_creation(self, client, test_db, sample_user, sample_audio_file):
        """Test creating multiple jobs concurrently."""
        # Setup user
        test_db.add(sample_user)
        test_db.commit()
        
        async def create_job(job_index):
            with open(sample_audio_file, 'rb') as audio_file:
                response = client.post(
                    f"/api/v1/users/{sample_user.id}/jobs",
                    files={"audio_file": (f"test_audio_{job_index}.wav", audio_file, "audio/wav")}
                )
            return response
        
        # Create 10 jobs concurrently
        tasks = [create_job(i) for i in range(10)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all jobs were created successfully
        successful_responses = [r for r in responses if not isinstance(r, Exception)]
        assert len(successful_responses) == 10
        
        for response in successful_responses:
            assert response.status_code == 201
        
        # Verify all jobs exist in database
        response = client.get(f"/api/v1/users/{sample_user.id}/jobs")
        assert response.status_code == 200
        jobs = response.json()
        assert len(jobs) == 10
    
    async def test_concurrent_status_updates(self, client, test_db, sample_user, sample_audio_file):
        """Test concurrent status updates don't cause conflicts."""
        # Setup user and job
        test_db.add(sample_user)
        test_db.commit()
        
        with open(sample_audio_file, 'rb') as audio_file:
            response = client.post(
                f"/api/v1/users/{sample_user.id}/jobs",
                files={"audio_file": ("test_audio.wav", audio_file, "audio/wav")}
            )
        
        job_id = response.json()["id"]
        
        # Attempt concurrent status updates
        async def update_status(status_value):
            response = client.put(
                f"/api/v1/jobs/{job_id}/status",
                json={"status": status_value}
            )
            return response
        
        # Try to update to processing status concurrently
        tasks = [update_status(JobStatus.PROCESSING.value) for _ in range(5)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # At least one should succeed
        successful_responses = [r for r in responses if not isinstance(r, Exception) and r.status_code == 200]
        assert len(successful_responses) >= 1
        
        # Verify final status
        response = client.get(f"/api/v1/jobs/{job_id}")
        job_data = response.json()
        assert job_data["status"] == JobStatus.PROCESSING.value
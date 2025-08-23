"""
End-to-end tests for the complete transcription flow.
Tests the entire system from API request to final result.
"""

import pytest
import asyncio
import time
import uuid
from typing import Dict, Any
from unittest.mock import patch, AsyncMock

from app.domain.shared.enums import JobStatus


@pytest.mark.e2e
@pytest.mark.slow
class TestFullTranscriptionFlow:
    """End-to-end tests for complete transcription workflow."""
    
    async def test_complete_transcription_pipeline(
        self,
        async_client,
        test_db,
        sample_user,
        sample_audio_file,
        mock_ai_models
    ):
        """Test the complete transcription pipeline from upload to result."""
        
        # Step 1: Setup user in database
        test_db.add(sample_user)
        test_db.commit()
        
        # Step 2: Upload audio file and create transcription job
        with open(sample_audio_file, 'rb') as audio_file:
            files = {"audio_file": ("test_audio.wav", audio_file, "audio/wav")}
            response = await async_client.post(
                f"/api/v1/users/{sample_user.id}/jobs",
                files=files
            )
        
        assert response.status_code == 201
        job_data = response.json()
        job_id = job_data["id"]
        
        # Verify initial job state
        assert job_data["status"] == JobStatus.PENDING.value
        assert job_data["user_id"] == str(sample_user.id)
        assert job_data["original_filename"] == "test_audio.wav"
        
        # Step 3: Simulate job processing (normally done by Celery worker)
        with patch('app.infrastructure.external.ai_models.get_model_registry') as mock_registry:
            # Mock AI models
            mock_registry.return_value.get_whisper_models.return_value = (
                mock_ai_models["whisper_processor"],
                mock_ai_models["whisper_model"]
            )
            mock_registry.return_value.get_diarization_pipeline.return_value = (
                mock_ai_models["diarization_pipeline"]
            )
            
            # Start processing
            response = await async_client.put(
                f"/api/v1/jobs/{job_id}/status",
                json={"status": JobStatus.PROCESSING.value}
            )
            assert response.status_code == 200
            
            # Verify processing status
            response = await async_client.get(f"/api/v1/jobs/{job_id}")
            job_data = response.json()
            assert job_data["status"] == JobStatus.PROCESSING.value
            assert "started_at" in job_data
            
            # Simulate processing completion with results
            transcription_result = {
                "text": "Hello, this is a test transcription from the end-to-end test.",
                "confidence": 0.95,
                "processing_time_seconds": 15.8,
                "model_version": "whisper-large-v3",
                "language": "en",
                "segments": [
                    {
                        "start_time": 0.0,
                        "end_time": 3.0,
                        "text": "Hello, this is a test",
                        "confidence": 0.96,
                        "speaker_label": "SPEAKER_00"
                    },
                    {
                        "start_time": 3.0,
                        "end_time": 6.0,
                        "text": "transcription from the end-to-end test.",
                        "confidence": 0.94,
                        "speaker_label": "SPEAKER_00"
                    }
                ]
            }
            
            # Complete processing
            response = await async_client.put(
                f"/api/v1/jobs/{job_id}/status",
                json={
                    "status": JobStatus.COMPLETED.value,
                    "result": transcription_result
                }
            )
            assert response.status_code == 200
        
        # Step 4: Verify final result
        response = await async_client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        
        final_job = response.json()
        assert final_job["status"] == JobStatus.COMPLETED.value
        assert "completed_at" in final_job
        
        # Verify transcription result
        result = final_job["transcription_result"]
        assert result["text"] == transcription_result["text"]
        assert result["confidence"] == transcription_result["confidence"]
        assert result["model_version"] == transcription_result["model_version"]
        assert result["language"] == transcription_result["language"]
        assert len(result["segments"]) == 2
        
        # Verify segments
        for i, segment in enumerate(result["segments"]):
            expected_segment = transcription_result["segments"][i]
            assert segment["text"] == expected_segment["text"]
            assert segment["confidence"] == expected_segment["confidence"]
            assert segment["speaker_label"] == expected_segment["speaker_label"]
        
        # Step 5: Verify job appears in user's job list
        response = await async_client.get(f"/api/v1/users/{sample_user.id}/jobs")
        assert response.status_code == 200
        
        user_jobs = response.json()
        assert len(user_jobs) == 1
        assert user_jobs[0]["id"] == job_id
        assert user_jobs[0]["status"] == JobStatus.COMPLETED.value
    
    async def test_transcription_failure_flow(
        self,
        async_client,
        test_db,
        sample_user,
        sample_audio_file
    ):
        """Test the transcription flow when processing fails."""
        
        # Setup user
        test_db.add(sample_user)
        test_db.commit()
        
        # Upload audio file
        with open(sample_audio_file, 'rb') as audio_file:
            files = {"audio_file": ("corrupted_audio.wav", audio_file, "audio/wav")}
            response = await async_client.post(
                f"/api/v1/users/{sample_user.id}/jobs",
                files=files
            )
        
        job_id = response.json()["id"]
        
        # Start processing
        await async_client.put(
            f"/api/v1/jobs/{job_id}/status",
            json={"status": JobStatus.PROCESSING.value}
        )
        
        # Simulate processing failure
        error_message = "Audio file format is not supported or corrupted"
        response = await async_client.put(
            f"/api/v1/jobs/{job_id}/status",
            json={
                "status": JobStatus.FAILED.value,
                "error_message": error_message
            }
        )
        assert response.status_code == 200
        
        # Verify failure state
        response = await async_client.get(f"/api/v1/jobs/{job_id}")
        job_data = response.json()
        
        assert job_data["status"] == JobStatus.FAILED.value
        assert job_data["error_message"] == error_message
        assert "completed_at" in job_data
    
    async def test_job_cancellation_flow(
        self,
        async_client,
        test_db,
        sample_user,
        sample_audio_file
    ):
        """Test job cancellation during processing."""
        
        # Setup user
        test_db.add(sample_user)
        test_db.commit()
        
        # Upload audio file
        with open(sample_audio_file, 'rb') as audio_file:
            files = {"audio_file": ("test_audio.wav", audio_file, "audio/wav")}
            response = await async_client.post(
                f"/api/v1/users/{sample_user.id}/jobs",
                files=files
            )
        
        job_id = response.json()["id"]
        
        # Cancel job while pending
        response = await async_client.post(f"/api/v1/jobs/{job_id}/cancel")
        assert response.status_code == 200
        
        # Verify cancellation
        response = await async_client.get(f"/api/v1/jobs/{job_id}")
        job_data = response.json()
        assert job_data["status"] == JobStatus.CANCELLED.value
    
    async def test_multiple_concurrent_jobs(
        self,
        async_client,
        test_db,
        sample_user,
        sample_audio_file,
        mock_ai_models
    ):
        """Test handling multiple concurrent transcription jobs."""
        
        # Setup user
        test_db.add(sample_user)
        test_db.commit()
        
        # Create multiple jobs concurrently
        async def create_and_process_job(job_index: int) -> Dict[str, Any]:
            # Upload audio file
            with open(sample_audio_file, 'rb') as audio_file:
                files = {"audio_file": (f"test_audio_{job_index}.wav", audio_file, "audio/wav")}
                response = await async_client.post(
                    f"/api/v1/users/{sample_user.id}/jobs",
                    files=files
                )
            
            job_id = response.json()["id"]
            
            # Process job
            with patch('app.infrastructure.external.ai_models.get_model_registry') as mock_registry:
                mock_registry.return_value.get_whisper_models.return_value = (
                    mock_ai_models["whisper_processor"],
                    mock_ai_models["whisper_model"]
                )
                
                # Start processing
                await async_client.put(
                    f"/api/v1/jobs/{job_id}/status",
                    json={"status": JobStatus.PROCESSING.value}
                )
                
                # Complete processing
                result = {
                    "text": f"This is transcription result for job {job_index}",
                    "confidence": 0.90 + (job_index * 0.01),  # Slightly different confidence
                    "processing_time_seconds": 10.0 + job_index,
                    "model_version": "whisper-large-v3",
                    "language": "en",
                    "segments": []
                }
                
                await async_client.put(
                    f"/api/v1/jobs/{job_id}/status",
                    json={
                        "status": JobStatus.COMPLETED.value,
                        "result": result
                    }
                )
            
            return {"job_id": job_id, "index": job_index}
        
        # Process 5 jobs concurrently
        num_jobs = 5
        tasks = [create_and_process_job(i) for i in range(num_jobs)]
        results = await asyncio.gather(*tasks)
        
        # Verify all jobs completed successfully
        assert len(results) == num_jobs
        
        # Check all jobs in user's job list
        response = await async_client.get(f"/api/v1/users/{sample_user.id}/jobs")
        assert response.status_code == 200
        
        user_jobs = response.json()
        assert len(user_jobs) == num_jobs
        
        # Verify all jobs are completed
        for job in user_jobs:
            assert job["status"] == JobStatus.COMPLETED.value
            assert job["transcription_result"] is not None
    
    async def test_user_statistics_after_multiple_jobs(
        self,
        async_client,
        test_db,
        sample_user,
        sample_audio_file
    ):
        """Test user statistics after processing multiple jobs with different outcomes."""
        
        # Setup user
        test_db.add(sample_user)
        test_db.commit()
        
        # Create jobs with different outcomes
        job_outcomes = [
            (JobStatus.COMPLETED, None),
            (JobStatus.COMPLETED, None),
            (JobStatus.FAILED, "Processing error"),
            (JobStatus.CANCELLED, None),
            (JobStatus.PENDING, None)
        ]
        
        job_ids = []
        
        # Create all jobs
        for i, (status, error) in enumerate(job_outcomes):
            with open(sample_audio_file, 'rb') as audio_file:
                files = {"audio_file": (f"test_audio_{i}.wav", audio_file, "audio/wav")}
                response = await async_client.post(
                    f"/api/v1/users/{sample_user.id}/jobs",
                    files=files
                )
            
            job_id = response.json()["id"]
            job_ids.append(job_id)
            
            # Update job status based on outcome
            if status == JobStatus.COMPLETED:
                # Start processing first
                await async_client.put(
                    f"/api/v1/jobs/{job_id}/status",
                    json={"status": JobStatus.PROCESSING.value}
                )
                
                # Then complete
                result = {
                    "text": f"Completed transcription {i}",
                    "confidence": 0.95,
                    "processing_time_seconds": 10.0,
                    "model_version": "whisper-large-v3",
                    "language": "en",
                    "segments": []
                }
                
                await async_client.put(
                    f"/api/v1/jobs/{job_id}/status",
                    json={
                        "status": status.value,
                        "result": result
                    }
                )
            
            elif status == JobStatus.FAILED:
                # Start processing first
                await async_client.put(
                    f"/api/v1/jobs/{job_id}/status",
                    json={"status": JobStatus.PROCESSING.value}
                )
                
                # Then fail
                await async_client.put(
                    f"/api/v1/jobs/{job_id}/status",
                    json={
                        "status": status.value,
                        "error_message": error
                    }
                )
            
            elif status == JobStatus.CANCELLED:
                await async_client.post(f"/api/v1/jobs/{job_id}/cancel")
            
            # PENDING jobs remain as-is
        
        # Get user statistics
        response = await async_client.get(f"/api/v1/users/{sample_user.id}/statistics")
        assert response.status_code == 200
        
        stats = response.json()
        assert stats["total_jobs"] == 5
        assert stats["completed_jobs"] == 2
        assert stats["failed_jobs"] == 1
        assert stats["pending_jobs"] == 1  # Cancelled might be counted separately
        
        # Verify job filtering works
        response = await async_client.get(
            f"/api/v1/users/{sample_user.id}/jobs?status={JobStatus.COMPLETED.value}"
        )
        completed_jobs = response.json()
        assert len(completed_jobs) == 2
        
        response = await async_client.get(
            f"/api/v1/users/{sample_user.id}/jobs?status={JobStatus.FAILED.value}"
        )
        failed_jobs = response.json()
        assert len(failed_jobs) == 1
    
    async def test_job_lifecycle_with_progress_tracking(
        self,
        async_client,
        test_db,
        sample_user,
        sample_audio_file
    ):
        """Test complete job lifecycle with progress tracking via Redis."""
        
        # Setup user
        test_db.add(sample_user)
        test_db.commit()
        
        # Upload audio file
        with open(sample_audio_file, 'rb') as audio_file:
            files = {"audio_file": ("test_audio.wav", audio_file, "audio/wav")}
            response = await async_client.post(
                f"/api/v1/users/{sample_user.id}/jobs",
                files=files
            )
        
        job_id = response.json()["id"]
        
        # Mock Redis client for progress tracking
        with patch('app.infrastructure.external.redis_client.get_redis_client') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            
            # Start processing with progress updates
            await async_client.put(
                f"/api/v1/jobs/{job_id}/status",
                json={"status": JobStatus.PROCESSING.value}
            )
            
            # Simulate progress updates
            progress_steps = [
                {"step": "audio_conversion", "progress": 25},
                {"step": "feature_extraction", "progress": 50},
                {"step": "transcription", "progress": 75},
                {"step": "diarization", "progress": 90},
            ]
            
            for progress in progress_steps:
                # Cache progress
                await mock_client.cache_job_progress(job_id, progress)
                
                # Small delay to simulate processing time
                await asyncio.sleep(0.1)
            
            # Complete processing
            result = {
                "text": "Final transcription result",
                "confidence": 0.95,
                "processing_time_seconds": 25.5,
                "model_version": "whisper-large-v3",
                "language": "en",
                "segments": []
            }
            
            await async_client.put(
                f"/api/v1/jobs/{job_id}/status",
                json={
                    "status": JobStatus.COMPLETED.value,
                    "result": result
                }
            )
            
            # Clear progress cache after completion
            await mock_client.clear_job_cache(job_id)
        
        # Verify final state
        response = await async_client.get(f"/api/v1/jobs/{job_id}")
        job_data = response.json()
        
        assert job_data["status"] == JobStatus.COMPLETED.value
        assert job_data["transcription_result"]["text"] == "Final transcription result"
    
    async def test_file_cleanup_after_job_deletion(
        self,
        async_client,
        test_db,
        sample_user,
        sample_audio_file,
        file_manager
    ):
        """Test that files are properly cleaned up when jobs are deleted."""
        
        # Setup user
        test_db.add(sample_user)
        test_db.commit()
        
        # Upload audio file
        with open(sample_audio_file, 'rb') as audio_file:
            files = {"audio_file": ("test_audio.wav", audio_file, "audio/wav")}
            response = await async_client.post(
                f"/api/v1/users/{sample_user.id}/jobs",
                files=files
            )
        
        job_data = response.json()
        job_id = job_data["id"]
        audio_file_path = job_data["audio_file_path"]
        
        # Complete the job
        await async_client.put(
            f"/api/v1/jobs/{job_id}/status",
            json={"status": JobStatus.COMPLETED.value}
        )
        
        # Verify file exists
        file_exists = await file_manager.file_exists(audio_file_path)
        assert file_exists is True
        
        # Delete the job
        response = await async_client.delete(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 204
        
        # Verify job is deleted
        response = await async_client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 404
        
        # Verify file is cleaned up
        file_exists = await file_manager.file_exists(audio_file_path)
        assert file_exists is False


@pytest.mark.e2e
class TestSystemHealthAndPerformance:
    """End-to-end tests for system health and performance."""
    
    async def test_system_health_under_load(
        self,
        async_client,
        test_db,
        sample_user,
        sample_audio_file
    ):
        """Test system health monitoring under load."""
        
        # Setup user
        test_db.add(sample_user)
        test_db.commit()
        
        # Check initial system health
        response = await async_client.get("/health/detailed")
        assert response.status_code == 200
        
        initial_health = response.json()
        assert initial_health["database"]["status"] == "healthy"
        
        # Create multiple jobs to simulate load
        for i in range(10):
            with open(sample_audio_file, 'rb') as audio_file:
                files = {"audio_file": (f"load_test_{i}.wav", audio_file, "audio/wav")}
                response = await async_client.post(
                    f"/api/v1/users/{sample_user.id}/jobs",
                    files=files
                )
            assert response.status_code == 201
        
        # Check system health after load
        response = await async_client.get("/health/detailed")
        assert response.status_code == 200
        
        health_under_load = response.json()
        assert health_under_load["database"]["status"] == "healthy"
        assert health_under_load["storage"]["status"] == "healthy"
    
    async def test_api_response_times(
        self,
        async_client,
        test_db,
        sample_user,
        sample_audio_file
    ):
        """Test API response times are within acceptable limits."""
        
        # Setup user
        test_db.add(sample_user)
        test_db.commit()
        
        # Test health endpoint response time
        start_time = time.time()
        response = await async_client.get("/health")
        health_response_time = time.time() - start_time
        
        assert response.status_code == 200
        assert health_response_time < 1.0  # Should respond within 1 second
        
        # Test job creation response time
        with open(sample_audio_file, 'rb') as audio_file:
            files = {"audio_file": ("performance_test.wav", audio_file, "audio/wav")}
            
            start_time = time.time()
            response = await async_client.post(
                f"/api/v1/users/{sample_user.id}/jobs",
                files=files
            )
            job_creation_time = time.time() - start_time
        
        assert response.status_code == 201
        assert job_creation_time < 5.0  # Should create job within 5 seconds
        
        job_id = response.json()["id"]
        
        # Test job retrieval response time
        start_time = time.time()
        response = await async_client.get(f"/api/v1/jobs/{job_id}")
        job_retrieval_time = time.time() - start_time
        
        assert response.status_code == 200
        assert job_retrieval_time < 0.5  # Should retrieve job within 0.5 seconds
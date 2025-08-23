"""
Integration tests for API endpoints.
"""

import pytest
import uuid
import json
from fastapi.testclient import TestClient

from app.domain.transcription.models import User
from app.domain.shared.enums import JobStatus


@pytest.mark.integration
class TestJobsEndpoints:
    """Integration tests for jobs API endpoints."""
    
    def test_create_job_endpoint(self, client, test_db, sample_user, sample_audio_file):
        """Test job creation endpoint."""
        # Setup user in database
        test_db.add(sample_user)
        test_db.commit()
        
        # Test successful job creation
        with open(sample_audio_file, 'rb') as audio_file:
            response = client.post(
                f"/api/v1/users/{sample_user.id}/jobs",
                files={"audio_file": ("test_audio.wav", audio_file, "audio/wav")}
            )
        
        assert response.status_code == 201
        job_data = response.json()
        
        # Verify response structure
        assert "id" in job_data
        assert job_data["user_id"] == str(sample_user.id)
        assert job_data["original_filename"] == "test_audio.wav"
        assert job_data["status"] == JobStatus.PENDING.value
        assert "created_at" in job_data
        assert "audio_metadata" in job_data
    
    def test_create_job_invalid_user(self, client, sample_audio_file):
        """Test job creation with invalid user."""
        fake_user_id = uuid.uuid4()
        
        with open(sample_audio_file, 'rb') as audio_file:
            response = client.post(
                f"/api/v1/users/{fake_user_id}/jobs",
                files={"audio_file": ("test_audio.wav", audio_file, "audio/wav")}
            )
        
        assert response.status_code == 404
        assert "User not found" in response.json()["detail"]
    
    def test_create_job_invalid_file_format(self, client, test_db, sample_user):
        """Test job creation with invalid file format."""
        test_db.add(sample_user)
        test_db.commit()
        
        # Create a fake text file
        fake_file_content = b"This is not an audio file"
        
        response = client.post(
            f"/api/v1/users/{sample_user.id}/jobs",
            files={"audio_file": ("test_file.txt", fake_file_content, "text/plain")}
        )
        
        assert response.status_code == 400
        assert "Invalid audio file format" in response.json()["detail"]
    
    def test_get_job_endpoint(self, client, test_db, sample_user, sample_audio_file):
        """Test job retrieval endpoint."""
        # Setup user and create job
        test_db.add(sample_user)
        test_db.commit()
        
        with open(sample_audio_file, 'rb') as audio_file:
            create_response = client.post(
                f"/api/v1/users/{sample_user.id}/jobs",
                files={"audio_file": ("test_audio.wav", audio_file, "audio/wav")}
            )
        
        job_id = create_response.json()["id"]
        
        # Test successful job retrieval
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        
        job_data = response.json()
        assert job_data["id"] == job_id
        assert job_data["user_id"] == str(sample_user.id)
    
    def test_get_job_not_found(self, client):
        """Test job retrieval with non-existent job."""
        fake_job_id = uuid.uuid4()
        
        response = client.get(f"/api/v1/jobs/{fake_job_id}")
        assert response.status_code == 404
        assert "Transcription job not found" in response.json()["detail"]
    
    def test_update_job_status_endpoint(self, client, test_db, sample_user, sample_audio_file):
        """Test job status update endpoint."""
        # Setup user and create job
        test_db.add(sample_user)
        test_db.commit()
        
        with open(sample_audio_file, 'rb') as audio_file:
            create_response = client.post(
                f"/api/v1/users/{sample_user.id}/jobs",
                files={"audio_file": ("test_audio.wav", audio_file, "audio/wav")}
            )
        
        job_id = create_response.json()["id"]
        
        # Test status update to processing
        response = client.put(
            f"/api/v1/jobs/{job_id}/status",
            json={"status": JobStatus.PROCESSING.value}
        )
        
        assert response.status_code == 200
        job_data = response.json()
        assert job_data["status"] == JobStatus.PROCESSING.value
        assert "started_at" in job_data
    
    def test_update_job_status_with_result(self, client, test_db, sample_user, sample_audio_file):
        """Test job status update with transcription result."""
        # Setup user and create job
        test_db.add(sample_user)
        test_db.commit()
        
        with open(sample_audio_file, 'rb') as audio_file:
            create_response = client.post(
                f"/api/v1/users/{sample_user.id}/jobs",
                files={"audio_file": ("test_audio.wav", audio_file, "audio/wav")}
            )
        
        job_id = create_response.json()["id"]
        
        # First update to processing
        client.put(
            f"/api/v1/jobs/{job_id}/status",
            json={"status": JobStatus.PROCESSING.value}
        )
        
        # Then update to completed with result
        transcription_result = {
            "text": "Hello, this is a test transcription.",
            "confidence": 0.95,
            "processing_time_seconds": 10.5,
            "model_version": "whisper-large-v3",
            "language": "en",
            "segments": []
        }
        
        response = client.put(
            f"/api/v1/jobs/{job_id}/status",
            json={
                "status": JobStatus.COMPLETED.value,
                "result": transcription_result
            }
        )
        
        assert response.status_code == 200
        job_data = response.json()
        assert job_data["status"] == JobStatus.COMPLETED.value
        assert job_data["transcription_result"]["text"] == transcription_result["text"]
        assert "completed_at" in job_data
    
    def test_cancel_job_endpoint(self, client, test_db, sample_user, sample_audio_file):
        """Test job cancellation endpoint."""
        # Setup user and create job
        test_db.add(sample_user)
        test_db.commit()
        
        with open(sample_audio_file, 'rb') as audio_file:
            create_response = client.post(
                f"/api/v1/users/{sample_user.id}/jobs",
                files={"audio_file": ("test_audio.wav", audio_file, "audio/wav")}
            )
        
        job_id = create_response.json()["id"]
        
        # Test job cancellation
        response = client.post(f"/api/v1/jobs/{job_id}/cancel")
        assert response.status_code == 200
        
        job_data = response.json()
        assert job_data["status"] == JobStatus.CANCELLED.value
    
    def test_cancel_completed_job(self, client, test_db, sample_user, sample_audio_file):
        """Test cancelling already completed job."""
        # Setup user and create job
        test_db.add(sample_user)
        test_db.commit()
        
        with open(sample_audio_file, 'rb') as audio_file:
            create_response = client.post(
                f"/api/v1/users/{sample_user.id}/jobs",
                files={"audio_file": ("test_audio.wav", audio_file, "audio/wav")}
            )
        
        job_id = create_response.json()["id"]
        
        # Complete the job first
        client.put(
            f"/api/v1/jobs/{job_id}/status",
            json={"status": JobStatus.COMPLETED.value}
        )
        
        # Try to cancel completed job
        response = client.post(f"/api/v1/jobs/{job_id}/cancel")
        assert response.status_code == 400
        assert "Cannot cancel completed or failed job" in response.json()["detail"]
    
    def test_delete_job_endpoint(self, client, test_db, sample_user, sample_audio_file):
        """Test job deletion endpoint."""
        # Setup user and create job
        test_db.add(sample_user)
        test_db.commit()
        
        with open(sample_audio_file, 'rb') as audio_file:
            create_response = client.post(
                f"/api/v1/users/{sample_user.id}/jobs",
                files={"audio_file": ("test_audio.wav", audio_file, "audio/wav")}
            )
        
        job_id = create_response.json()["id"]
        
        # Complete the job first (only completed jobs can be deleted)
        client.put(
            f"/api/v1/jobs/{job_id}/status",
            json={"status": JobStatus.COMPLETED.value}
        )
        
        # Test job deletion
        response = client.delete(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 204
        
        # Verify job is deleted
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 404


@pytest.mark.integration
class TestUsersEndpoints:
    """Integration tests for users API endpoints."""
    
    def test_get_user_jobs_endpoint(self, client, test_db, sample_user, sample_audio_file):
        """Test getting user jobs endpoint."""
        # Setup user
        test_db.add(sample_user)
        test_db.commit()
        
        # Create multiple jobs
        job_ids = []
        for i in range(3):
            with open(sample_audio_file, 'rb') as audio_file:
                response = client.post(
                    f"/api/v1/users/{sample_user.id}/jobs",
                    files={"audio_file": (f"test_audio_{i}.wav", audio_file, "audio/wav")}
                )
            job_ids.append(response.json()["id"])
        
        # Test getting all user jobs
        response = client.get(f"/api/v1/users/{sample_user.id}/jobs")
        assert response.status_code == 200
        
        jobs = response.json()
        assert len(jobs) == 3
        
        # Verify all jobs belong to the user
        for job in jobs:
            assert job["user_id"] == str(sample_user.id)
            assert job["id"] in job_ids
    
    def test_get_user_jobs_with_pagination(self, client, test_db, sample_user, sample_audio_file):
        """Test user jobs endpoint with pagination."""
        # Setup user
        test_db.add(sample_user)
        test_db.commit()
        
        # Create 5 jobs
        for i in range(5):
            with open(sample_audio_file, 'rb') as audio_file:
                client.post(
                    f"/api/v1/users/{sample_user.id}/jobs",
                    files={"audio_file": (f"test_audio_{i}.wav", audio_file, "audio/wav")}
                )
        
        # Test pagination
        response = client.get(f"/api/v1/users/{sample_user.id}/jobs?limit=2&offset=0")
        assert response.status_code == 200
        
        jobs = response.json()
        assert len(jobs) == 2
        
        # Test second page
        response = client.get(f"/api/v1/users/{sample_user.id}/jobs?limit=2&offset=2")
        assert response.status_code == 200
        
        jobs = response.json()
        assert len(jobs) == 2
    
    def test_get_user_jobs_with_status_filter(self, client, test_db, sample_user, sample_audio_file):
        """Test user jobs endpoint with status filter."""
        # Setup user
        test_db.add(sample_user)
        test_db.commit()
        
        # Create jobs with different statuses
        job_ids = []
        for i in range(3):
            with open(sample_audio_file, 'rb') as audio_file:
                response = client.post(
                    f"/api/v1/users/{sample_user.id}/jobs",
                    files={"audio_file": (f"test_audio_{i}.wav", audio_file, "audio/wav")}
                )
            job_ids.append(response.json()["id"])
        
        # Complete one job
        client.put(
            f"/api/v1/jobs/{job_ids[0]}/status",
            json={"status": JobStatus.COMPLETED.value}
        )
        
        # Fail one job
        client.put(
            f"/api/v1/jobs/{job_ids[1]}/status",
            json={
                "status": JobStatus.FAILED.value,
                "error_message": "Test error"
            }
        )
        
        # Test filtering by completed status
        response = client.get(f"/api/v1/users/{sample_user.id}/jobs?status={JobStatus.COMPLETED.value}")
        assert response.status_code == 200
        
        jobs = response.json()
        assert len(jobs) == 1
        assert jobs[0]["status"] == JobStatus.COMPLETED.value
        
        # Test filtering by pending status
        response = client.get(f"/api/v1/users/{sample_user.id}/jobs?status={JobStatus.PENDING.value}")
        assert response.status_code == 200
        
        jobs = response.json()
        assert len(jobs) == 1
        assert jobs[0]["status"] == JobStatus.PENDING.value
    
    def test_get_user_statistics_endpoint(self, client, test_db, sample_user, sample_audio_file):
        """Test user statistics endpoint."""
        # Setup user
        test_db.add(sample_user)
        test_db.commit()
        
        # Create jobs with different statuses
        job_ids = []
        for i in range(4):
            with open(sample_audio_file, 'rb') as audio_file:
                response = client.post(
                    f"/api/v1/users/{sample_user.id}/jobs",
                    files={"audio_file": (f"test_audio_{i}.wav", audio_file, "audio/wav")}
                )
            job_ids.append(response.json()["id"])
        
        # Update job statuses
        client.put(f"/api/v1/jobs/{job_ids[0]}/status", json={"status": JobStatus.COMPLETED.value})
        client.put(f"/api/v1/jobs/{job_ids[1]}/status", json={"status": JobStatus.COMPLETED.value})
        client.put(f"/api/v1/jobs/{job_ids[2]}/status", json={"status": JobStatus.FAILED.value, "error_message": "Test error"})
        # job_ids[3] remains pending
        
        # Test statistics endpoint
        response = client.get(f"/api/v1/users/{sample_user.id}/statistics")
        assert response.status_code == 200
        
        stats = response.json()
        assert stats["total_jobs"] == 4
        assert stats["completed_jobs"] == 2
        assert stats["failed_jobs"] == 1
        assert stats["pending_jobs"] == 1
        assert "total_processing_time" in stats
    
    def test_get_user_not_found(self, client):
        """Test getting jobs for non-existent user."""
        fake_user_id = uuid.uuid4()
        
        response = client.get(f"/api/v1/users/{fake_user_id}/jobs")
        assert response.status_code == 404


@pytest.mark.integration
class TestHealthEndpoints:
    """Integration tests for health check endpoints."""
    
    def test_health_check_endpoint(self, client):
        """Test basic health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert health_data["status"] == "healthy"
        assert "timestamp" in health_data
        assert "version" in health_data
    
    def test_detailed_health_check_endpoint(self, client):
        """Test detailed health check endpoint."""
        response = client.get("/health/detailed")
        assert response.status_code == 200
        
        health_data = response.json()
        assert "database" in health_data
        assert "redis" in health_data
        assert "storage" in health_data
        assert "ai_models" in health_data
        
        # Each component should have status
        for component in ["database", "redis", "storage", "ai_models"]:
            assert "status" in health_data[component]


@pytest.mark.integration
class TestErrorHandling:
    """Integration tests for error handling."""
    
    def test_validation_error_handling(self, client, test_db, sample_user):
        """Test API validation error handling."""
        test_db.add(sample_user)
        test_db.commit()
        
        # Test invalid job status update
        response = client.put(
            f"/api/v1/jobs/{uuid.uuid4()}/status",
            json={"status": "INVALID_STATUS"}
        )
        
        assert response.status_code == 422
        error_data = response.json()
        assert "detail" in error_data
    
    def test_internal_server_error_handling(self, client):
        """Test internal server error handling."""
        # This would test how the API handles unexpected errors
        # In a real scenario, you might mock a service to raise an exception
        pass
    
    def test_rate_limiting(self, client, test_db, sample_user, sample_audio_file):
        """Test rate limiting (if implemented)."""
        # This would test rate limiting functionality
        # The implementation depends on your rate limiting strategy
        pass


@pytest.mark.integration
class TestCORS:
    """Integration tests for CORS configuration."""
    
    def test_cors_headers(self, client):
        """Test CORS headers are properly set."""
        response = client.options("/api/v1/health")
        
        # Check for CORS headers
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
        assert "access-control-allow-headers" in response.headers
    
    def test_preflight_request(self, client):
        """Test CORS preflight request handling."""
        headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type"
        }
        
        response = client.options("/api/v1/jobs", headers=headers)
        assert response.status_code == 200


@pytest.mark.integration
class TestContentNegotiation:
    """Integration tests for content negotiation."""
    
    def test_json_response(self, client):
        """Test JSON response format."""
        response = client.get("/health", headers={"Accept": "application/json"})
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
    
    def test_unsupported_media_type(self, client, test_db, sample_user):
        """Test unsupported media type handling."""
        test_db.add(sample_user)
        test_db.commit()
        
        # Try to send XML data (not supported)
        response = client.post(
            f"/api/v1/users/{sample_user.id}/jobs",
            data="<xml>data</xml>",
            headers={"Content-Type": "application/xml"}
        )
        
        assert response.status_code == 415  # Unsupported Media Type
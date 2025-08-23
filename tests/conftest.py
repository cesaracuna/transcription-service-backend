"""
Pytest configuration and shared fixtures.
"""

import asyncio
import os
import tempfile
import uuid
from typing import Generator, AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import create_app
from app.core.config import settings
from app.infrastructure.database.models import Base
from app.infrastructure.database.connection import get_db
from app.domain.transcription.models import TranscriptionJob, User
from app.domain.shared.enums import JobStatus
from app.infrastructure.storage.file_storage import LocalFileStorage
from app.infrastructure.storage.base import FileManager


# Test database configuration
TEST_DATABASE_URL = "sqlite:///./test.db"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def test_db():
    """Create a test database for each test function."""
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    # Create session
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()
        # Clean up
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(test_db):
    """Create a test client with test database."""
    def override_get_db():
        try:
            yield test_db
        finally:
            pass
    
    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="function")
def temp_storage_dir():
    """Create a temporary directory for storage tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture(scope="function")
def file_manager(temp_storage_dir):
    """Create a file manager with temporary storage."""
    storage = LocalFileStorage(str(temp_storage_dir))
    return FileManager(storage)


@pytest.fixture
def sample_user() -> User:
    """Create a sample user for testing."""
    return User(
        id=uuid.uuid4(),
        username="testuser",
        email="test@example.com"
    )


@pytest.fixture
def sample_job(sample_user) -> TranscriptionJob:
    """Create a sample transcription job for testing."""
    return TranscriptionJob(
        id=uuid.uuid4(),
        user_id=sample_user.id,
        original_filename="test_audio.wav",
        audio_file_path="/tmp/test_audio.wav",
        status=JobStatus.PENDING
    )


@pytest.fixture
def sample_audio_file():
    """Create a sample audio file for testing."""
    # Create a simple WAV file with silence
    import wave
    import numpy as np
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        # Generate 1 second of silence at 16kHz
        sample_rate = 16000
        duration = 1.0
        samples = int(sample_rate * duration)
        audio_data = np.zeros(samples, dtype=np.int16)
        
        with wave.open(temp_file.name, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        yield temp_file.name
        
        # Cleanup
        try:
            os.unlink(temp_file.name)
        except OSError:
            pass


@pytest.fixture
def mock_ai_models():
    """Mock AI models for testing."""
    class MockWhisperProcessor:
        def __call__(self, audio, sampling_rate, return_tensors):
            return {"input_features": [[0.0] * 80 * 3000]}  # Mock features
        
        def batch_decode(self, token_ids, skip_special_tokens=True):
            return ["Mock transcription text"]
    
    class MockWhisperModel:
        def generate(self, **kwargs):
            return [[1, 2, 3, 4, 5]]  # Mock token IDs
        
        def to(self, device):
            return self
    
    class MockDiarizationPipeline:
        def __call__(self, audio_data):
            # Mock diarization result with two speakers
            class MockSegment:
                def __init__(self, start, end):
                    self.start = start
                    self.end = end
            
            class MockDiarization:
                def itertracks(self, yield_label=True):
                    yield MockSegment(0.0, 5.0), None, "SPEAKER_00"
                    yield MockSegment(5.0, 10.0), None, "SPEAKER_01"
                
                def get_timeline(self):
                    return True
            
            return MockDiarization()
        
        def to(self, device):
            return self
    
    return {
        "whisper_processor": MockWhisperProcessor(),
        "whisper_model": MockWhisperModel(),
        "diarization_pipeline": MockDiarizationPipeline()
    }


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment variables."""
    # Override settings for testing
    monkeypatch.setenv("ENVIRONMENT", "testing")
    monkeypatch.setenv("DB_URL", TEST_DATABASE_URL)
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/15")  # Use different DB for tests
    monkeypatch.setenv("LOGGING_LEVEL", "DEBUG")
    monkeypatch.setenv("AI_DEVICE", "cpu")  # Force CPU for tests


# Async fixtures
@pytest_asyncio.fixture
async def async_client():
    """Create an async test client."""
    from httpx import AsyncClient
    from app.main import create_app
    
    app = create_app()
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


# Markers for test categorization
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.e2e = pytest.mark.e2e
pytest.mark.slow = pytest.mark.slow
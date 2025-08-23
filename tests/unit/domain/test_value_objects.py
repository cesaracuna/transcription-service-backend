"""
Unit tests for domain value objects.
"""

import pytest
from app.domain.shared.value_objects import (
    AudioMetadata, 
    TranscriptionResult, 
    ModelConfiguration
)
from app.domain.shared.enums import DeviceType, ModelType
from app.domain.shared.exceptions import DomainValidationError


class TestAudioMetadata:
    """Test cases for AudioMetadata value object."""
    
    def test_valid_audio_metadata_creation(self):
        """Test creating valid audio metadata."""
        metadata = AudioMetadata(
            duration_seconds=120.5,
            sample_rate=16000,
            channels=2,
            format="wav",
            file_size_bytes=2048000,
            rms_level=0.5,
            peak_level=0.8
        )
        
        assert metadata.duration_seconds == 120.5
        assert metadata.sample_rate == 16000
        assert metadata.channels == 2
        assert metadata.format == "wav"
        assert metadata.file_size_bytes == 2048000
        assert metadata.rms_level == 0.5
        assert metadata.peak_level == 0.8
    
    def test_audio_metadata_validation_negative_duration(self):
        """Test validation for negative duration."""
        with pytest.raises(DomainValidationError, match="Duration cannot be negative"):
            AudioMetadata(
                duration_seconds=-10.0,
                sample_rate=16000,
                channels=1,
                format="wav",
                file_size_bytes=1024,
                rms_level=0.5,
                peak_level=0.8
            )
    
    def test_audio_metadata_validation_invalid_sample_rate(self):
        """Test validation for invalid sample rate."""
        with pytest.raises(DomainValidationError, match="Sample rate must be positive"):
            AudioMetadata(
                duration_seconds=60.0,
                sample_rate=0,
                channels=1,
                format="wav",
                file_size_bytes=1024,
                rms_level=0.5,
                peak_level=0.8
            )
    
    def test_audio_metadata_validation_invalid_channels(self):
        """Test validation for invalid channels."""
        with pytest.raises(DomainValidationError, match="Channels must be positive"):
            AudioMetadata(
                duration_seconds=60.0,
                sample_rate=16000,
                channels=0,
                format="wav",
                file_size_bytes=1024,
                rms_level=0.5,
                peak_level=0.8
            )
    
    def test_audio_metadata_validation_empty_format(self):
        """Test validation for empty format."""
        with pytest.raises(DomainValidationError, match="Format cannot be empty"):
            AudioMetadata(
                duration_seconds=60.0,
                sample_rate=16000,
                channels=1,
                format="",
                file_size_bytes=1024,
                rms_level=0.5,
                peak_level=0.8
            )
    
    def test_audio_metadata_validation_negative_file_size(self):
        """Test validation for negative file size."""
        with pytest.raises(DomainValidationError, match="File size cannot be negative"):
            AudioMetadata(
                duration_seconds=60.0,
                sample_rate=16000,
                channels=1,
                format="wav",
                file_size_bytes=-1024,
                rms_level=0.5,
                peak_level=0.8
            )
    
    def test_audio_metadata_validation_invalid_levels(self):
        """Test validation for invalid audio levels."""
        # RMS level out of range
        with pytest.raises(DomainValidationError, match="RMS level must be between 0.0 and 1.0"):
            AudioMetadata(
                duration_seconds=60.0,
                sample_rate=16000,
                channels=1,
                format="wav",
                file_size_bytes=1024,
                rms_level=1.5,
                peak_level=0.8
            )
        
        # Peak level out of range
        with pytest.raises(DomainValidationError, match="Peak level must be between 0.0 and 1.0"):
            AudioMetadata(
                duration_seconds=60.0,
                sample_rate=16000,
                channels=1,
                format="wav",
                file_size_bytes=1024,
                rms_level=0.5,
                peak_level=-0.1
            )
    
    def test_audio_metadata_equality(self):
        """Test audio metadata equality."""
        metadata1 = AudioMetadata(
            duration_seconds=60.0,
            sample_rate=16000,
            channels=1,
            format="wav",
            file_size_bytes=1024,
            rms_level=0.5,
            peak_level=0.8
        )
        
        metadata2 = AudioMetadata(
            duration_seconds=60.0,
            sample_rate=16000,
            channels=1,
            format="wav",
            file_size_bytes=1024,
            rms_level=0.5,
            peak_level=0.8
        )
        
        metadata3 = AudioMetadata(
            duration_seconds=120.0,  # Different duration
            sample_rate=16000,
            channels=1,
            format="wav",
            file_size_bytes=1024,
            rms_level=0.5,
            peak_level=0.8
        )
        
        assert metadata1 == metadata2
        assert metadata1 != metadata3
    
    def test_audio_metadata_size_in_mb(self):
        """Test file size conversion to MB."""
        metadata = AudioMetadata(
            duration_seconds=60.0,
            sample_rate=16000,
            channels=1,
            format="wav",
            file_size_bytes=2048000,  # 2MB
            rms_level=0.5,
            peak_level=0.8
        )
        
        assert metadata.file_size_mb == 2.0


class TestTranscriptionResult:
    """Test cases for TranscriptionResult value object."""
    
    def test_valid_transcription_result_creation(self):
        """Test creating valid transcription result."""
        result = TranscriptionResult(
            text="Hello, this is a test transcription.",
            confidence=0.95,
            processing_time_seconds=45.2,
            model_version="whisper-large-v3",
            language="en",
            segments=[]
        )
        
        assert result.text == "Hello, this is a test transcription."
        assert result.confidence == 0.95
        assert result.processing_time_seconds == 45.2
        assert result.model_version == "whisper-large-v3"
        assert result.language == "en"
        assert result.segments == []
    
    def test_transcription_result_validation_empty_text(self):
        """Test validation for empty text."""
        with pytest.raises(DomainValidationError, match="Text cannot be empty"):
            TranscriptionResult(
                text="",
                confidence=0.95,
                processing_time_seconds=45.2,
                model_version="whisper-large-v3",
                language="en",
                segments=[]
            )
    
    def test_transcription_result_validation_invalid_confidence(self):
        """Test validation for invalid confidence."""
        with pytest.raises(DomainValidationError, match="Confidence must be between 0.0 and 1.0"):
            TranscriptionResult(
                text="Test text",
                confidence=1.5,
                processing_time_seconds=45.2,
                model_version="whisper-large-v3",
                language="en",
                segments=[]
            )
    
    def test_transcription_result_validation_negative_processing_time(self):
        """Test validation for negative processing time."""
        with pytest.raises(DomainValidationError, match="Processing time cannot be negative"):
            TranscriptionResult(
                text="Test text",
                confidence=0.95,
                processing_time_seconds=-10.0,
                model_version="whisper-large-v3",
                language="en",
                segments=[]
            )
    
    def test_transcription_result_validation_empty_model_version(self):
        """Test validation for empty model version."""
        with pytest.raises(DomainValidationError, match="Model version cannot be empty"):
            TranscriptionResult(
                text="Test text",
                confidence=0.95,
                processing_time_seconds=45.2,
                model_version="",
                language="en",
                segments=[]
            )
    
    def test_transcription_result_validation_empty_language(self):
        """Test validation for empty language."""
        with pytest.raises(DomainValidationError, match="Language cannot be empty"):
            TranscriptionResult(
                text="Test text",
                confidence=0.95,
                processing_time_seconds=45.2,
                model_version="whisper-large-v3",
                language="",
                segments=[]
            )
    
    def test_transcription_result_word_count(self):
        """Test word count calculation."""
        result = TranscriptionResult(
            text="Hello, this is a test transcription with multiple words.",
            confidence=0.95,
            processing_time_seconds=45.2,
            model_version="whisper-large-v3",
            language="en",
            segments=[]
        )
        
        assert result.word_count == 10
    
    def test_transcription_result_character_count(self):
        """Test character count calculation."""
        result = TranscriptionResult(
            text="Hello, world!",
            confidence=0.95,
            processing_time_seconds=45.2,
            model_version="whisper-large-v3",
            language="en",
            segments=[]
        )
        
        assert result.character_count == 13
    
    def test_transcription_result_equality(self):
        """Test transcription result equality."""
        result1 = TranscriptionResult(
            text="Test text",
            confidence=0.95,
            processing_time_seconds=45.2,
            model_version="whisper-large-v3",
            language="en",
            segments=[]
        )
        
        result2 = TranscriptionResult(
            text="Test text",
            confidence=0.95,
            processing_time_seconds=45.2,
            model_version="whisper-large-v3",
            language="en",
            segments=[]
        )
        
        result3 = TranscriptionResult(
            text="Different text",
            confidence=0.95,
            processing_time_seconds=45.2,
            model_version="whisper-large-v3",
            language="en",
            segments=[]
        )
        
        assert result1 == result2
        assert result1 != result3


class TestModelConfiguration:
    """Test cases for ModelConfiguration value object."""
    
    def test_valid_model_configuration_creation(self):
        """Test creating valid model configuration."""
        config = ModelConfiguration(
            model_path="openai/whisper-large-v3",
            device="cuda:0",
            batch_size=8,
            additional_params={"temperature": 0.0, "beam_size": 5}
        )
        
        assert config.model_path == "openai/whisper-large-v3"
        assert config.device == "cuda:0"
        assert config.batch_size == 8
        assert config.additional_params == {"temperature": 0.0, "beam_size": 5}
    
    def test_model_configuration_validation_empty_model_path(self):
        """Test validation for empty model path."""
        with pytest.raises(DomainValidationError, match="Model path cannot be empty"):
            ModelConfiguration(
                model_path="",
                device="cpu",
                batch_size=1,
                additional_params={}
            )
    
    def test_model_configuration_validation_empty_device(self):
        """Test validation for empty device."""
        with pytest.raises(DomainValidationError, match="Device cannot be empty"):
            ModelConfiguration(
                model_path="openai/whisper-base",
                device="",
                batch_size=1,
                additional_params={}
            )
    
    def test_model_configuration_validation_invalid_batch_size(self):
        """Test validation for invalid batch size."""
        with pytest.raises(DomainValidationError, match="Batch size must be positive"):
            ModelConfiguration(
                model_path="openai/whisper-base",
                device="cpu",
                batch_size=0,
                additional_params={}
            )
    
    def test_model_configuration_device_type_detection(self):
        """Test device type detection."""
        # CPU device
        cpu_config = ModelConfiguration(
            model_path="openai/whisper-base",
            device="cpu",
            batch_size=1,
            additional_params={}
        )
        assert cpu_config.device_type == DeviceType.CPU
        
        # CUDA device
        cuda_config = ModelConfiguration(
            model_path="openai/whisper-base",
            device="cuda:0",
            batch_size=1,
            additional_params={}
        )
        assert cuda_config.device_type == DeviceType.CUDA
        
        # Auto device
        auto_config = ModelConfiguration(
            model_path="openai/whisper-base",
            device="auto",
            batch_size=1,
            additional_params={}
        )
        assert auto_config.device_type == DeviceType.AUTO
    
    def test_model_configuration_memory_usage_estimation(self):
        """Test memory usage estimation."""
        config = ModelConfiguration(
            model_path="openai/whisper-large-v3",
            device="cuda:0",
            batch_size=4,
            additional_params={}
        )
        
        # Should return estimated memory usage based on model size and batch size
        memory_usage = config.estimated_memory_usage_mb
        assert memory_usage > 0
        assert isinstance(memory_usage, (int, float))
    
    def test_model_configuration_equality(self):
        """Test model configuration equality."""
        config1 = ModelConfiguration(
            model_path="openai/whisper-base",
            device="cpu",
            batch_size=1,
            additional_params={"temperature": 0.0}
        )
        
        config2 = ModelConfiguration(
            model_path="openai/whisper-base",
            device="cpu",
            batch_size=1,
            additional_params={"temperature": 0.0}
        )
        
        config3 = ModelConfiguration(
            model_path="openai/whisper-large-v3",  # Different model
            device="cpu",
            batch_size=1,
            additional_params={"temperature": 0.0}
        )
        
        assert config1 == config2
        assert config1 != config3
    
    def test_model_configuration_to_dict(self):
        """Test configuration serialization to dictionary."""
        config = ModelConfiguration(
            model_path="openai/whisper-base",
            device="cuda:0",
            batch_size=2,
            additional_params={"temperature": 0.5, "beam_size": 3}
        )
        
        config_dict = config.to_dict()
        
        expected = {
            "model_path": "openai/whisper-base",
            "device": "cuda:0",
            "batch_size": 2,
            "additional_params": {"temperature": 0.5, "beam_size": 3}
        }
        
        assert config_dict == expected
    
    def test_model_configuration_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            "model_path": "openai/whisper-base",
            "device": "cpu",
            "batch_size": 1,
            "additional_params": {"temperature": 0.0}
        }
        
        config = ModelConfiguration.from_dict(config_dict)
        
        assert config.model_path == "openai/whisper-base"
        assert config.device == "cpu"
        assert config.batch_size == 1
        assert config.additional_params == {"temperature": 0.0}
"""
Application configuration management using Pydantic Settings.
Supports environment-specific configurations and environment variables.
"""

import logging
from typing import Optional, List, Any, Dict
from pathlib import Path
from functools import lru_cache

from pydantic import BaseSettings, Field, validator
from pydantic.env_settings import SettingsSourceCallable


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    url: str = Field(
        default="mssql+pyodbc://transcribeapp:transcribeapp@localhost,51392/Transcribe?driver=ODBC+Driver+17+for+SQL+Server",
        description="Database connection URL"
    )
    echo: bool = Field(default=False, description="Enable SQL query logging")
    pool_size: int = Field(default=5, description="Connection pool size")
    max_overflow: int = Field(default=10, description="Max overflow connections")
    pool_timeout: int = Field(default=30, description="Pool timeout in seconds")
    pool_recycle: int = Field(default=3600, description="Connection recycle time in seconds")
    
    class Config:
        env_prefix = "DB_"


class RedisSettings(BaseSettings):
    """Redis configuration settings."""
    
    url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    max_connections: int = Field(default=20, description="Max Redis connections")
    socket_timeout: int = Field(default=30, description="Socket timeout in seconds")
    socket_connect_timeout: int = Field(default=30, description="Socket connect timeout")
    
    class Config:
        env_prefix = "REDIS_"


class CelerySettings(BaseSettings):
    """Celery task queue configuration."""
    
    broker_url: str = Field(default="redis://localhost:6379/0", description="Celery broker URL")
    result_backend: str = Field(default="redis://localhost:6379/0", description="Celery result backend")
    task_track_started: bool = Field(default=True, description="Track task start time")
    task_serializer: str = Field(default="json", description="Task serialization format")
    result_serializer: str = Field(default="json", description="Result serialization format")
    accept_content: List[str] = Field(default=["json"], description="Accepted content types")
    timezone: str = Field(default="UTC", description="Celery timezone")
    enable_utc: bool = Field(default=True, description="Enable UTC timezone")
    worker_cancel_long_running_tasks_on_connection_loss: bool = Field(
        default=True, description="Cancel long-running tasks on connection loss"
    )
    
    class Config:
        env_prefix = "CELERY_"


class AIModelSettings(BaseSettings):
    """AI model configuration settings."""
    
    whisper_model_path: str = Field(
        default="openai/whisper-large-v3", 
        description="Whisper model path or name"
    )
    diarization_model_path: str = Field(
        default="pyannote/speaker-diarization-3.1", 
        description="Diarization model path or name"
    )
    device: str = Field(default="auto", description="Computing device (auto, cpu, cuda:0)")
    batch_size: int = Field(default=16, description="Processing batch size")
    beam_size: int = Field(default=5, description="Whisper beam search size")
    temperature: float = Field(default=0.0, description="Generation temperature")
    no_repeat_ngram_size: int = Field(default=3, description="No repeat n-gram size")
    
    # Hugging Face token for model access
    hf_token: Optional[str] = Field(default=None, description="Hugging Face access token")
    
    class Config:
        env_prefix = "AI_"


class AudioProcessingSettings(BaseSettings):
    """Audio processing configuration."""
    
    sample_rate: int = Field(default=16000, description="Audio sample rate")
    min_segment_length: int = Field(default=160, description="Minimum segment length in samples")
    max_segment_duration: float = Field(default=28.0, description="Max segment duration in seconds")
    min_chunk_duration: float = Field(default=5.0, description="Min chunk duration in seconds")
    chunk_overlap: float = Field(default=1.0, description="Chunk overlap in seconds")
    silence_threshold: int = Field(default=-40, description="Silence threshold in dB")
    min_silence_duration: float = Field(default=0.3, description="Min silence duration in seconds")
    
    # Storage paths
    audio_storage_path: str = Field(default="./audio_files", description="Audio files storage path")
    
    class Config:
        env_prefix = "AUDIO_"


class HallucinationDetectionSettings(BaseSettings):
    """Hallucination detection configuration."""
    
    silence_rms_threshold: float = Field(default=0.005, description="Silence RMS threshold")
    repetition_compression_threshold: float = Field(default=0.45, description="Repetition compression threshold")
    lone_utterance_max_segments: int = Field(default=2, description="Max segments for lone utterance")
    lone_utterance_max_duration: float = Field(default=3.0, description="Max duration for lone utterance")
    sandwich_max_pause: float = Field(default=1.5, description="Max pause for sandwich detection")
    sandwich_max_duration: float = Field(default=4.0, description="Max duration for sandwich detection")
    outlier_lang_max_duration: float = Field(default=5.0, description="Max duration for language outlier")
    
    class Config:
        env_prefix = "HALLUCINATION_"


class DiarizationSettings(BaseSettings):
    """Speaker diarization configuration."""
    
    onset: float = Field(default=0.005, description="Diarization onset threshold")
    offset: float = Field(default=0.005, description="Diarization offset threshold")
    min_duration_on: float = Field(default=0.015, description="Min duration on")
    min_duration_off: float = Field(default=0.04, description="Min duration off")
    min_speaker_segment_duration: float = Field(default=1.0, description="Min speaker segment duration")
    
    class Config:
        env_prefix = "DIARIZATION_"


class SecuritySettings(BaseSettings):
    """Security configuration."""
    
    secret_key: str = Field(default="your-secret-key-change-in-production", description="Application secret key")
    access_token_expire_minutes: int = Field(default=30, description="Access token expiration time")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    
    # CORS settings
    cors_origins: List[str] = Field(
        default=[
            "http://localhost",
            "http://localhost:8080",
            "https://localhost:7003",
            "http://localhost:5149",
        ],
        description="Allowed CORS origins"
    )
    cors_allow_credentials: bool = Field(default=True, description="Allow CORS credentials")
    cors_allow_methods: List[str] = Field(default=["*"], description="Allowed CORS methods")
    cors_allow_headers: List[str] = Field(default=["*"], description="Allowed CORS headers")
    
    class Config:
        env_prefix = "SECURITY_"


class LoggingSettings(BaseSettings):
    """Logging configuration."""
    
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    file_path: Optional[str] = Field(default=None, description="Log file path")
    max_file_size: int = Field(default=10_000_000, description="Max log file size in bytes")
    backup_count: int = Field(default=5, description="Number of backup log files")
    
    # Structured logging
    use_json: bool = Field(default=False, description="Use JSON logging format")
    
    class Config:
        env_prefix = "LOGGING_"


class Settings(BaseSettings):
    """Main application settings."""
    
    # Application info
    app_name: str = Field(default="Transcription Service Backend", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    environment: str = Field(default="development", description="Environment (development, production, testing)")
    
    # API settings
    api_v1_prefix: str = Field(default="/api/v1", description="API v1 prefix")
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: int = Field(default=8000, description="Port to bind to")
    reload: bool = Field(default=False, description="Enable auto-reload")
    
    # Component settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    celery: CelerySettings = Field(default_factory=CelerySettings)
    ai_models: AIModelSettings = Field(default_factory=AIModelSettings)
    audio: AudioProcessingSettings = Field(default_factory=AudioProcessingSettings)
    hallucination: HallucinationDetectionSettings = Field(default_factory=HallucinationDetectionSettings)
    diarization: DiarizationSettings = Field(default_factory=DiarizationSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    
    @validator("environment")
    def validate_environment(cls, v):
        allowed = ["development", "production", "testing"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v
    
    @validator("ai_models", pre=True, always=True)
    def set_ai_device(cls, v, values):
        """Auto-detect computing device if set to 'auto'."""
        if isinstance(v, dict) and v.get("device") == "auto":
            try:
                import torch
                v["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                v["device"] = "cpu"
        elif isinstance(v, AIModelSettings) and v.device == "auto":
            try:
                import torch
                v.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                v.device = "cpu"
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        @classmethod
        def customise_sources(
            cls,
            init_settings: SettingsSourceCallable,
            env_settings: SettingsSourceCallable,
            file_secret_settings: SettingsSourceCallable,
        ) -> tuple[SettingsSourceCallable, ...]:
            """Customize settings sources priority."""
            return (
                init_settings,
                env_settings,
                file_secret_settings,
            )


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    Returns:
        Application settings instance
    """
    return Settings()


def get_environment_config(environment: str) -> Dict[str, Any]:
    """
    Get environment-specific configuration.
    
    Args:
        environment: Environment name (development, production, testing)
        
    Returns:
        Environment-specific configuration dictionary
    """
    base_config = {
        "development": {
            "debug": True,
            "reload": True,
            "logging": {"level": "DEBUG"},
            "database": {"echo": True},
        },
        "production": {
            "debug": False,
            "reload": False,
            "logging": {"level": "INFO", "use_json": True},
            "database": {"echo": False},
        },
        "testing": {
            "debug": True,
            "reload": False,
            "logging": {"level": "WARNING"},
            "database": {"echo": False},
        },
    }
    
    return base_config.get(environment, base_config["development"])


# Global settings instance
settings = get_settings()
"""
Configuration management module.
Centralizes all application configuration with environment variable support.
"""

import os
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Auto-detect CUDA availability
def _detect_device():
    """Detects the best available device for AI processing."""
    # Allow manual override via environment variable
    manual_device = os.getenv("DEVICE")
    if manual_device:
        logger.info(f"Using manually configured device: {manual_device}")
        return manual_device
    
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"CUDA detected with {device_count} GPU(s). Using cuda:0")
            return "cuda:0"
        else:
            logger.warning("CUDA not available. Falling back to CPU processing")
            return "cpu"
    except ImportError:
        logger.warning("PyTorch not available. Defaulting to CPU")
        return "cpu"

class Config:
    """Centralized configuration class with environment variable support."""
    
    # --- APPLICATION SETTINGS ---
    APP_NAME = os.getenv("APP_NAME", "Pluralia Transcription Backend")
    APP_VERSION = os.getenv("APP_VERSION", "1.1.0")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    
    # --- SERVER SETTINGS ---
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    WORKERS = int(os.getenv("WORKERS", "1"))
    
    # --- SECURITY SETTINGS ---
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "500"))
    MAX_FILENAME_LENGTH = int(os.getenv("MAX_FILENAME_LENGTH", "255"))
    ALLOWED_AUDIO_EXTENSIONS = {
        '.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a', '.wma'
    }
    ALLOWED_MIME_TYPES = {
        'audio/wav', 'audio/wave', 'audio/x-wav',
        'audio/mpeg', 'audio/mp3',
        'audio/flac', 'audio/x-flac', 
        'audio/aac', 'audio/x-aac',
        'audio/ogg', 'audio/x-ogg',
        'audio/mp4', 'audio/x-m4a',
        'audio/x-ms-wma'
    }
    
    # --- CORS SETTINGS ---
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else [
        "http://localhost",
        "http://localhost:8080",
        "https://localhost:7003",
        "http://localhost:5149",
        "http://servicesdev.oas.org",
        "https://servicesdev.oas.org",
        "http://servicestest.oas.org",
        "https://servicestest.oas.org",
    ]
    
    # --- DATABASE SETTINGS ---
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        r"mssql+pyodbc://transapp:transapp@(localdb)\MSSQLLocalDB/Pluralia?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes"
    )
    DB_ECHO = os.getenv("DB_ECHO", "False").lower() == "true"
    DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
    DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))
    
    # --- REDIS/CELERY SETTINGS ---
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
    CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)
    
    # --- AI MODEL SETTINGS ---
    WHISPER_MODEL_PATH = os.getenv("WHISPER_MODEL_PATH", "openai/whisper-large-v3")
    DIARIZATION_MODEL_PATH = os.getenv("DIARIZATION_MODEL_PATH", "pyannote/speaker-diarization-3.1")
    HF_TOKEN = os.getenv("HF_TOKEN")
    DEVICE = _detect_device()
    
    # --- LLM SETTINGS ---
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
    OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
    
    # --- AUDIO PROCESSING SETTINGS ---
    SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
    MIN_SEGMENT_LENGTH = int(os.getenv("MIN_SEGMENT_LENGTH", "160"))
    MAX_SEGMENT_DURATION = float(os.getenv("MAX_SEGMENT_DURATION", "30.0"))
    MIN_CHUNK_DURATION = float(os.getenv("MIN_CHUNK_DURATION", "5.0"))
    CHUNK_OVERLAP = float(os.getenv("CHUNK_OVERLAP", "1.0"))
    
    # --- WHISPER GENERATION SETTINGS ---
    WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "3"))
    WHISPER_BATCH_SIZE = int(os.getenv("WHISPER_BATCH_SIZE", "16"))
    WHISPER_TEMPERATURE = float(os.getenv("WHISPER_TEMPERATURE", "0.2"))
    WHISPER_NO_REPEAT_NGRAM_SIZE = int(os.getenv("WHISPER_NO_REPEAT_NGRAM_SIZE", "0"))
    WHISPER_MAX_NEW_TOKENS = int(os.getenv("WHISPER_MAX_NEW_TOKENS", "224"))  # Limit output length - reduced to leave room for special tokens
    
    # --- DIARIZATION SETTINGS ---
    DIARIZATION_ONSET = float(os.getenv("DIARIZATION_ONSET", "0.1")) # 0.005
    DIARIZATION_OFFSET = float(os.getenv("DIARIZATION_OFFSET", "0.1")) # 0.005
    DIARIZATION_MIN_DURATION_ON = float(os.getenv("DIARIZATION_MIN_DURATION_ON", "0.015")) # 0.015
    DIARIZATION_MIN_DURATION_OFF = float(os.getenv("DIARIZATION_MIN_DURATION_OFF", "0.04")) # 0.04
    
    # --- LANGUAGE SETTINGS ---
    VALID_LANGUAGES = {'en', 'es', 'fr', 'pt'}
    
    # --- HALLUCINATION DETECTION SETTINGS ---
    SILENCE_RMS_THRESHOLD = float(os.getenv("SILENCE_RMS_THRESHOLD", "0.005"))
    REPETITION_COMPRESSION_THRESHOLD = float(os.getenv("REPETITION_COMPRESSION_THRESHOLD", "0.45"))
    LONE_UTTERANCE_MAX_SEGMENTS = int(os.getenv("LONE_UTTERANCE_MAX_SEGMENTS", "2"))
    LONE_UTTERANCE_MAX_DURATION = float(os.getenv("LONE_UTTERANCE_MAX_DURATION", "3.0"))
    SANDWICH_MAX_PAUSE = float(os.getenv("SANDWICH_MAX_PAUSE", "1.5"))
    SANDWICH_MAX_DURATION = float(os.getenv("SANDWICH_MAX_DURATION", "4.0"))
    
    # --- FILE STORAGE SETTINGS ---
    AUDIO_STORAGE_PATH = Path(os.getenv("AUDIO_STORAGE_PATH", "./storage/audio"))
    LOG_STORAGE_PATH = Path(os.getenv("LOG_STORAGE_PATH", "./storage/logs"))
    TEMP_STORAGE_PATH = Path(os.getenv("TEMP_STORAGE_PATH", "./storage/temp"))

    def __init__(self):
        """Initialize configuration and create necessary directories."""
        self.AUDIO_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
        self.LOG_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
        self.TEMP_STORAGE_PATH.mkdir(parents=True, exist_ok=True)


# Global configuration instance
_config = None

def initialize_config():
    """Initialize the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
        logger.info("Configuration initialized successfully")
    return _config

def get_config() -> Config:
    """Get the global configuration instance."""
    if _config is None:
        return initialize_config()
    return _config
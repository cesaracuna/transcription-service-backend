"""
Shared enums used across the domain.
"""

from enum import Enum


class JobStatus(str, Enum):
    """Job processing status."""
    PENDING = "pending"
    DIARIZING = "diarizing"
    TRANSCRIBING = "transcribing"
    PROCESSING = "processing"
    POST_PROCESSING = "post_processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingStage(str, Enum):
    """Processing pipeline stages."""
    AUDIO_LOADING = "audio_loading"
    AUDIO_CONVERSION = "audio_conversion"
    DIARIZATION = "diarization"
    TRANSCRIPTION = "transcription"
    POST_PROCESSING = "post_processing"
    HALLUCINATION_DETECTION = "hallucination_detection"
    SPEAKER_REMAPPING = "speaker_remapping"
    COMPLETED = "completed"


class AudioFormat(str, Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    M4A = "m4a"
    FLAC = "flac"
    OGG = "ogg"
    WEBM = "webm"


class Language(str, Enum):
    """Supported languages."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    PORTUGUESE = "pt"
    UNKNOWN = "unknown"


class LanguageCode(str, Enum):
    """ISO language codes."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    PORTUGUESE = "pt"
    GERMAN = "de"
    ITALIAN = "it"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    RUSSIAN = "ru"
    ARABIC = "ar"
    UNKNOWN = "unknown"


class DeviceType(str, Enum):
    """Computing device types."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon


class ModelType(str, Enum):
    """AI model types."""
    WHISPER = "whisper"
    DIARIZATION = "diarization"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class SegmentType(str, Enum):
    """Transcription segment types."""
    SPEECH = "speech"
    SILENCE = "silence"
    NOISE = "noise"
    MUSIC = "music"


class ConfidenceLevel(str, Enum):
    """Confidence level categories."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"
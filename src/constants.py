"""
Configuration constants for the transcription and diarization system.
All constants are centralized in this file to facilitate maintenance.
"""

# --- HALLUCINATION DETECTION CONSTANTS ---
SILENCE_RMS_THRESHOLD = 0.005
REPETITION_COMPRESSION_THRESHOLD = 0.45
LONE_UTTERANCE_MAX_SEGMENTS = 2
LONE_UTTERANCE_MAX_DURATION = 3.0
SANDWICH_MAX_PAUSE = 1.5
SANDWICH_MAX_DURATION = 4.0
OUTLIER_LANG_MAX_DURATION = 5.0

# --- DIARIZATION CONSTANTS ---
DIARIZATION_ONSET = 0.005
DIARIZATION_OFFSET = 0.005
DIARIZATION_MIN_DURATION_ON = 0.015
DIARIZATION_MIN_DURATION_OFF = 0.04

# --- SEGMENT MERGING CONSTANTS ---
MIN_DURATION_FOR_MERGE = 2.0
MAX_PAUSE_FOR_MERGE = 1.0

# --- MODEL CONSTANTS ---
WHISPER_MODEL_PATH = "openai/whisper-large-v3"
DIARIZATION_MODEL_PATH = "pyannote/speaker-diarization-3.1"

# --- LANGUAGE CONSTANTS ---
VALID_LANGUAGES = {'en', 'es', 'fr', 'pt'}

# --- AUDIO CONSTANTS ---
SAMPLE_RATE = 16000
MIN_SEGMENT_LENGTH = 160  # Minimum audio segment length in samples

# --- SEGMENT CHUNKING CONSTANTS ---
MAX_SEGMENT_DURATION = 28.0  # Maximum duration before chunking (seconds)
MIN_CHUNK_DURATION = 5.0     # Minimum duration for a chunk (seconds)
SILENCE_THRESHOLD = -40      # dB threshold for silence detection
MIN_SILENCE_DURATION = 0.3   # Minimum silence duration for natural pause (seconds)
CHUNK_OVERLAP = 1.0          # Overlap between chunks to avoid cutting words (seconds)
WORD_BOUNDARY_BUFFER = 0.1   # Extra buffer around split points to avoid cutting words (seconds)

# --- AUDIO CHUNKING CONSTANTS (for long files) ---
MAX_AUDIO_CHUNK_DURATION = 30.0 * 60  # Maximum audio chunk duration in seconds (30 minutes)
MIN_SPEAKER_SEGMENT_DURATION = 1.0     # Minimum speaker segment duration to include (seconds)

# --- WHISPER GENERATION CONSTANTS ---
WHISPER_BEAM_SIZE = 5          # Beam search size for better quality (1-10, default: 1)
WHISPER_BATCH_SIZE = 16        # Batch size for parallel processing (1-32, default: 1)
WHISPER_TEMPERATURE = 0.0      # Temperature for generation (0.0 = deterministic)
WHISPER_NO_REPEAT_NGRAM_SIZE = 3  # Prevent repetition of n-grams

# --- DATABASE CONSTANTS ---
SQLALCHEMY_DATABASE_URL = r"mssql+pyodbc://transcribeapp:transcribeapp@localhost,51392/Transcribe?driver=ODBC+Driver+17+for+SQL+Server"
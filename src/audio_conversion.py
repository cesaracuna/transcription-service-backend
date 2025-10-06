"""
Audio conversion utilities for standardizing audio formats.
Separated from main.py to avoid circular imports with Celery tasks.
"""

import os
import logging
import subprocess
from pydub import AudioSegment
from pydub.effects import normalize
from .config import get_config

config = get_config()

def convert_to_wav_and_resample_ffmpeg(input_audio_path, output_dir):
    """
    Converts any audio/video file to mono 16kHz WAV format using ffmpeg with streaming normalization.
    Uses dynaudnorm for efficient single-pass dynamic audio normalization without loading entire file into memory.
    Optimized for large files (>10MB) with proper timeout and memory management.
    """
    import json

    logging.info(f"Checking and standardizing audio for {input_audio_path}...")
    base_name = os.path.splitext(os.path.basename(input_audio_path))[0]
    output_wav_path = os.path.join(output_dir, f"{base_name}_standardized.wav")

    try:
        # Check if ffmpeg is available
        logging.info("Checking ffmpeg availability...")
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=5)
        logging.info("ffmpeg is available")

        # Get file size for timeout calculation
        file_size_mb = os.path.getsize(input_audio_path) / (1024 * 1024)
        logging.info(f"Input file size: {file_size_mb:.2f} MB")

        # Calculate dynamic timeout: 60s base + 10s per MB (conservative)
        timeout_seconds = max(120, int(60 + file_size_mb * 10))
        logging.info(f"Setting timeout to {timeout_seconds} seconds")

        # Single-pass conversion with streaming normalization using dynaudnorm
        # dynaudnorm applies dynamic normalization without needing two passes
        # Parameters optimized for speech:
        # - framelen=500: Frame length for analysis (500ms works well for speech)
        # - gausssize=31: Gaussian filter size for smoothing
        # - targetrms=0.25: Target RMS level (good for speech clarity)
        # - compress=0.9: Compression factor to reduce dynamic range
        cmd = [
            "ffmpeg",
            "-nostdin",               # CRITICAL: Disable stdin to prevent hanging
            "-y",                     # Overwrite output files without asking
            "-i", input_audio_path,
            "-af", "dynaudnorm=framelen=500:gausssize=31:targetrms=0.25:compress=0.9",
            "-ar", "16000",           # Resample to 16kHz
            "-ac", "1",               # Convert to mono
            "-c:a", "pcm_s16le",      # 16-bit PCM encoding
            "-map", "0:a:0",          # Only process first audio stream
            "-hide_banner",           # Reduce log verbosity
            "-loglevel", "warning",   # Only show warnings and errors
            output_wav_path
        ]

        logging.info("Starting audio conversion and normalization (single-pass streaming)...")
        logging.info(f"Command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,  # CRITICAL: Close stdin to prevent hanging
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout_seconds,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0  # Hide console on Windows
        )

        logging.info(f"Successfully converted and normalized audio: {output_wav_path}")

        # Verify output file exists and has content
        if not os.path.exists(output_wav_path) or os.path.getsize(output_wav_path) == 0:
            raise Exception("Output file was not created or is empty")

        output_size = os.path.getsize(output_wav_path)
        logging.info(f"Output file size: {output_size} bytes ({output_size/(1024*1024):.2f} MB)")

        return output_wav_path

    except FileNotFoundError as e:
        logging.error(f"ffmpeg not found in PATH: {e}")
        logging.error("ffmpeg is required for audio processing. Please install ffmpeg.")
        raise Exception("ffmpeg not available - cannot process audio files")

    except subprocess.TimeoutExpired as e:
        logging.error(f"Audio conversion timed out after {timeout_seconds} seconds")
        logging.error(f"File size: {file_size_mb:.2f} MB - may be too large or corrupted")
        raise Exception(f"Audio conversion timeout - file too large or corrupted")

    except subprocess.CalledProcessError as e:
        logging.error(f"ffmpeg command failed. Return code: {e.returncode}")
        logging.error(f"Command: {' '.join(e.cmd)}")
        logging.error(f"Stdout: {e.stdout}")
        logging.error(f"Stderr: {e.stderr}")
        raise Exception(f"ffmpeg conversion failed: {e.stderr}")

    except Exception as e:
        logging.error(f"Audio conversion failed: {e}", exc_info=True)
        raise
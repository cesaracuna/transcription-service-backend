# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is a **transcription service backend** that provides audio transcription and speaker diarization capabilities through a FastAPI web service with Celery-based asynchronous processing.

### Core Components

- **FastAPI API Server** (`src/main.py`): REST API endpoints for job submission, status checking, and result retrieval
- **Celery Workers** (`src/tasks.py`): Background processing for transcription and diarization tasks
- **Database Layer** (`src/database.py`, `src/models.py`): SQLAlchemy-based data persistence with SQL Server
- **Audio Processing Pipeline** (`src/audio_processing.py`): Whisper + PyAnnote integration for transcription and diarization
- **Hallucination Detection** (`src/hallucination_detection.py`): Post-processing to remove AI-generated artifacts

### Processing Pipeline

1. **Job Creation**: Audio files uploaded via `/jobs/` endpoint, converted to standardized 16kHz mono WAV
2. **Transcription Task**: Celery worker processes audio through Whisper large-v3 model
3. **Diarization**: PyAnnote speaker-diarization-3.1 identifies speaker segments
4. **Post-Processing**: Hallucination detection and speaker remapping
5. **Completion**: Results available via `/jobs/{job_id}` endpoint

## Development Commands

### Running the Application

**Start the API server:**
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

**Start Celery worker:**
```bash
celery -A src.tasks:celery_app worker --loglevel=info --pool=eventlet
```

**Start Redis (required for Celery):**
```bash
docker-compose up -d redis
```

**Using Docker:**
```bash
docker-compose up
```

### Dependencies Installation

**First install PyTorch with CUDA:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Then install other requirements:**
```bash
pip install -r requirements.txt
```

### Testing the API

**Submit a transcription job:**
```bash
curl -X POST -F "file=@./audio_file.wav" -F "user_id=some-uuid" http://127.0.0.1:8000/jobs/
```

**Check job status:**
```bash
curl http://127.0.0.1:8000/jobs/{job_id}
```

## Database Schema

- **Job**: Tracks transcription jobs with status (pending → processing → post_processing → completed/failed)
- **Segment**: Individual transcribed segments with speaker labels and timestamps
- **DiarizationSegment**: Raw speaker diarization data before post-processing
- **Hallucination**: Predefined patterns for detecting AI artifacts
- **User**: User management for job ownership

## Configuration

- **Models**: Uses Whisper large-v3 and PyAnnote speaker-diarization-3.1 (configurable in `config.yaml`)
- **Database**: SQL Server connection via `SQLALCHEMY_DATABASE_URL` in `src/constants.py`
- **Audio**: Automatic conversion to 16kHz mono WAV format using FFmpeg
- **Processing**: CUDA acceleration when available, fallback to CPU

## Key Implementation Details

- **Async Processing**: FastAPI returns job ID immediately, actual processing happens in Celery workers
- **Audio Standardization**: All audio converted to 16kHz mono WAV for consistent processing
- **Speaker Remapping**: Sequential speaker labeling (SPEAKER_01, SPEAKER_02, etc.) based on chronological appearance
- **Hallucination Detection**: Removes repetitive or nonsensical segments using pattern matching and contextual analysis
- **Error Handling**: Comprehensive logging and status tracking throughout the pipeline
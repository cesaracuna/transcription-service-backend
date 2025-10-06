# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pluralia is a production-ready audio transcription service with speaker diarization capabilities. It processes audio files through a multi-stage pipeline:
1. **Audio Preprocessing**: Security validation and format conversion
2. **Speaker Segmentation**: Speaker diarization using PyAnnote
3. **Transcription**: Whisper-based speech-to-text with hallucination detection
4. **Post-processing**: Grouping segments into paragraphs and formatting

The architecture uses FastAPI for the API layer and Celery for asynchronous task processing.

## Development Commands

### Starting the Application

```bash
# Start the FastAPI server (development)
python start.py

# Start the FastAPI server (production with uvicorn directly)
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4

# Start Celery worker (required for processing jobs)
# IMPORTANT: Use --concurrency=1 to process only ONE job at a time
celery -A src.celery_app worker --loglevel=info --pool=eventlet --concurrency=1

# Start Celery with beat scheduler (for periodic tasks)
celery -A src.celery_app beat --loglevel=info

# Monitor Celery with Flower
celery -A src.celery_app flower
```

### Environment Setup

```bash
# Install PyTorch with CUDA support first (if available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your configuration
```

### Testing & Validation

```bash
# Test database connection
python test_db_connection.py

# Test simple database operations
python test_simple_db.py

# Validate security configuration
python validate_security.py

# Run security tests
python test_security.py

# Check Celery status
python check_celery_status.py

# Diagnose stuck jobs
python diagnose_stuck_job.py
python fix_stuck_job.py
```

## Architecture Overview

### Core Structure

The application uses a **flat module structure** within `src/`:

- **main.py**: FastAPI application with REST endpoints for job management
- **tasks.py**: Celery tasks for the 4-stage processing pipeline
- **celery_app.py**: Celery configuration and initialization
- **config.py**: Centralized configuration with environment variable support
- **database.py**: SQLAlchemy database setup and session management
- **models.py**: SQLAlchemy ORM models (Job, Segment, DiarizationSegment, Paragraph, Hallucination, etc.)
- **schemas.py**: Pydantic schemas for API validation and serialization
- **audio_processing.py**: Core AI processing (Whisper transcription, PyAnnote diarization)
- **audio_conversion.py**: Audio format conversion using FFmpeg
- **hallucination_detection.py**: Detection and removal of Whisper hallucinations
- **security.py**: File validation, MIME type checking, path sanitization
- **logging_config.py**: Structured logging configuration
- **utils.py**: Shared utilities (timestamp formatting, silence detection, etc.)

### Database Models

Key relationships:
- **Job**: Main transcription job entity (belongs to User, has many Segments/Paragraphs/DiarizationSegments)
- **DiarizationSegment**: Raw speaker segments from PyAnnote (temporary, used for transcription grouping)
- **Segment**: Individual transcribed segments with speaker labels (word-level granularity)
- **Paragraph**: Grouped segments forming coherent paragraphs (used for final output and formatting)
- **Hallucination**: Reusable hallucination detection rules
- **JobHallucination**: Tracking of removed hallucinations per job

### Processing Pipeline

The Celery task chain in `tasks.py` executes sequentially (one job at a time):

1. **audio_preprocessing_task**: Validates security, converts audio to WAV 16kHz mono
   - Status: Job remains in `PENDING` during this task

2. **speaker_segmentation_task**: Runs PyAnnote diarization, stores DiarizationSegments
   - Status: Updates to `DIARIZING` at task start

3. **generate_transcript_task**: Groups diarization segments, transcribes with Whisper, detects hallucinations, creates Segments
   - Status: Updates to `TRANSCRIBING` at task start

4. **complete_processing_task**: Hallucination cleanup, speaker remapping, paragraph grouping, final status update
   - Status: Updates to `FINALIZING` at task start, then `COMPLETED` at end

**Job Status Flow**: PENDING → DIARIZING → TRANSCRIBING → FINALIZING → COMPLETED (or FAILED on error)

**CRITICAL**: Each task updates the job status at its **start**, not at the end. This ensures accurate status tracking as jobs move through the pipeline.

### AI Models

- **Whisper**: `openai/whisper-large-v3` for transcription (configurable via `WHISPER_MODEL_PATH`)
- **PyAnnote**: `pyannote/speaker-diarization-3.1` for speaker diarization (requires HuggingFace token)
- Models are loaded once at worker startup in `tasks.py` via `initialize_ai_models()`

### Hallucination Detection

The system implements sophisticated hallucination detection in `hallucination_detection.py`:
- **Sandwiched hallucinations**: Short segments between long pauses
- **Question mark patterns**: Repeated "?" indicating model uncertainty
- **Compression ratio**: Detecting repetitive text
- **Silence detection**: Removing transcriptions from silent audio
- **Database-driven rules**: Reusable patterns stored in `Hallucination` table

### Security Features

All uploaded files go through comprehensive validation in `security.py`:
- MIME type verification using python-magic
- File extension validation
- Filename sanitization
- File size limits (configurable via `MAX_FILE_SIZE_MB`)
- Path traversal prevention
- Malware signature checking

### Configuration

All configuration is centralized in `config.py` using environment variables. Key settings:
- Database: SQL Server via pyodbc (configurable to PostgreSQL)
- Redis: Required for Celery broker and result backend
- Audio: Sample rate (16kHz), chunk sizes, overlap settings
- Whisper: Beam size, batch size, temperature, max tokens
- Diarization: Onset/offset thresholds, min duration settings
- Storage: `storage/audio/`, `storage/logs/`, `storage/temp/`

## Important Notes

### Working with Celery Tasks

- Tasks use `eventlet` monkey-patching (see top of `tasks.py`)
- Each task uses `database_session()` context manager for DB access
- Tasks update job status at each stage for client visibility
- Error handling pattern: catch exceptions, update job status to FAILED, log details
- Tasks are configured with retry logic and timeouts in decorators

**Task Registration**: Tasks are automatically registered when `celery_app.py` imports the `tasks` module at the end of the file. This import happens AFTER celery_app is configured to avoid circular import issues. All tasks are registered with the prefix `"tasks."` (e.g., `"tasks.audio_preprocessing"`).

**Sequential Processing**: The worker is configured with `--concurrency=1` to ensure only ONE job processes at a time. This prevents resource contention and ensures predictable behavior with large AI models.

### Database Sessions

- Use `get_db()` dependency in FastAPI endpoints
- Use `database_session()` context manager in Celery tasks
- Never share sessions between tasks or threads
- Always commit changes before passing job IDs to Celery tasks

### Audio Processing

- All audio is converted to WAV 16kHz mono before processing
- FFmpeg is required for audio conversion
- Large files are streamed in 8KB chunks during upload
- Audio files are stored with secure filenames in `storage/audio/`
- Original filenames are preserved in `Job.original_filename`

### Speaker Diarization

- DiarizationSegments are created first, then used to group audio for transcription
- Segments are grouped to avoid cutting mid-word (using `group_segments_for_transcription()`)
- Speakers are remapped chronologically (SPEAKER_00, SPEAKER_01, etc.)
- Very short segments (<5s by default) are merged with adjacent segments

### Common Pitfalls

1. **Import Path**: The module structure is flat under `src/`, not hierarchical. Use `from .module import function` within src/, not nested paths.

2. **Celery Worker Pool**: Must use `--pool=eventlet` on Windows due to monkey-patching requirements. MUST also use `--concurrency=1` as a CLI argument (not in config) to ensure sequential job processing.

3. **Celery Concurrency Configuration**: The `worker_concurrency=1` setting in `celery_app.conf.update()` is IGNORED when using eventlet pool. Always use `--concurrency=1` in the command line.

4. **Database Connection Strings**: SQL Server requires `TrustServerCertificate=yes` for local development. Connection string format is critical (see `.env.example`).

5. **Model Loading**: AI models are loaded at worker startup, not per-task. Don't try to reload models in tasks.

6. **Job Status Updates**: Always use `update_job_status()` helper function in tasks to ensure consistent error tracking and logging. Each task updates status at its START, not end.

7. **UUID Handling**: User IDs and Job IDs are UUIDs. Always sanitize and validate UUID strings (`.strip().lower()`) before parsing.

8. **File Paths**: Use `Path` objects from config for storage paths. Always use `create_secure_storage_path()` for user-provided filenames.

## API Endpoints

All endpoints are defined in `src/main.py`:

- `POST /jobs/`: Create transcription job (multipart/form-data with user_id and file)
- `GET /jobs/{job_id}`: Get job status and results
- `GET /users/{user_id}/jobs/`: Paginated list of user jobs (with optional status filter)
- `DELETE /jobs/{job_id}`: Delete job and audio file
- `DELETE /jobs/{job_id}/segments/{segment_id}`: Delete specific segment
- `PATCH /jobs/{job_id}/mark-as-viewed`: Mark job as viewed
- `PATCH /jobs/{job_id}/mark-as-deleted`: Soft-delete job (status=DELETED)
- `GET /jobs/{job_id}/paragraphs`: Get formatted paragraphs for job

## File Storage

```
storage/
├── audio/       # Uploaded audio files (original and converted)
├── logs/        # Application logs (FastAPI, Celery, security)
└── temp/        # Temporary processing files
```

## Logging

Structured logging is configured in `logging_config.py`:
- Console handler for development
- Rotating file handlers (size-based and time-based)
- Database handler for critical events
- Security-specific log file
- Process-specific loggers (FastAPI vs CeleryWorker)

## Migration Notes

The codebase has a legacy structure documented in `RESTRUCTURE_PLAN.md`. Current structure is flat in `src/`. The `app/` directory mentioned in `README.md` and `start.py` is aspirational - actual code is in `src/`.

## Dependencies

Critical dependencies:
- **torch**: PyTorch for AI models (install with CUDA if available)
- **transformers**: HuggingFace Transformers for Whisper
- **pyannote.audio==3.1.1**: Speaker diarization (requires HF token)
- **fastapi**: Web framework
- **celery**: Async task processing
- **redis**: Message broker
- **sqlalchemy**: ORM
- **pyodbc**: SQL Server driver
- **ffmpeg-python**: Audio conversion wrapper
- **librosa**: Audio analysis
- **pydub**: Audio manipulation
- **eventlet**: Async I/O for Celery on Windows

## Troubleshooting

### Celery "Unregistered Task" Error

**Error**: `Received unregistered task of type 'tasks.audio_preprocessing'`

**Causes**:
1. Tasks module not imported in `celery_app.py`
2. Worker started before code changes were applied
3. Multiple workers running with different code versions

**Solutions**:
1. Verify `celery_app.py` has `from . import tasks` at the end
2. Restart the Celery worker completely (Ctrl+C and restart)
3. Clear Redis queue: `redis-cli FLUSHALL`
4. Ensure only ONE worker is running: `tasklist | findstr celery` (Windows) or `ps aux | grep celery` (Linux)

### Multiple Jobs Processing in Parallel

**Problem**: Multiple jobs showing DIARIZING/TRANSCRIBING at the same time

**Cause**: Worker not started with `--concurrency=1`

**Solution**: Always use the full command:
```bash
celery -A src.celery_app worker --loglevel=info --pool=eventlet --concurrency=1
```

### Jobs Stuck in PENDING Status

**Causes**:
1. Celery worker not running
2. Redis connection failed
3. Tasks not registered

**Solutions**:
1. Check worker is running and connected
2. Verify Redis is accessible: `redis-cli ping`
3. Check worker logs for "Tasks module imported successfully"

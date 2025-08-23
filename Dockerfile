# Multi-stage Docker build for transcription service
# Stage 1: Build dependencies and compile Python packages
FROM python:3.12-slim as builder

# Set environment variables for build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies needed for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements files
COPY requirements/base.txt requirements/ai.txt requirements/production.txt ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r base.txt && \
    pip install -r ai.txt && \
    pip install -r production.txt

# Stage 2: Runtime environment
FROM python:3.12-slim as runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    ENVIRONMENT=production

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create application directory
WORKDIR /app

# Copy application code
COPY app/ ./app/
COPY pyproject.toml ./
COPY alembic.ini ./
COPY alembic/ ./alembic/

# Create directories for data and logs
RUN mkdir -p /app/data/audio_files /app/logs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 3: Development environment
FROM runtime as development

USER root

# Install development dependencies
COPY requirements/development.txt ./
RUN pip install -r development.txt

# Install additional development tools
RUN apt-get update && apt-get install -y \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Copy test files and development configuration
COPY tests/ ./tests/
COPY .env.development .env.test ./

# Development user setup
USER appuser

# Development command with hot reload
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Stage 4: Testing environment
FROM development as testing

USER root

# Copy additional test files and configurations
COPY pytest.ini ./
COPY .coveragerc ./

USER appuser

# Command for running tests
CMD ["pytest", "-v", "--cov=app", "--cov-report=html", "--cov-report=term"]

# Stage 5: Production environment with optimizations
FROM runtime as production

# Copy production configuration
COPY logging.yaml ./
COPY .env.production ./

# Production optimizations
ENV PYTHONOPTIMIZE=1

# Use gunicorn for production
RUN pip install gunicorn[gthread]

# Production command with gunicorn
CMD ["gunicorn", "app.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--access-logfile", "-", "--error-logfile", "-"]
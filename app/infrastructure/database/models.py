"""
SQLAlchemy models for database persistence.
These models are separate from domain models and handle database-specific concerns.
"""

import uuid
from typing import List

from sqlalchemy import (
    Column, String, DateTime, ForeignKey, Text, Float, Boolean, Integer
)
from sqlalchemy.orm import relationship, DeclarativeBase
from sqlalchemy.dialects.mssql import UNIQUEIDENTIFIER
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


class UserModel(Base):
    """SQLAlchemy model for users."""
    __tablename__ = "users"
    
    id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    jobs = relationship("JobModel", back_populates="owner", cascade="all, delete-orphan")


class JobModel(Base):
    """SQLAlchemy model for transcription jobs."""
    __tablename__ = "jobs"
    
    id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    user_id = Column(UNIQUEIDENTIFIER, ForeignKey("users.id"), nullable=False, index=True)
    status = Column(String(50), nullable=False, default="pending", index=True)
    original_filename = Column(String(255), nullable=False)
    audio_file_path = Column(String(1024), nullable=False)
    error_message = Column(Text, nullable=True)
    audio_duration = Column(Float, nullable=True)
    is_post_processed = Column(Boolean, default=False, nullable=False)
    is_viewed = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)
    
    # Relationships
    owner = relationship("UserModel", back_populates="jobs")
    segments = relationship(
        "SegmentModel", 
        back_populates="job", 
        cascade="all, delete-orphan",
        order_by="SegmentModel.start_timestamp"
    )
    diarization_segments = relationship(
        "DiarizationSegmentModel", 
        back_populates="job", 
        cascade="all, delete-orphan"
    )
    job_hallucinations = relationship(
        "JobHallucinationModel", 
        back_populates="job", 
        cascade="all, delete-orphan"
    )


class SegmentModel(Base):
    """SQLAlchemy model for transcription segments."""
    __tablename__ = "segments"
    
    id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    job_id = Column(UNIQUEIDENTIFIER, ForeignKey("jobs.id"), nullable=False, index=True)
    start_timestamp = Column(String(20), nullable=True)
    end_timestamp = Column(String(20), nullable=True)
    speaker = Column(String(50), nullable=True)
    text = Column(Text, nullable=True)
    language = Column(String(10), nullable=True)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    job = relationship("JobModel", back_populates="segments")


class DiarizationSegmentModel(Base):
    """SQLAlchemy model for speaker diarization segments."""
    __tablename__ = "diarization_segments"
    
    id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    job_id = Column(UNIQUEIDENTIFIER, ForeignKey("jobs.id"), nullable=False, index=True)
    speaker_tag = Column(String(50), nullable=False)
    start_seconds = Column(Float, nullable=False)
    end_seconds = Column(Float, nullable=False)
    status = Column(String(50), default="pending", nullable=False)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    job = relationship("JobModel", back_populates="diarization_segments")


class HallucinationModel(Base):
    """SQLAlchemy model for hallucination detection patterns."""
    __tablename__ = "hallucinations"
    
    id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    text_to_match = Column(String(512), nullable=False, index=True)
    language = Column(String(10), nullable=True, index=True)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    is_regex = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)
    
    # Relationships
    job_hallucinations = relationship(
        "JobHallucinationModel", 
        back_populates="hallucination_rule"
    )


class JobHallucinationModel(Base):
    """SQLAlchemy model for tracking removed hallucinations per job."""
    __tablename__ = "job_hallucinations"
    
    id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    job_id = Column(UNIQUEIDENTIFIER, ForeignKey("jobs.id"), nullable=False, index=True)
    hallucination_id = Column(UNIQUEIDENTIFIER, ForeignKey("hallucinations.id"), nullable=False)
    removed_text = Column(Text, nullable=False)
    speaker = Column(String(50), nullable=True)
    start_timestamp = Column(String(20), nullable=True)
    end_timestamp = Column(String(20), nullable=True)
    language = Column(String(10), nullable=True)
    removed_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    job = relationship("JobModel", back_populates="job_hallucinations")
    hallucination_rule = relationship("HallucinationModel", back_populates="job_hallucinations")


class LogModel(Base):
    """SQLAlchemy model for application logs."""
    __tablename__ = "logs"
    
    id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    level = Column(String(50), nullable=False, index=True)
    message = Column(Text, nullable=False)
    process_name = Column(String(100), nullable=True, index=True)
    job_id = Column(UNIQUEIDENTIFIER, nullable=True, index=True)
    user_id = Column(UNIQUEIDENTIFIER, nullable=True, index=True)
    extra_data = Column(Text, nullable=True)  # JSON string for additional context


class ModelConfigurationModel(Base):
    """SQLAlchemy model for AI model configurations."""
    __tablename__ = "model_configurations"
    
    id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    model_type = Column(String(50), nullable=False, index=True)  # whisper, diarization
    model_path = Column(String(512), nullable=False)
    device = Column(String(50), nullable=False)
    batch_size = Column(Integer, default=1, nullable=False)
    additional_params = Column(Text, nullable=True)  # JSON string for extra parameters
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)


class ProcessingMetricsModel(Base):
    """SQLAlchemy model for storing processing metrics."""
    __tablename__ = "processing_metrics"
    
    id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    job_id = Column(UNIQUEIDENTIFIER, ForeignKey("jobs.id"), nullable=False, index=True)
    stage = Column(String(50), nullable=False)  # audio_loading, diarization, transcription, etc.
    processing_time = Column(Float, nullable=False)
    audio_duration = Column(Float, nullable=True)
    segments_processed = Column(Integer, default=0, nullable=False)
    segments_skipped = Column(Integer, default=0, nullable=False)
    error_count = Column(Integer, default=0, nullable=False)
    memory_usage_mb = Column(Float, nullable=True)
    additional_metrics = Column(Text, nullable=True)  # JSON string for stage-specific metrics
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    job = relationship("JobModel", foreign_keys=[job_id])


# Indexes for better query performance
from sqlalchemy import Index

# Job status and user indexes
Index('idx_jobs_user_status', JobModel.user_id, JobModel.status)
Index('idx_jobs_status_created', JobModel.status, JobModel.created_at)

# Segment query optimization
Index('idx_segments_job_timestamp', SegmentModel.job_id, SegmentModel.start_timestamp)

# Diarization segments optimization
Index('idx_diarization_job_time', DiarizationSegmentModel.job_id, DiarizationSegmentModel.start_seconds)

# Hallucination pattern lookup
Index('idx_hallucinations_active_language', HallucinationModel.is_active, HallucinationModel.language)

# Log querying
Index('idx_logs_timestamp_level', LogModel.timestamp, LogModel.level)
Index('idx_logs_job_process', LogModel.job_id, LogModel.process_name)
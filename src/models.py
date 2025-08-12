import uuid
from sqlalchemy import Column, String, DateTime, ForeignKey, Text, Enum as SQLAlchemyEnum, Float, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.mssql import UNIQUEIDENTIFIER
from sqlalchemy.sql import func
from .database import Base
import enum


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    DIARIZING = "diarizing"
    TRANSCRIBING = "transcribing"
    PROCESSING = "processing"
    POST_PROCESSING = "post_processing"
    COMPLETED = "completed"
    FAILED = "failed"


class User(Base):
    __tablename__ = "users"
    id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    jobs = relationship("Job", back_populates="owner")


class Job(Base):
    __tablename__ = "jobs"
    id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    user_id = Column(UNIQUEIDENTIFIER, ForeignKey("users.id"), nullable=False)
    status = Column(SQLAlchemyEnum(JobStatus), default=JobStatus.PENDING)
    original_filename = Column(String(255), nullable=False)
    audio_file_path = Column(String(1024), nullable=False)
    error_message = Column(Text, nullable=True)
    audio_duration = Column(Float, nullable=True)  # Duración en segundos Cesar no se me olvide
    is_post_processed = Column(Boolean, default=False, nullable=False)
    is_viewed = Column(Boolean, default=False, nullable=False, server_default='0')
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    owner = relationship("User", back_populates="jobs")
    segments = relationship("Segment", back_populates="job", cascade="all, delete-orphan", order_by="Segment.start_timestamp")
    job_hallucinations = relationship("JobHallucination", back_populates="job", cascade="all, delete-orphan")
    diarization_segments = relationship("DiarizationSegment", back_populates="job", cascade="all, delete-orphan")


class DiarizationSegment(Base):
    __tablename__ = "diarization_segments"
    id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    job_id = Column(UNIQUEIDENTIFIER, ForeignKey("jobs.id"), nullable=False, index=True)
    speaker_tag = Column(String(50), nullable=False)
    start_seconds = Column(Float, nullable=False)
    end_seconds = Column(Float, nullable=False)
    status = Column(String(50), default="pending", nullable=False)

    job = relationship("Job", back_populates="diarization_segments")


class Segment(Base):
    __tablename__ = "segments"
    id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    job_id = Column(UNIQUEIDENTIFIER, ForeignKey("jobs.id"), nullable=False)
    start_timestamp = Column(String(20))
    end_timestamp = Column(String(20))
    speaker = Column(String(50))
    text = Column(Text)
    language = Column(String(10))

    job = relationship("Job", back_populates="segments")

class Hallucination(Base):
    __tablename__ = "hallucinations"
    id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    text_to_match = Column(String(512), nullable=False, index=True)
    language = Column(String(10), nullable=True, index=True)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class JobHallucination(Base):
    __tablename__ = "job_hallucinations"
    id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    job_id = Column(UNIQUEIDENTIFIER, ForeignKey("jobs.id"), nullable=False)
    hallucination_id = Column(UNIQUEIDENTIFIER, ForeignKey("hallucinations.id"), nullable=False)
    removed_text = Column(Text, nullable=False)
    speaker = Column(String(50))
    start_timestamp = Column(String(20))
    end_timestamp = Column(String(20))
    language = Column(String(10))
    removed_at = Column(DateTime(timezone=True), server_default=func.now())

    job = relationship("Job", back_populates="job_hallucinations")
    rule = relationship("Hallucination")

class Log(Base):
    __tablename__ = "logs"
    id = Column(UNIQUEIDENTIFIER(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    level = Column(String(50))
    message = Column(Text)
    process_name = Column(String(100))

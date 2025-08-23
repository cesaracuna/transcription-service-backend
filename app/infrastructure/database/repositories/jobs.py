"""
Job repository implementation.
Handles database operations for transcription jobs.
"""

from typing import List, Optional, Tuple, Dict, Any
from uuid import UUID
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session, selectinload
from sqlalchemy import select, func, desc

from .base import SQLAlchemyRepository, SyncSQLAlchemyRepository
from ..models import JobModel, SegmentModel, DiarizationSegmentModel
from ....domain.transcription.models import TranscriptionJob
from ....domain.shared.enums import JobStatus
from ....domain.shared.value_objects import (
    AudioMetadata, 
    TimeInterval, 
    TranscriptionSegment,
    SpeakerSegment,
    ProcessingMetrics
)
from ....core.exceptions import DatabaseError


class JobRepository(SQLAlchemyRepository[TranscriptionJob, JobModel]):
    """Repository for transcription jobs using async SQLAlchemy."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, JobModel)
    
    def to_domain(self, model: JobModel) -> TranscriptionJob:
        """Convert SQLAlchemy model to domain entity."""
        # Convert audio metadata
        audio_metadata = None
        if model.audio_duration is not None:
            audio_metadata = AudioMetadata(
                duration_seconds=model.audio_duration,
                sample_rate=16000,  # Default value, could be stored in DB
                channels=1,  # Default value
                format="wav",  # Default value
                file_size_bytes=0,  # Could be calculated or stored
                rms_level=0.0,  # Could be calculated or stored
                peak_level=0.0  # Could be calculated or stored
            )
        
        # Convert segments
        segments = []
        if model.segments:
            for seg_model in model.segments:
                segment = TranscriptionSegment(
                    text=seg_model.text or "",
                    speaker_id=seg_model.speaker or "",
                    interval=TimeInterval(
                        start=self._parse_timestamp(seg_model.start_timestamp),
                        end=self._parse_timestamp(seg_model.end_timestamp)
                    ),
                    language=seg_model.language or "unknown",
                    confidence=None  # Could be added to model
                )
                segments.append(segment)
        
        # Convert diarization segments
        diarization_segments = []
        if model.diarization_segments:
            for diaz_model in model.diarization_segments:
                diarization_segment = SpeakerSegment(
                    speaker_id=diaz_model.speaker_tag,
                    interval=TimeInterval(
                        start=diaz_model.start_seconds,
                        end=diaz_model.end_seconds
                    ),
                    confidence=None  # Could be added to model
                )
                diarization_segments.append(diarization_segment)
        
        # Create domain job
        job = TranscriptionJob(
            id=model.id,
            user_id=model.user_id,
            original_filename=model.original_filename,
            audio_file_path=model.audio_file_path,
            status=JobStatus(model.status),
            created_at=model.created_at,
            updated_at=model.updated_at or model.created_at,
            error_message=model.error_message,
            audio_metadata=audio_metadata,
            is_post_processed=model.is_post_processed,
            is_viewed=model.is_viewed,
            segments=segments,
            diarization_segments=diarization_segments
        )
        
        return job
    
    def to_model(self, entity: TranscriptionJob) -> JobModel:
        """Convert domain entity to SQLAlchemy model."""
        model = JobModel(
            id=entity.id,
            user_id=entity.user_id,
            original_filename=entity.original_filename,
            audio_file_path=entity.audio_file_path,
            status=entity.status.value,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            error_message=entity.error_message,
            audio_duration=entity.audio_metadata.duration_seconds if entity.audio_metadata else None,
            is_post_processed=entity.is_post_processed,
            is_viewed=entity.is_viewed
        )
        
        return model
    
    def update_model(self, model: JobModel, entity: TranscriptionJob) -> JobModel:
        """Update SQLAlchemy model with domain entity data."""
        model.status = entity.status.value
        model.updated_at = entity.updated_at
        model.error_message = entity.error_message
        model.audio_duration = entity.audio_metadata.duration_seconds if entity.audio_metadata else None
        model.is_post_processed = entity.is_post_processed
        model.is_viewed = entity.is_viewed
        
        return model
    
    async def get_by_id_with_segments(self, job_id: UUID) -> Optional[TranscriptionJob]:
        """Get job by ID with all segments loaded."""
        try:
            stmt = (
                select(JobModel)
                .options(
                    selectinload(JobModel.segments),
                    selectinload(JobModel.diarization_segments)
                )
                .where(JobModel.id == job_id)
            )
            result = await self.session.execute(stmt)
            model = result.scalar_one_or_none()
            
            if model is None:
                return None
            
            return self.to_domain(model)
        
        except Exception as e:
            raise DatabaseError(f"Failed to get job with segments: {e}")
    
    async def get_by_user(
        self, 
        user_id: UUID, 
        status: Optional[JobStatus] = None,
        page: int = 1,
        size: int = 10
    ) -> Tuple[List[TranscriptionJob], int]:
        """Get jobs for a user with pagination."""
        try:
            # Build query
            stmt = select(JobModel).where(JobModel.user_id == user_id)
            count_stmt = select(func.count(JobModel.id)).where(JobModel.user_id == user_id)
            
            # Apply status filter
            if status is not None:
                stmt = stmt.where(JobModel.status == status.value)
                count_stmt = count_stmt.where(JobModel.status == status.value)
            
            # Apply ordering and pagination
            stmt = stmt.order_by(desc(JobModel.created_at))
            stmt = stmt.offset((page - 1) * size).limit(size)
            
            # Execute queries
            result = await self.session.execute(stmt)
            count_result = await self.session.execute(count_stmt)
            
            models = result.scalars().all()
            total_count = count_result.scalar_one()
            
            jobs = [self.to_domain(model) for model in models]
            
            return jobs, total_count
        
        except Exception as e:
            raise DatabaseError(f"Failed to get user jobs: {e}")
    
    async def get_by_status(self, status: JobStatus) -> List[TranscriptionJob]:
        """Get all jobs with a specific status."""
        try:
            stmt = select(JobModel).where(JobModel.status == status.value)
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            return [self.to_domain(model) for model in models]
        
        except Exception as e:
            raise DatabaseError(f"Failed to get jobs by status: {e}")
    
    def _parse_timestamp(self, timestamp_str: Optional[str]) -> float:
        """Parse timestamp string to float seconds."""
        if not timestamp_str:
            return 0.0
        
        try:
            # Handle different timestamp formats
            if ":" in timestamp_str:
                # Format: "HH:MM:SS.mmm" or "MM:SS.mmm"
                parts = timestamp_str.split(":")
                if len(parts) == 3:
                    hours, minutes, seconds = parts
                    return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
                elif len(parts) == 2:
                    minutes, seconds = parts
                    return float(minutes) * 60 + float(seconds)
            else:
                # Assume direct float value
                return float(timestamp_str)
        except (ValueError, AttributeError):
            return 0.0
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds to timestamp string."""
        minutes, secs = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        
        if hours > 0:
            return f"{int(hours):02d}:{int(minutes):02d}:{secs:.3f}"
        else:
            return f"{int(minutes):02d}:{secs:.3f}"


class SyncJobRepository(SyncSQLAlchemyRepository[TranscriptionJob, JobModel]):
    """Synchronous job repository for compatibility with existing code."""
    
    def __init__(self, session: Session):
        super().__init__(session, JobModel)
    
    def to_domain(self, model: JobModel) -> TranscriptionJob:
        """Convert SQLAlchemy model to domain entity."""
        # Convert audio metadata
        audio_metadata = None
        if model.audio_duration is not None:
            audio_metadata = AudioMetadata(
                duration_seconds=model.audio_duration,
                sample_rate=16000,
                channels=1,
                format="wav",
                file_size_bytes=0,
                rms_level=0.0,
                peak_level=0.0
            )
        
        # Convert segments
        segments = []
        if model.segments:
            for seg_model in model.segments:
                segment = TranscriptionSegment(
                    text=seg_model.text or "",
                    speaker_id=seg_model.speaker or "",
                    interval=TimeInterval(
                        start=self._parse_timestamp(seg_model.start_timestamp),
                        end=self._parse_timestamp(seg_model.end_timestamp)
                    ),
                    language=seg_model.language or "unknown"
                )
                segments.append(segment)
        
        # Convert diarization segments
        diarization_segments = []
        if model.diarization_segments:
            for diaz_model in model.diarization_segments:
                diarization_segment = SpeakerSegment(
                    speaker_id=diaz_model.speaker_tag,
                    interval=TimeInterval(
                        start=diaz_model.start_seconds,
                        end=diaz_model.end_seconds
                    )
                )
                diarization_segments.append(diarization_segment)
        
        job = TranscriptionJob(
            id=model.id,
            user_id=model.user_id,
            original_filename=model.original_filename,
            audio_file_path=model.audio_file_path,
            status=JobStatus(model.status),
            created_at=model.created_at,
            updated_at=model.updated_at or model.created_at,
            error_message=model.error_message,
            audio_metadata=audio_metadata,
            is_post_processed=model.is_post_processed,
            is_viewed=model.is_viewed,
            segments=segments,
            diarization_segments=diarization_segments
        )
        
        return job
    
    def to_model(self, entity: TranscriptionJob) -> JobModel:
        """Convert domain entity to SQLAlchemy model."""
        model = JobModel(
            id=entity.id,
            user_id=entity.user_id,
            original_filename=entity.original_filename,
            audio_file_path=entity.audio_file_path,
            status=entity.status.value,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            error_message=entity.error_message,
            audio_duration=entity.audio_metadata.duration_seconds if entity.audio_metadata else None,
            is_post_processed=entity.is_post_processed,
            is_viewed=entity.is_viewed
        )
        
        return model
    
    def update_model(self, model: JobModel, entity: TranscriptionJob) -> JobModel:
        """Update SQLAlchemy model with domain entity data."""
        model.status = entity.status.value
        model.updated_at = entity.updated_at
        model.error_message = entity.error_message
        model.audio_duration = entity.audio_metadata.duration_seconds if entity.audio_metadata else None
        model.is_post_processed = entity.is_post_processed
        model.is_viewed = entity.is_viewed
        
        return model
    
    def get_by_user(
        self, 
        user_id: UUID, 
        status: Optional[JobStatus] = None,
        page: int = 1,
        size: int = 10
    ) -> Tuple[List[TranscriptionJob], int]:
        """Get jobs for a user with pagination."""
        try:
            # Build base query
            query = self.session.query(JobModel).filter(JobModel.user_id == user_id)
            
            # Apply status filter
            if status is not None:
                query = query.filter(JobModel.status == status.value)
            
            # Get total count
            total_count = query.count()
            
            # Apply ordering and pagination
            jobs_query = query.order_by(desc(JobModel.created_at))
            jobs_query = jobs_query.offset((page - 1) * size).limit(size)
            
            models = jobs_query.all()
            jobs = [self.to_domain(model) for model in models]
            
            return jobs, total_count
        
        except Exception as e:
            raise DatabaseError(f"Failed to get user jobs: {e}")
    
    def _parse_timestamp(self, timestamp_str: Optional[str]) -> float:
        """Parse timestamp string to float seconds."""
        if not timestamp_str:
            return 0.0
        
        try:
            if ":" in timestamp_str:
                parts = timestamp_str.split(":")
                if len(parts) == 3:
                    hours, minutes, seconds = parts
                    return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
                elif len(parts) == 2:
                    minutes, seconds = parts
                    return float(minutes) * 60 + float(seconds)
            else:
                return float(timestamp_str)
        except (ValueError, AttributeError):
            return 0.0
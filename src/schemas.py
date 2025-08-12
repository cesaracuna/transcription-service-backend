import uuid
from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from datetime import datetime
from .models import JobStatus

class SegmentBase(BaseModel):
    start_timestamp: str
    end_timestamp: str
    speaker: str
    text: str
    language: str

class Segment(SegmentBase):
    id: uuid.UUID
    model_config = ConfigDict(from_attributes=True)

class JobBase(BaseModel):
    id: uuid.UUID
    status: JobStatus
    original_filename: str
    audio_duration: Optional[float] = None
    is_post_processed: bool
    is_viewed: bool
    created_at: datetime

class JobHistoryItemSchema(JobBase):
    updated_at: datetime | None = None
    model_config = ConfigDict(from_attributes=True)

class Job(JobBase):
    segments: List[Segment] = []
    full_text: Optional[str] = None
    model_config = ConfigDict(from_attributes=True)

class JobCreateResponse(JobBase):
    model_config = ConfigDict(from_attributes=True)

class PaginatedJobResponse(BaseModel):
    total_items: int
    total_pages: int
    current_page: int
    items: List[JobHistoryItemSchema]

"""
Common Pydantic schemas used across API endpoints.
"""

from typing import Optional, List, Any, Dict
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


class BaseSchema(BaseModel):
    """Base schema with common configuration."""
    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        str_strip_whitespace=True
    )


class TimestampMixin(BaseModel):
    """Mixin for timestamp fields."""
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


class PaginationRequest(BaseModel):
    """Request schema for pagination parameters."""
    page: int = Field(default=1, ge=1, description="Page number (1-based)")
    size: int = Field(default=10, ge=1, le=100, description="Page size")


class PaginationResponse(BaseModel):
    """Response schema for paginated results."""
    total_items: int = Field(..., description="Total number of items")
    total_pages: int = Field(..., description="Total number of pages")
    current_page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="Service version")
    environment: str = Field(..., description="Environment name")
    services: Dict[str, Any] = Field(default_factory=dict, description="Service dependencies status")


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


class SuccessResponse(BaseModel):
    """Generic success response schema."""
    success: bool = Field(True, description="Operation success flag")
    message: str = Field(..., description="Success message")
    data: Optional[Dict[str, Any]] = Field(None, description="Optional response data")


class TimeInterval(BaseModel):
    """Time interval schema."""
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    
    @property
    def duration(self) -> float:
        """Calculate duration."""
        return self.end - self.start


class ProcessingMetrics(BaseModel):
    """Processing metrics schema."""
    processing_time: float = Field(..., description="Total processing time in seconds")
    audio_duration: float = Field(..., description="Audio duration in seconds")
    segments_processed: int = Field(..., description="Number of segments processed")
    segments_skipped: int = Field(default=0, description="Number of segments skipped")
    error_count: int = Field(default=0, description="Number of errors encountered")
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in MB")
    
    @property
    def real_time_factor(self) -> float:
        """Calculate real-time factor."""
        return self.processing_time / self.audio_duration if self.audio_duration > 0 else 0.0


class MetricsResponse(BaseModel):
    """Metrics response schema."""
    system_metrics: Dict[str, Any] = Field(default_factory=dict, description="System metrics")
    service_metrics: Dict[str, Any] = Field(default_factory=dict, description="Service metrics")
    processing_metrics: Optional[ProcessingMetrics] = Field(None, description="Processing metrics")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Metrics timestamp")


class MessageResponse(BaseModel):
    """Generic message response schema."""
    message: str = Field(..., description="Response message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
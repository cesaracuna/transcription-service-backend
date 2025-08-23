"""
Pydantic schemas for API v1.

This module contains all the request and response schemas for API version 1.
Schemas are organized by domain and include:
- Request schemas for input validation
- Response schemas for output serialization
- Common schemas used across endpoints
"""

from .jobs import (
    JobCreateRequest,
    JobResponse,
    JobStatusUpdateRequest,
    TranscriptionResultResponse,
    SegmentResponse
)

from .users import (
    UserCreateRequest,
    UserResponse,
    UserUpdateRequest,
    UserJobsResponse,
    UserStatisticsResponse
)

from .common import (
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    PaginationResponse,
    MessageResponse
)

# Export all schemas for external use
__all__ = [
    # Job schemas
    "JobCreateRequest",
    "JobResponse", 
    "JobStatusUpdateRequest",
    "TranscriptionResultResponse",
    "SegmentResponse",
    
    # User schemas
    "UserCreateRequest",
    "UserResponse",
    "UserUpdateRequest", 
    "UserJobsResponse",
    "UserStatisticsResponse",
    
    # Common schemas
    "ErrorResponse",
    "HealthResponse",
    "MetricsResponse",
    "PaginationResponse",
    "MessageResponse"
]
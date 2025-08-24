"""
Pydantic schemas for user-related API endpoints.
"""

from typing import Optional
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, EmailStr, field_validator

from .common import BaseSchema, TimestampMixin


class UserCreate(BaseSchema):
    """Schema for creating a new user."""
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: EmailStr = Field(..., description="Email address")
    
    @field_validator('username')
    @classmethod
    def validate_username(cls, v):
        """Validate username format."""
        if not v.isalnum():
            raise ValueError('Username must contain only alphanumeric characters')
        return v.lower()


class User(BaseSchema, TimestampMixin):
    """Complete user schema."""
    id: UUID = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: EmailStr = Field(..., description="Email address")
    is_active: bool = Field(default=True, description="User active status")


class UserSummary(BaseSchema):
    """Summary user schema."""
    id: UUID = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: EmailStr = Field(..., description="Email address")


class UserUpdate(BaseSchema):
    """Schema for updating user information."""
    username: Optional[str] = Field(None, min_length=3, max_length=50, description="Username")
    email: Optional[EmailStr] = Field(None, description="Email address")
    
    @field_validator('username')
    @classmethod
    def validate_username(cls, v):
        """Validate username format."""
        if v is not None and not v.isalnum():
            raise ValueError('Username must contain only alphanumeric characters')
        return v.lower() if v else v


class UserExistsRequest(BaseSchema):
    """Request schema for checking if user exists."""
    username: str = Field(..., min_length=3, max_length=50, description="Username to check")


class UserExistsResponse(BaseSchema):
    """Response schema for user existence check."""
    exists: bool = Field(..., description="Whether the user exists")
    user_id: Optional[UUID] = Field(None, description="User ID if exists")


class UserStatsResponse(BaseSchema):
    """Response schema for user statistics."""
    user_id: UUID = Field(..., description="User ID")
    total_jobs: int = Field(..., description="Total number of jobs")
    completed_jobs: int = Field(..., description="Number of completed jobs")
    failed_jobs: int = Field(..., description="Number of failed jobs")
    pending_jobs: int = Field(..., description="Number of pending jobs")
    total_audio_duration: float = Field(..., description="Total audio duration processed (seconds)")
    total_segments: int = Field(..., description="Total number of segments created")
    account_created: datetime = Field(..., description="Account creation date")


# Aliases for compatibility
UserCreateRequest = UserCreate
UserResponse = User
UserUpdateRequest = UserUpdate
UserJobsResponse = UserSummary  # Could also be a list of jobs, but using summary for now
UserStatisticsResponse = UserStatsResponse
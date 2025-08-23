"""
API version 1 router configuration.

This module configures the main router for API version 1, including:
- All endpoint routers
- Common middleware
- API documentation tags
- Response models
"""

from fastapi import APIRouter

# Import from the centralized endpoints module
from .endpoints import api_router as endpoints_router
from .schemas import (
    JobResponse,
    UserResponse,
    ErrorResponse,
    HealthResponse
)

# Create API v1 router with metadata
api_router = APIRouter(
    prefix="/api/v1",
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Forbidden"},
        404: {"model": ErrorResponse, "description": "Not Found"},
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
        503: {"model": ErrorResponse, "description": "Service Unavailable"}
    }
)

# Include all endpoints
api_router.include_router(endpoints_router)

# Export for use in main application
__all__ = [
    "api_router",
    # Schema exports for external use
    "JobResponse",
    "UserResponse", 
    "ErrorResponse",
    "HealthResponse"
]
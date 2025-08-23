"""
API v1 endpoints.

This module aggregates all API endpoints for version 1 of the transcription service.
It provides a centralized router that includes all endpoint modules.
"""

from fastapi import APIRouter

from .health import router as health_router
from .jobs import router as jobs_router
from .users import router as users_router
from .metrics import router as metrics_router

# Create the main API router for v1
api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    health_router,
    tags=["health"],
    responses={
        200: {"description": "Health check successful"},
        503: {"description": "Service unavailable"}
    }
)

api_router.include_router(
    jobs_router,
    prefix="/jobs",
    tags=["transcription-jobs"],
    responses={
        404: {"description": "Job not found"},
        400: {"description": "Invalid request"},
        422: {"description": "Validation error"}
    }
)

api_router.include_router(
    users_router,
    prefix="/users",
    tags=["users"],
    responses={
        404: {"description": "User not found"},
        400: {"description": "Invalid request"},
        422: {"description": "Validation error"}
    }
)

api_router.include_router(
    metrics_router,
    tags=["metrics"],
    responses={
        200: {"description": "Metrics retrieved successfully"},
        500: {"description": "Internal server error"}
    }
)

# Export the main router and individual routers for flexibility
__all__ = [
    "api_router",
    "health_router",
    "jobs_router", 
    "users_router",
    "metrics_router"
]
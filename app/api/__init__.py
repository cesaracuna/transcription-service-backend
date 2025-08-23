"""
API layer for the transcription service.

This package contains all the API-related components including:
- API versioning (v1, future v2, etc.)
- Route handlers and endpoints
- Request/response schemas
- API middleware and dependencies
"""

from .v1 import api_router as api_v1_router

# Future versions can be added here
# from .v2 import api_router as api_v2_router

__all__ = [
    "api_v1_router",
    # "api_v2_router",  # Future version
]
"""
Health check and system status endpoints.
"""

import logging
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..schemas.common import HealthResponse
from ....infrastructure.database.connection import get_db
from ....core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint.
    
    Returns service status and basic information.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version=settings.app_version,
        environment=settings.environment,
        services={}
    )


@router.get("/detailed", response_model=HealthResponse)
async def detailed_health_check(db: Session = Depends(get_db)):
    """
    Detailed health check including database and external service status.
    
    Checks:
    - Database connectivity
    - Redis connectivity (if available)
    - AI models status (if available)
    """
    logger.info("Performing detailed health check")
    
    services_status = {}
    overall_status = "healthy"
    
    # Check database
    try:
        db.execute("SELECT 1")
        services_status["database"] = {
            "status": "healthy",
            "message": "Database connection successful"
        }
        logger.debug("Database health check passed")
    except Exception as e:
        services_status["database"] = {
            "status": "unhealthy",
            "message": f"Database connection failed: {str(e)}"
        }
        overall_status = "degraded"
        logger.warning(f"Database health check failed: {e}")
    
    # Check Redis (if configured)
    try:
        # TODO: Implement Redis health check
        services_status["redis"] = {
            "status": "unknown",
            "message": "Redis health check not implemented"
        }
    except Exception as e:
        services_status["redis"] = {
            "status": "unhealthy",
            "message": f"Redis connection failed: {str(e)}"
        }
        if overall_status == "healthy":
            overall_status = "degraded"
    
    # Check AI models (if loaded)
    try:
        # TODO: Implement AI models health check
        services_status["ai_models"] = {
            "status": "unknown",
            "message": "AI models health check not implemented"
        }
    except Exception as e:
        services_status["ai_models"] = {
            "status": "unhealthy",
            "message": f"AI models check failed: {str(e)}"
        }
        if overall_status == "healthy":
            overall_status = "degraded"
    
    # Check Celery workers (if available)
    try:
        # TODO: Implement Celery health check
        services_status["celery"] = {
            "status": "unknown",
            "message": "Celery health check not implemented"
        }
    except Exception as e:
        services_status["celery"] = {
            "status": "unhealthy",
            "message": f"Celery check failed: {str(e)}"
        }
        if overall_status == "healthy":
            overall_status = "degraded"
    
    # Add system information
    services_status["system"] = {
        "status": "healthy",
        "uptime": "unknown",  # TODO: Calculate uptime
        "memory_usage": "unknown",  # TODO: Get memory usage
        "cpu_usage": "unknown"  # TODO: Get CPU usage
    }
    
    response = HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        version=settings.app_version,
        environment=settings.environment,
        services=services_status
    )
    
    logger.info(f"Detailed health check completed with status: {overall_status}")
    
    # Return appropriate HTTP status
    if overall_status == "unhealthy":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=response.dict()
        )
    
    return response


@router.get("/readiness")
async def readiness_check(db: Session = Depends(get_db)):
    """
    Kubernetes-style readiness probe.
    
    Returns 200 if service is ready to accept traffic.
    Returns 503 if service is not ready.
    """
    try:
        # Check database connection
        db.execute("SELECT 1")
        
        # TODO: Check if AI models are loaded
        # TODO: Check if required services are available
        
        return {"status": "ready"}
        
    except Exception as e:
        logger.warning(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not_ready", "reason": str(e)}
        )


@router.get("/liveness")
async def liveness_check():
    """
    Kubernetes-style liveness probe.
    
    Returns 200 if service is alive.
    Returns 503 if service should be restarted.
    """
    try:
        # Basic liveness checks
        # Check if the application is responding
        
        return {"status": "alive"}
        
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "dead", "reason": str(e)}
        )
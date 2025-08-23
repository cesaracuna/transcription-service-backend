"""
Main FastAPI application module.

This module configures and creates the FastAPI application instance
using the modern layered architecture with proper dependency injection,
middleware configuration, and API routing.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import from organized structure
from . import __version__, __title__, __description__
from .core import (
    get_settings,
    setup_logging,
    get_logger,
    BaseAppException,
    DomainValidationError,
    ResourceNotFoundError,
    BusinessRuleViolationError,
    DatabaseError,
    ExternalServiceError
)
from .utils import get_utc_now
from .api import api_v1_router
from .infrastructure import (
    get_database_session,
    get_redis_client,
    close_redis_client,
    initialize_models,
    cleanup_models
)

# Set up logging
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager.
    Handles startup and shutdown events for proper resource management.
    """
    logger.info("🚀 Starting Transcription Service Backend...")
    settings = get_settings()
    
    try:
        # Startup events
        logger.info("Initializing database connection...")
        # Database connection is handled by dependency injection
        
        logger.info("Initializing Redis connection...")
        redis_client = await get_redis_client()
        if await redis_client.is_connected():
            logger.info("✅ Redis connection established")
        else:
            logger.warning("⚠️ Redis connection failed")
        
        logger.info("Initializing AI models...")
        await initialize_models()
        logger.info("✅ AI models initialized")
        
        logger.info("🎉 Application startup completed successfully")
        
        yield  # Application runs here
        
    except Exception as e:
        logger.error(f"❌ Failed to start application: {str(e)}", exc_info=True)
        raise
    finally:
        # Shutdown events
        logger.info("🔄 Shutting down Transcription Service Backend...")
        
        try:
            logger.info("Cleaning up AI models...")
            await cleanup_models()
            
            logger.info("Closing Redis connection...")
            await close_redis_client()
            
            logger.info("✅ Application shutdown completed")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {str(e)}", exc_info=True)


def create_application() -> FastAPI:
    """
    Application factory function.
    Creates and configures the FastAPI application instance.
    """
    settings = get_settings()
    
    # Create FastAPI instance
    app = FastAPI(
        title=__title__,
        description=__description__,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url="/redoc" if settings.environment != "production" else None,
        openapi_url="/openapi.json" if settings.environment != "production" else None,
    )
    
    # Configure middleware
    configure_middleware(app, settings)
    
    # Configure exception handlers
    configure_exception_handlers(app)
    
    # Include routers
    configure_routes(app)
    
    return app


def configure_middleware(app: FastAPI, settings) -> None:
    """Configure application middleware."""
    
    # CORS middleware
    if settings.environment in ["development", "testing"]:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # More permissive for development
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    else:
        # Production CORS settings
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "https://services.oas.org",
                "https://services.oas.org/pluralia"
            ],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
            allow_headers=["*"],
        )
    
    # Trusted hosts middleware for production
    if settings.environment == "production":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["services.oas.org", "*.services.oas.org"]
        )
    
    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all HTTP requests."""
        start_time = time.time()
        
        # Log request
        logger.info(
            f"📨 {request.method} {request.url.path}",
            extra={
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
                "client_ip": request.client.host if request.client else None
            }
        )
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"📤 {response.status_code} {request.method} {request.url.path} - {process_time:.3f}s",
                extra={
                    "status_code": response.status_code,
                    "process_time": process_time,
                    "method": request.method,
                    "url": str(request.url)
                }
            )
            
            # Add performance headers
            response.headers["X-Process-Time"] = str(process_time)
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"💥 Error processing {request.method} {request.url.path} - {process_time:.3f}s: {str(e)}",
                exc_info=True,
                extra={
                    "error": str(e),
                    "process_time": process_time,
                    "method": request.method,
                    "url": str(request.url)
                }
            )
            raise


def configure_exception_handlers(app: FastAPI) -> None:
    """Configure global exception handlers."""
    
    @app.exception_handler(DomainValidationError)
    async def domain_validation_error_handler(request: Request, exc: DomainValidationError):
        """Handle domain validation errors."""
        logger.warning(f"Domain validation error: {str(exc)}")
        return JSONResponse(
            status_code=400,
            content={
                "error": "Validation Error",
                "message": str(exc),
                "type": "domain_validation_error"
            }
        )
    
    @app.exception_handler(ResourceNotFoundError)
    async def resource_not_found_handler(request: Request, exc: ResourceNotFoundError):
        """Handle resource not found errors."""
        logger.warning(f"Resource not found: {str(exc)}")
        return JSONResponse(
            status_code=404,
            content={
                "error": "Not Found",
                "message": str(exc),
                "type": "resource_not_found"
            }
        )
    
    @app.exception_handler(BusinessRuleViolationError)
    async def business_rule_violation_handler(request: Request, exc: BusinessRuleViolationError):
        """Handle business rule violations."""
        logger.warning(f"Business rule violation: {str(exc)}")
        return JSONResponse(
            status_code=422,
            content={
                "error": "Business Rule Violation",
                "message": str(exc),
                "type": "business_rule_violation"
            }
        )
    
    @app.exception_handler(DatabaseError)
    async def database_error_handler(request: Request, exc: DatabaseError):
        """Handle database errors."""
        logger.error(f"Database error: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=503,
            content={
                "error": "Database Error",
                "message": "A database error occurred. Please try again later.",
                "type": "database_error"
            }
        )
    
    @app.exception_handler(ExternalServiceError)
    async def external_service_error_handler(request: Request, exc: ExternalServiceError):
        """Handle external service errors."""
        logger.error(f"External service error: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=503,
            content={
                "error": "Service Unavailable",
                "message": "An external service is currently unavailable. Please try again later.",
                "type": "external_service_error"
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred. Please try again later.",
                "type": "internal_server_error"
            }
        )


def configure_routes(app: FastAPI) -> None:
    """Configure application routes."""
    
    # Include API v1 router
    app.include_router(
        api_v1_router,
        prefix="/api/v1",
        tags=["api-v1"]
    )
    
    # Health check endpoint (outside of API versioning)
    @app.get("/health", tags=["health"])
    async def health_check():
        """Basic health check endpoint."""
        try:
            settings = get_settings()
            
            # Check database connection
            db_status = "healthy"
            try:
                db_session = next(get_database_session())
                db_session.execute("SELECT 1")
                db_session.close()
            except Exception as e:
                logger.warning(f"Database health check failed: {str(e)}")
                db_status = "unhealthy"
            
            # Check Redis connection
            redis_status = "healthy"
            try:
                redis_client = await get_redis_client()
                if not await redis_client.is_connected():
                    redis_status = "unhealthy"
            except Exception as e:
                logger.warning(f"Redis health check failed: {str(e)}")
                redis_status = "unhealthy"
            
            overall_status = "healthy" if all([
                db_status == "healthy",
                redis_status == "healthy"
            ]) else "degraded"
            
            return {
                "status": overall_status,
                "timestamp": get_utc_now().isoformat(),
                "version": __version__,
                "environment": settings.environment,
                "components": {
                    "database": {"status": db_status},
                    "redis": {"status": redis_status}
                }
            }
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "timestamp": get_utc_now().isoformat(),
                    "error": "Health check failed"
                }
            )
    
    # Root endpoint
    @app.get("/", tags=["root"])
    async def root():
        """Root endpoint with API information."""
        return {
            "name": __title__,
            "version": __version__,
            "description": __description__,
            "docs_url": "/docs",
            "health_url": "/health",
            "api": {
                "v1": {
                    "base_url": "/api/v1",
                    "docs": "/docs",
                }
            }
        }


# Create the FastAPI application
app = create_application()


# Configure startup logging
setup_logging()
logger.info(f"🔧 {__title__} v{__version__} configured successfully")


if __name__ == "__main__":
    """Run the application directly for development."""
    settings = get_settings()
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.environment == "development",
        log_level="info" if settings.environment == "production" else "debug",
        access_log=True,
    )
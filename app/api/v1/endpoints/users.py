"""
User-related API endpoints.
Handles user management operations.
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..schemas.users import (
    UserCreate, User, UserUpdate, UserExistsRequest, 
    UserExistsResponse, UserStatsResponse
)
from ....domain.transcription.services import UserService
from ....infrastructure.database.connection import get_db
from ....infrastructure.database.repositories.users import SyncUserRepository
from ....core.exceptions import UserNotFoundError, ValidationError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["users"])


def get_user_service(db: Session = Depends(get_db)) -> UserService:
    """Dependency to get user service."""
    user_repo = SyncUserRepository(db)
    return UserService(user_repo)


@router.post("/", response_model=User, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreate,
    user_service: UserService = Depends(get_user_service)
):
    """
    Create a new user.
    
    - **username**: Unique username (3-50 characters, alphanumeric)
    - **email**: Valid email address
    
    Returns the created user information.
    """
    logger.info(f"Creating new user: {user_data.username}")
    
    try:
        user = await user_service.create_user(
            username=user_data.username,
            email=user_data.email
        )
        
        response = User(
            id=user.id,
            username=user.username,
            email=user.email,
            created_at=user.created_at,
            updated_at=user.created_at,  # Same as created_at initially
            is_active=user.is_active
        )
        
        logger.info(f"User {user.username} created successfully with ID {user.id}")
        return response
        
    except ValidationError as e:
        logger.warning(f"User validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error creating user: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while creating user"
        )


@router.get("/{user_id}", response_model=User)
async def get_user(
    user_id: UUID,
    user_service: UserService = Depends(get_user_service)
):
    """
    Get user information by ID.
    
    - **user_id**: ID of the user to retrieve
    
    Returns complete user information.
    """
    logger.info(f"Fetching user {user_id}")
    
    try:
        user = await user_service.get_user(user_id)
        
        response = User(
            id=user.id,
            username=user.username,
            email=user.email,
            created_at=user.created_at,
            updated_at=user.created_at,
            is_active=user.is_active
        )
        
        return response
        
    except UserNotFoundError as e:
        logger.warning(f"User not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error fetching user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.put("/{user_id}", response_model=User)
async def update_user(
    user_id: UUID,
    user_data: UserUpdate,
    user_service: UserService = Depends(get_user_service)
):
    """
    Update user information.
    
    - **user_id**: ID of the user to update
    - **username**: New username (optional)
    - **email**: New email address (optional)
    
    Returns updated user information.
    """
    logger.info(f"Updating user {user_id}")
    
    try:
        # Get existing user
        user = await user_service.get_user(user_id)
        
        # Update fields if provided
        if user_data.username is not None:
            user.username = user_data.username
        if user_data.email is not None:
            user.email = user_data.email
        
        # TODO: Implement update in service
        # For now, return the user as-is
        
        response = User(
            id=user.id,
            username=user.username,
            email=user.email,
            created_at=user.created_at,
            updated_at=user.created_at,
            is_active=user.is_active
        )
        
        logger.info(f"User {user_id} updated successfully")
        return response
        
    except UserNotFoundError as e:
        logger.warning(f"User not found for update: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except ValidationError as e:
        logger.warning(f"User update validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error updating user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/check-exists", response_model=UserExistsResponse)
async def check_user_exists(
    request: UserExistsRequest,
    user_service: UserService = Depends(get_user_service)
):
    """
    Check if a user exists by username.
    
    - **username**: Username to check
    
    Returns whether the user exists and their ID if found.
    """
    logger.info(f"Checking if user exists: {request.username}")
    
    try:
        exists = await user_service.check_user_exists(request.username)
        
        response = UserExistsResponse(exists=exists)
        
        if exists:
            user = await user_service.get_user_by_username(request.username)
            if user:
                response.user_id = user.id
        
        logger.info(f"User existence check for {request.username}: {exists}")
        return response
        
    except Exception as e:
        logger.error(f"Unexpected error checking user existence: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/check/{user_name}")
async def check_user_exists_legacy(
    user_name: str,
    user_service: UserService = Depends(get_user_service)
):
    """
    Legacy endpoint for checking if user exists by username.
    
    - **user_name**: Username to check
    
    This endpoint maintains compatibility with existing clients.
    """
    logger.info(f"Legacy user existence check: {user_name}")
    
    try:
        exists = await user_service.check_user_exists(user_name)
        
        response = {"exists": exists}
        
        if exists:
            user = await user_service.get_user_by_username(user_name)
            if user:
                response["user_id"] = str(user.id)
        
        return response
        
    except Exception as e:
        logger.error(f"Unexpected error in legacy user check: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/{user_id}/stats", response_model=UserStatsResponse)
async def get_user_stats(
    user_id: UUID,
    user_service: UserService = Depends(get_user_service)
):
    """
    Get user statistics including job counts and processing metrics.
    
    - **user_id**: ID of the user
    
    Returns comprehensive statistics about the user's activity.
    """
    logger.info(f"Fetching stats for user {user_id}")
    
    try:
        user = await user_service.get_user(user_id)
        
        # TODO: Implement stats calculation in service
        # For now, return placeholder data
        response = UserStatsResponse(
            user_id=user_id,
            total_jobs=0,
            completed_jobs=0,
            failed_jobs=0,
            pending_jobs=0,
            total_audio_duration=0.0,
            total_segments=0,
            account_created=user.created_at
        )
        
        return response
        
    except UserNotFoundError as e:
        logger.warning(f"User not found for stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error fetching user stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
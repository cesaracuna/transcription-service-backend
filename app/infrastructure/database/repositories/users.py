"""
User repository implementation.
Handles database operations for users.
"""

from typing import Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlalchemy import select

from .base import SQLAlchemyRepository, SyncSQLAlchemyRepository
from ..models import UserModel
from ....domain.transcription.models import User
from ....core.exceptions import DatabaseError


class UserRepository(SQLAlchemyRepository[User, UserModel]):
    """Repository for users using async SQLAlchemy."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, UserModel)
    
    def to_domain(self, model: UserModel) -> User:
        """Convert SQLAlchemy model to domain entity."""
        return User(
            id=model.id,
            username=model.username,
            email=model.email,
            created_at=model.created_at if hasattr(model, 'created_at') else None,
            is_active=True  # Default value, could be added to model
        )
    
    def to_model(self, entity: User) -> UserModel:
        """Convert domain entity to SQLAlchemy model."""
        return UserModel(
            id=entity.id,
            username=entity.username,
            email=entity.email
        )
    
    def update_model(self, model: UserModel, entity: User) -> UserModel:
        """Update SQLAlchemy model with domain entity data."""
        model.username = entity.username
        model.email = entity.email
        return model
    
    async def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        try:
            stmt = select(UserModel).where(UserModel.username == username)
            result = await self.session.execute(stmt)
            model = result.scalar_one_or_none()
            
            if model is None:
                return None
            
            return self.to_domain(model)
        
        except Exception as e:
            raise DatabaseError(f"Failed to get user by username: {e}")
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        try:
            stmt = select(UserModel).where(UserModel.email == email)
            result = await self.session.execute(stmt)
            model = result.scalar_one_or_none()
            
            if model is None:
                return None
            
            return self.to_domain(model)
        
        except Exception as e:
            raise DatabaseError(f"Failed to get user by email: {e}")


class SyncUserRepository(SyncSQLAlchemyRepository[User, UserModel]):
    """Synchronous user repository for compatibility with existing code."""
    
    def __init__(self, session: Session):
        super().__init__(session, UserModel)
    
    def to_domain(self, model: UserModel) -> User:
        """Convert SQLAlchemy model to domain entity."""
        return User(
            id=model.id,
            username=model.username,
            email=model.email,
            is_active=True
        )
    
    def to_model(self, entity: User) -> UserModel:
        """Convert domain entity to SQLAlchemy model."""
        return UserModel(
            id=entity.id,
            username=entity.username,
            email=entity.email
        )
    
    def update_model(self, model: UserModel, entity: User) -> UserModel:
        """Update SQLAlchemy model with domain entity data."""
        model.username = entity.username
        model.email = entity.email
        return model
    
    def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        try:
            model = self.session.query(UserModel).filter(UserModel.username == username).first()
            
            if model is None:
                return None
            
            return self.to_domain(model)
        
        except Exception as e:
            raise DatabaseError(f"Failed to get user by username: {e}")
    
    def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        try:
            model = self.session.query(UserModel).filter(UserModel.email == email).first()
            
            if model is None:
                return None
            
            return self.to_domain(model)
        
        except Exception as e:
            raise DatabaseError(f"Failed to get user by email: {e}")
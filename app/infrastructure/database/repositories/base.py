"""
Base repository interface and abstract implementation.
Defines the contract for data access operations.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Optional, Dict, Any, Tuple
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlalchemy import select, update, delete, func
from sqlalchemy.exc import SQLAlchemyError

from ....core.exceptions import DatabaseError

# Type variables for generic repository
T = TypeVar('T')  # Domain model type
M = TypeVar('M')  # SQLAlchemy model type


class Repository(ABC, Generic[T]):
    """Abstract base repository interface."""
    
    @abstractmethod
    async def get_by_id(self, id: UUID) -> Optional[T]:
        """Get entity by ID."""
        pass
    
    @abstractmethod
    async def create(self, entity: T) -> T:
        """Create new entity."""
        pass
    
    @abstractmethod
    async def update(self, entity: T) -> T:
        """Update existing entity."""
        pass
    
    @abstractmethod
    async def delete(self, id: UUID) -> None:
        """Delete entity by ID."""
        pass
    
    @abstractmethod
    async def list(
        self, 
        limit: Optional[int] = None, 
        offset: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[T]:
        """List entities with optional pagination and filtering."""
        pass
    
    @abstractmethod
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count entities with optional filtering."""
        pass


class SQLAlchemyRepository(Repository[T], Generic[T, M]):
    """
    Base SQLAlchemy repository implementation.
    Provides common database operations using SQLAlchemy.
    """
    
    def __init__(self, session: AsyncSession, model_class: type[M]):
        self.session = session
        self.model_class = model_class
    
    @abstractmethod
    def to_domain(self, model: M) -> T:
        """Convert SQLAlchemy model to domain entity."""
        pass
    
    @abstractmethod
    def to_model(self, entity: T) -> M:
        """Convert domain entity to SQLAlchemy model."""
        pass
    
    @abstractmethod
    def update_model(self, model: M, entity: T) -> M:
        """Update SQLAlchemy model with domain entity data."""
        pass
    
    async def get_by_id(self, id: UUID) -> Optional[T]:
        """Get entity by ID."""
        try:
            stmt = select(self.model_class).where(self.model_class.id == id)
            result = await self.session.execute(stmt)
            model = result.scalar_one_or_none()
            
            if model is None:
                return None
            
            return self.to_domain(model)
        
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to get entity by ID: {e}")
    
    async def create(self, entity: T) -> T:
        """Create new entity."""
        try:
            model = self.to_model(entity)
            self.session.add(model)
            await self.session.commit()
            await self.session.refresh(model)
            
            return self.to_domain(model)
        
        except SQLAlchemyError as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to create entity: {e}")
    
    async def update(self, entity: T) -> T:
        """Update existing entity."""
        try:
            # Get existing model
            stmt = select(self.model_class).where(self.model_class.id == entity.id)
            result = await self.session.execute(stmt)
            model = result.scalar_one_or_none()
            
            if model is None:
                raise DatabaseError(f"Entity with ID {entity.id} not found for update")
            
            # Update model with entity data
            updated_model = self.update_model(model, entity)
            await self.session.commit()
            await self.session.refresh(updated_model)
            
            return self.to_domain(updated_model)
        
        except SQLAlchemyError as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to update entity: {e}")
    
    async def delete(self, id: UUID) -> None:
        """Delete entity by ID."""
        try:
            stmt = delete(self.model_class).where(self.model_class.id == id)
            result = await self.session.execute(stmt)
            
            if result.rowcount == 0:
                raise DatabaseError(f"Entity with ID {id} not found for deletion")
            
            await self.session.commit()
        
        except SQLAlchemyError as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to delete entity: {e}")
    
    async def list(
        self, 
        limit: Optional[int] = None, 
        offset: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[T]:
        """List entities with optional pagination and filtering."""
        try:
            stmt = select(self.model_class)
            
            # Apply filters
            if filters:
                for field, value in filters.items():
                    if hasattr(self.model_class, field):
                        stmt = stmt.where(getattr(self.model_class, field) == value)
            
            # Apply pagination
            if offset is not None:
                stmt = stmt.offset(offset)
            if limit is not None:
                stmt = stmt.limit(limit)
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            return [self.to_domain(model) for model in models]
        
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to list entities: {e}")
    
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count entities with optional filtering."""
        try:
            stmt = select(func.count(self.model_class.id))
            
            # Apply filters
            if filters:
                for field, value in filters.items():
                    if hasattr(self.model_class, field):
                        stmt = stmt.where(getattr(self.model_class, field) == value)
            
            result = await self.session.execute(stmt)
            return result.scalar_one()
        
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to count entities: {e}")
    
    async def exists(self, id: UUID) -> bool:
        """Check if entity exists by ID."""
        try:
            stmt = select(func.count(self.model_class.id)).where(self.model_class.id == id)
            result = await self.session.execute(stmt)
            count = result.scalar_one()
            return count > 0
        
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to check entity existence: {e}")


class SyncSQLAlchemyRepository(Repository[T], Generic[T, M]):
    """
    Synchronous SQLAlchemy repository implementation.
    For compatibility with existing synchronous code.
    """
    
    def __init__(self, session: Session, model_class: type[M]):
        self.session = session
        self.model_class = model_class
    
    @abstractmethod
    def to_domain(self, model: M) -> T:
        """Convert SQLAlchemy model to domain entity."""
        pass
    
    @abstractmethod
    def to_model(self, entity: T) -> M:
        """Convert domain entity to SQLAlchemy model."""
        pass
    
    @abstractmethod
    def update_model(self, model: M, entity: T) -> M:
        """Update SQLAlchemy model with domain entity data."""
        pass
    
    def get_by_id(self, id: UUID) -> Optional[T]:
        """Get entity by ID."""
        try:
            model = self.session.query(self.model_class).filter(self.model_class.id == id).first()
            
            if model is None:
                return None
            
            return self.to_domain(model)
        
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to get entity by ID: {e}")
    
    def create(self, entity: T) -> T:
        """Create new entity."""
        try:
            model = self.to_model(entity)
            self.session.add(model)
            self.session.commit()
            self.session.refresh(model)
            
            return self.to_domain(model)
        
        except SQLAlchemyError as e:
            self.session.rollback()
            raise DatabaseError(f"Failed to create entity: {e}")
    
    def update(self, entity: T) -> T:
        """Update existing entity."""
        try:
            # Get existing model
            model = self.session.query(self.model_class).filter(self.model_class.id == entity.id).first()
            
            if model is None:
                raise DatabaseError(f"Entity with ID {entity.id} not found for update")
            
            # Update model with entity data
            updated_model = self.update_model(model, entity)
            self.session.commit()
            self.session.refresh(updated_model)
            
            return self.to_domain(updated_model)
        
        except SQLAlchemyError as e:
            self.session.rollback()
            raise DatabaseError(f"Failed to update entity: {e}")
    
    def delete(self, id: UUID) -> None:
        """Delete entity by ID."""
        try:
            result = self.session.query(self.model_class).filter(self.model_class.id == id).delete()
            
            if result == 0:
                raise DatabaseError(f"Entity with ID {id} not found for deletion")
            
            self.session.commit()
        
        except SQLAlchemyError as e:
            self.session.rollback()
            raise DatabaseError(f"Failed to delete entity: {e}")
    
    def list(
        self, 
        limit: Optional[int] = None, 
        offset: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[T]:
        """List entities with optional pagination and filtering."""
        try:
            query = self.session.query(self.model_class)
            
            # Apply filters
            if filters:
                for field, value in filters.items():
                    if hasattr(self.model_class, field):
                        query = query.filter(getattr(self.model_class, field) == value)
            
            # Apply pagination
            if offset is not None:
                query = query.offset(offset)
            if limit is not None:
                query = query.limit(limit)
            
            models = query.all()
            
            return [self.to_domain(model) for model in models]
        
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to list entities: {e}")
    
    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count entities with optional filtering."""
        try:
            query = self.session.query(func.count(self.model_class.id))
            
            # Apply filters
            if filters:
                for field, value in filters.items():
                    if hasattr(self.model_class, field):
                        query = query.filter(getattr(self.model_class, field) == value)
            
            return query.scalar()
        
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to count entities: {e}")
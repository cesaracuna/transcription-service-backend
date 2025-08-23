"""
Redis client for caching and session management.
"""

import json
from typing import Any, Optional, Union, Dict
from datetime import timedelta

import redis.asyncio as redis
from redis.exceptions import RedisError

from ...core.config import settings
from ...core.exceptions import DatabaseError
from ...core.logging import get_logger

logger = get_logger(__name__)


class RedisClient:
    """
    Async Redis client wrapper with error handling and serialization.
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or settings.redis.url
        self.client: Optional[redis.Redis] = None
        self._connected = False
    
    async def connect(self) -> None:
        """Establish connection to Redis."""
        try:
            self.client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=settings.redis.max_connections,
                socket_timeout=settings.redis.socket_timeout,
                socket_connect_timeout=settings.redis.socket_connect_timeout,
                health_check_interval=30
            )
            
            # Test connection
            await self.client.ping()
            self._connected = True
            
            logger.info("Connected to Redis successfully")
            
        except RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise DatabaseError(f"Redis connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self.client:
            await self.client.close()
            self._connected = False
            logger.info("Disconnected from Redis")
    
    async def is_connected(self) -> bool:
        """Check if Redis is connected."""
        if not self.client or not self._connected:
            return False
        
        try:
            await self.client.ping()
            return True
        except RedisError:
            self._connected = False
            return False
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        expire: Optional[Union[int, timedelta]] = None
    ) -> bool:
        """
        Set a key-value pair in Redis.
        
        Args:
            key: Redis key
            value: Value to store (will be JSON serialized)
            expire: Optional expiration time in seconds or timedelta
            
        Returns:
            True if successful
        """
        try:
            if not await self.is_connected():
                await self.connect()
            
            # Serialize value
            serialized_value = json.dumps(value) if not isinstance(value, str) else value
            
            # Set with optional expiration
            result = await self.client.set(key, serialized_value, ex=expire)
            
            logger.debug(f"Set Redis key: {key}")
            return bool(result)
            
        except (RedisError, json.JSONEncodeError) as e:
            logger.error(f"Failed to set Redis key {key}: {e}")
            return False
    
    async def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from Redis.
        
        Args:
            key: Redis key
            default: Default value if key doesn't exist
            
        Returns:
            Value from Redis or default
        """
        try:
            if not await self.is_connected():
                await self.connect()
            
            value = await self.client.get(key)
            
            if value is None:
                return default
            
            # Try to deserialize JSON
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                # Return as string if not JSON
                return value
                
        except RedisError as e:
            logger.error(f"Failed to get Redis key {key}: {e}")
            return default
    
    async def delete(self, *keys: str) -> int:
        """
        Delete one or more keys from Redis.
        
        Args:
            keys: Keys to delete
            
        Returns:
            Number of keys deleted
        """
        try:
            if not await self.is_connected():
                await self.connect()
            
            result = await self.client.delete(*keys)
            logger.debug(f"Deleted Redis keys: {keys}")
            return result
            
        except RedisError as e:
            logger.error(f"Failed to delete Redis keys {keys}: {e}")
            return 0
    
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in Redis.
        
        Args:
            key: Redis key
            
        Returns:
            True if key exists
        """
        try:
            if not await self.is_connected():
                await self.connect()
            
            result = await self.client.exists(key)
            return bool(result)
            
        except RedisError as e:
            logger.error(f"Failed to check Redis key existence {key}: {e}")
            return False
    
    async def expire(self, key: str, seconds: Union[int, timedelta]) -> bool:
        """
        Set expiration for a key.
        
        Args:
            key: Redis key
            seconds: Expiration time in seconds or timedelta
            
        Returns:
            True if successful
        """
        try:
            if not await self.is_connected():
                await self.connect()
            
            if isinstance(seconds, timedelta):
                seconds = int(seconds.total_seconds())
            
            result = await self.client.expire(key, seconds)
            return bool(result)
            
        except RedisError as e:
            logger.error(f"Failed to set expiration for Redis key {key}: {e}")
            return False
    
    async def ttl(self, key: str) -> int:
        """
        Get time to live for a key.
        
        Args:
            key: Redis key
            
        Returns:
            TTL in seconds (-1 if no expiration, -2 if key doesn't exist)
        """
        try:
            if not await self.is_connected():
                await self.connect()
            
            return await self.client.ttl(key)
            
        except RedisError as e:
            logger.error(f"Failed to get TTL for Redis key {key}: {e}")
            return -2
    
    async def hset(self, key: str, mapping: Dict[str, Any]) -> int:
        """
        Set multiple fields in a hash.
        
        Args:
            key: Hash key
            mapping: Dictionary of field-value pairs
            
        Returns:
            Number of fields added
        """
        try:
            if not await self.is_connected():
                await self.connect()
            
            # Serialize values
            serialized_mapping = {}
            for field, value in mapping.items():
                serialized_mapping[field] = json.dumps(value) if not isinstance(value, str) else value
            
            result = await self.client.hset(key, mapping=serialized_mapping)
            logger.debug(f"Set Redis hash fields: {key}")
            return result
            
        except (RedisError, json.JSONEncodeError) as e:
            logger.error(f"Failed to set Redis hash {key}: {e}")
            return 0
    
    async def hget(self, key: str, field: str, default: Any = None) -> Any:
        """
        Get a field from a hash.
        
        Args:
            key: Hash key
            field: Field name
            default: Default value if field doesn't exist
            
        Returns:
            Field value or default
        """
        try:
            if not await self.is_connected():
                await self.connect()
            
            value = await self.client.hget(key, field)
            
            if value is None:
                return default
            
            # Try to deserialize JSON
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
                
        except RedisError as e:
            logger.error(f"Failed to get Redis hash field {key}.{field}: {e}")
            return default
    
    async def hgetall(self, key: str) -> Dict[str, Any]:
        """
        Get all fields and values from a hash.
        
        Args:
            key: Hash key
            
        Returns:
            Dictionary of field-value pairs
        """
        try:
            if not await self.is_connected():
                await self.connect()
            
            result = await self.client.hgetall(key)
            
            # Deserialize values
            deserialized_result = {}
            for field, value in result.items():
                try:
                    deserialized_result[field] = json.loads(value)
                except json.JSONDecodeError:
                    deserialized_result[field] = value
            
            return deserialized_result
            
        except RedisError as e:
            logger.error(f"Failed to get Redis hash {key}: {e}")
            return {}
    
    async def cache_job_progress(
        self, 
        job_id: str, 
        progress_data: Dict[str, Any],
        expire_seconds: int = 3600
    ) -> bool:
        """
        Cache job progress information.
        
        Args:
            job_id: Job ID
            progress_data: Progress information
            expire_seconds: Cache expiration time
            
        Returns:
            True if successful
        """
        key = f"job_progress:{job_id}"
        return await self.set(key, progress_data, expire_seconds)
    
    async def get_job_progress(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get cached job progress information.
        
        Args:
            job_id: Job ID
            
        Returns:
            Progress data or None
        """
        key = f"job_progress:{job_id}"
        return await self.get(key)
    
    async def clear_job_cache(self, job_id: str) -> bool:
        """
        Clear all cached data for a job.
        
        Args:
            job_id: Job ID
            
        Returns:
            True if successful
        """
        pattern = f"job_*:{job_id}"
        
        try:
            if not await self.is_connected():
                await self.connect()
            
            # Get all keys matching pattern
            keys = await self.client.keys(pattern)
            
            if keys:
                deleted = await self.delete(*keys)
                logger.debug(f"Cleared {deleted} cache entries for job {job_id}")
                return deleted > 0
            
            return True
            
        except RedisError as e:
            logger.error(f"Failed to clear job cache for {job_id}: {e}")
            return False


# Global Redis client instance
_redis_client: Optional[RedisClient] = None


async def get_redis_client() -> RedisClient:
    """
    Get the global Redis client instance.
    
    Returns:
        Redis client
    """
    global _redis_client
    
    if _redis_client is None:
        _redis_client = RedisClient()
        await _redis_client.connect()
    
    return _redis_client


async def close_redis_client() -> None:
    """Close the global Redis client."""
    global _redis_client
    
    if _redis_client:
        await _redis_client.disconnect()
        _redis_client = None
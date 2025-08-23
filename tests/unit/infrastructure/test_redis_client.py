"""
Unit tests for Redis client.
"""

import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import timedelta

from app.infrastructure.external.redis_client import RedisClient
from app.core.exceptions import DatabaseError


class TestRedisClient:
    """Test cases for RedisClient."""
    
    @pytest.fixture
    def redis_client(self):
        """Create Redis client for testing."""
        return RedisClient(redis_url="redis://localhost:6379/15")
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis connection."""
        mock = AsyncMock()
        mock.ping = AsyncMock(return_value=True)
        mock.set = AsyncMock(return_value=True)
        mock.get = AsyncMock(return_value=None)
        mock.delete = AsyncMock(return_value=1)
        mock.exists = AsyncMock(return_value=True)
        mock.expire = AsyncMock(return_value=True)
        mock.ttl = AsyncMock(return_value=3600)
        mock.hset = AsyncMock(return_value=1)
        mock.hget = AsyncMock(return_value=None)
        mock.hgetall = AsyncMock(return_value={})
        mock.keys = AsyncMock(return_value=[])
        mock.close = AsyncMock()
        return mock

    async def test_connect_success(self, redis_client, mock_redis):
        """Test successful Redis connection."""
        with patch('redis.asyncio.from_url', return_value=mock_redis):
            await redis_client.connect()
            
            assert redis_client._connected is True
            assert redis_client.client == mock_redis
            mock_redis.ping.assert_called_once()
    
    async def test_connect_failure(self, redis_client):
        """Test Redis connection failure."""
        with patch('redis.asyncio.from_url', side_effect=Exception("Connection failed")):
            with pytest.raises(DatabaseError, match="Redis connection failed"):
                await redis_client.connect()
            
            assert redis_client._connected is False
    
    async def test_disconnect(self, redis_client, mock_redis):
        """Test Redis disconnection."""
        redis_client.client = mock_redis
        redis_client._connected = True
        
        await redis_client.disconnect()
        
        assert redis_client._connected is False
        mock_redis.close.assert_called_once()
    
    async def test_is_connected_true(self, redis_client, mock_redis):
        """Test connection check when connected."""
        redis_client.client = mock_redis
        redis_client._connected = True
        
        result = await redis_client.is_connected()
        
        assert result is True
        mock_redis.ping.assert_called_once()
    
    async def test_is_connected_false_no_client(self, redis_client):
        """Test connection check when no client."""
        redis_client.client = None
        redis_client._connected = False
        
        result = await redis_client.is_connected()
        
        assert result is False
    
    async def test_is_connected_ping_fails(self, redis_client, mock_redis):
        """Test connection check when ping fails."""
        redis_client.client = mock_redis
        redis_client._connected = True
        mock_redis.ping.side_effect = Exception("Ping failed")
        
        result = await redis_client.is_connected()
        
        assert result is False
        assert redis_client._connected is False
    
    async def test_set_string_value(self, redis_client, mock_redis):
        """Test setting string value."""
        redis_client.client = mock_redis
        redis_client._connected = True
        
        result = await redis_client.set("test_key", "test_value")
        
        assert result is True
        mock_redis.set.assert_called_once_with("test_key", "test_value", ex=None)
    
    async def test_set_json_value(self, redis_client, mock_redis):
        """Test setting JSON value."""
        redis_client.client = mock_redis
        redis_client._connected = True
        
        test_data = {"key": "value", "number": 42}
        
        result = await redis_client.set("test_key", test_data)
        
        assert result is True
        expected_json = json.dumps(test_data)
        mock_redis.set.assert_called_once_with("test_key", expected_json, ex=None)
    
    async def test_set_with_expiration(self, redis_client, mock_redis):
        """Test setting value with expiration."""
        redis_client.client = mock_redis
        redis_client._connected = True
        
        result = await redis_client.set("test_key", "test_value", expire=3600)
        
        assert result is True
        mock_redis.set.assert_called_once_with("test_key", "test_value", ex=3600)
    
    async def test_set_with_timedelta_expiration(self, redis_client, mock_redis):
        """Test setting value with timedelta expiration."""
        redis_client.client = mock_redis
        redis_client._connected = True
        
        expire_time = timedelta(hours=1)
        
        result = await redis_client.set("test_key", "test_value", expire=expire_time)
        
        assert result is True
        mock_redis.set.assert_called_once_with("test_key", "test_value", ex=expire_time)
    
    async def test_set_auto_connect(self, redis_client, mock_redis):
        """Test auto-connect when setting value."""
        redis_client._connected = False
        
        with patch.object(redis_client, 'is_connected', return_value=False), \
             patch.object(redis_client, 'connect') as mock_connect:
            redis_client.client = mock_redis
            
            await redis_client.set("test_key", "test_value")
            
            mock_connect.assert_called_once()
    
    async def test_get_string_value(self, redis_client, mock_redis):
        """Test getting string value."""
        redis_client.client = mock_redis
        redis_client._connected = True
        mock_redis.get.return_value = "test_value"
        
        result = await redis_client.get("test_key")
        
        assert result == "test_value"
        mock_redis.get.assert_called_once_with("test_key")
    
    async def test_get_json_value(self, redis_client, mock_redis):
        """Test getting JSON value."""
        redis_client.client = mock_redis
        redis_client._connected = True
        
        test_data = {"key": "value", "number": 42}
        mock_redis.get.return_value = json.dumps(test_data)
        
        result = await redis_client.get("test_key")
        
        assert result == test_data
    
    async def test_get_nonexistent_key(self, redis_client, mock_redis):
        """Test getting non-existent key."""
        redis_client.client = mock_redis
        redis_client._connected = True
        mock_redis.get.return_value = None
        
        result = await redis_client.get("nonexistent_key", default="default_value")
        
        assert result == "default_value"
    
    async def test_get_invalid_json(self, redis_client, mock_redis):
        """Test getting invalid JSON value."""
        redis_client.client = mock_redis
        redis_client._connected = True
        mock_redis.get.return_value = "invalid json {"
        
        result = await redis_client.get("test_key")
        
        # Should return as string when JSON parsing fails
        assert result == "invalid json {"
    
    async def test_delete_keys(self, redis_client, mock_redis):
        """Test deleting keys."""
        redis_client.client = mock_redis
        redis_client._connected = True
        mock_redis.delete.return_value = 2
        
        result = await redis_client.delete("key1", "key2")
        
        assert result == 2
        mock_redis.delete.assert_called_once_with("key1", "key2")
    
    async def test_exists_key(self, redis_client, mock_redis):
        """Test checking key existence."""
        redis_client.client = mock_redis
        redis_client._connected = True
        mock_redis.exists.return_value = 1
        
        result = await redis_client.exists("test_key")
        
        assert result is True
        mock_redis.exists.assert_called_once_with("test_key")
    
    async def test_expire_key(self, redis_client, mock_redis):
        """Test setting key expiration."""
        redis_client.client = mock_redis
        redis_client._connected = True
        
        result = await redis_client.expire("test_key", 3600)
        
        assert result is True
        mock_redis.expire.assert_called_once_with("test_key", 3600)
    
    async def test_expire_key_timedelta(self, redis_client, mock_redis):
        """Test setting key expiration with timedelta."""
        redis_client.client = mock_redis
        redis_client._connected = True
        
        expire_time = timedelta(hours=1)
        
        result = await redis_client.expire("test_key", expire_time)
        
        assert result is True
        mock_redis.expire.assert_called_once_with("test_key", 3600)
    
    async def test_ttl_key(self, redis_client, mock_redis):
        """Test getting key TTL."""
        redis_client.client = mock_redis
        redis_client._connected = True
        mock_redis.ttl.return_value = 3600
        
        result = await redis_client.ttl("test_key")
        
        assert result == 3600
        mock_redis.ttl.assert_called_once_with("test_key")
    
    async def test_hset_hash(self, redis_client, mock_redis):
        """Test setting hash fields."""
        redis_client.client = mock_redis
        redis_client._connected = True
        mock_redis.hset.return_value = 2
        
        mapping = {"field1": "value1", "field2": {"nested": "value"}}
        
        result = await redis_client.hset("test_hash", mapping)
        
        assert result == 2
        expected_mapping = {
            "field1": "value1",
            "field2": json.dumps({"nested": "value"})
        }
        mock_redis.hset.assert_called_once_with("test_hash", mapping=expected_mapping)
    
    async def test_hget_hash_field(self, redis_client, mock_redis):
        """Test getting hash field."""
        redis_client.client = mock_redis
        redis_client._connected = True
        mock_redis.hget.return_value = json.dumps({"nested": "value"})
        
        result = await redis_client.hget("test_hash", "field1")
        
        assert result == {"nested": "value"}
        mock_redis.hget.assert_called_once_with("test_hash", "field1")
    
    async def test_hgetall_hash(self, redis_client, mock_redis):
        """Test getting all hash fields."""
        redis_client.client = mock_redis
        redis_client._connected = True
        
        hash_data = {
            "field1": "value1",
            "field2": json.dumps({"nested": "value"})
        }
        mock_redis.hgetall.return_value = hash_data
        
        result = await redis_client.hgetall("test_hash")
        
        expected = {
            "field1": "value1",
            "field2": {"nested": "value"}
        }
        assert result == expected
    
    async def test_cache_job_progress(self, redis_client, mock_redis):
        """Test caching job progress."""
        redis_client.client = mock_redis
        redis_client._connected = True
        
        job_id = "test-job-123"
        progress_data = {"status": "processing", "progress": 50}
        
        with patch.object(redis_client, 'set', return_value=True) as mock_set:
            result = await redis_client.cache_job_progress(job_id, progress_data)
            
            assert result is True
            mock_set.assert_called_once_with(f"job_progress:{job_id}", progress_data, 3600)
    
    async def test_get_job_progress(self, redis_client, mock_redis):
        """Test getting job progress."""
        redis_client.client = mock_redis
        redis_client._connected = True
        
        job_id = "test-job-123"
        progress_data = {"status": "processing", "progress": 75}
        
        with patch.object(redis_client, 'get', return_value=progress_data) as mock_get:
            result = await redis_client.get_job_progress(job_id)
            
            assert result == progress_data
            mock_get.assert_called_once_with(f"job_progress:{job_id}")
    
    async def test_clear_job_cache(self, redis_client, mock_redis):
        """Test clearing job cache."""
        redis_client.client = mock_redis
        redis_client._connected = True
        
        job_id = "test-job-123"
        cache_keys = [f"job_progress:{job_id}", f"job_result:{job_id}"]
        
        mock_redis.keys.return_value = cache_keys
        
        with patch.object(redis_client, 'delete', return_value=2) as mock_delete:
            result = await redis_client.clear_job_cache(job_id)
            
            assert result is True
            mock_redis.keys.assert_called_once_with(f"job_*:{job_id}")
            mock_delete.assert_called_once_with(*cache_keys)
    
    async def test_clear_job_cache_no_keys(self, redis_client, mock_redis):
        """Test clearing job cache with no keys."""
        redis_client.client = mock_redis
        redis_client._connected = True
        
        job_id = "test-job-123"
        mock_redis.keys.return_value = []
        
        result = await redis_client.clear_job_cache(job_id)
        
        assert result is True
        mock_redis.keys.assert_called_once_with(f"job_*:{job_id}")


class TestRedisClientGlobalFunctions:
    """Test global Redis client functions."""
    
    async def test_get_redis_client_singleton(self):
        """Test Redis client singleton behavior."""
        from app.infrastructure.external.redis_client import get_redis_client, _redis_client
        
        # Reset global client
        import app.infrastructure.external.redis_client as redis_module
        redis_module._redis_client = None
        
        with patch.object(RedisClient, 'connect') as mock_connect:
            client1 = await get_redis_client()
            client2 = await get_redis_client()
            
            assert client1 is client2
            mock_connect.assert_called_once()
    
    async def test_close_redis_client(self):
        """Test closing global Redis client."""
        from app.infrastructure.external.redis_client import close_redis_client
        import app.infrastructure.external.redis_client as redis_module
        
        # Set up a mock client
        mock_client = AsyncMock()
        redis_module._redis_client = mock_client
        
        await close_redis_client()
        
        mock_client.disconnect.assert_called_once()
        assert redis_module._redis_client is None
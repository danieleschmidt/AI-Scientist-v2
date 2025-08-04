#!/usr/bin/env python3
"""
Distributed Caching Framework

Advanced multi-layer caching system for AI Scientist v2 with Redis backend,
local memory cache, and intelligent cache invalidation strategies.
"""

import asyncio
import hashlib
import json
import pickle
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
from functools import wraps
import zlib

# Optional Redis support
try:
    import redis
    import redis.asyncio as aioredis
    HAS_REDIS = True
except ImportError:
    redis = None
    aioredis = None
    HAS_REDIS = False

logger = logging.getLogger(__name__)


class CacheBackend(Enum):
    """Available cache backend types."""
    MEMORY = "memory"
    REDIS = "redis"
    HYBRID = "hybrid"


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"          # Least Recently Used
    LFU = "lfu"          # Least Frequently Used
    TTL = "ttl"          # Time To Live
    FIFO = "fifo"        # First In First Out


@dataclass
class CacheConfig:
    """Configuration for cache behavior."""
    backend: CacheBackend = CacheBackend.HYBRID
    default_ttl: int = 3600                    # Default TTL in seconds
    max_memory_items: int = 1000               # Max items in memory cache
    max_memory_size: int = 100 * 1024 * 1024   # Max memory cache size (100MB)
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    compression_threshold: int = 1024          # Compress values larger than this
    
    # Redis configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_ssl: bool = False
    redis_cluster: bool = False
    
    # Advanced configuration
    prefetch_enabled: bool = True              # Enable cache prefetching
    write_behind: bool = False                 # Enable write-behind caching
    circuit_breaker: bool = True               # Enable circuit breaker for Redis
    stats_enabled: bool = True                 # Enable cache statistics
    invalidation_patterns: List[str] = field(default_factory=list)


@dataclass
class CacheItem:
    """Cache item with metadata."""
    key: str
    value: Any
    timestamp: float
    ttl: Optional[int] = None
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    size: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache item is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def touch(self):
        """Update access information."""
        self.access_count += 1
        self.last_access = time.time()


@dataclass
class CacheStats:
    """Cache statistics tracking."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    writes: int = 0
    deletes: int = 0
    memory_usage: int = 0
    items_count: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 1.0 - self.hit_rate


class MemoryCache:
    """In-memory cache implementation with multiple eviction policies."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._cache: Dict[str, CacheItem] = {}
        self._access_order: List[str] = []  # For LRU
        self._stats = CacheStats()
        self._lock = threading.RLock()
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage."""
        data = pickle.dumps(value)
        
        # Compress if above threshold
        if len(data) > self.config.compression_threshold:
            data = zlib.compress(data)
        
        return data
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        try:
            # Try to decompress first
            decompressed = zlib.decompress(data)
            return pickle.loads(decompressed)
        except zlib.error:
            # Not compressed
            return pickle.loads(data)
    
    def _evict_if_needed(self):
        """Evict items based on policy if cache is full."""
        if len(self._cache) < self.config.max_memory_items:
            return
        
        keys_to_evict = []
        
        if self.config.eviction_policy == EvictionPolicy.LRU:
            # Evict least recently used
            keys_to_evict = [self._access_order[0]] if self._access_order else []
        
        elif self.config.eviction_policy == EvictionPolicy.LFU:
            # Evict least frequently used
            if self._cache:
                min_access = min(item.access_count for item in self._cache.values())
                keys_to_evict = [k for k, v in self._cache.items() if v.access_count == min_access][:1]
        
        elif self.config.eviction_policy == EvictionPolicy.TTL:
            # Evict expired items first
            now = time.time()
            expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
            if expired_keys:
                keys_to_evict = expired_keys[:1]
            elif self._access_order:  # Fallback to LRU
                keys_to_evict = [self._access_order[0]]
        
        elif self.config.eviction_policy == EvictionPolicy.FIFO:
            # Evict oldest item
            if self._cache:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].timestamp)
                keys_to_evict = [oldest_key]
        
        # Perform eviction
        for key in keys_to_evict:
            self._delete_item(key)
    
    def _delete_item(self, key: str):
        """Delete item from cache."""
        if key in self._cache:
            item = self._cache.pop(key)
            if key in self._access_order:
                self._access_order.remove(key)
            self._stats.memory_usage -= item.size
            self._stats.items_count -= 1
            self._stats.evictions += 1
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return None
            
            item = self._cache[key]
            
            # Check expiration
            if item.is_expired():
                self._delete_item(key)
                self._stats.misses += 1
                return None
            
            # Update access information
            item.touch()
            
            # Update LRU order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            
            self._stats.hits += 1
            return item.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        with self._lock:
            try:
                # Serialize and calculate size
                serialized = self._serialize_value(value)
                size = len(serialized)
                
                # Check if we need to evict
                self._evict_if_needed()
                
                # Create cache item
                item = CacheItem(
                    key=key,
                    value=value,
                    timestamp=time.time(),
                    ttl=ttl or self.config.default_ttl,
                    size=size
                )
                
                # Remove old item if exists
                if key in self._cache:
                    old_item = self._cache[key]
                    self._stats.memory_usage -= old_item.size
                else:
                    self._stats.items_count += 1
                
                # Add new item
                self._cache[key] = item
                self._stats.memory_usage += size
                self._stats.writes += 1
                
                # Update access order
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
                
                return True
                
            except Exception as e:
                logger.error(f"Error setting cache item {key}: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        with self._lock:
            if key in self._cache:
                self._delete_item(key)
                self._stats.deletes += 1
                return True
            return False
    
    def clear(self):
        """Clear all cache items."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._stats = CacheStats()
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            stats = CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                writes=self._stats.writes,
                deletes=self._stats.deletes,
                memory_usage=self._stats.memory_usage,
                items_count=self._stats.items_count
            )
            return stats


class RedisCache:
    """Redis-based cache implementation."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._client = None
        self._async_client = None
        self._stats = CacheStats()
        self._connect()
    
    def _connect(self):
        """Connect to Redis."""
        if not HAS_REDIS:
            logger.warning("Redis not available, falling back to memory cache")
            return
        
        try:
            # Synchronous client
            self._client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                ssl=self.config.redis_ssl,
                decode_responses=False  # We handle serialization
            )
            
            # Test connection
            self._client.ping()
            logger.info(f"Connected to Redis at {self.config.redis_host}:{self.config.redis_port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._client = None
    
    async def _get_async_client(self):
        """Get async Redis client."""
        if not HAS_REDIS or not self._client:
            return None
        
        if self._async_client is None:
            self._async_client = aioredis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                ssl=self.config.redis_ssl
            )
        
        return self._async_client
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for Redis storage."""
        data = pickle.dumps(value)
        
        # Compress if above threshold
        if len(data) > self.config.compression_threshold:
            data = zlib.compress(data)
        
        return data
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from Redis storage."""
        if data is None:
            return None
        
        try:
            # Try to decompress first
            decompressed = zlib.decompress(data)
            return pickle.loads(decompressed)
        except zlib.error:
            # Not compressed
            return pickle.loads(data)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if not self._client:
            return None
        
        try:
            data = self._client.get(key)
            if data is None:
                self._stats.misses += 1
                return None
            
            value = self._deserialize_value(data)
            self._stats.hits += 1
            return value
            
        except Exception as e:
            logger.error(f"Error getting from Redis cache {key}: {e}")
            self._stats.misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache."""
        if not self._client:
            return False
        
        try:
            data = self._serialize_value(value)
            ttl = ttl or self.config.default_ttl
            
            result = self._client.setex(key, ttl, data)
            if result:
                self._stats.writes += 1
            return result
            
        except Exception as e:
            logger.error(f"Error setting Redis cache {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from Redis cache."""
        if not self._client:
            return False
        
        try:
            result = self._client.delete(key)
            if result:
                self._stats.deletes += 1
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error deleting from Redis cache {key}: {e}")
            return False
    
    def clear(self):
        """Clear Redis cache."""
        if not self._client:
            return
        
        try:
            self._client.flushdb()
            self._stats = CacheStats()
        except Exception as e:
            logger.error(f"Error clearing Redis cache: {e}")
    
    async def async_get(self, key: str) -> Optional[Any]:
        """Async get value from Redis cache."""
        client = await self._get_async_client()
        if not client:
            return None
        
        try:
            data = await client.get(key)
            if data is None:
                self._stats.misses += 1
                return None
            
            value = self._deserialize_value(data)
            self._stats.hits += 1
            return value
            
        except Exception as e:
            logger.error(f"Error async getting from Redis cache {key}: {e}")
            self._stats.misses += 1
            return None
    
    async def async_set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Async set value in Redis cache."""
        client = await self._get_async_client()
        if not client:
            return False
        
        try:
            data = self._serialize_value(value)
            ttl = ttl or self.config.default_ttl
            
            result = await client.setex(key, ttl, data)
            if result:
                self._stats.writes += 1
            return result
            
        except Exception as e:
            logger.error(f"Error async setting Redis cache {key}: {e}")
            return False
    
    def get_stats(self) -> CacheStats:
        """Get Redis cache statistics."""
        return CacheStats(
            hits=self._stats.hits,
            misses=self._stats.misses,
            evictions=self._stats.evictions,
            writes=self._stats.writes,
            deletes=self._stats.deletes,
            memory_usage=0,  # Not tracked for Redis
            items_count=0    # Not tracked for Redis
        )


class HybridCache:
    """Hybrid cache combining memory and Redis for optimal performance."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.memory_cache = MemoryCache(config)
        self.redis_cache = RedisCache(config) if HAS_REDIS else None
        
    def get(self, key: str) -> Optional[Any]:
        """Get value with L1 (memory) -> L2 (Redis) fallback."""
        # Try memory cache first (L1)
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try Redis cache (L2)
        if self.redis_cache:
            value = self.redis_cache.get(key)
            if value is not None:
                # Populate L1 cache
                self.memory_cache.set(key, value)
                return value
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in both caches."""
        # Set in memory cache (L1)
        memory_success = self.memory_cache.set(key, value, ttl)
        
        # Set in Redis cache (L2)
        redis_success = True
        if self.redis_cache:
            redis_success = self.redis_cache.set(key, value, ttl)
        
        return memory_success and redis_success
    
    def delete(self, key: str) -> bool:
        """Delete value from both caches."""
        memory_success = self.memory_cache.delete(key)
        redis_success = True
        
        if self.redis_cache:
            redis_success = self.redis_cache.delete(key)
        
        return memory_success or redis_success
    
    def clear(self):
        """Clear both caches."""
        self.memory_cache.clear()
        if self.redis_cache:
            self.redis_cache.clear()
    
    def get_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for both cache layers."""
        stats = {"memory": self.memory_cache.get_stats()}
        if self.redis_cache:
            stats["redis"] = self.redis_cache.get_stats()
        return stats


class DistributedCache:
    """Main distributed cache interface."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize distributed cache."""
        self.config = config or CacheConfig()
        
        if self.config.backend == CacheBackend.MEMORY:
            self._cache = MemoryCache(self.config)
        elif self.config.backend == CacheBackend.REDIS:
            self._cache = RedisCache(self.config)
        else:  # HYBRID
            self._cache = HybridCache(self.config)
        
        logger.info(f"Distributed cache initialized with backend: {self.config.backend.value}")
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from function arguments."""
        # Create a deterministic key from arguments
        key_data = f"{prefix}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self._cache.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        return self._cache.set(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        return self._cache.delete(key)
    
    def clear(self):
        """Clear all cache data."""
        self._cache.clear()
    
    def get_stats(self) -> Union[CacheStats, Dict[str, CacheStats]]:
        """Get cache statistics."""
        return self._cache.get_stats()
    
    def cache_function(self, ttl: Optional[int] = None, key_prefix: Optional[str] = None):
        """Decorator for caching function results."""
        def decorator(func: Callable) -> Callable:
            func_prefix = key_prefix or f"{func.__module__}.{func.__name__}"
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_key(func_prefix, *args, **kwargs)
                
                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl)
                return result
            
            return wrapper
        return decorator
    
    def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern."""
        # This is a placeholder - full implementation would depend on backend
        logger.info(f"Cache invalidation pattern: {pattern}")


# Global cache instance
_cache = None


def get_cache(config: Optional[CacheConfig] = None) -> DistributedCache:
    """Get global cache instance."""
    global _cache
    if _cache is None:
        _cache = DistributedCache(config)
    return _cache


def init_cache(config: CacheConfig) -> DistributedCache:
    """Initialize global cache with configuration."""
    global _cache
    _cache = DistributedCache(config)
    return _cache


# Convenience decorators
def cache(ttl: Optional[int] = None, key_prefix: Optional[str] = None):
    """Decorator for caching function results using global cache."""
    return get_cache().cache_function(ttl, key_prefix)


def llm_cache(ttl: int = 3600):
    """Specialized cache decorator for LLM responses."""
    return cache(ttl=ttl, key_prefix="llm")


def semantic_scholar_cache(ttl: int = 86400):  # 24 hours
    """Specialized cache decorator for Semantic Scholar API responses."""
    return cache(ttl=ttl, key_prefix="semantic_scholar")


if __name__ == "__main__":
    # Example usage and testing
    import random
    
    # Initialize cache
    config = CacheConfig(backend=CacheBackend.HYBRID)
    cache_instance = DistributedCache(config)
    
    # Test basic operations
    cache_instance.set("test_key", {"data": "test_value"}, ttl=60)
    result = cache_instance.get("test_key")
    print(f"Cache test: {result}")
    
    # Test function caching
    @cache_instance.cache_function(ttl=30)
    def expensive_calculation(n: int) -> int:
        """Simulate expensive calculation."""
        time.sleep(0.1)  # Simulate work
        return n * n
    
    # Test cached function
    start = time.time()
    result1 = expensive_calculation(42)
    time1 = time.time() - start
    
    start = time.time()
    result2 = expensive_calculation(42)  # Should be cached
    time2 = time.time() - start
    
    print(f"First call: {result1} ({time1:.3f}s)")
    print(f"Second call: {result2} ({time2:.3f}s)")
    print(f"Speedup: {time1/time2:.1f}x")
    
    # Print statistics
    stats = cache_instance.get_stats()
    print(f"\nCache statistics: {stats}")
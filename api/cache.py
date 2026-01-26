"""
Thread-safe LRU cache with bounded size for WINDMAR API.

Provides a production-grade caching solution that:
- Bounds memory usage with configurable max entries
- Uses LRU eviction when cache is full
- Supports TTL (time-to-live) for entries
- Is thread-safe for concurrent access
- Provides metrics for monitoring
"""
import threading
import logging
from typing import TypeVar, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
import functools

logger = logging.getLogger(__name__)

K = TypeVar('K')
V = TypeVar('V')


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    value: Any
    created_at: datetime
    expires_at: Optional[datetime]
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)


class BoundedLRUCache:
    """
    Thread-safe LRU cache with bounded size and TTL support.

    Features:
    - Maximum entry limit with LRU eviction
    - Optional TTL for automatic expiration
    - Thread-safe for concurrent read/write
    - Metrics tracking (hits, misses, evictions)

    Usage:
        cache = BoundedLRUCache(max_size=1000, default_ttl_seconds=3600)
        cache.set("key", value)
        result = cache.get("key")
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl_seconds: Optional[int] = 3600,
        name: str = "default"
    ):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of entries before LRU eviction
            default_ttl_seconds: Default TTL for entries (None = no expiration)
            name: Cache name for logging/metrics
        """
        self.max_size = max_size
        self.default_ttl_seconds = default_ttl_seconds
        self.name = name

        self._cache: OrderedDict[Any, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        # Metrics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    def get(self, key: K) -> Optional[V]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check expiration
            if entry.expires_at and datetime.utcnow() > entry.expires_at:
                self._remove(key)
                self._expirations += 1
                self._misses += 1
                return None

            # Update access metadata and move to end (most recently used)
            entry.access_count += 1
            entry.last_accessed = datetime.utcnow()
            self._cache.move_to_end(key)

            self._hits += 1
            return entry.value

    def set(
        self,
        key: K,
        value: V,
        ttl_seconds: Optional[int] = None
    ) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: TTL for this entry (None = use default)
        """
        with self._lock:
            # Determine expiration
            ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
            expires_at = None
            if ttl:
                expires_at = datetime.utcnow() + timedelta(seconds=ttl)

            # Create entry
            entry = CacheEntry(
                value=value,
                created_at=datetime.utcnow(),
                expires_at=expires_at,
            )

            # Update or insert
            if key in self._cache:
                self._cache[key] = entry
                self._cache.move_to_end(key)
            else:
                # Check if we need to evict
                while len(self._cache) >= self.max_size:
                    self._evict_oldest()

                self._cache[key] = entry

    def delete(self, key: K) -> bool:
        """
        Delete entry from cache.

        Args:
            key: Cache key

        Returns:
            True if key was present and deleted
        """
        with self._lock:
            if key in self._cache:
                self._remove(key)
                return True
            return False

    def clear(self) -> int:
        """
        Clear all entries from cache.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Cache '{self.name}' cleared: {count} entries removed")
            return count

    def _remove(self, key: K) -> None:
        """Remove entry without lock (internal use)."""
        del self._cache[key]

    def _evict_oldest(self) -> None:
        """Evict oldest (least recently used) entry."""
        if self._cache:
            oldest_key = next(iter(self._cache))
            self._remove(oldest_key)
            self._evictions += 1
            logger.debug(f"Cache '{self.name}' evicted: {oldest_key}")

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        with self._lock:
            now = datetime.utcnow()
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.expires_at and now > entry.expires_at
            ]

            for key in expired_keys:
                self._remove(key)
                self._expirations += 1

            if expired_keys:
                logger.debug(f"Cache '{self.name}' cleanup: {len(expired_keys)} expired entries removed")

            return len(expired_keys)

    def get_or_set(
        self,
        key: K,
        factory: Callable[[], V],
        ttl_seconds: Optional[int] = None
    ) -> V:
        """
        Get value from cache, or compute and cache it if missing.

        Args:
            key: Cache key
            factory: Function to compute value if not cached
            ttl_seconds: TTL for this entry

        Returns:
            Cached or computed value
        """
        # Try to get from cache first
        value = self.get(key)
        if value is not None:
            return value

        # Compute value
        value = factory()

        # Cache it
        self.set(key, value, ttl_seconds)

        return value

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache metrics
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            return {
                'name': self.name,
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': round(hit_rate, 4),
                'evictions': self._evictions,
                'expirations': self._expirations,
                'default_ttl_seconds': self.default_ttl_seconds,
            }

    def __len__(self) -> int:
        """Get number of entries in cache."""
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: K) -> bool:
        """Check if key is in cache (without updating access time)."""
        with self._lock:
            if key not in self._cache:
                return False
            entry = self._cache[key]
            if entry.expires_at and datetime.utcnow() > entry.expires_at:
                return False
            return True


def cached(
    cache: BoundedLRUCache,
    key_func: Optional[Callable[..., str]] = None,
    ttl_seconds: Optional[int] = None,
):
    """
    Decorator to cache function results.

    Args:
        cache: BoundedLRUCache instance to use
        key_func: Function to generate cache key from args (default: str of args)
        ttl_seconds: TTL for cached results

    Usage:
        weather_cache = BoundedLRUCache(max_size=100, name="weather")

        @cached(weather_cache, key_func=lambda lat, lon: f"{lat:.1f},{lon:.1f}")
        def get_weather(lat: float, lon: float):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = f"{func.__name__}:{args}:{sorted(kwargs.items())}"

            # Check cache
            result = cache.get(key)
            if result is not None:
                return result

            # Compute and cache
            result = func(*args, **kwargs)
            cache.set(key, result, ttl_seconds)

            return result

        # Attach cache reference for testing/inspection
        wrapper._cache = cache

        return wrapper

    return decorator


# Pre-configured caches for common use cases
weather_cache = BoundedLRUCache(
    max_size=500,
    default_ttl_seconds=3600,  # 1 hour
    name="weather"
)

route_cache = BoundedLRUCache(
    max_size=100,
    default_ttl_seconds=1800,  # 30 minutes
    name="routes"
)

calculation_cache = BoundedLRUCache(
    max_size=200,
    default_ttl_seconds=900,  # 15 minutes
    name="calculations"
)


def get_all_cache_stats() -> Dict[str, Dict[str, Any]]:
    """Get stats for all registered caches."""
    return {
        'weather': weather_cache.get_stats(),
        'routes': route_cache.get_stats(),
        'calculations': calculation_cache.get_stats(),
    }


def cleanup_all_caches() -> Dict[str, int]:
    """Cleanup expired entries from all caches."""
    return {
        'weather': weather_cache.cleanup_expired(),
        'routes': route_cache.cleanup_expired(),
        'calculations': calculation_cache.cleanup_expired(),
    }

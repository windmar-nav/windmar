"""
Rate limiting for WINDMAR API using Redis and SlowAPI.
"""
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
import redis
import logging

from api.config import settings

logger = logging.getLogger(__name__)

# Initialize Redis client
redis_client = None
if settings.redis_enabled:
    try:
        redis_client = redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=5
        )
        # Test connection
        redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        logger.warning("Rate limiting will not work without Redis")
        redis_client = None


def get_api_key_identifier(request: Request) -> str:
    """
    Get identifier for rate limiting.
    Uses API key if present, otherwise falls back to IP address.

    Args:
        request: FastAPI request object

    Returns:
        str: Identifier for rate limiting
    """
    # Try to get API key from header
    api_key = request.headers.get(settings.api_key_header)
    if api_key:
        # Use first 8 characters of API key for identification
        return f"key:{api_key[:8]}"

    # Fall back to IP address
    return f"ip:{get_remote_address(request)}"


# Initialize limiter
limiter = Limiter(
    key_func=get_api_key_identifier,
    enabled=settings.rate_limit_enabled and redis_client is not None,
    storage_uri=settings.redis_url if redis_client else None,
    strategy="fixed-window"  # or "moving-window" for more sophisticated limiting
)


def get_rate_limit_string() -> str:
    """
    Get rate limit string for use with @limiter.limit() decorator.

    Returns:
        str: Rate limit string (e.g., "60/minute")
    """
    return f"{settings.rate_limit_per_minute}/minute"


def get_rate_limit_hourly_string() -> str:
    """
    Get hourly rate limit string.

    Returns:
        str: Rate limit string (e.g., "1000/hour")
    """
    return f"{settings.rate_limit_per_hour}/hour"


async def check_rate_limit(request: Request, redis_key: str, limit: int, window: int) -> bool:
    """
    Check if request is within rate limit.

    Args:
        request: FastAPI request object
        redis_key: Redis key for this rate limit
        limit: Maximum number of requests
        window: Time window in seconds

    Returns:
        bool: True if within limit, False otherwise
    """
    if not settings.rate_limit_enabled:
        return True

    if not redis_client:
        logger.warning("Rate limit check denied: Redis unavailable")
        return False

    try:
        # Get current count
        current = redis_client.get(redis_key)
        if current is None:
            # First request in window
            redis_client.setex(redis_key, window, 1)
            return True

        current = int(current)
        if current >= limit:
            return False

        # Increment count
        redis_client.incr(redis_key)
        return True

    except Exception as e:
        logger.error(f"Rate limit check error: {e}")
        # Fail closed: deny request when rate limiter is broken
        return False


async def get_rate_limit_status(identifier: str) -> dict:
    """
    Get current rate limit status for an identifier.

    Args:
        identifier: Identifier (API key or IP)

    Returns:
        dict: Rate limit status information
    """
    if not redis_client or not settings.rate_limit_enabled:
        return {
            "enabled": False,
            "message": "Rate limiting is disabled"
        }

    try:
        # Check minute window
        minute_key = f"rate_limit:minute:{identifier}"
        minute_count = redis_client.get(minute_key)
        minute_ttl = redis_client.ttl(minute_key)

        # Check hour window
        hour_key = f"rate_limit:hour:{identifier}"
        hour_count = redis_client.get(hour_key)
        hour_ttl = redis_client.ttl(hour_key)

        return {
            "enabled": True,
            "per_minute": {
                "limit": settings.rate_limit_per_minute,
                "remaining": max(0, settings.rate_limit_per_minute - int(minute_count or 0)),
                "reset_in": minute_ttl if minute_ttl > 0 else 0
            },
            "per_hour": {
                "limit": settings.rate_limit_per_hour,
                "remaining": max(0, settings.rate_limit_per_hour - int(hour_count or 0)),
                "reset_in": hour_ttl if hour_ttl > 0 else 0
            }
        }

    except Exception as e:
        logger.error(f"Error getting rate limit status: {e}")
        return {
            "enabled": True,
            "error": str(e)
        }

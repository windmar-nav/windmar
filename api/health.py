"""
Comprehensive health check module for WINDMAR API.

Provides detailed health checks for all dependencies and system components.
Designed for Kubernetes liveness/readiness probes and load balancer health checks.
"""
import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import redis

from api.config import settings
from api.cache import get_all_cache_stats
from api.resilience import get_all_circuit_breaker_status

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health status of a single component."""
    name: str
    status: HealthStatus
    latency_ms: Optional[float] = None
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


def check_database_health() -> ComponentHealth:
    """
    Check PostgreSQL database connectivity.

    Returns:
        ComponentHealth with database status
    """
    start = datetime.utcnow()

    try:
        from api.database import SessionLocal

        db = SessionLocal()
        try:
            # Execute simple query
            result = db.execute("SELECT 1").scalar()
            latency_ms = (datetime.utcnow() - start).total_seconds() * 1000

            if result == 1:
                return ComponentHealth(
                    name="database",
                    status=HealthStatus.HEALTHY,
                    latency_ms=round(latency_ms, 2),
                    message="PostgreSQL connected",
                )
            else:
                return ComponentHealth(
                    name="database",
                    status=HealthStatus.UNHEALTHY,
                    message="Unexpected query result",
                )
        finally:
            db.close()

    except Exception as e:
        latency_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.error(f"Database health check failed: {e}")
        return ComponentHealth(
            name="database",
            status=HealthStatus.UNHEALTHY,
            latency_ms=round(latency_ms, 2),
            message=f"Connection failed: {type(e).__name__}",
        )


def check_redis_health() -> ComponentHealth:
    """
    Check Redis connectivity.

    Returns:
        ComponentHealth with Redis status
    """
    start = datetime.utcnow()

    if not settings.redis_enabled:
        return ComponentHealth(
            name="redis",
            status=HealthStatus.HEALTHY,
            message="Redis disabled (not required)",
        )

    try:
        client = redis.from_url(
            settings.redis_url,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        pong = client.ping()
        latency_ms = (datetime.utcnow() - start).total_seconds() * 1000

        if pong:
            # Get some stats
            info = client.info(section="memory")
            return ComponentHealth(
                name="redis",
                status=HealthStatus.HEALTHY,
                latency_ms=round(latency_ms, 2),
                message="Redis connected",
                details={
                    "used_memory_human": info.get("used_memory_human"),
                    "connected_clients": info.get("connected_clients", "N/A"),
                },
            )
        else:
            return ComponentHealth(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                message="Ping failed",
            )

    except Exception as e:
        latency_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.error(f"Redis health check failed: {e}")
        return ComponentHealth(
            name="redis",
            status=HealthStatus.DEGRADED if not settings.redis_enabled else HealthStatus.UNHEALTHY,
            latency_ms=round(latency_ms, 2),
            message=f"Connection failed: {type(e).__name__}",
        )


def check_weather_provider_health() -> ComponentHealth:
    """
    Check weather data provider status.

    Returns:
        ComponentHealth with provider status
    """
    try:
        from api.state import get_app_state

        app_state = get_app_state()
        providers = app_state.weather_providers

        if providers is None:
            return ComponentHealth(
                name="weather_provider",
                status=HealthStatus.DEGRADED,
                message="Providers not initialized",
            )

        copernicus = providers.get('copernicus')
        has_cds = copernicus._has_cdsapi if copernicus else False
        has_cmems = copernicus._has_copernicusmarine if copernicus else False

        if has_cds and has_cmems:
            status = HealthStatus.HEALTHY
            message = "Full Copernicus access available"
        elif has_cds or has_cmems:
            status = HealthStatus.DEGRADED
            message = "Partial Copernicus access"
        else:
            status = HealthStatus.DEGRADED
            message = "Using synthetic data fallback"

        return ComponentHealth(
            name="weather_provider",
            status=status,
            message=message,
            details={
                "cds_available": has_cds,
                "cmems_available": has_cmems,
                "fallback_available": True,
            },
        )

    except Exception as e:
        logger.error(f"Weather provider health check failed: {e}")
        return ComponentHealth(
            name="weather_provider",
            status=HealthStatus.DEGRADED,
            message=f"Check failed: {type(e).__name__}",
        )


async def perform_full_health_check() -> Dict[str, Any]:
    """
    Perform comprehensive health check of all components.

    Returns:
        Dict with overall status and component details
    """
    start = datetime.utcnow()

    # Run health checks
    db_health = check_database_health()
    redis_health = check_redis_health()
    weather_health = check_weather_provider_health()

    components = [db_health, redis_health, weather_health]

    # Determine overall status
    unhealthy_count = sum(1 for c in components if c.status == HealthStatus.UNHEALTHY)
    degraded_count = sum(1 for c in components if c.status == HealthStatus.DEGRADED)

    if unhealthy_count > 0:
        overall_status = HealthStatus.UNHEALTHY
    elif degraded_count > 0:
        overall_status = HealthStatus.DEGRADED
    else:
        overall_status = HealthStatus.HEALTHY

    total_time_ms = (datetime.utcnow() - start).total_seconds() * 1000

    return {
        "status": overall_status.value,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": "2.1.0",
        "check_duration_ms": round(total_time_ms, 2),
        "components": {
            c.name: {
                "status": c.status.value,
                "latency_ms": c.latency_ms,
                "message": c.message,
                **({"details": c.details} if c.details else {}),
            }
            for c in components
        },
    }


async def perform_liveness_check() -> Dict[str, Any]:
    """
    Simple liveness check for Kubernetes probes.

    This should be fast and only check if the service is alive,
    not if all dependencies are healthy.

    Returns:
        Dict with basic status
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


async def perform_readiness_check() -> Dict[str, Any]:
    """
    Readiness check for Kubernetes probes.

    Checks if the service is ready to accept traffic.
    Includes database connectivity check.

    Returns:
        Dict with readiness status
    """
    db_health = check_database_health()

    # Service is ready if database is connected
    is_ready = db_health.status == HealthStatus.HEALTHY

    return {
        "status": "ready" if is_ready else "not_ready",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "database": db_health.status.value,
    }


async def get_detailed_status() -> Dict[str, Any]:
    """
    Get detailed system status including metrics and cache stats.

    Returns:
        Dict with comprehensive system information
    """
    health = await perform_full_health_check()

    # Add cache stats
    cache_stats = get_all_cache_stats()

    # Add circuit breaker status
    circuit_breakers = get_all_circuit_breaker_status()

    # Get uptime
    from api.state import get_app_state
    app_state = get_app_state()

    return {
        **health,
        "uptime_seconds": round(app_state.uptime_seconds, 2),
        "environment": settings.environment,
        "caches": cache_stats,
        "circuit_breakers": circuit_breakers,
        "config": {
            "auth_enabled": settings.auth_enabled,
            "rate_limit_enabled": settings.rate_limit_enabled,
            "metrics_enabled": settings.metrics_enabled,
        },
    }

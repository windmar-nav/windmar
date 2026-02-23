"""
System / health / metrics / data-sources API router.

Handles health checks, Prometheus metrics, log streaming,
data-source status, and the root endpoint.
"""

import asyncio
import collections
import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse, JSONResponse

from api.config import settings
from api.middleware import metrics_collector, get_request_id
from api.rate_limit import limiter
from api.state import get_app_state

router = APIRouter(tags=["System"])

logger = logging.getLogger(__name__)


# =============================================================================
# Server-Side Event Log Stream (for frontend DebugConsole)
# =============================================================================
_log_buffer: collections.deque = collections.deque(maxlen=200)
_log_event = asyncio.Event()


class _BufferHandler(logging.Handler):
    """Captures log records into a ring buffer for SSE streaming."""
    def emit(self, record):
        try:
            entry = {
                "ts": record.created,
                "level": record.levelname.lower(),
                "msg": self.format(record),
            }
            _log_buffer.append(entry)
            # Signal waiting SSE clients (thread-safe via asyncio)
            try:
                _log_event.set()
            except Exception:
                pass
        except Exception:
            pass


_buf_handler = _BufferHandler()
_buf_handler.setLevel(logging.INFO)
_buf_handler.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger().addHandler(_buf_handler)
logging.getLogger("uvicorn.access").addHandler(_buf_handler)


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/")
async def root():
    """
    API root endpoint.

    Returns basic API information and available endpoint categories.
    """
    return {
        "name": "WINDMAR API",
        "version": "2.1.0",
        "status": "operational",
        "docs": "/api/docs",
        "endpoints": {
            "health": "/api/health",
            "metrics": "/api/metrics",
            "weather": "/api/weather/...",
            "routes": "/api/routes/...",
            "voyage": "/api/voyage/...",
            "vessel": "/api/vessel/...",
            "zones": "/api/zones/...",
        }
    }


@router.get("/api/logs/stream")
async def log_stream():
    """SSE endpoint streaming backend log entries to the frontend console."""
    async def _generate():
        last_idx = len(_log_buffer)
        # Send recent history first
        for entry in list(_log_buffer)[-50:]:
            yield f"data: {json.dumps(entry)}\n\n"
        while True:
            _log_event.clear()
            try:
                await asyncio.wait_for(_log_event.wait(), timeout=15.0)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            # Drain new entries
            buf = list(_log_buffer)
            for entry in buf[last_idx:]:
                yield f"data: {json.dumps(entry)}\n\n"
            last_idx = len(buf)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/api/health")
async def health_check():
    """
    Comprehensive health check endpoint for load balancers and orchestrators.

    Checks connectivity to all dependencies:
    - Database (PostgreSQL)
    - Cache (Redis)
    - Weather data providers

    Returns:
        - status: Overall health status (healthy/degraded/unhealthy)
        - timestamp: Current UTC timestamp
        - version: API version
        - components: Individual component health status
    """
    from api.health import perform_full_health_check
    result = await perform_full_health_check()
    result["request_id"] = get_request_id()
    return result


@router.get("/api/health/live")
async def liveness_check():
    """
    Kubernetes liveness probe endpoint.

    Simple check that the service is alive.
    Use this for K8s livenessProbe configuration.
    """
    from api.health import perform_liveness_check
    return await perform_liveness_check()


@router.get("/api/health/ready")
async def readiness_check():
    """
    Kubernetes readiness probe endpoint.

    Checks if the service is ready to accept traffic.
    Use this for K8s readinessProbe configuration.
    """
    from api.health import perform_readiness_check
    result = await perform_readiness_check()

    # Return 503 if not ready
    if result.get("status") != "ready":
        raise HTTPException(status_code=503, detail="Service not ready")

    return result


@router.get("/api/status")
async def detailed_status():
    """
    Detailed system status endpoint.

    Returns comprehensive information about the system including:
    - Health status of all components
    - Cache statistics
    - Circuit breaker states
    - Configuration summary
    """
    from api.health import get_detailed_status
    return await get_detailed_status()


@router.get("/api/metrics", response_class=PlainTextResponse)
async def get_metrics():
    """
    Prometheus-compatible metrics endpoint.

    Returns metrics in Prometheus exposition format for scraping.
    Includes:
    - Request counts by endpoint and status
    - Request duration summaries
    - Error counts
    - Service uptime
    """
    return metrics_collector.get_prometheus_metrics()


@router.get("/api/metrics/json")
async def get_metrics_json():
    """
    Metrics endpoint in JSON format.

    Alternative to Prometheus format for custom dashboards.
    """
    return metrics_collector.get_metrics()


@router.get("/api/data-sources")
async def get_data_sources():
    """
    Get status of available data sources.

    Shows which Copernicus APIs are configured and available.
    """
    # Resolve copernicus provider from app state
    _app_state = get_app_state()
    copernicus_provider = _app_state.weather_providers['copernicus']

    # Check if pygrib is available for GFS
    try:
        import pygrib
        has_pygrib = True
    except ImportError:
        has_pygrib = False

    return {
        "gfs": {
            "available": has_pygrib,
            "description": "NOAA GFS 0.25\u00b0 near-real-time wind (updated every 6h, ~3.5h lag)",
            "requires": "pygrib + libeccodes (no credentials needed)",
        },
        "copernicus": {
            "cds": {
                "available": copernicus_provider._has_cdsapi,
                "configured": settings.has_cds_credentials,
                "description": "Climate Data Store (ERA5 reanalysis wind, ~5-day lag)",
                "setup": "Set CDSAPI_KEY in .env (register at https://cds.climate.copernicus.eu)",
            },
            "cmems": {
                "available": copernicus_provider._has_copernicusmarine,
                "configured": settings.has_cmems_credentials,
                "description": "Copernicus Marine Service (waves, currents)",
                "setup": "Set COPERNICUSMARINE_SERVICE_USERNAME/PASSWORD in .env (register at https://marine.copernicus.eu)",
            },
            "xarray": {
                "available": copernicus_provider._has_xarray,
                "description": "NetCDF data handling",
                "setup": "pip install xarray netcdf4",
            },
        },
        "fallback": {
            "synthetic": {
                "available": True,
                "description": "Synthetic data generator (always available)",
            }
        },
        "wind_provider_chain": "GFS \u2192 ERA5 \u2192 Synthetic",
        "active_wind_source": "gfs" if has_pygrib else (
            "era5" if (copernicus_provider._has_cdsapi and settings.has_cds_credentials)
            else "synthetic"
        ),
    }


# =============================================================================
# Coastline GeoJSON (GSHHS vector overlay for crisp land boundaries)
# =============================================================================

_coastline_cache: dict = {}


@router.get("/api/coastline")
async def get_coastline(
    lat_min: float = -90,
    lat_max: float = 90,
    lon_min: float = -180,
    lon_max: float = 180,
    simplify: float = 0.005,
):
    """
    Return simplified GSHHS land polygons as GeoJSON, clipped to the viewport.

    Used by the frontend to render a crisp vector coastline overlay above
    weather grid layers, replacing the blocky raster mask.
    """
    cache_key = f"{lat_min:.1f},{lat_max:.1f},{lon_min:.1f},{lon_max:.1f},{simplify}"
    if cache_key in _coastline_cache:
        return _coastline_cache[cache_key]

    try:
        from src.data.land_mask import get_land_geometry
        from shapely.geometry import box, mapping

        land = get_land_geometry()
        if land is None:
            return JSONResponse(
                status_code=404,
                content={"detail": "GSHHS coastline data not available"},
            )

        viewport = box(lon_min, lat_min, lon_max, lat_max)
        clipped = land.intersection(viewport)

        if clipped.is_empty:
            result = {"type": "FeatureCollection", "features": []}
        else:
            simplified = clipped.simplify(simplify, preserve_topology=True)
            result = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"type": "land"},
                        "geometry": mapping(simplified),
                    }
                ],
            }

        # Cache (bounded size — evict oldest when > 20 entries)
        if len(_coastline_cache) > 20:
            oldest = next(iter(_coastline_cache))
            del _coastline_cache[oldest]
        _coastline_cache[cache_key] = result
        return result

    except Exception as e:
        logger.warning(f"Coastline endpoint error: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Failed to generate coastline: {str(e)}"},
        )


# =============================================================================
# Ocean / land point check (waypoint validation)
# =============================================================================

@router.get("/api/check-ocean")
async def check_ocean(lat: float, lon: float):
    """Return whether a point is over ocean (True) or land (False)."""
    from src.data.land_mask import is_ocean as _is_ocean
    return {"ocean": _is_ocean(lat, lon)}


# =============================================================================
# Demo Authentication
# =============================================================================

@router.post("/api/demo/verify")
@limiter.limit("5/minute")
async def verify_demo_key(request: Request):
    """
    Verify a licence key and return the user's access tier.

    Accepts X-API-Key header. Returns 200 with tier if valid, 401 if not.
    Only functional when DEMO_MODE=true.
    """
    from api.demo import get_user_tier

    if not settings.demo_mode:
        return {"authenticated": True, "demo_mode": False, "tier": "full"}

    api_key = request.headers.get("X-API-Key", "")
    if not api_key:
        return JSONResponse(
            status_code=401,
            content={"authenticated": False, "detail": "Licence key is required"},
        )

    tier = get_user_tier(request)
    if tier in ("full", "demo"):
        return {"authenticated": True, "demo_mode": True, "tier": tier}

    return JSONResponse(
        status_code=401,
        content={"authenticated": False, "detail": "Invalid licence key"},
    )

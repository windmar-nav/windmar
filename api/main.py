"""
FastAPI Backend for WINDMAR Maritime Route Optimizer.

Provides REST API endpoints for:
- Weather data visualization (wind/wave fields)
- Route management (waypoints, RTZ import)
- Voyage calculation (per-leg SOG, ETA, fuel)
- Vessel configuration

Version: 2.1.0
License: Apache 2.0 - See LICENSE file
"""

import asyncio
import collections
import io
import json
import logging
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, File, Query, Response, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from api.demo import require_not_demo, is_demo, demo_mode_response
from fastapi.exceptions import RequestValidationError
from slowapi.errors import RateLimitExceeded
import uvicorn

# File upload size limits (security)
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB general limit
MAX_RTZ_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB for RTZ files
MAX_CSV_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB for CSV files

# Import WINDMAR modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimization.vessel_model import VesselModel, VesselSpecs
from src.optimization.voyage import VoyageCalculator, LegWeather
from src.optimization.base_optimizer import OptimizedRoute
from src.optimization.route_optimizer import RouteOptimizer
from src.optimization.visir_optimizer import VisirOptimizer
from src.optimization.grid_weather_provider import GridWeatherProvider
from src.optimization.temporal_weather_provider import TemporalGridWeatherProvider, WeatherProvenance
from src.optimization.weather_assessment import RouteWeatherAssessment
from src.optimization.monte_carlo import MonteCarloSimulator, MonteCarloResult as MCResult
from src.optimization.vessel_calibration import (
    VesselCalibrator, NoonReport, CalibrationFactors, create_calibrated_model
)
from src.routes.rtz_parser import (
    Route, Waypoint, parse_rtz_string, create_route_from_waypoints,
    haversine_distance, calculate_bearing
)
from src.data.land_mask import is_ocean

from src.data.copernicus import (
    CopernicusDataProvider, SyntheticDataProvider, GFSDataProvider, WeatherData,
    ClimatologyProvider, UnifiedWeatherProvider, WeatherDataSource
)


from src.compliance.cii import (
    CIICalculator, VesselType as CIIVesselType, CIIRating, CIIResult,
    CIIProjection, estimate_cii_from_route, annualize_voyage_cii
)
from src.data.regulatory_zones import (
    get_zone_checker, Zone, ZoneProperties, ZoneType, ZoneInteraction
)
from api.config import settings
from api.middleware import (
    setup_middleware,
    metrics_collector,
    structured_logger,
    get_request_id,
)
from api.auth import get_api_key, get_optional_api_key
from api.rate_limit import limiter, get_rate_limit_string
from api.state import get_app_state, get_vessel_state
from api.cache import weather_cache, get_all_cache_stats
from api.resilience import get_all_circuit_breaker_status

# Configure structured logging for production
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format='%(message)s',  # JSON logs are self-contained
)
logger = logging.getLogger(__name__)


# =============================================================================
# Application Factory
# =============================================================================

def create_app() -> FastAPI:
    """
    Application factory for WINDMAR API.

    Creates and configures the FastAPI application with all middleware,
    routes, and dependencies. Supports both production and development modes.

    Returns:
        FastAPI: Configured application instance
    """
    application = FastAPI(
        title="WINDMAR API",
        description="""
## Maritime Route Optimization API

Professional-grade API for maritime route optimization, weather routing,
and voyage planning.

### Features
- Real-time weather data integration (Copernicus CDS/CMEMS)
- A* pathfinding route optimization
- Vessel performance modeling with calibration
- Regulatory zone management (ECA, HRA, TSS)
- Fuel consumption prediction

### Authentication
API key authentication required for all endpoints except health checks.
Include your API key in the `X-API-Key` header.

### Rate Limiting
- 60 requests per minute
- 1000 requests per hour

### Support
Contact: contact@slmar.co
        """,
        version="2.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        license_info={
            "name": "Apache 2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0",
        },
        contact={
            "name": "WINDMAR Support",
            "url": "https://slmar.co",
            "email": "contact@slmar.co",
        },
    )

    # Setup production middleware (security headers, logging, metrics, etc.)
    setup_middleware(
        application,
        debug=settings.is_development,
        enable_hsts=settings.is_production,
    )

    # CORS middleware - use configured origins only (NO WILDCARDS)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=["*"],
    )

    # Add rate limiter to app state
    application.state.limiter = limiter

    # Add rate limit exception handler
    @application.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "detail": str(exc.detail),
                "retry_after": getattr(exc, 'retry_after', 60),
            },
            headers={"Retry-After": str(getattr(exc, 'retry_after', 60))},
        )

    return application


# =============================================================================
# Database Migration Runner
# =============================================================================

def _run_weather_migrations():
    """Apply weather table migrations if they don't exist yet."""
    db_url = os.environ.get("DATABASE_URL", settings.database_url)
    if not db_url.startswith("postgresql"):
        logger.info("Skipping weather migrations (non-PostgreSQL database)")
        return
    try:
        import psycopg2
    except ImportError:
        logger.warning("psycopg2 not installed, skipping weather migrations")
        return

    try:
        conn = psycopg2.connect(db_url)
        conn.autocommit = True
        cur = conn.cursor()

        # Use advisory lock to prevent concurrent migration by multiple workers
        cur.execute("SELECT pg_try_advisory_lock(20250208)")
        got_lock = cur.fetchone()[0]
        if not got_lock:
            logger.info("Another worker is running weather migrations, skipping")
            conn.close()
            return

        try:
            # Check if tables already exist
            cur.execute(
                "SELECT 1 FROM information_schema.tables WHERE table_name = 'weather_forecast_runs'"
            )
            if cur.fetchone():
                logger.info("Weather tables already exist")
                return

            # Apply migration
            migration_path = Path(__file__).parent.parent / "docker" / "migrations" / "001_weather_tables.sql"
            if migration_path.exists():
                sql = migration_path.read_text()
                cur.execute(sql)
                logger.info("Weather database migration applied successfully")
            else:
                logger.warning(f"Migration file not found: {migration_path}")
        finally:
            cur.execute("SELECT pg_advisory_unlock(20250208)")
            conn.close()
    except Exception as e:
        logger.error(f"Failed to run weather migrations: {e}")


# Create the application
app = create_app()

# Include live sensor API router
try:
    from api.live import include_in_app as include_live_routes
    include_live_routes(app)
except ImportError:
    logging.getLogger(__name__).info("Live sensor module not available, skipping live routes")

# Initialize application state (thread-safe singleton)
_ = get_app_state()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    logging.error(f"Validation error on {request.method} {request.url.path}: {exc.errors()}")
    logging.error(f"Request body: {body[:500]}")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


# ============================================================================
# Pydantic Models (extracted to api/schemas/)
# ============================================================================

from api.schemas import (  # noqa: E402
    Position,
    WaypointModel,
    RouteModel,
    VoyageRequest,
    LegResultModel,
    DataSourceSummary,
    VoyageResponse,
    MonteCarloRequest,
    PercentileFloat,
    PercentileString,
    MonteCarloResponse,
    WindDataPoint,
    WeatherGridResponse,
    VelocityDataResponse,
    VesselConfig,
    NoonReportModel,
    CalibrationFactorsModel,
    CalibrationResponse,
    OptimizationRequest,
    OptimizationLegModel,
    WeatherProvenanceModel,
    SafetySummary,
    SpeedScenarioModel,
    OptimizationResponse,
    PerformancePredictionRequest,
    ZoneCoordinate,
    CreateZoneRequest,
    ZoneResponse,
    CIIFuelConsumption,
    CIICalculateRequest,
    CIIProjectRequest,
    CIIReductionRequest,
    EngineLogUploadResponse,
    EngineLogEntryResponse,
    EngineLogSummaryResponse,
    EngineLogCalibrateResponse,
)


# ============================================================================
# Global State
# ============================================================================

# =============================================================================
# Centralized state (vessel + weather providers)
# All vessel globals replaced by VesselState singleton (api/state.py).
# =============================================================================
from api.state import get_app_state, get_vessel_state  # noqa: E402

_app_state = get_app_state()
_vs = get_vessel_state()
_wx = _app_state.weather_providers

# Shim aliases — endpoints still reference these names directly
copernicus_provider = _wx['copernicus']
climatology_provider = _wx['climatology']
unified_weather_provider = _wx['unified']
synthetic_provider = _wx['synthetic']
gfs_provider = _wx['gfs']
db_weather = _wx.get('db_weather')
weather_ingestion = _wx.get('weather_ingestion')

from api.weather_service import (  # noqa: E402
    get_wind_field,
    get_wave_field,
    get_current_field,
    get_sst_field,
    get_visibility_field,
    get_ice_field,
    get_weather_at_point,
    weather_provider,
    supplement_temporal_wind as _supplement_temporal_wind,
    build_ocean_mask as _build_ocean_mask,
    apply_ocean_mask_velocity as _apply_ocean_mask_velocity,
    reset_voyage_data_sources,
    get_voyage_data_sources,
    CACHE_TTL_MINUTES,
)
from api.forecast_layer_manager import (  # noqa: E402
    wind_layer, wave_layer, current_layer, ice_layer, sst_layer, vis_layer,
    is_cache_complete, cache_covers_bounds, find_covering_cache,
)

# Backward compat: get_voyage_data_sources() was a module-level list in main.py.
# Endpoints that referenced it now use get_voyage_data_sources().

# ============================================================================
# API Endpoints - Core
# ============================================================================

@app.get("/", tags=["System"])
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


@app.get("/api/logs/stream", tags=["System"])
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


@app.get("/api/health", tags=["System"])
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


@app.get("/api/health/live", tags=["System"])
async def liveness_check():
    """
    Kubernetes liveness probe endpoint.

    Simple check that the service is alive.
    Use this for K8s livenessProbe configuration.
    """
    from api.health import perform_liveness_check
    return await perform_liveness_check()


@app.get("/api/health/ready", tags=["System"])
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


@app.get("/api/status", tags=["System"])
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


@app.get("/api/metrics", tags=["System"], response_class=PlainTextResponse)
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


@app.get("/api/metrics/json", tags=["System"])
async def get_metrics_json():
    """
    Metrics endpoint in JSON format.

    Alternative to Prometheus format for custom dashboards.
    """
    return metrics_collector.get_metrics()


@app.get("/api/data-sources")
async def get_data_sources():
    """
    Get status of available data sources.

    Shows which Copernicus APIs are configured and available.
    """
    # Check if pygrib is available for GFS
    try:
        import pygrib
        has_pygrib = True
    except ImportError:
        has_pygrib = False

    return {
        "gfs": {
            "available": has_pygrib,
            "description": "NOAA GFS 0.25° near-real-time wind (updated every 6h, ~3.5h lag)",
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
        "wind_provider_chain": "GFS → ERA5 → Synthetic",
        "active_wind_source": "gfs" if has_pygrib else (
            "era5" if (copernicus_provider._has_cdsapi and settings.has_cds_credentials)
            else "synthetic"
        ),
    }


# ============================================================================
# API Endpoints - Weather (Layer 1)
# ============================================================================




@app.get("/api/weather/health")
async def api_weather_health():
    """Return per-source health status for all weather sources.

    Purely informational — returns per-layer presence, completeness,
    ingested_at, and age_hours. No longer drives automation.
    """
    if db_weather is None:
        raise HTTPException(status_code=503, detail="Database weather provider not configured")

    health = await asyncio.to_thread(db_weather.get_health)
    return health


# Layer name → ingestion source mapping for per-layer resync
_LAYER_TO_SOURCE = {
    "wind": "gfs",
    "waves": "cmems_wave",
    "currents": "cmems_current",
    "ice": "cmems_ice",
    "sst": "cmems_sst",
    "visibility": "gfs_visibility",
    "swell": "cmems_wave",  # swell reuses wave data
}

_LAYER_INGEST_FN = {
    "wind": lambda wi: wi.ingest_wind(True),
    "waves": lambda wi: wi.ingest_waves(True),
    "currents": lambda wi: wi.ingest_currents(True),
    "ice": lambda wi: wi.ingest_ice(True),
    "sst": lambda wi: wi.ingest_sst(True),
    "visibility": lambda wi: wi.ingest_visibility(True),
    "swell": lambda wi: wi.ingest_waves(True),  # swell reuses wave ingestion
}


@app.post("/api/weather/{layer}/resync", dependencies=[Depends(require_not_demo("Weather resync"))])
async def api_weather_layer_resync(
    layer: str,
    lat_min: Optional[float] = Query(None, ge=-90, le=90),
    lat_max: Optional[float] = Query(None, ge=-90, le=90),
    lon_min: Optional[float] = Query(None, ge=-180, le=180),
    lon_max: Optional[float] = Query(None, ge=-180, le=180),
):
    """Re-ingest a single weather layer and return fresh ingested_at.

    Synchronous — blocks until ingestion completes (30-120s for CMEMS layers).
    Clears the layer's frame cache so the next timeline request rebuilds it.
    When bbox params are provided, CMEMS layers use them instead of defaults.
    """
    if layer not in _LAYER_INGEST_FN:
        raise HTTPException(status_code=400, detail=f"Unknown layer: {layer}. Valid: {list(_LAYER_INGEST_FN.keys())}")
    if weather_ingestion is None:
        raise HTTPException(status_code=503, detail="Weather ingestion not configured")

    # Cap CMEMS bbox to safe max size (40° lat × 60° lon) centered on viewport
    # to avoid OOM — 0.083° resolution × 9 params × 41 timesteps gets huge fast.
    _CMEMS_MAX_LAT_SPAN = 40.0
    _CMEMS_MAX_LON_SPAN = 60.0
    has_bbox = all(v is not None for v in (lat_min, lat_max, lon_min, lon_max))
    if has_bbox:
        lat_center = (lat_min + lat_max) / 2
        lon_center = (lon_min + lon_max) / 2
        lat_half = min((lat_max - lat_min) / 2, _CMEMS_MAX_LAT_SPAN / 2)
        lon_half = min((lon_max - lon_min) / 2, _CMEMS_MAX_LON_SPAN / 2)
        lat_min = max(-90.0, lat_center - lat_half)
        lat_max = min(90.0, lat_center + lat_half)
        lon_min = max(-180.0, lon_center - lon_half)
        lon_max = min(180.0, lon_center + lon_half)

    logger.info(f"Per-layer resync starting: {layer}" + (f" bbox=[{lat_min:.1f},{lat_max:.1f}]x[{lon_min:.1f},{lon_max:.1f}]" if has_bbox else ""))
    try:
        # CMEMS layers accept viewport bbox; GFS layers ignore it (global)
        if has_bbox and layer in ("waves", "currents", "swell", "ice", "sst"):
            ingest_method = {
                "waves": weather_ingestion.ingest_waves,
                "currents": weather_ingestion.ingest_currents,
                "swell": weather_ingestion.ingest_waves,
                "ice": weather_ingestion.ingest_ice,
                "sst": weather_ingestion.ingest_sst,
            }[layer]
            await asyncio.to_thread(
                ingest_method, True,
                lat_min, lat_max, lon_min, lon_max,
            )
        else:
            ingest_fn = _LAYER_INGEST_FN[layer]
            await asyncio.to_thread(ingest_fn, weather_ingestion)

        # Supersede old runs and clean orphans — scoped to this source only
        source = _LAYER_TO_SOURCE[layer]
        weather_ingestion._supersede_old_runs(source)
        weather_ingestion.cleanup_orphaned_grid_data(source)

        # Clear layer-specific frame cache (delete files, keep directory)
        cache_dir = Path("/tmp/windmar_cache")
        cache_names = {
            "wind": "wind", "waves": "wave", "currents": "current",
            "ice": "ice", "sst": "sst", "visibility": "vis", "swell": "wave",
        }
        layer_cache = cache_dir / cache_names.get(layer, layer)
        if layer_cache.exists():
            for f in layer_cache.iterdir():
                f.unlink(missing_ok=True)
        else:
            layer_cache.mkdir(parents=True, exist_ok=True)

        # Clean stale file caches
        _cleanup_stale_caches()

        # Get actual ingested_at from the newly-created DB run
        _, db_ingested_at = db_weather._find_latest_run(source)
        ingested_at = db_ingested_at or datetime.now(timezone.utc)
        logger.info(f"Per-layer resync complete: {layer}, ingested_at={ingested_at.isoformat()}")
        return {"status": "complete", "ingested_at": ingested_at.isoformat()}
    except Exception as e:
        logger.error(f"Per-layer resync failed ({layer}): {e}")
        raise HTTPException(status_code=500, detail=f"Resync failed: {e}")


@app.get("/api/weather/wind")
async def api_get_wind_field(
    lat_min: float = Query(30.0, ge=-90, le=90),
    lat_max: float = Query(60.0, ge=-90, le=90),
    lon_min: float = Query(-15.0, ge=-180, le=180),
    lon_max: float = Query(40.0, ge=-180, le=180),
    resolution: float = Query(1.0, ge=0.25, le=5.0),
    time: Optional[datetime] = None,
):
    """
    Get wind field data for visualization.

    Returns U/V wind components on a grid.
    DB-first: returns cached data if available, otherwise fetches from API.
    """
    if time is None:
        time = datetime.utcnow()

    ingested_at = None
    # Try DB first
    if db_weather is not None:
        wind_data, ingested_at = db_weather.get_wind_from_db(lat_min, lat_max, lon_min, lon_max, time)
        if wind_data is not None:
            logger.debug("Wind endpoint: serving from DB")
        else:
            wind_data = get_wind_field(lat_min, lat_max, lon_min, lon_max, resolution, time)
            ingested_at = datetime.now(timezone.utc)
    else:
        wind_data = get_wind_field(lat_min, lat_max, lon_min, lon_max, resolution, time)

    # High-resolution ocean mask (0.05° ≈ 5.5km) via vectorized numpy
    mask_lats, mask_lons, ocean_mask = _build_ocean_mask(lat_min, lat_max, lon_min, lon_max, step=0.05)

    # SPEC-P1: Piggyback SST on wind endpoint (same bounding box)
    sst_data = get_sst_field(lat_min, lat_max, lon_min, lon_max, resolution, time)

    response = {
        "parameter": "wind",
        "time": time.isoformat(),
        "bbox": {
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
        },
        "resolution": resolution,
        "nx": len(wind_data.lons),
        "ny": len(wind_data.lats),
        "lats": wind_data.lats.tolist(),
        "lons": wind_data.lons.tolist(),
        "u": wind_data.u_component.tolist() if wind_data.u_component is not None else [],
        "v": wind_data.v_component.tolist() if wind_data.v_component is not None else [],
        "ocean_mask": ocean_mask,
        "ocean_mask_lats": mask_lats,
        "ocean_mask_lons": mask_lons,
        "source": (
            "gfs" if (wind_data.time and (datetime.utcnow() - wind_data.time).total_seconds() < 43200)
            else "copernicus" if copernicus_provider._has_cdsapi
            else "synthetic"
        ),
    }
    if sst_data is not None and sst_data.values is not None:
        response["sst"] = {
            "lats": sst_data.lats.tolist(),
            "lons": sst_data.lons.tolist(),
            "data": sst_data.values.tolist(),
            "unit": "°C",
        }
    if ingested_at is not None:
        response["ingested_at"] = ingested_at.isoformat()
    return response


@app.get("/api/weather/wind/velocity")
async def api_get_wind_velocity_format(
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
    resolution: float = Query(1.0),
    time: Optional[datetime] = None,
    forecast_hour: int = Query(0, ge=0, le=120),
):
    """
    Get wind data in leaflet-velocity compatible format.

    Returns array of [U-component, V-component] data with headers.
    DB-first: returns cached data if available, otherwise fetches from API.
    """
    if time is None:
        time = datetime.utcnow()

    if forecast_hour > 0:
        # Direct GFS fetch for specific forecast hour
        wind_data = gfs_provider.fetch_wind_data(lat_min, lat_max, lon_min, lon_max, time, forecast_hour)
        if wind_data is None:
            raise HTTPException(status_code=404, detail=f"Forecast hour f{forecast_hour:03d} not available")
    else:
        wind_data = get_wind_field(lat_min, lat_max, lon_min, lon_max, resolution, time)

    # Apply ocean mask — zero out land so particles don't render there
    u_masked, v_masked = _apply_ocean_mask_velocity(
        wind_data.u_component, wind_data.v_component,
        wind_data.lats, wind_data.lons,
    )

    # leaflet-velocity format — derive header from actual data arrays
    # (data may come at native resolution, e.g. 0.25° from ERA5)
    actual_lats = wind_data.lats
    actual_lons = wind_data.lons
    actual_dx = abs(float(actual_lons[1] - actual_lons[0])) if len(actual_lons) > 1 else resolution
    actual_dy = abs(float(actual_lats[1] - actual_lats[0])) if len(actual_lats) > 1 else resolution

    # leaflet-velocity expects data ordered N→S (descending lat), W→E (ascending lon)
    if len(actual_lats) > 1 and actual_lats[1] > actual_lats[0]:
        # Ascending (S→N from np.arange) — flip to N→S
        u_ordered = u_masked[::-1]
        v_ordered = v_masked[::-1]
        lat_north = float(actual_lats[-1])
        lat_south = float(actual_lats[0])
    else:
        # Already descending (N→S, typical for ERA5 netCDF)
        u_ordered = u_masked
        v_ordered = v_masked
        lat_north = float(actual_lats[0])
        lat_south = float(actual_lats[-1])

    header = {
        "parameterCategory": 2,
        "parameterNumber": 2,
        "lo1": float(actual_lons[0]),
        "la1": lat_north,
        "lo2": float(actual_lons[-1]),
        "la2": lat_south,
        "dx": actual_dx,
        "dy": actual_dy,
        "nx": len(actual_lons),
        "ny": len(actual_lats),
        "refTime": (wind_data.time.isoformat() if isinstance(wind_data.time, datetime) else time.isoformat()),
    }

    u_flat = u_ordered.flatten().tolist()
    v_flat = v_ordered.flatten().tolist()

    return [
        {"header": {**header, "parameterNumber": 2}, "data": u_flat},
        {"header": {**header, "parameterNumber": 3}, "data": v_flat},
    ]


# ============================================================================
# API Endpoints - GFS Forecast Timeline
# ============================================================================

# (Per-layer state, locks, cache I/O, and shared utilities are now in
#  api.forecast_layer_manager — imported as wind_layer, wave_layer, etc.)


def _build_wind_frames(lat_min, lat_max, lon_min, lon_max, run_date, run_hour):
    """Process all cached GRIB files into leaflet-velocity frames dict.

    Called once after prefetch completes. Result is saved to file cache.
    """
    run_time = datetime.strptime(f"{run_date}{run_hour}", "%Y%m%d%H")
    hours_status = gfs_provider.get_cached_forecast_hours(lat_min, lat_max, lon_min, lon_max, run_date, run_hour)

    frames = {}
    for h_info in hours_status:
        if not h_info["cached"]:
            continue
        fh = h_info["forecast_hour"]
        wind_data = gfs_provider.fetch_wind_data(lat_min, lat_max, lon_min, lon_max, forecast_hour=fh, run_date=run_date, run_hour=run_hour)
        if wind_data is None:
            continue

        u_masked, v_masked = _apply_ocean_mask_velocity(
            wind_data.u_component, wind_data.v_component,
            wind_data.lats, wind_data.lons,
        )
        actual_lats = wind_data.lats
        actual_lons = wind_data.lons
        actual_dx = abs(float(actual_lons[1] - actual_lons[0])) if len(actual_lons) > 1 else 0.25
        actual_dy = abs(float(actual_lats[1] - actual_lats[0])) if len(actual_lats) > 1 else 0.25

        if len(actual_lats) > 1 and actual_lats[1] > actual_lats[0]:
            u_ordered = u_masked[::-1]
            v_ordered = v_masked[::-1]
            lat_north = float(actual_lats[-1])
            lat_south = float(actual_lats[0])
        else:
            u_ordered = u_masked
            v_ordered = v_masked
            lat_north = float(actual_lats[0])
            lat_south = float(actual_lats[-1])

        valid_time = run_time + timedelta(hours=fh)
        header = {
            "parameterCategory": 2,
            "parameterNumber": 2,
            "lo1": float(actual_lons[0]),
            "la1": lat_north,
            "lo2": float(actual_lons[-1]),
            "la2": lat_south,
            "dx": actual_dx,
            "dy": actual_dy,
            "nx": len(actual_lons),
            "ny": len(actual_lats),
            "refTime": run_time.isoformat() if run_time else valid_time.isoformat(),
            "forecastHour": fh,
        }
        frames[str(fh)] = [
            {"header": {**header, "parameterNumber": 2}, "data": u_ordered.flatten().tolist()},
            {"header": {**header, "parameterNumber": 3}, "data": v_ordered.flatten().tolist()},
        ]
        logger.info(f"Wind frame f{fh:03d} processed ({len(actual_lats)}x{len(actual_lons)})")

    result = {
        "run_date": run_date,
        "run_hour": run_hour,
        "run_time": run_time.isoformat(),
        "total_hours": len(GFSDataProvider.FORECAST_HOURS),
        "cached_hours": len(frames),
        "source": "gfs",
        "frames": frames,
    }

    cache_key = wind_layer.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    wind_layer.cache_put(cache_key, result)
    logger.info(f"Wind frames cache saved: {len(frames)} frames, key={cache_key}")
    return result


def _rebuild_wind_cache_from_db(cache_key, lat_min, lat_max, lon_min, lon_max):
    """Rebuild wind forecast file cache from PostgreSQL data.

    Fallback when GRIB file cache is missing or partial (e.g. latest GFS
    run on NOMADS is still publishing). Mirrors the wave/current/ice
    pattern: read complete DB run → build leaflet-velocity frames → save.
    """
    if db_weather is None:
        return None

    run_time, hours = db_weather.get_available_hours_by_source("gfs")
    if not hours:
        return None

    logger.info(f"Rebuilding wind cache from DB: {len(hours)} hours")

    grids = db_weather.get_grids_for_timeline(
        "gfs", ["wind_u", "wind_v"], lat_min, lat_max, lon_min, lon_max, hours
    )

    if not grids or "wind_u" not in grids or not grids["wind_u"]:
        return None

    frames = {}
    for fh in sorted(hours):
        if fh not in grids.get("wind_u", {}) or fh not in grids.get("wind_v", {}):
            continue

        lats, lons, u_data = grids["wind_u"][fh]
        _, _, v_data = grids["wind_v"][fh]

        u_masked, v_masked = _apply_ocean_mask_velocity(u_data, v_data, lats, lons)

        actual_dx = abs(float(lons[1] - lons[0])) if len(lons) > 1 else 0.25
        actual_dy = abs(float(lats[1] - lats[0])) if len(lats) > 1 else 0.25

        if len(lats) > 1 and lats[1] > lats[0]:
            u_ordered = u_masked[::-1]
            v_ordered = v_masked[::-1]
            lat_north = float(lats[-1])
            lat_south = float(lats[0])
        else:
            u_ordered = u_masked
            v_ordered = v_masked
            lat_north = float(lats[0])
            lat_south = float(lats[-1])

        valid_time = run_time + timedelta(hours=fh) if run_time else datetime.utcnow()
        header = {
            "parameterCategory": 2,
            "parameterNumber": 2,
            "lo1": float(lons[0]),
            "la1": lat_north,
            "lo2": float(lons[-1]),
            "la2": lat_south,
            "dx": actual_dx,
            "dy": actual_dy,
            "nx": len(lons),
            "ny": len(lats),
            "refTime": run_time.isoformat() if run_time else valid_time.isoformat(),
            "forecastHour": fh,
        }
        frames[str(fh)] = [
            {"header": {**header, "parameterNumber": 2}, "data": u_ordered.flatten().tolist()},
            {"header": {**header, "parameterNumber": 3}, "data": v_ordered.flatten().tolist()},
        ]

    run_date_str = run_time.strftime("%Y%m%d") if run_time else ""
    run_hour_str = run_time.strftime("%H") if run_time else "00"

    result = {
        "run_date": run_date_str,
        "run_hour": run_hour_str,
        "run_time": run_time.isoformat() if run_time else "",
        "total_hours": len(GFSDataProvider.FORECAST_HOURS),
        "cached_hours": len(frames),
        "source": "gfs",
        "frames": frames,
    }

    wind_layer.cache_put(cache_key, result)
    logger.info(f"Wind cache rebuilt from DB: {len(frames)} frames")
    return result


@app.get("/api/weather/forecast/status")
async def api_get_forecast_status(
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    """
    Get GFS forecast prefetch status.

    Checks the file cache first (instant). Falls back to scanning GRIB files.
    """
    cache_key = wind_layer.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    cached = wind_layer.cache_get(cache_key)
    if cached and not wind_layer.is_running:
        cached_hours = len(cached.get("frames", {}))
        return {
            "run_date": cached["run_date"],
            "run_hour": cached["run_hour"],
            "total_hours": cached.get("total_hours", 41),
            "cached_hours": cached_hours,
            "complete": True,
            "prefetch_running": False,
        }

    # No file cache — fall back to scanning GRIB files
    if wind_layer.last_run:
        run_date, run_hour = wind_layer.last_run
    else:
        run_date, run_hour = gfs_provider._get_latest_run()
    hours = gfs_provider.get_cached_forecast_hours(lat_min, lat_max, lon_min, lon_max, run_date, run_hour)
    cached_count = sum(1 for h in hours if h["cached"])

    if cached_count == 0 and not wind_layer.is_running:
        best = gfs_provider.find_best_cached_run(lat_min, lat_max, lon_min, lon_max)
        if best:
            run_date, run_hour = best
            hours = gfs_provider.get_cached_forecast_hours(lat_min, lat_max, lon_min, lon_max, run_date, run_hour)
    cached_count = sum(1 for h in hours if h["cached"])
    total_count = len(hours)

    # If GRIB cache is empty but DB has data, report DB availability
    # so the frontend knows frames will be served via DB fallback
    if cached_count == 0 and db_weather is not None:
        db_run_time, db_hours = db_weather.get_available_hours_by_source("gfs")
        if db_hours:
            db_run_date = db_run_time.strftime("%Y%m%d") if db_run_time else run_date
            db_run_hour = db_run_time.strftime("%H") if db_run_time else run_hour
            return {
                "run_date": db_run_date,
                "run_hour": db_run_hour,
                "total_hours": len(GFSDataProvider.FORECAST_HOURS),
                "cached_hours": len(db_hours),
                "complete": True,
                "prefetch_running": False,
            }

    return {
        "run_date": run_date,
        "run_hour": run_hour,
        "total_hours": total_count,
        "cached_hours": cached_count,
        "complete": cached_count == total_count and not wind_layer.is_running,
        "prefetch_running": wind_layer.is_running,
    }


def _do_wind_prefetch(mgr, lat_min, lat_max, lon_min, lon_max):
    """Download all GFS forecast hours and build wind frames cache."""
    run_date, run_hour = gfs_provider._get_latest_run()
    mgr.last_run = (run_date, run_hour)
    logger.info(f"GFS forecast prefetch started (run {run_date}/{run_hour}z)")
    gfs_provider.prefetch_forecast_hours(lat_min, lat_max, lon_min, lon_max)
    logger.info("GFS forecast prefetch completed, building frames cache...")
    _build_wind_frames(lat_min, lat_max, lon_min, lon_max, run_date, run_hour)
    logger.info("Wind frames cache ready")


@app.post("/api/weather/forecast/prefetch", dependencies=[Depends(require_not_demo("Weather prefetch"))])
async def api_trigger_forecast_prefetch(
    background_tasks: BackgroundTasks,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    """Trigger background download of all GFS forecast hours (f000-f120)."""
    return wind_layer.trigger_response(background_tasks, _do_wind_prefetch, lat_min, lat_max, lon_min, lon_max)


@app.get("/api/weather/forecast/frames")
async def api_get_forecast_frames(
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    """
    Return all wind forecast frames from file cache (instant).

    The cache is built once during prefetch. No GRIB parsing happens here.
    Serves the raw JSON file to avoid parse+re-serialize overhead.
    """
    from starlette.responses import Response
    cache_key = wind_layer.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    cache_file = wind_layer.cache_path(cache_key)
    if cache_file.exists():
        return Response(content=cache_file.read_bytes(), media_type="application/json")

    # No file cache — fallback: rebuild from PostgreSQL
    cached = await asyncio.to_thread(
        _rebuild_wind_cache_from_db, cache_key, lat_min, lat_max, lon_min, lon_max
    )

    if cached:
        return cached

    # No DB data either — return empty
    run_date, run_hour = gfs_provider._get_latest_run()
    run_time = datetime.strptime(f"{run_date}{run_hour}", "%Y%m%d%H")
    return {
        "run_date": run_date,
        "run_hour": run_hour,
        "run_time": run_time.isoformat(),
        "total_hours": len(GFSDataProvider.FORECAST_HOURS),
        "cached_hours": 0,
        "source": "gfs",
        "frames": {},
    }


# =========================================================================
# Wave Forecast Endpoints (CMEMS)
# =========================================================================

def _rebuild_wave_cache_from_db(cache_key, lat_min, lat_max, lon_min, lon_max):
    """Rebuild wave forecast file cache from PostgreSQL data."""
    if db_weather is None:
        return None

    run_time, hours = db_weather.get_available_hours_by_source("cmems_wave")
    if not hours:
        return None

    logger.info(f"Rebuilding wave cache from DB: {len(hours)} hours")

    params = ["wave_hs", "wave_dir", "swell_hs", "swell_tp", "swell_dir",
              "windwave_hs", "windwave_tp", "windwave_dir"]
    grids = db_weather.get_grids_for_timeline(
        "cmems_wave", params, lat_min, lat_max, lon_min, lon_max, hours
    )

    if not grids or "wave_hs" not in grids or not grids["wave_hs"]:
        return None

    first_fh = min(grids["wave_hs"].keys())
    lats_full, lons_full, _ = grids["wave_hs"][first_fh]
    max_dim = max(len(lats_full), len(lons_full))
    STEP = max(1, round(max_dim / 250))
    shared_lats = lats_full[::STEP].tolist()
    shared_lons = lons_full[::STEP].tolist()

    def _sub(arr):
        if arr is None:
            return None
        return np.round(arr[::STEP, ::STEP], 2).tolist()

    mask_lats_arr, mask_lons_arr, ocean_mask_arr = _build_ocean_mask(
        lat_min, lat_max, lon_min, lon_max
    )

    frames = {}
    for fh in sorted(hours):
        frame = {}
        if "wave_hs" in grids and fh in grids["wave_hs"]:
            _, _, d = grids["wave_hs"][fh]
            frame["data"] = _sub(d)
        if "wave_dir" in grids and fh in grids["wave_dir"]:
            _, _, d = grids["wave_dir"][fh]
            frame["direction"] = _sub(d)

        has_decomp = (fh in grids.get("windwave_hs", {}) and
                      fh in grids.get("swell_hs", {}))
        if has_decomp:
            frame["windwave"] = {}
            for p, k in [("windwave_hs", "height"), ("windwave_tp", "period"), ("windwave_dir", "direction")]:
                if fh in grids.get(p, {}):
                    _, _, d = grids[p][fh]
                    frame["windwave"][k] = _sub(d)
            frame["swell"] = {}
            for p, k in [("swell_hs", "height"), ("swell_tp", "period"), ("swell_dir", "direction")]:
                if fh in grids.get(p, {}):
                    _, _, d = grids[p][fh]
                    frame["swell"][k] = _sub(d)

        if frame:
            frames[str(fh)] = frame

    cache_data = {
        "run_time": run_time.isoformat() if run_time else "",
        "total_hours": 41,
        "cached_hours": len(frames),
        "source": "cmems",
        "lats": shared_lats,
        "lons": shared_lons,
        "ny": len(shared_lats),
        "nx": len(shared_lons),
        "ocean_mask": ocean_mask_arr,
        "ocean_mask_lats": mask_lats_arr,
        "ocean_mask_lons": mask_lons_arr,
        "colorscale": {"min": 0, "max": 6, "colors": ["#00ff00", "#ffff00", "#ff8800", "#ff0000", "#800000"]},
        "frames": frames,
    }

    wave_layer.cache_put(cache_key, cache_data)
    logger.info(f"Wave cache rebuilt from DB: {len(frames)} frames")
    return cache_data


def _rebuild_current_cache_from_db(cache_key, lat_min, lat_max, lon_min, lon_max):
    """Rebuild current forecast file cache from PostgreSQL data."""
    if db_weather is None:
        return None

    run_time, hours = db_weather.get_available_hours_by_source("cmems_current")
    if not hours:
        return None

    logger.info(f"Rebuilding current cache from DB: {len(hours)} hours")

    grids = db_weather.get_grids_for_timeline(
        "cmems_current", ["current_u", "current_v"],
        lat_min, lat_max, lon_min, lon_max, hours
    )

    if not grids or "current_u" not in grids or not grids["current_u"]:
        return None

    first_fh = min(grids["current_u"].keys())
    lats_full, lons_full, _ = grids["current_u"][first_fh]
    max_dim = max(len(lats_full), len(lons_full))
    STEP = max(1, round(max_dim / 250))
    sub_lats = lats_full[::STEP]  # numpy, S→N order
    sub_lons = lons_full[::STEP]

    frames = {}
    for fh in sorted(hours):
        u_sub = v_sub = None
        if fh in grids.get("current_u", {}):
            _, _, d = grids["current_u"][fh]
            u_sub = d[::STEP, ::STEP]
        if fh in grids.get("current_v", {}):
            _, _, d = grids["current_v"][fh]
            v_sub = d[::STEP, ::STEP]
        if u_sub is not None and v_sub is not None:
            # Ocean mask: zero out land points
            u_m, v_m = _apply_ocean_mask_velocity(u_sub, v_sub, sub_lats, sub_lons)
            # Flip N→S for leaflet-velocity (lats stay S→N for header)
            frames[str(fh)] = {
                "u": np.round(u_m[::-1], 2).tolist(),
                "v": np.round(v_m[::-1], 2).tolist(),
            }

    cache_data = {
        "run_time": run_time.isoformat() if run_time else "",
        "total_hours": 41,
        "cached_hours": len(frames),
        "source": "cmems",
        "lats": sub_lats.tolist(),
        "lons": sub_lons.tolist(),
        "ny": len(sub_lats),
        "nx": len(sub_lons),
        "frames": frames,
    }

    current_layer.cache_put(cache_key, cache_data)
    logger.info(f"Current cache rebuilt from DB: {len(frames)} frames")
    return cache_data


@app.get("/api/weather/forecast/wave/status")
async def api_get_wave_forecast_status(
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    """Get wave forecast prefetch status."""
    cache_key = wave_layer.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    cached = wave_layer.cache_get(cache_key)
    total_hours = 41  # 0-120h every 3h

    prefetch_running = wave_layer.is_running

    if cached:
        cached_hours = len(cached.get("frames", {}))
        return {
            "run_date": cached.get("run_time", "")[:8] if cached.get("run_time") else "",
            "run_hour": cached.get("run_time", "")[9:11] if cached.get("run_time") else "00",
            "total_hours": total_hours,
            "cached_hours": cached_hours,
            "complete": cached_hours >= total_hours,
            "prefetch_running": prefetch_running,
            "hours": [],
        }

    # File cache miss — check DB for data from the auto-prefetch
    if db_weather is not None:
        try:
            run_time, hours = db_weather.get_available_hours_by_source("cmems_wave")
            if hours:
                return {
                    "run_date": run_time.strftime("%Y%m%d") if run_time else "",
                    "run_hour": run_time.strftime("%H") if run_time else "00",
                    "total_hours": total_hours,
                    "cached_hours": len(hours),
                    "complete": len(hours) >= total_hours and not prefetch_running,
                    "prefetch_running": prefetch_running,
                    "hours": [],
                }
        except Exception:
            pass

    return {
        "run_date": "",
        "run_hour": "00",
        "total_hours": total_hours,
        "cached_hours": 0,
        "complete": False,
        "prefetch_running": prefetch_running,
        "hours": [],
    }


def _do_wave_prefetch(mgr, lat_min, lat_max, lon_min, lon_max):
    """Download CMEMS wave forecast and build frames cache."""
    cache_key_chk = mgr.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    existing = mgr.cache_get(cache_key_chk)
    if existing and len(existing.get("frames", {})) >= 41 and cache_covers_bounds(existing, lat_min, lat_max, lon_min, lon_max):
        logger.info("Wave forecast file cache already complete, skipping CMEMS download")
        return

    if db_weather is not None:
        rebuilt = _rebuild_wave_cache_from_db(cache_key_chk, lat_min, lat_max, lon_min, lon_max)
        if rebuilt and len(rebuilt.get("frames", {})) >= 41 and cache_covers_bounds(rebuilt, lat_min, lat_max, lon_min, lon_max):
            logger.info("Wave forecast rebuilt from DB, skipping CMEMS download")
            return

    stale_path = mgr.cache_path(cache_key_chk)
    if stale_path.exists():
        stale_path.unlink(missing_ok=True)
        logger.info(f"Removed stale wave cache: {cache_key_chk}")

    logger.info("CMEMS wave forecast prefetch started")

    result = copernicus_provider.fetch_wave_forecast(lat_min, lat_max, lon_min, lon_max)
    if result is None:
        logger.error("Wave forecast fetch returned None")
        return

    mask_lats_arr, mask_lons_arr, ocean_mask_arr = _build_ocean_mask(
        lat_min, lat_max, lon_min, lon_max
    )

    first_wd = next(iter(result.values()))
    max_dim = max(len(first_wd.lats), len(first_wd.lons))
    STEP = max(1, round(max_dim / 250))
    logger.info(f"Wave forecast: grid {len(first_wd.lats)}x{len(first_wd.lons)}, STEP={STEP}")
    shared_lats = first_wd.lats[::STEP].tolist()
    shared_lons = first_wd.lons[::STEP].tolist()

    def _subsample_round(arr):
        if arr is None:
            return None
        sub = arr[::STEP, ::STEP]
        return np.round(sub, 2).tolist()

    frames = {}
    for fh, wd in sorted(result.items()):
        frame = {"data": _subsample_round(wd.values)}
        if wd.wave_direction is not None:
            frame["direction"] = _subsample_round(wd.wave_direction)
        has_decomp = wd.windwave_height is not None and wd.swell_height is not None
        if has_decomp:
            frame["windwave"] = {
                "height": _subsample_round(wd.windwave_height),
                "period": _subsample_round(wd.windwave_period),
                "direction": _subsample_round(wd.windwave_direction),
            }
            frame["swell"] = {
                "height": _subsample_round(wd.swell_height),
                "period": _subsample_round(wd.swell_period),
                "direction": _subsample_round(wd.swell_direction),
            }
        frames[str(fh)] = frame

    cache_key = mgr.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    mgr.cache_put(cache_key, {
        "run_time": first_wd.time.isoformat() if first_wd.time else "",
        "total_hours": 41,
        "cached_hours": len(frames),
        "source": "cmems",
        "lats": shared_lats,
        "lons": shared_lons,
        "ny": len(shared_lats),
        "nx": len(shared_lons),
        "ocean_mask": ocean_mask_arr,
        "ocean_mask_lats": mask_lats_arr,
        "ocean_mask_lons": mask_lons_arr,
        "colorscale": {"min": 0, "max": 6, "colors": ["#00ff00", "#ffff00", "#ff8800", "#ff0000", "#800000"]},
        "frames": frames,
    })
    logger.info(f"Wave forecast cached: {len(frames)} frames")

    if weather_ingestion is not None:
        try:
            logger.info("Ingesting wave forecast frames into PostgreSQL...")
            weather_ingestion.ingest_wave_forecast_frames(result)
        except Exception as db_e:
            logger.error(f"Wave forecast DB ingestion failed: {db_e}")


@app.post("/api/weather/forecast/wave/prefetch", dependencies=[Depends(require_not_demo("Weather prefetch"))])
async def api_trigger_wave_forecast_prefetch(
    background_tasks: BackgroundTasks,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    """Trigger background download of CMEMS wave forecast (0-120h)."""
    return wave_layer.trigger_response(background_tasks, _do_wave_prefetch, lat_min, lat_max, lon_min, lon_max)


@app.get("/api/weather/forecast/wave/frames")
async def api_get_wave_forecast_frames(
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    """Return all cached CMEMS wave forecast frames.

    Serves the raw JSON file to avoid parse+re-serialize overhead.
    """
    cache_key = wave_layer.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    raw = wave_layer.serve_frames_file(cache_key)
    if raw is not None:
        return raw

    # Fallback: rebuild from PostgreSQL
    cached = await asyncio.to_thread(
        _rebuild_wave_cache_from_db, cache_key, lat_min, lat_max, lon_min, lon_max
    )

    if not cached:
        return {
            "run_time": "",
            "total_hours": 41,
            "cached_hours": 0,
            "source": "none",
            "lats": [],
            "lons": [],
            "ny": 0,
            "nx": 0,
            "colorscale": {"min": 0, "max": 6, "colors": []},
            "frames": {},
        }

    return cached


# =========================================================================
# Current Forecast Endpoints (CMEMS)
# =========================================================================

@app.get("/api/weather/forecast/current/status")
async def api_get_current_forecast_status(
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    """Get current forecast prefetch status."""
    cache_key = current_layer.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    cached = current_layer.cache_get(cache_key)
    total_hours = 41

    prefetch_running = current_layer.is_running

    if cached:
        cached_hours = len(cached.get("frames", {}))
        return {
            "total_hours": total_hours,
            "cached_hours": cached_hours,
            "complete": cached_hours >= total_hours,
            "prefetch_running": prefetch_running,
        }

    # File cache miss — check DB for data from the auto-prefetch
    if db_weather is not None:
        try:
            run_time, hours = db_weather.get_available_hours_by_source("cmems_current")
            if hours:
                return {
                    "total_hours": total_hours,
                    "cached_hours": len(hours),
                    "complete": len(hours) >= total_hours and not prefetch_running,
                    "prefetch_running": prefetch_running,
                }
        except Exception:
            pass

    return {
        "total_hours": total_hours,
        "cached_hours": 0,
        "complete": False,
        "prefetch_running": prefetch_running,
    }


def _do_current_prefetch(mgr, lat_min, lat_max, lon_min, lon_max):
    """Download CMEMS current forecast and build frames cache."""
    cache_key_chk = mgr.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    existing = mgr.cache_get(cache_key_chk)
    if existing and len(existing.get("frames", {})) >= 41 and cache_covers_bounds(existing, lat_min, lat_max, lon_min, lon_max):
        logger.info("Current forecast file cache already complete, skipping CMEMS download")
        return

    if db_weather is not None:
        rebuilt = _rebuild_current_cache_from_db(cache_key_chk, lat_min, lat_max, lon_min, lon_max)
        if rebuilt and len(rebuilt.get("frames", {})) >= 41 and cache_covers_bounds(rebuilt, lat_min, lat_max, lon_min, lon_max):
            logger.info("Current forecast rebuilt from DB, skipping CMEMS download")
            return

    stale_path = mgr.cache_path(cache_key_chk)
    if stale_path.exists():
        stale_path.unlink(missing_ok=True)
        logger.info(f"Removed stale current cache: {cache_key_chk}")

    logger.info("CMEMS current forecast prefetch started")

    result = copernicus_provider.fetch_current_forecast(lat_min, lat_max, lon_min, lon_max)
    if result is None:
        logger.error("Current forecast fetch returned None")
        return

    first_wd = next(iter(result.values()))
    max_dim = max(len(first_wd.lats), len(first_wd.lons))
    STEP = max(1, round(max_dim / 250))
    logger.info(f"Current forecast: grid {len(first_wd.lats)}x{len(first_wd.lons)}, STEP={STEP}")
    sub_lats = first_wd.lats[::STEP]
    sub_lons = first_wd.lons[::STEP]

    frames = {}
    for fh, wd in sorted(result.items()):
        if wd.u_component is not None and wd.v_component is not None:
            u_sub = wd.u_component[::STEP, ::STEP]
            v_sub = wd.v_component[::STEP, ::STEP]
            u_m, v_m = _apply_ocean_mask_velocity(u_sub, v_sub, sub_lats, sub_lons)
            frames[str(fh)] = {
                "u": np.round(u_m[::-1], 2).tolist(),
                "v": np.round(v_m[::-1], 2).tolist(),
            }

    cache_key = mgr.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    mgr.cache_put(cache_key, {
        "run_time": first_wd.time.isoformat() if first_wd.time else "",
        "total_hours": 41,
        "cached_hours": len(frames),
        "source": "cmems",
        "lats": sub_lats.tolist(),
        "lons": sub_lons.tolist(),
        "ny": len(sub_lats),
        "nx": len(sub_lons),
        "frames": frames,
    })
    logger.info(f"Current forecast cached: {len(frames)} frames")

    if weather_ingestion is not None:
        try:
            logger.info("Ingesting current forecast frames into PostgreSQL...")
            weather_ingestion.ingest_current_forecast_frames(result)
        except Exception as db_e:
            logger.error(f"Current forecast DB ingestion failed: {db_e}")


@app.post("/api/weather/forecast/current/prefetch", dependencies=[Depends(require_not_demo("Weather prefetch"))])
async def api_trigger_current_forecast_prefetch(
    background_tasks: BackgroundTasks,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    """Trigger background download of CMEMS current forecast (0-120h)."""
    return current_layer.trigger_response(background_tasks, _do_current_prefetch, lat_min, lat_max, lon_min, lon_max)


@app.get("/api/weather/forecast/current/frames")
async def api_get_current_forecast_frames(
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    """Return all cached CMEMS current forecast frames.

    Serves the raw JSON file to avoid parse+re-serialize overhead.
    """
    cache_key = current_layer.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    raw = current_layer.serve_frames_file(cache_key)
    if raw is not None:
        return raw

    # Fallback: rebuild from PostgreSQL
    cached = await asyncio.to_thread(
        _rebuild_current_cache_from_db, cache_key, lat_min, lat_max, lon_min, lon_max
    )

    if not cached:
        return {
            "run_time": "",
            "total_hours": 41,
            "cached_hours": 0,
            "source": "none",
            "lats": [],
            "lons": [],
            "ny": 0,
            "nx": 0,
            "frames": {},
        }

    return cached


# =========================================================================
# Ice Forecast Endpoints (CMEMS, 10-day daily)
# =========================================================================

def _rebuild_ice_cache_from_db(cache_key, lat_min, lat_max, lon_min, lon_max):
    """Rebuild ice forecast file cache from PostgreSQL data."""
    if db_weather is None:
        return None

    run_time, hours = db_weather.get_available_hours_by_source("cmems_ice")
    if not hours:
        return None

    logger.info(f"Rebuilding ice cache from DB: {len(hours)} hours")

    grids = db_weather.get_grids_for_timeline(
        "cmems_ice", ["ice_siconc"],
        lat_min, lat_max, lon_min, lon_max, hours
    )

    if not grids or "ice_siconc" not in grids or not grids["ice_siconc"]:
        return None

    first_fh = min(grids["ice_siconc"].keys())
    lats_full, lons_full, _ = grids["ice_siconc"][first_fh]
    max_dim = max(len(lats_full), len(lons_full))
    STEP = max(1, round(max_dim / 250))
    shared_lats = lats_full[::STEP].tolist()
    shared_lons = lons_full[::STEP].tolist()

    mask_lats_arr, mask_lons_arr, ocean_mask_arr = _build_ocean_mask(
        lat_min, lat_max, lon_min, lon_max
    )

    frames = {}
    for fh in sorted(hours):
        if fh in grids["ice_siconc"]:
            _, _, d = grids["ice_siconc"][fh]
            frames[str(fh)] = {
                "data": np.round(d[::STEP, ::STEP], 4).tolist(),
            }

    cache_data = {
        "run_time": run_time.isoformat() if run_time else "",
        "total_hours": len(frames),
        "cached_hours": len(frames),
        "source": "cmems",
        "lats": shared_lats,
        "lons": shared_lons,
        "ny": len(shared_lats),
        "nx": len(shared_lons),
        "ocean_mask": ocean_mask_arr,
        "ocean_mask_lats": mask_lats_arr,
        "ocean_mask_lons": mask_lons_arr,
        "frames": frames,
    }

    ice_layer.cache_put(cache_key, cache_data)
    logger.info(f"Ice cache rebuilt from DB: {len(frames)} frames")
    return cache_data


@app.get("/api/weather/forecast/ice/status")
async def api_get_ice_forecast_status(
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    """Get ice forecast prefetch status."""
    cache_key = ice_layer.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    cached = ice_layer.cache_get(cache_key)

    prefetch_running = ice_layer.is_running

    if cached:
        cached_hours = len(cached.get("frames", {}))
        # total_hours matches cached_hours once prefetch finishes (CMEMS may have < 10 days)
        total_hours = cached_hours if not prefetch_running else max(cached_hours + 1, 10)
        return {
            "total_hours": total_hours,
            "cached_hours": cached_hours,
            "complete": cached_hours > 0 and not prefetch_running,
            "prefetch_running": prefetch_running,
        }

    # File cache miss — check DB for data from the auto-prefetch
    if db_weather is not None:
        try:
            run_time, hours = db_weather.get_available_hours_by_source("cmems_ice")
            if hours:
                return {
                    "total_hours": len(hours),
                    "cached_hours": len(hours),
                    "complete": len(hours) > 0 and not prefetch_running,
                    "prefetch_running": prefetch_running,
                }
        except Exception:
            pass

    return {
        "total_hours": 10,
        "cached_hours": 0,
        "complete": False,
        "prefetch_running": prefetch_running,
    }


def _do_ice_prefetch(mgr, lat_min, lat_max, lon_min, lon_max):
    """Download CMEMS ice forecast and build frames cache."""
    cache_key_chk = mgr.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    existing = mgr.cache_get(cache_key_chk)
    if existing and len(existing.get("frames", {})) >= 10 and cache_covers_bounds(existing, lat_min, lat_max, lon_min, lon_max):
        logger.info("Ice forecast file cache already complete, skipping CMEMS download")
        return

    if db_weather is not None:
        rebuilt = _rebuild_ice_cache_from_db(cache_key_chk, lat_min, lat_max, lon_min, lon_max)
        if rebuilt and len(rebuilt.get("frames", {})) >= 10 and cache_covers_bounds(rebuilt, lat_min, lat_max, lon_min, lon_max):
            logger.info("Ice forecast rebuilt from DB, skipping CMEMS download")
            return

    stale_path = mgr.cache_path(cache_key_chk)
    if stale_path.exists():
        stale_path.unlink(missing_ok=True)
        logger.info(f"Removed stale ice cache: {cache_key_chk}")

    logger.info("CMEMS ice forecast prefetch started")

    result = copernicus_provider.fetch_ice_forecast(lat_min, lat_max, lon_min, lon_max)
    if result is None:
        logger.info("CMEMS ice forecast unavailable, generating synthetic")
        result = synthetic_provider.generate_ice_forecast(lat_min, lat_max, lon_min, lon_max)

    if not result:
        logger.error("Ice forecast fetch returned empty")
        return

    first_wd = next(iter(result.values()))
    max_dim = max(len(first_wd.lats), len(first_wd.lons))
    STEP = max(1, round(max_dim / 250))
    logger.info(f"Ice forecast: grid {len(first_wd.lats)}x{len(first_wd.lons)}, STEP={STEP}")
    sub_lats = first_wd.lats[::STEP]
    sub_lons = first_wd.lons[::STEP]

    mask_lats_arr, mask_lons_arr, ocean_mask_arr = _build_ocean_mask(
        lat_min, lat_max, lon_min, lon_max
    )

    frames = {}
    for fh, wd in sorted(result.items()):
        siconc = wd.ice_concentration if wd.ice_concentration is not None else wd.values
        if siconc is not None:
            frames[str(fh)] = {
                "data": np.round(siconc[::STEP, ::STEP], 4).tolist(),
            }

    cache_key = mgr.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    mgr.cache_put(cache_key, {
        "run_time": first_wd.time.isoformat() if first_wd.time else "",
        "total_hours": len(frames),
        "cached_hours": len(frames),
        "source": "cmems",
        "lats": sub_lats.tolist() if hasattr(sub_lats, 'tolist') else list(sub_lats),
        "lons": sub_lons.tolist() if hasattr(sub_lons, 'tolist') else list(sub_lons),
        "ny": len(sub_lats),
        "nx": len(sub_lons),
        "ocean_mask": ocean_mask_arr,
        "ocean_mask_lats": mask_lats_arr,
        "ocean_mask_lons": mask_lons_arr,
        "frames": frames,
    })
    logger.info(f"Ice forecast cached: {len(frames)} frames")

    if weather_ingestion is not None:
        try:
            logger.info("Ingesting ice forecast frames into PostgreSQL...")
            weather_ingestion.ingest_ice_forecast_frames(result)
        except Exception as db_e:
            logger.error(f"Ice forecast DB ingestion failed: {db_e}")


@app.post("/api/weather/forecast/ice/prefetch", dependencies=[Depends(require_not_demo("Weather prefetch"))])
async def api_trigger_ice_forecast_prefetch(
    background_tasks: BackgroundTasks,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    """Trigger background download of CMEMS ice forecast (10-day daily)."""
    return ice_layer.trigger_response(background_tasks, _do_ice_prefetch, lat_min, lat_max, lon_min, lon_max)


@app.get("/api/weather/forecast/ice/frames")
async def api_get_ice_forecast_frames(
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    """Return all cached CMEMS ice forecast frames.

    Serves the raw JSON file to avoid parse+re-serialize overhead.
    """
    cache_key = ice_layer.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    raw = ice_layer.serve_frames_file(cache_key)
    if raw is not None:
        return raw

    # Fallback: rebuild from PostgreSQL
    cached = await asyncio.to_thread(
        _rebuild_ice_cache_from_db, cache_key, lat_min, lat_max, lon_min, lon_max
    )

    if not cached:
        return {
            "run_time": "",
            "total_hours": 0,
            "cached_hours": 0,
            "source": "none",
            "lats": [],
            "lons": [],
            "ny": 0,
            "nx": 0,
            "frames": {},
        }

    return cached


# ======================================================================
# SST Forecast Prefetch Pipeline
# ======================================================================

@app.get("/api/weather/forecast/sst/status")
async def api_get_sst_forecast_status(
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    """Get SST forecast prefetch status."""
    cache_key = sst_layer.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    cached = sst_layer.cache_get(cache_key)

    prefetch_running = sst_layer.is_running

    if cached:
        cached_hours = len(cached.get("frames", {}))
        total_hours = cached_hours if not prefetch_running else max(cached_hours + 1, 41)
        return {
            "total_hours": total_hours,
            "cached_hours": cached_hours,
            "complete": cached_hours > 0 and not prefetch_running,
            "prefetch_running": prefetch_running,
        }

    return {
        "total_hours": 41,
        "cached_hours": 0,
        "complete": False,
        "prefetch_running": prefetch_running,
    }


def _do_sst_prefetch(mgr, lat_min, lat_max, lon_min, lon_max):
    """Download CMEMS SST forecast and build frames cache."""
    cache_key_chk = mgr.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    existing = mgr.cache_get(cache_key_chk)
    if existing and len(existing.get("frames", {})) >= 30 and cache_covers_bounds(existing, lat_min, lat_max, lon_min, lon_max):
        logger.info("SST forecast file cache already complete, skipping CMEMS download")
        return

    stale_path = mgr.cache_path(cache_key_chk)
    if stale_path.exists():
        stale_path.unlink(missing_ok=True)
        logger.info(f"Removed stale SST cache: {cache_key_chk}")

    logger.info("CMEMS SST forecast prefetch started")

    result = copernicus_provider.fetch_sst_forecast(lat_min, lat_max, lon_min, lon_max)
    if not result:
        logger.error("SST forecast fetch returned empty")
        return

    first_wd = next(iter(result.values()))
    max_dim = max(len(first_wd.lats), len(first_wd.lons))
    STEP = max(1, round(max_dim / 250))
    logger.info(f"SST forecast: grid {len(first_wd.lats)}x{len(first_wd.lons)}, STEP={STEP}")
    sub_lats = first_wd.lats[::STEP]
    sub_lons = first_wd.lons[::STEP]

    mask_lats_arr, mask_lons_arr, ocean_mask_arr = _build_ocean_mask(
        lat_min, lat_max, lon_min, lon_max
    )

    frames = {}
    for fh, wd in sorted(result.items()):
        sst_vals = wd.sst if wd.sst is not None else wd.values
        if sst_vals is not None:
            frames[str(fh)] = {
                "data": np.round(sst_vals[::STEP, ::STEP], 2).tolist(),
            }

    cache_key = mgr.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    mgr.cache_put(cache_key, {
        "run_time": first_wd.time.isoformat() if first_wd.time else "",
        "total_hours": len(frames),
        "cached_hours": len(frames),
        "source": "cmems",
        "lats": sub_lats.tolist() if hasattr(sub_lats, 'tolist') else list(sub_lats),
        "lons": sub_lons.tolist() if hasattr(sub_lons, 'tolist') else list(sub_lons),
        "ny": len(sub_lats),
        "nx": len(sub_lons),
        "ocean_mask": ocean_mask_arr,
        "ocean_mask_lats": mask_lats_arr,
        "ocean_mask_lons": mask_lons_arr,
        "colorscale": {"min": -2, "max": 32, "colors": ["#0000ff", "#00ccff", "#00ff88", "#ffff00", "#ff8800", "#ff0000"]},
        "frames": frames,
    })
    logger.info(f"SST forecast cached: {len(frames)} frames")

    # Also persist to DB so data survives container restarts
    if weather_ingestion is not None:
        try:
            weather_ingestion.ingest_sst_forecast_frames(result)
            logger.info("SST forecast also ingested to DB")
        except Exception as db_err:
            logger.error(f"SST DB ingestion after prefetch failed: {db_err}")


@app.post("/api/weather/forecast/sst/prefetch", dependencies=[Depends(require_not_demo("Weather prefetch"))])
async def api_trigger_sst_forecast_prefetch(
    background_tasks: BackgroundTasks,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    """Trigger background download of CMEMS SST forecast (0-120h, 3h steps)."""
    return sst_layer.trigger_response(background_tasks, _do_sst_prefetch, lat_min, lat_max, lon_min, lon_max)


def _rebuild_sst_cache_from_db(cache_key, lat_min, lat_max, lon_min, lon_max):
    """Rebuild SST forecast file cache from PostgreSQL data."""
    if db_weather is None:
        return None

    run_time, hours = db_weather.get_available_hours_by_source("cmems_sst")
    if not hours:
        return None

    logger.info(f"Rebuilding SST cache from DB: {len(hours)} hours")

    grids = db_weather.get_grids_for_timeline(
        "cmems_sst", ["sst"],
        lat_min, lat_max, lon_min, lon_max, hours
    )

    if not grids or "sst" not in grids or not grids["sst"]:
        return None

    first_fh = min(grids["sst"].keys())
    lats_full, lons_full, _ = grids["sst"][first_fh]
    max_dim = max(len(lats_full), len(lons_full))
    STEP = max(1, round(max_dim / 250))
    shared_lats = lats_full[::STEP].tolist()
    shared_lons = lons_full[::STEP].tolist()

    mask_lats_arr, mask_lons_arr, ocean_mask_arr = _build_ocean_mask(
        lat_min, lat_max, lon_min, lon_max
    )

    frames = {}
    for fh in sorted(hours):
        if fh in grids["sst"]:
            _, _, d = grids["sst"][fh]
            frames[str(fh)] = {
                "data": np.round(d[::STEP, ::STEP], 2).tolist(),
            }

    cache_data = {
        "run_time": run_time.isoformat() if run_time else "",
        "total_hours": len(frames),
        "cached_hours": len(frames),
        "source": "cmems",
        "lats": shared_lats,
        "lons": shared_lons,
        "ny": len(shared_lats),
        "nx": len(shared_lons),
        "ocean_mask": ocean_mask_arr,
        "ocean_mask_lats": mask_lats_arr,
        "ocean_mask_lons": mask_lons_arr,
        "colorscale": {"min": -2, "max": 32, "colors": ["#0000ff", "#00ccff", "#00ff88", "#ffff00", "#ff8800", "#ff0000"]},
        "frames": frames,
    }

    sst_layer.cache_put(cache_key, cache_data)
    logger.info(f"SST cache rebuilt from DB: {len(frames)} frames")
    return cache_data


@app.get("/api/weather/forecast/sst/frames")
async def api_get_sst_forecast_frames(
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    """Return all cached CMEMS SST forecast frames.

    Serves the raw JSON file to avoid parse+re-serialize overhead.
    Falls back to any cache file whose bounds cover the requested viewport,
    then to PostgreSQL if no file cache exists.
    """
    cache_key = sst_layer.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    raw = sst_layer.serve_frames_file(cache_key, lat_min, lat_max, lon_min, lon_max, use_covering=True)
    if raw is not None:
        return raw

    # Fallback: rebuild from PostgreSQL
    cached = await asyncio.to_thread(
        _rebuild_sst_cache_from_db, cache_key, lat_min, lat_max, lon_min, lon_max
    )

    if cached:
        return cached

    return {
        "run_time": "",
        "total_hours": 0,
        "cached_hours": 0,
        "source": "none",
        "lats": [],
        "lons": [],
        "ny": 0,
        "nx": 0,
        "frames": {},
    }


# ======================================================================
# Visibility Forecast Prefetch Pipeline
# ======================================================================

def _rebuild_vis_cache_from_db(cache_key, lat_min, lat_max, lon_min, lon_max):
    """Rebuild visibility forecast file cache from PostgreSQL data."""
    if db_weather is None:
        return None

    run_time, hours = db_weather.get_available_hours_by_source("gfs_visibility")
    if not hours:
        return None

    logger.info(f"Rebuilding visibility cache from DB: {len(hours)} hours")

    grids = db_weather.get_grids_for_timeline(
        "gfs_visibility", ["visibility"],
        lat_min, lat_max, lon_min, lon_max, hours
    )

    if not grids or "visibility" not in grids or not grids["visibility"]:
        return None

    first_fh = min(grids["visibility"].keys())
    lats_full, lons_full, _ = grids["visibility"][first_fh]
    max_dim = max(len(lats_full), len(lons_full))
    STEP = max(1, round(max_dim / 250))
    shared_lats = lats_full[::STEP].tolist()
    shared_lons = lons_full[::STEP].tolist()

    frames = {}
    for fh in sorted(hours):
        if fh in grids["visibility"]:
            _, _, d = grids["visibility"][fh]
            frames[str(fh)] = {
                "data": np.round(d[::STEP, ::STEP], 1).tolist(),
            }

    cache_data = {
        "run_time": run_time.isoformat() if run_time else "",
        "total_hours": len(frames),
        "cached_hours": len(frames),
        "source": "gfs",
        "lats": shared_lats,
        "lons": shared_lons,
        "ny": len(shared_lats),
        "nx": len(shared_lons),
        "frames": frames,
    }

    vis_layer.cache_put(cache_key, cache_data)
    logger.info(f"Visibility cache rebuilt from DB: {len(frames)} frames")
    return cache_data


@app.get("/api/weather/forecast/visibility/status")
async def api_get_vis_forecast_status(
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    """Get visibility forecast prefetch status."""
    cache_key = vis_layer.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    cached = vis_layer.cache_get(cache_key)

    prefetch_running = vis_layer.is_running

    if cached:
        cached_hours = len(cached.get("frames", {}))
        total_hours = cached_hours if not prefetch_running else max(cached_hours + 1, 41)
        return {
            "total_hours": total_hours,
            "cached_hours": cached_hours,
            "complete": cached_hours > 0 and not prefetch_running,
            "prefetch_running": prefetch_running,
        }

    return {
        "total_hours": 41,
        "cached_hours": 0,
        "complete": False,
        "prefetch_running": prefetch_running,
    }


def _do_vis_prefetch(mgr, lat_min, lat_max, lon_min, lon_max):
    """Download GFS visibility forecast and build frames cache."""
    cache_key_chk = mgr.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    existing = mgr.cache_get(cache_key_chk)
    if existing and len(existing.get("frames", {})) >= 30 and cache_covers_bounds(existing, lat_min, lat_max, lon_min, lon_max):
        logger.info("Visibility forecast file cache already complete, skipping GFS download")
        return

    stale_path = mgr.cache_path(cache_key_chk)
    if stale_path.exists():
        stale_path.unlink(missing_ok=True)
        logger.info(f"Removed stale visibility cache: {cache_key_chk}")

    logger.info("GFS visibility forecast prefetch started")

    result = gfs_provider.fetch_visibility_forecast(lat_min, lat_max, lon_min, lon_max)
    if not result:
        logger.error("Visibility forecast fetch returned empty")
        return

    first_wd = next(iter(result.values()))
    max_dim = max(len(first_wd.lats), len(first_wd.lons))
    STEP = max(1, round(max_dim / 250))
    logger.info(f"Visibility forecast: grid {len(first_wd.lats)}x{len(first_wd.lons)}, STEP={STEP}")
    sub_lats = first_wd.lats[::STEP]
    sub_lons = first_wd.lons[::STEP]

    mask_lats_arr, mask_lons_arr, ocean_mask_arr = _build_ocean_mask(
        lat_min, lat_max, lon_min, lon_max
    )

    frames = {}
    for fh, wd in sorted(result.items()):
        vis_vals = wd.visibility if wd.visibility is not None else wd.values
        if vis_vals is not None:
            frames[str(fh)] = {
                "data": np.round(vis_vals[::STEP, ::STEP], 1).tolist(),
            }

    cache_key = mgr.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    mgr.cache_put(cache_key, {
        "run_time": first_wd.time.isoformat() if first_wd.time else "",
        "total_hours": len(frames),
        "cached_hours": len(frames),
        "source": "gfs",
        "lats": sub_lats.tolist() if hasattr(sub_lats, 'tolist') else list(sub_lats),
        "lons": sub_lons.tolist() if hasattr(sub_lons, 'tolist') else list(sub_lons),
        "ny": len(sub_lats),
        "nx": len(sub_lons),
        "ocean_mask": ocean_mask_arr,
        "ocean_mask_lats": mask_lats_arr,
        "ocean_mask_lons": mask_lons_arr,
        "colorscale": {"min": 0, "max": 50, "colors": ["#ff0000", "#ff8800", "#ffff00", "#88ff00", "#00ff00"]},
        "frames": frames,
    })
    logger.info(f"Visibility forecast cached: {len(frames)} frames")

    # Also persist to DB so data survives container restarts
    if weather_ingestion is not None:
        try:
            weather_ingestion.ingest_visibility_forecast_frames(result)
            logger.info("Visibility forecast also ingested to DB")
        except Exception as db_err:
            logger.error(f"Visibility DB ingestion after prefetch failed: {db_err}")


@app.post("/api/weather/forecast/visibility/prefetch", dependencies=[Depends(require_not_demo("Weather prefetch"))])
async def api_trigger_vis_forecast_prefetch(
    background_tasks: BackgroundTasks,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    """Trigger background download of GFS visibility forecast (0-120h, 3h steps)."""
    return vis_layer.trigger_response(background_tasks, _do_vis_prefetch, lat_min, lat_max, lon_min, lon_max)


@app.get("/api/weather/forecast/visibility/frames")
async def api_get_vis_forecast_frames(
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    """Return all cached GFS visibility forecast frames.

    Serves the raw JSON file to avoid parse+re-serialize overhead.
    Falls back to any cache file whose bounds cover the requested viewport.
    """
    cache_key = vis_layer.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    raw = vis_layer.serve_frames_file(cache_key, lat_min, lat_max, lon_min, lon_max, use_covering=True)
    if raw is not None:
        return raw

    # Fallback: rebuild from PostgreSQL
    cached = await asyncio.to_thread(
        _rebuild_vis_cache_from_db, cache_key, lat_min, lat_max, lon_min, lon_max
    )

    if cached:
        return cached

    return {
        "run_time": "",
        "total_hours": 0,
        "cached_hours": 0,
        "source": "none",
        "lats": [],
        "lons": [],
        "ny": 0,
        "nx": 0,
        "frames": {},
    }


# --- Overlay grid subsampling (prevents browser OOM on large viewports) ---
_OVERLAY_MAX_DIM = 500  # max grid points per axis for single-frame overlays


def _overlay_step(lats, lons):
    """Compute subsample step for overlay grids exceeding target size."""
    return max(1, math.ceil(max(len(lats), len(lons)) / _OVERLAY_MAX_DIM))


def _sub2d(arr, step, decimals=2):
    """Subsample and round a 2D numpy array. Returns list or None."""
    if arr is None:
        return None
    return np.round(arr[::step, ::step], decimals).tolist()


def _dynamic_mask_step(lat_min, lat_max, lon_min, lon_max):
    """Compute ocean mask step that keeps grid under 500 points per axis."""
    span = max(lat_max - lat_min, lon_max - lon_min)
    return round(max(0.05, span / 500), 3)


@app.get("/api/weather/waves")
async def api_get_wave_field(
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
    resolution: float = Query(1.0),
    time: Optional[datetime] = None,
):
    """
    Get wave height field for visualization.

    DB-first: returns cached data if available, otherwise fetches from API.
    """
    if time is None:
        time = datetime.utcnow()

    ingested_at = None
    if db_weather is not None:
        wave_data, ingested_at = db_weather.get_wave_from_db(lat_min, lat_max, lon_min, lon_max)
        if wave_data is not None:
            logger.debug("Waves endpoint: serving from DB")
        else:
            wind_data = get_wind_field(lat_min, lat_max, lon_min, lon_max, resolution, time)
            wave_data = get_wave_field(lat_min, lat_max, lon_min, lon_max, resolution, wind_data)
            ingested_at = datetime.now(timezone.utc)
    else:
        wind_data = get_wind_field(lat_min, lat_max, lon_min, lon_max, resolution, time)
        wave_data = get_wave_field(lat_min, lat_max, lon_min, lon_max, resolution, wind_data)

    # Subsample large CMEMS grids to prevent browser OOM
    step = _overlay_step(wave_data.lats, wave_data.lons)
    if step > 1:
        logger.info(f"Waves overlay: subsampling grid by step={step}")
    sub_lats = wave_data.lats[::step].tolist()
    sub_lons = wave_data.lons[::step].tolist()

    mask_lats, mask_lons, ocean_mask = _build_ocean_mask(
        lat_min, lat_max, lon_min, lon_max,
        step=_dynamic_mask_step(lat_min, lat_max, lon_min, lon_max),
    )

    # Build response with combined data
    response = {
        "parameter": "wave_height",
        "time": time.isoformat(),
        "bbox": {
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
        },
        "resolution": resolution,
        "nx": len(sub_lons),
        "ny": len(sub_lats),
        "lats": sub_lats,
        "lons": sub_lons,
        "data": _sub2d(wave_data.values, step),
        "unit": "m",
        "ocean_mask": ocean_mask,
        "ocean_mask_lats": mask_lats,
        "ocean_mask_lons": mask_lons,
        "source": "copernicus" if copernicus_provider._has_copernicusmarine else "synthetic",
        "colorscale": {
            "min": 0,
            "max": 6,
            "colors": ["#00ff00", "#ffff00", "#ff8800", "#ff0000", "#800000"],
        },
    }

    # Include mean wave direction grid (degrees, meteorological convention)
    if wave_data.wave_direction is not None:
        response["direction"] = _sub2d(wave_data.wave_direction, step, 1)

    # Include wave decomposition when available
    has_decomp = wave_data.windwave_height is not None and wave_data.swell_height is not None
    response["has_decomposition"] = has_decomp
    if has_decomp:
        response["windwave"] = {
            "height": _sub2d(wave_data.windwave_height, step),
            "period": _sub2d(wave_data.windwave_period, step, 1),
            "direction": _sub2d(wave_data.windwave_direction, step, 1),
        }
        response["swell"] = {
            "height": _sub2d(wave_data.swell_height, step),
            "period": _sub2d(wave_data.swell_period, step, 1),
            "direction": _sub2d(wave_data.swell_direction, step, 1),
        }

    if ingested_at is not None:
        response["ingested_at"] = ingested_at.isoformat()
    return response


@app.get("/api/weather/currents")
async def api_get_current_field(
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
    resolution: float = Query(1.0),
    time: Optional[datetime] = None,
):
    """
    Get ocean current field for visualization.

    Uses Copernicus CMEMS when available, falls back to synthetic data.
    """
    if time is None:
        time = datetime.utcnow()

    current_data = get_current_field(lat_min, lat_max, lon_min, lon_max, resolution)

    # Subsample large CMEMS grids to prevent browser OOM
    step = _overlay_step(current_data.lats, current_data.lons)
    if step > 1:
        logger.info(f"Currents overlay: subsampling grid by step={step}")

    return {
        "parameter": "current",
        "time": time.isoformat(),
        "available": True,
        "bbox": {
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
        },
        "resolution": resolution,
        "nx": len(current_data.lons[::step]),
        "ny": len(current_data.lats[::step]),
        "lats": current_data.lats[::step].tolist(),
        "lons": current_data.lons[::step].tolist(),
        "u": _sub2d(current_data.u_component, step) if current_data.u_component is not None else [],
        "v": _sub2d(current_data.v_component, step) if current_data.v_component is not None else [],
        "unit": "m/s",
        "source": "copernicus" if copernicus_provider._has_copernicusmarine else "synthetic",
    }


@app.get("/api/weather/currents/velocity")
async def api_get_current_velocity_format(
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
    resolution: float = Query(1.0),
    time: Optional[datetime] = None,
):
    """
    Get ocean current data in leaflet-velocity compatible format.

    Returns array of [U-component, V-component] data with headers.
    DB-first: returns cached data if available, otherwise fetches from API.
    """
    if time is None:
        time = datetime.utcnow()

    current_data = get_current_field(lat_min, lat_max, lon_min, lon_max, resolution)

    # Apply ocean mask — zero out land so particles don't render there
    u_masked, v_masked = _apply_ocean_mask_velocity(
        current_data.u_component, current_data.v_component,
        current_data.lats, current_data.lons,
    )

    # Subsample large CMEMS grids to prevent browser OOM
    step = _overlay_step(current_data.lats, current_data.lons)
    if step > 1:
        logger.info(f"Currents velocity: subsampling grid by step={step}")
        u_masked = u_masked[::step, ::step]
        v_masked = v_masked[::step, ::step]
        actual_lats = current_data.lats[::step]
        actual_lons = current_data.lons[::step]
    else:
        actual_lats = current_data.lats
        actual_lons = current_data.lons

    # Derive header from actual data arrays (native resolution may differ from request)
    actual_dx = abs(float(actual_lons[1] - actual_lons[0])) if len(actual_lons) > 1 else resolution
    actual_dy = abs(float(actual_lats[1] - actual_lats[0])) if len(actual_lats) > 1 else resolution

    # leaflet-velocity expects data ordered N→S, W→E
    if len(actual_lats) > 1 and actual_lats[1] > actual_lats[0]:
        u_ordered = u_masked[::-1]
        v_ordered = v_masked[::-1]
        lat_north = float(actual_lats[-1])
        lat_south = float(actual_lats[0])
    else:
        u_ordered = u_masked
        v_ordered = v_masked
        lat_north = float(actual_lats[0])
        lat_south = float(actual_lats[-1])

    header = {
        "parameterCategory": 2,
        "parameterNumber": 2,
        "lo1": float(actual_lons[0]),
        "la1": lat_north,
        "lo2": float(actual_lons[-1]),
        "la2": lat_south,
        "dx": actual_dx,
        "dy": actual_dy,
        "nx": len(actual_lons),
        "ny": len(actual_lats),
        "refTime": time.isoformat(),
    }

    u_flat = u_ordered.flatten().tolist()
    v_flat = v_ordered.flatten().tolist()

    return [
        {"header": {**header, "parameterNumber": 2}, "data": u_flat},
        {"header": {**header, "parameterNumber": 3}, "data": v_flat},
    ]


@app.get("/api/weather/point")
async def api_get_weather_point(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    time: Optional[datetime] = None,
):
    """
    Get weather at a specific point.

    Returns wind, waves, and currents (if available).
    """
    if time is None:
        time = datetime.utcnow()

    wx = get_weather_at_point(lat, lon, time)

    return {
        "position": {"lat": lat, "lon": lon},
        "time": time.isoformat(),
        "wind": {
            "speed_ms": wx['wind_speed_ms'],
            "speed_kts": wx['wind_speed_ms'] * 1.94384,
            "dir_deg": wx['wind_dir_deg'],
        },
        "waves": {
            "height_m": wx['sig_wave_height_m'],
            "dir_deg": wx['wave_dir_deg'],
        },
        "current": {
            "speed_ms": wx['current_speed_ms'],
            "speed_kts": wx['current_speed_ms'] * 1.94384,
            "dir_deg": wx['current_dir_deg'],
        }
    }


# ============================================================================
# API Endpoints - Extended Weather Fields (SPEC-P1)
# ============================================================================

@app.get("/api/weather/sst")
async def api_get_sst_field(
    lat_min: float = Query(30.0, ge=-90, le=90),
    lat_max: float = Query(60.0, ge=-90, le=90),
    lon_min: float = Query(-15.0, ge=-180, le=180),
    lon_max: float = Query(40.0, ge=-180, le=180),
    resolution: float = Query(1.0, ge=0.25, le=5.0),
    time: Optional[datetime] = None,
):
    """
    Get sea surface temperature field for visualization.

    DB-first: returns cached data if available, otherwise fetches from API.
    """
    if time is None:
        time = datetime.utcnow()

    ingested_at = None
    if db_weather is not None:
        sst_data, ingested_at = db_weather.get_sst_from_db(lat_min, lat_max, lon_min, lon_max, time)
        if sst_data is not None:
            logger.debug("SST endpoint: serving from DB")
        else:
            sst_data = get_sst_field(lat_min, lat_max, lon_min, lon_max, resolution, time)
            ingested_at = datetime.now(timezone.utc)
    else:
        sst_data = get_sst_field(lat_min, lat_max, lon_min, lon_max, resolution, time)

    # Subsample large grids to prevent browser OOM
    step = _overlay_step(sst_data.lats, sst_data.lons)
    if step > 1:
        logger.info(f"SST overlay: subsampling grid by step={step}")
    sub_lats = sst_data.lats[::step].tolist()
    sub_lons = sst_data.lons[::step].tolist()

    mask_lats, mask_lons, ocean_mask = _build_ocean_mask(
        lat_min, lat_max, lon_min, lon_max,
        step=_dynamic_mask_step(lat_min, lat_max, lon_min, lon_max),
    )

    response = {
        "parameter": "sst",
        "time": time.isoformat(),
        "bbox": {
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
        },
        "resolution": resolution,
        "nx": len(sub_lons),
        "ny": len(sub_lats),
        "lats": sub_lats,
        "lons": sub_lons,
        "data": np.round(np.nan_to_num(sst_data.values[::step, ::step], nan=15.0), 2).tolist(),
        "unit": "°C",
        "ocean_mask": ocean_mask,
        "ocean_mask_lats": mask_lats,
        "ocean_mask_lons": mask_lons,
        "source": "copernicus" if copernicus_provider._has_copernicusmarine else "synthetic",
        "colorscale": {
            "min": -2,
            "max": 32,
            "colors": ["#0000ff", "#00ccff", "#00ff88", "#ffff00", "#ff8800", "#ff0000"],
        },
    }
    if ingested_at is not None:
        response["ingested_at"] = ingested_at.isoformat()
    return response


@app.get("/api/weather/visibility")
async def api_get_visibility_field(
    lat_min: float = Query(30.0, ge=-90, le=90),
    lat_max: float = Query(60.0, ge=-90, le=90),
    lon_min: float = Query(-15.0, ge=-180, le=180),
    lon_max: float = Query(40.0, ge=-180, le=180),
    resolution: float = Query(1.0, ge=0.25, le=5.0),
    time: Optional[datetime] = None,
):
    """
    Get visibility field for visualization.

    DB-first: returns cached data if available, otherwise fetches from API.
    """
    if time is None:
        time = datetime.utcnow()

    ingested_at = None
    if db_weather is not None:
        vis_data, ingested_at = db_weather.get_visibility_from_db(lat_min, lat_max, lon_min, lon_max, time)
        if vis_data is not None:
            logger.debug("Visibility endpoint: serving from DB")
        else:
            vis_data = get_visibility_field(lat_min, lat_max, lon_min, lon_max, resolution, time)
            ingested_at = datetime.now(timezone.utc)
    else:
        vis_data = get_visibility_field(lat_min, lat_max, lon_min, lon_max, resolution, time)

    # Subsample large grids to prevent browser OOM
    step = _overlay_step(vis_data.lats, vis_data.lons)
    if step > 1:
        logger.info(f"Visibility overlay: subsampling grid by step={step}")
    sub_lats = vis_data.lats[::step].tolist()
    sub_lons = vis_data.lons[::step].tolist()

    mask_lats, mask_lons, ocean_mask = _build_ocean_mask(
        lat_min, lat_max, lon_min, lon_max,
        step=_dynamic_mask_step(lat_min, lat_max, lon_min, lon_max),
    )

    response = {
        "parameter": "visibility",
        "time": time.isoformat(),
        "bbox": {
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
        },
        "resolution": resolution,
        "nx": len(sub_lons),
        "ny": len(sub_lats),
        "lats": sub_lats,
        "lons": sub_lons,
        "data": np.round(np.nan_to_num(vis_data.values[::step, ::step], nan=50.0), 1).tolist(),
        "unit": "km",
        "ocean_mask": ocean_mask,
        "ocean_mask_lats": mask_lats,
        "ocean_mask_lons": mask_lons,
        "source": "gfs" if vis_data.time is not None else "synthetic",
        "colorscale": {
            "min": 0,
            "max": 50,
            "colors": ["#ff0000", "#ff8800", "#ffff00", "#88ff00", "#00ff00"],
        },
    }
    if ingested_at is not None:
        response["ingested_at"] = ingested_at.isoformat()
    return response


@app.get("/api/weather/ice")
async def api_get_ice_field(
    lat_min: float = Query(30.0, ge=-90, le=90),
    lat_max: float = Query(60.0, ge=-90, le=90),
    lon_min: float = Query(-15.0, ge=-180, le=180),
    lon_max: float = Query(40.0, ge=-180, le=180),
    resolution: float = Query(1.0, ge=0.25, le=5.0),
    time: Optional[datetime] = None,
):
    """
    Get sea ice concentration field for visualization.

    DB-first: returns cached data if available, otherwise fetches from API.
    """
    if time is None:
        time = datetime.utcnow()

    ingested_at = None
    if db_weather is not None:
        ice_data, ingested_at = db_weather.get_ice_from_db(lat_min, lat_max, lon_min, lon_max, time)
        if ice_data is not None:
            logger.debug("Ice endpoint: serving from DB")
        else:
            ice_data = get_ice_field(lat_min, lat_max, lon_min, lon_max, resolution, time)
            ingested_at = datetime.now(timezone.utc)
    else:
        ice_data = get_ice_field(lat_min, lat_max, lon_min, lon_max, resolution, time)

    mask_lats, mask_lons, ocean_mask = _build_ocean_mask(lat_min, lat_max, lon_min, lon_max, step=0.05)

    response = {
        "parameter": "ice_concentration",
        "time": time.isoformat(),
        "bbox": {
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
        },
        "resolution": resolution,
        "nx": len(ice_data.lons),
        "ny": len(ice_data.lats),
        "lats": ice_data.lats.tolist(),
        "lons": ice_data.lons.tolist(),
        "data": np.nan_to_num(ice_data.values, nan=0.0).tolist(),
        "unit": "fraction",
        "ocean_mask": ocean_mask,
        "ocean_mask_lats": mask_lats,
        "ocean_mask_lons": mask_lons,
        "source": "copernicus" if copernicus_provider._has_copernicusmarine else "synthetic",
        "colorscale": {
            "min": 0,
            "max": 1,
            "colors": ["#ffffff", "#ccddff", "#6688ff", "#0033cc", "#001166"],
        },
    }
    if ingested_at is not None:
        response["ingested_at"] = ingested_at.isoformat()
    return response


@app.get("/api/weather/swell")
async def api_get_swell_field(
    lat_min: float = Query(30.0, ge=-90, le=90),
    lat_max: float = Query(60.0, ge=-90, le=90),
    lon_min: float = Query(-15.0, ge=-180, le=180),
    lon_max: float = Query(40.0, ge=-180, le=180),
    resolution: float = Query(1.0, ge=0.25, le=5.0),
    time: Optional[datetime] = None,
):
    """
    Get partitioned swell field (primary swell + wind-sea).

    Returns swell and wind-sea decomposition from CMEMS wave data.
    Same query params as /api/weather/waves.
    """
    if time is None:
        time = datetime.utcnow()

    wave_data = get_wave_field(lat_min, lat_max, lon_min, lon_max, resolution)

    # Subsample large CMEMS grids to prevent browser OOM
    step = _overlay_step(wave_data.lats, wave_data.lons)
    if step > 1:
        logger.info(f"Swell overlay: subsampling grid by step={step}")
    sub_lats = wave_data.lats[::step].tolist()
    sub_lons = wave_data.lons[::step].tolist()

    mask_lats, mask_lons, ocean_mask = _build_ocean_mask(
        lat_min, lat_max, lon_min, lon_max,
        step=_dynamic_mask_step(lat_min, lat_max, lon_min, lon_max),
    )

    # Build swell decomposition response
    has_decomposition = wave_data.swell_height is not None

    return {
        "parameter": "swell",
        "time": time.isoformat(),
        "bbox": {
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
        },
        "resolution": resolution,
        "has_decomposition": has_decomposition,
        "nx": len(sub_lons),
        "ny": len(sub_lats),
        "lats": sub_lats,
        "lons": sub_lons,
        "total_hs": _sub2d(wave_data.values, step),
        "data": _sub2d(np.nan_to_num(wave_data.values, nan=0.0), step),
        "swell_hs": _sub2d(wave_data.swell_height, step),
        "swell_tp": _sub2d(wave_data.swell_period, step, 1),
        "swell_dir": _sub2d(wave_data.swell_direction, step, 1),
        "windsea_hs": _sub2d(wave_data.windwave_height, step),
        "windsea_tp": _sub2d(wave_data.windwave_period, step, 1),
        "windsea_dir": _sub2d(wave_data.windwave_direction, step, 1),
        "unit": "m",
        "ocean_mask": ocean_mask,
        "ocean_mask_lats": mask_lats,
        "ocean_mask_lons": mask_lons,
        "source": "copernicus" if has_decomposition else "synthetic",
    }


# ============================================================================
# API Endpoints - Routes (Layer 2)
# ============================================================================

@app.post("/api/routes/parse-rtz")
@limiter.limit(get_rate_limit_string())
async def parse_rtz(
    request: Request,
    file: UploadFile = File(...),
):
    """
    Parse an uploaded RTZ route file.

    Maximum file size: 5 MB.
    Returns waypoints in standard format.
    """
    try:
        content = await file.read()

        # Validate file size
        if len(content) > MAX_RTZ_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {MAX_RTZ_SIZE_BYTES // (1024*1024)} MB"
            )

        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        rtz_string = content.decode('utf-8')

        route = parse_rtz_string(rtz_string)

        return {
            "name": route.name,
            "waypoints": [
                {
                    "id": wp.id,
                    "name": wp.name,
                    "lat": wp.lat,
                    "lon": wp.lon,
                }
                for wp in route.waypoints
            ],
            "total_distance_nm": route.total_distance_nm,
            "legs": [
                {
                    "from": leg.from_wp.name,
                    "to": leg.to_wp.name,
                    "distance_nm": leg.distance_nm,
                    "bearing_deg": leg.bearing_deg,
                }
                for leg in route.legs
            ]
        }
    except Exception as e:
        logger.error(f"Failed to parse RTZ: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid RTZ file: {str(e)}")


@app.post("/api/routes/from-waypoints")
async def create_route_from_wps(
    waypoints: List[Position],
    name: str = "Custom Route",
):
    """
    Create a route from a list of waypoints.

    Returns route with calculated distances and bearings.
    """
    if len(waypoints) < 2:
        raise HTTPException(status_code=400, detail="At least 2 waypoints required")

    wps = [(wp.lat, wp.lon) for wp in waypoints]
    route = create_route_from_waypoints(wps, name)

    return {
        "name": route.name,
        "waypoints": [
            {
                "id": wp.id,
                "name": wp.name,
                "lat": wp.lat,
                "lon": wp.lon,
            }
            for wp in route.waypoints
        ],
        "total_distance_nm": route.total_distance_nm,
        "legs": [
            {
                "from": leg.from_wp.name,
                "to": leg.to_wp.name,
                "distance_nm": leg.distance_nm,
                "bearing_deg": leg.bearing_deg,
            }
            for leg in route.legs
        ]
    }


# ============================================================================
# API Endpoints - Voyage Calculation (Layer 3)
# ============================================================================

@app.post("/api/voyage/calculate", response_model=VoyageResponse)
async def calculate_voyage(request: VoyageRequest):
    """
    Calculate voyage with per-leg SOG, ETA, and fuel.

    Takes waypoints, calm speed, and vessel condition.
    Returns detailed per-leg results including weather impact.

    Weather data is sourced from:
    - Forecast: Copernicus data for first 10 days
    - Blended: Transition from forecast to climatology (days 10-12)
    - Climatology: ERA5 monthly averages beyond forecast horizon
    """
    if len(request.waypoints) < 2:
        raise HTTPException(status_code=400, detail="At least 2 waypoints required")

    import time as _time

    departure = request.departure_time or datetime.utcnow()
    t_start = _time.monotonic()
    logger.info(f"Voyage calculation started: {len(request.waypoints)} waypoints, speed={request.calm_speed_kts}kts, weather={request.use_weather}")

    # Create route from waypoints
    wps = [(wp.lat, wp.lon) for wp in request.waypoints]
    route = create_route_from_waypoints(wps, "Voyage Route")

    # Reset data source tracking
    reset_voyage_data_sources()

    # Pre-fetch weather grids for entire route bounding box
    # Try temporal (time-varying) weather first, fall back to single-snapshot
    wp_func = None
    data_source_type = None
    used_temporal = False
    if request.use_weather:
        margin = 3.0
        lats = [wp.lat for wp in request.waypoints]
        lons = [wp.lon for wp in request.waypoints]
        lat_min = max(min(lats) - margin, -85)
        lat_max = min(max(lats) + margin, 85)
        lon_min = min(lons) - margin
        lon_max = max(lons) + margin

        origin_pt = (request.waypoints[0].lat, request.waypoints[0].lon)
        dest_pt = (request.waypoints[-1].lat, request.waypoints[-1].lon)

        # ── Temporal weather provisioning (DB-first) ──────────────────
        temporal_wx = None
        if db_weather is not None:
            try:
                assessor = RouteWeatherAssessment(db_weather=db_weather)
                wx_needs = assessor.assess(
                    origin=origin_pt,
                    destination=dest_pt,
                    departure_time=departure,
                    calm_speed_kts=request.calm_speed_kts,
                )
                avail_parts = [f"{s}: {v.get('coverage_pct',0):.0f}%" for s, v in wx_needs.availability.items()]
                logger.info(
                    f"Voyage weather assessment: {wx_needs.estimated_passage_hours:.0f}h passage, "
                    f"need hours {wx_needs.required_forecast_hours[:5]}..., "
                    f"availability: {', '.join(avail_parts)}"
                )
                temporal_wx = assessor.provision(wx_needs)
                if temporal_wx is not None:
                    used_temporal = True
                    params_loaded = list(temporal_wx.grids.keys())
                    has_temporal_wind = any(p in temporal_wx.grids for p in ["wind_u", "wind_v"])
                    if not has_temporal_wind:
                        if _supplement_temporal_wind(temporal_wx, lat_min, lat_max, lon_min, lon_max, departure):
                            has_temporal_wind = True
                            params_loaded = list(temporal_wx.grids.keys())
                    logger.info(
                        f"Voyage using temporal weather: {len(params_loaded)} params ({params_loaded}), "
                        f"wind={'yes' if has_temporal_wind else 'NO (calm assumed)'}"
                    )
                else:
                    logger.warning("Temporal provisioning returned None — falling back to single-snapshot")
            except Exception as e:
                logger.warning(f"Temporal weather provisioning failed for voyage, falling back: {e}", exc_info=True)

        # ── Fallback: single-snapshot GridWeatherProvider ─────────────
        grid_wx = None
        if temporal_wx is None:
            t0 = _time.monotonic()
            logger.info(f"  Pre-fetching single-snapshot weather for bbox [{lat_min:.1f},{lat_max:.1f},{lon_min:.1f},{lon_max:.1f}]")
            wind = get_wind_field(lat_min, lat_max, lon_min, lon_max, 0.5, departure)
            logger.info(f"  Wind loaded in {_time.monotonic()-t0:.1f}s")
            t1 = _time.monotonic()
            waves = get_wave_field(lat_min, lat_max, lon_min, lon_max, 0.5, wind)
            logger.info(f"  Waves loaded in {_time.monotonic()-t1:.1f}s")
            t2 = _time.monotonic()
            currents = get_current_field(lat_min, lat_max, lon_min, lon_max, 0.5)
            logger.info(f"  Currents loaded in {_time.monotonic()-t2:.1f}s")
            # Extended fields (SPEC-P1)
            sst = get_sst_field(lat_min, lat_max, lon_min, lon_max, 0.5, departure)
            vis = get_visibility_field(lat_min, lat_max, lon_min, lon_max, 0.5, departure)
            ice = get_ice_field(lat_min, lat_max, lon_min, lon_max, 0.5, departure)
            logger.info(f"  Total prefetch: {_time.monotonic()-t0:.1f}s (incl. SST/vis/ice)")
            grid_wx = GridWeatherProvider(wind, waves, currents, sst, vis, ice)

        data_source_type = "temporal" if used_temporal else "forecast"
        wx_callable = temporal_wx.get_weather if temporal_wx else grid_wx.get_weather

        # Wrapper that tracks data sources per leg
        def tracked_weather_provider(lat: float, lon: float, time: datetime):
            leg_wx = wx_callable(lat, lon, time)
            get_voyage_data_sources().append({
                'lat': lat, 'lon': lon, 'time': time.isoformat(),
                'source': data_source_type, 'forecast_weight': 1.0,
                'message': f'{"Temporal" if used_temporal else "Single-snapshot"} grid',
            })
            return leg_wx

        wp_func = tracked_weather_provider

    result = _vs.voyage_calculator.calculate_voyage(
        route=route,
        calm_speed_kts=request.calm_speed_kts,
        is_laden=request.is_laden,
        departure_time=departure,
        weather_provider=wp_func,
    )
    logger.info(f"Voyage calculation completed in {_time.monotonic()-t_start:.1f}s: {len(result.legs)} legs, {result.total_distance_nm:.0f}nm, {result.total_fuel_mt:.1f}mt fuel")

    # Build data source summary
    forecast_legs = sum(1 for ds in get_voyage_data_sources() if ds['source'] == 'forecast')
    blended_legs = sum(1 for ds in get_voyage_data_sources() if ds['source'] == 'blended')
    climatology_legs = sum(1 for ds in get_voyage_data_sources() if ds['source'] == 'climatology')

    data_source_warning = None
    if climatology_legs > 0:
        data_source_warning = (
            f"Voyage extends beyond {ClimatologyProvider.FORECAST_HORIZON_DAYS}-day forecast horizon. "
            f"{climatology_legs} leg(s) use climatological averages with higher uncertainty."
        )
    elif blended_legs > 0:
        data_source_warning = (
            f"Voyage approaches forecast horizon. "
            f"{blended_legs} leg(s) use blended forecast/climatology data."
        )

    data_sources_summary = DataSourceSummary(
        forecast_legs=forecast_legs,
        blended_legs=blended_legs,
        climatology_legs=climatology_legs,
        forecast_horizon_days=ClimatologyProvider.FORECAST_HORIZON_DAYS,
        warning=data_source_warning,
    ) if request.use_weather else None

    # Format response with data source info per leg
    legs_response = []
    for i, leg in enumerate(result.legs):
        # Get data source info for this leg
        leg_source = get_voyage_data_sources()[i] if i < len(get_voyage_data_sources()) else None

        legs_response.append(LegResultModel(
            leg_index=leg.leg_index,
            from_wp=WaypointModel(
                id=leg.from_wp.id,
                name=leg.from_wp.name,
                lat=leg.from_wp.lat,
                lon=leg.from_wp.lon,
            ),
            to_wp=WaypointModel(
                id=leg.to_wp.id,
                name=leg.to_wp.name,
                lat=leg.to_wp.lat,
                lon=leg.to_wp.lon,
            ),
            distance_nm=round(leg.distance_nm, 2),
            bearing_deg=round(leg.bearing_deg, 1),
            wind_speed_kts=round(leg.weather.wind_speed_ms * 1.94384, 1),
            wind_dir_deg=round(leg.weather.wind_dir_deg, 0),
            wave_height_m=round(leg.weather.sig_wave_height_m, 1),
            wave_dir_deg=round(leg.weather.wave_dir_deg, 0),
            current_speed_ms=round(leg.weather.current_speed_ms, 2),
            current_dir_deg=round(leg.weather.current_dir_deg, 0),
            calm_speed_kts=round(leg.calm_speed_kts, 1),
            stw_kts=round(leg.stw_kts, 1),
            sog_kts=round(leg.sog_kts, 1),
            speed_loss_pct=round(leg.speed_loss_pct, 1),
            time_hours=round(leg.time_hours, 2),
            departure_time=leg.departure_time,
            arrival_time=leg.arrival_time,
            fuel_mt=round(leg.fuel_mt, 2),
            power_kw=round(leg.power_kw, 0),
            data_source=leg_source['source'] if leg_source else None,
            forecast_weight=leg_source['forecast_weight'] if leg_source else None,
        ))

    return VoyageResponse(
        route_name=result.route_name,
        departure_time=result.departure_time,
        arrival_time=result.arrival_time,
        total_distance_nm=round(result.total_distance_nm, 2),
        total_time_hours=round(result.total_time_hours, 2),
        total_fuel_mt=round(result.total_fuel_mt, 2),
        avg_sog_kts=round(result.avg_sog_kts, 1),
        avg_stw_kts=round(result.avg_stw_kts, 1),
        legs=legs_response,
        calm_speed_kts=request.calm_speed_kts,
        is_laden=request.is_laden,
        data_sources=data_sources_summary,
    )


@app.post("/api/voyage/monte-carlo", response_model=MonteCarloResponse)
async def monte_carlo_simulation(request: MonteCarloRequest):
    """
    Run Monte Carlo simulation on a voyage.

    Perturbs weather conditions across N simulations and returns
    P10/P50/P90 confidence intervals for ETA, fuel, and time.
    """
    import asyncio

    if len(request.waypoints) < 2:
        raise HTTPException(status_code=400, detail="At least 2 waypoints required")

    departure = request.departure_time or datetime.utcnow()

    wps = [(wp.lat, wp.lon) for wp in request.waypoints]
    route = create_route_from_waypoints(wps, "MC Simulation Route")

    # Pre-fetch wind grid for the route bbox so MC wind lookups are instant
    # (avoids per-leg unified provider calls that trigger live CMEMS/GFS downloads)
    mc_weather_fn = weather_provider  # default fallback
    try:
        lats = [wp.lat for wp in request.waypoints]
        lons = [wp.lon for wp in request.waypoints]
        margin = 2.0
        bbox = (min(lats) - margin, max(lats) + margin,
                min(lons) - margin, max(lons) + margin)
        wind_data = get_wind_field(*bbox, 0.5, departure)
        wave_data = get_wave_field(*bbox, 0.5, wind_data)
        current_data = get_current_field(bbox[0], bbox[1], bbox[2], bbox[3])
        if wind_data and wave_data and current_data:
            from src.optimization.grid_weather_provider import GridWeatherProvider
            grid_wx = GridWeatherProvider(wind_data, wave_data, current_data)
            mc_weather_fn = grid_wx.get_weather
            logger.info("MC: Pre-fetched route weather grid for fast wind lookups")
    except Exception as e:
        logger.warning(f"MC: Failed to pre-fetch route grid, using default provider: {e}")

    def _run():
        return _vs.monte_carlo_sim.run(
            route=route,
            calm_speed_kts=request.calm_speed_kts,
            is_laden=request.is_laden,
            departure_time=departure,
            weather_provider=mc_weather_fn,
            n_simulations=request.n_simulations,
            db_weather=db_weather,
        )

    try:
        mc_result = await asyncio.to_thread(_run)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Monte Carlo simulation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

    return MonteCarloResponse(
        n_simulations=mc_result.n_simulations,
        eta=PercentileString(
            p10=mc_result.eta_p10,
            p50=mc_result.eta_p50,
            p90=mc_result.eta_p90,
        ),
        fuel_mt=PercentileFloat(
            p10=mc_result.fuel_p10,
            p50=mc_result.fuel_p50,
            p90=mc_result.fuel_p90,
        ),
        total_time_hours=PercentileFloat(
            p10=mc_result.time_p10,
            p50=mc_result.time_p50,
            p90=mc_result.time_p90,
        ),
        computation_time_ms=mc_result.computation_time_ms,
    )


@app.get("/api/voyage/weather-along-route")
async def get_weather_along_route(
    waypoints: str = Query(..., description="Comma-separated lat,lon pairs: lat1,lon1;lat2,lon2;..."),
    time: Optional[datetime] = None,
    interpolation_points: int = Query(5, ge=1, le=20, description="Points to interpolate per leg"),
):
    """
    Get weather conditions along a route with distance-indexed interpolation.

    Returns weather at waypoints plus interpolated points along each leg,
    with cumulative distance for chart display.
    """
    if time is None:
        time = datetime.utcnow()

    # Parse waypoints
    try:
        wps = []
        for wp_str in waypoints.split(';'):
            lat, lon = wp_str.strip().split(',')
            wps.append((float(lat), float(lon)))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid waypoints format: {e}")

    if len(wps) < 2:
        raise HTTPException(status_code=400, detail="At least 2 waypoints required")

    # Build interpolated points along great circle per leg
    points = []
    cumulative_nm = 0.0

    for i in range(len(wps)):
        lat, lon = wps[i]

        if i > 0:
            prev_lat, prev_lon = wps[i - 1]

            # Interpolate along the leg (skip first point — already added as prev waypoint)
            for j in range(1, interpolation_points):
                frac = j / interpolation_points
                # Linear interpolation (good enough for nearby points)
                ilat = prev_lat + (lat - prev_lat) * frac
                ilon = prev_lon + (lon - prev_lon) * frac

                seg_dist = haversine_distance(
                    prev_lat if j == 1 else points[-1]["lat"],
                    prev_lon if j == 1 else points[-1]["lon"],
                    ilat, ilon,
                )
                cumulative_nm += seg_dist

                wx, _ = get_weather_at_point(ilat, ilon, time)
                points.append({
                    "distance_nm": round(cumulative_nm, 1),
                    "lat": round(ilat, 4),
                    "lon": round(ilon, 4),
                    "wind_speed_kts": round(wx['wind_speed_ms'] * 1.94384, 1),
                    "wind_dir_deg": round(wx['wind_dir_deg'], 0),
                    "wave_height_m": round(wx['sig_wave_height_m'], 1),
                    "wave_dir_deg": round(wx['wave_dir_deg'], 0),
                    "current_speed_ms": round(wx['current_speed_ms'], 2),
                    "current_dir_deg": round(wx['current_dir_deg'], 0),
                    "is_waypoint": False,
                    "waypoint_index": None,
                })

            # Distance from last interpolated point to this waypoint
            if points:
                seg_dist = haversine_distance(points[-1]["lat"], points[-1]["lon"], lat, lon)
                cumulative_nm += seg_dist
            # else first waypoint at distance 0

        # Add waypoint itself
        wx, _ = get_weather_at_point(lat, lon, time)
        points.append({
            "distance_nm": round(cumulative_nm, 1),
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "wind_speed_kts": round(wx['wind_speed_ms'] * 1.94384, 1),
            "wind_dir_deg": round(wx['wind_dir_deg'], 0),
            "wave_height_m": round(wx['sig_wave_height_m'], 1),
            "wave_dir_deg": round(wx['wave_dir_deg'], 0),
            "current_speed_ms": round(wx['current_speed_ms'], 2),
            "current_dir_deg": round(wx['current_dir_deg'], 0),
            "is_waypoint": True,
            "waypoint_index": i,
        })

    return {
        "time": time.isoformat(),
        "total_distance_nm": round(cumulative_nm, 1),
        "points": points,
    }


# ============================================================================
# API Endpoints - Route Optimization (Layer 4)
# ============================================================================

@app.post("/api/optimize/route", response_model=OptimizationResponse)
async def optimize_route(request: OptimizationRequest):
    """
    Find optimal route through weather.

    Supports two optimization engines selected via the ``engine`` field:
    - **astar** (default): A* grid search with weather-aware cost function
    - **visir**: VISIR-style Dijkstra with time-expanded graph and voluntary speed reduction

    Minimizes fuel consumption (or time) by routing around adverse weather.

    Grid resolution affects accuracy vs computation time:
    - 0.2° = ~12nm cells, good land avoidance (default for A*)
    - 0.25° = ~15nm cells, good balance (default for VISIR)
    - 0.5° = ~30nm cells, faster, less precise
    """
    # Run the entire optimization in a thread so the event loop stays
    # responsive (weather provisioning + VISIR can take 30s+, and an
    # unresponsive event loop causes the uvicorn worker supervisor to
    # kill the child process).
    return await asyncio.to_thread(_optimize_route_sync, request)


def _optimize_route_sync(request: "OptimizationRequest") -> "OptimizationResponse":
    """Synchronous route optimization logic (runs in a thread pool)."""
    departure = request.departure_time or datetime.utcnow()

    # Select optimization engine
    engine_name = request.engine.lower()
    if engine_name == "visir":
        active_optimizer = _vs.visir_optimizer
    else:
        active_optimizer = _vs.route_optimizer
    # VISIR uses coarser resolution than A* (0.25° vs 0.1°) for performance;
    # at 0.25° the edge land checks keep routes off land reliably.
    active_optimizer.resolution_deg = (
        max(request.grid_resolution_deg, 0.25) if engine_name == "visir"
        else request.grid_resolution_deg
    )
    active_optimizer.optimization_target = request.optimization_target
    active_optimizer.safety_weight = request.safety_weight

    try:
        # ── Temporal weather provisioning (DB-first) ──────────────────
        temporal_wx = None
        provenance_models = None
        used_temporal = False

        if db_weather is not None:
            try:
                assessor = RouteWeatherAssessment(db_weather=db_weather)
                wx_needs = assessor.assess(
                    origin=(request.origin.lat, request.origin.lon),
                    destination=(request.destination.lat, request.destination.lon),
                    departure_time=departure,
                    calm_speed_kts=request.calm_speed_kts,
                )
                avail_parts = [f"{s}: {v.get('coverage_pct',0):.0f}%" for s,v in wx_needs.availability.items()]
                logger.info(
                    f"Weather assessment: {wx_needs.estimated_passage_hours:.0f}h passage, "
                    f"need hours {wx_needs.required_forecast_hours[:5]}..., "
                    f"availability: {', '.join(avail_parts)}, "
                    f"warnings: {wx_needs.gap_warnings}"
                )
                # Provision temporal weather from DB (waves+currents may be available
                # even if GFS wind is not — provisioner handles partial data)
                temporal_wx = assessor.provision(wx_needs)
                if temporal_wx is not None:
                    used_temporal = True
                    params_loaded = list(temporal_wx.grids.keys())
                    has_temporal_wind = any(p in temporal_wx.grids for p in ["wind_u", "wind_v"])
                    if not has_temporal_wind:
                        bbox = wx_needs.corridor_bbox
                        if _supplement_temporal_wind(temporal_wx, bbox[0], bbox[1], bbox[2], bbox[3], departure):
                            has_temporal_wind = True
                            params_loaded = list(temporal_wx.grids.keys())
                    hours_per_param = {p: sorted(temporal_wx.grids[p].keys()) for p in params_loaded[:3]}
                    logger.info(
                        f"Temporal provider: {len(params_loaded)} params ({params_loaded}), "
                        f"wind={'yes' if has_temporal_wind else 'NO (calm assumed)'}, "
                        f"hours sample: {hours_per_param}"
                    )
                    provenance_models = [
                        WeatherProvenanceModel(
                            source_type=p.source_type,
                            model_name=p.model_name,
                            forecast_lead_hours=round(p.forecast_lead_hours, 1),
                            confidence=p.confidence,
                        )
                        for p in temporal_wx.provenance.values()
                    ]
                    logger.info("Using temporal weather provider for route optimization")
                else:
                    logger.warning("Temporal provisioning returned None — falling back to single-snapshot")
            except Exception as e:
                logger.warning(f"Temporal weather provisioning failed, falling back: {e}", exc_info=True)

        # ── Fallback: single-snapshot GridWeatherProvider ─────────────
        if temporal_wx is None:
            import time as _time
            margin = 5.0
            lat_min = min(request.origin.lat, request.destination.lat) - margin
            lat_max = max(request.origin.lat, request.destination.lat) + margin
            lon_min = min(request.origin.lon, request.destination.lon) - margin
            lon_max = max(request.origin.lon, request.destination.lon) + margin
            lat_min, lat_max = max(lat_min, -85), min(lat_max, 85)

            logger.info(f"Fallback: loading single-snapshot weather for bbox [{lat_min:.1f},{lat_max:.1f},{lon_min:.1f},{lon_max:.1f}]")
            t0 = _time.monotonic()
            wind = get_wind_field(lat_min, lat_max, lon_min, lon_max, request.grid_resolution_deg, departure)
            logger.info(f"  Wind loaded in {_time.monotonic()-t0:.1f}s: source={getattr(wind, 'source', '?')}")
            t1 = _time.monotonic()
            waves = get_wave_field(lat_min, lat_max, lon_min, lon_max, request.grid_resolution_deg, wind)
            logger.info(f"  Waves loaded in {_time.monotonic()-t1:.1f}s")
            t2 = _time.monotonic()
            currents = get_current_field(lat_min, lat_max, lon_min, lon_max, request.grid_resolution_deg)
            logger.info(f"  Currents loaded in {_time.monotonic()-t2:.1f}s")
            # Extended fields (SPEC-P1)
            sst = get_sst_field(lat_min, lat_max, lon_min, lon_max, request.grid_resolution_deg, departure)
            vis = get_visibility_field(lat_min, lat_max, lon_min, lon_max, request.grid_resolution_deg, departure)
            ice = get_ice_field(lat_min, lat_max, lon_min, lon_max, request.grid_resolution_deg, departure)
            logger.info(f"  Total fallback: {_time.monotonic()-t0:.1f}s (incl. SST/vis/ice)")
            grid_wx = GridWeatherProvider(wind, waves, currents, sst, vis, ice)

        # Select weather provider callable
        wx_provider = temporal_wx.get_weather if temporal_wx else grid_wx.get_weather

        # Convert route_waypoints for multi-segment optimization
        route_wps = None
        if request.route_waypoints and len(request.route_waypoints) > 2:
            route_wps = [(wp.lat, wp.lon) for wp in request.route_waypoints]

        # A* engine accepts extra dev params; VISIR uses base interface
        if engine_name == "visir":
            result = active_optimizer.optimize_route(
                origin=(request.origin.lat, request.origin.lon),
                destination=(request.destination.lat, request.destination.lon),
                departure_time=departure,
                calm_speed_kts=request.calm_speed_kts,
                is_laden=request.is_laden,
                weather_provider=wx_provider,
                max_time_factor=request.max_time_factor,
            )
        else:
            result = active_optimizer.optimize_route(
                origin=(request.origin.lat, request.origin.lon),
                destination=(request.destination.lat, request.destination.lon),
                departure_time=departure,
                calm_speed_kts=request.calm_speed_kts,
                is_laden=request.is_laden,
                weather_provider=wx_provider,
                max_time_factor=request.max_time_factor,
                baseline_time_hours=request.baseline_time_hours,
                baseline_fuel_mt=request.baseline_fuel_mt,
                baseline_distance_nm=request.baseline_distance_nm,
                route_waypoints=route_wps,
            )

        # Format response
        waypoints = [Position(lat=wp[0], lon=wp[1]) for wp in result.waypoints]

        # Compute cumulative time for provenance per leg
        cum_time_h = 0.0
        legs = []
        for leg in result.leg_details:
            # Per-leg provenance label
            data_source_label = None
            if used_temporal and temporal_wx is not None:
                from datetime import timedelta as _td
                leg_time = departure + _td(hours=cum_time_h + leg['time_hours'] / 2)
                prov = temporal_wx.get_provenance(leg_time)
                data_source_label = f"{prov.source_type} ({prov.confidence} confidence)"
            cum_time_h += leg['time_hours']

            legs.append(OptimizationLegModel(
                from_lat=leg['from'][0],
                from_lon=leg['from'][1],
                to_lat=leg['to'][0],
                to_lon=leg['to'][1],
                distance_nm=round(leg['distance_nm'], 2),
                bearing_deg=round(leg['bearing_deg'], 1),
                fuel_mt=round(leg['fuel_mt'], 3),
                time_hours=round(leg['time_hours'], 2),
                sog_kts=round(leg['sog_kts'], 1),
                stw_kts=round(leg.get('stw_kts', leg['sog_kts']), 1),
                wind_speed_ms=round(leg['wind_speed_ms'], 1),
                wave_height_m=round(leg['wave_height_m'], 1),
                safety_status=leg.get('safety_status'),
                roll_deg=round(leg['roll_deg'], 1) if leg.get('roll_deg') else None,
                pitch_deg=round(leg['pitch_deg'], 1) if leg.get('pitch_deg') else None,
                data_source=data_source_label,
                swell_hs_m=round(leg['swell_hs_m'], 2) if leg.get('swell_hs_m') is not None else None,
                windsea_hs_m=round(leg['windsea_hs_m'], 2) if leg.get('windsea_hs_m') is not None else None,
                current_effect_kts=round(leg['current_effect_kts'], 2) if leg.get('current_effect_kts') is not None else None,
                visibility_m=round(leg['visibility_m'], 0) if leg.get('visibility_m') is not None else None,
                sst_celsius=round(leg['sst_celsius'], 1) if leg.get('sst_celsius') is not None else None,
                ice_concentration=round(leg['ice_concentration'], 3) if leg.get('ice_concentration') is not None else None,
            ))

        # Build safety summary
        safety_summary = SafetySummary(
            status=result.safety_status,
            warnings=result.safety_warnings,
            max_roll_deg=round(result.max_roll_deg, 1),
            max_pitch_deg=round(result.max_pitch_deg, 1),
            max_accel_ms2=round(result.max_accel_ms2, 2),
        )

        # Build speed strategy scenarios
        scenario_models = []
        for sc in result.scenarios:
            sc_legs = []
            for leg in sc.leg_details:
                sc_legs.append(OptimizationLegModel(
                    from_lat=leg['from'][0],
                    from_lon=leg['from'][1],
                    to_lat=leg['to'][0],
                    to_lon=leg['to'][1],
                    distance_nm=round(leg['distance_nm'], 2),
                    bearing_deg=round(leg['bearing_deg'], 1),
                    fuel_mt=round(leg['fuel_mt'], 3),
                    time_hours=round(leg['time_hours'], 2),
                    sog_kts=round(leg['sog_kts'], 1),
                    stw_kts=round(leg.get('stw_kts', leg['sog_kts']), 1),
                    wind_speed_ms=round(leg['wind_speed_ms'], 1),
                    wave_height_m=round(leg['wave_height_m'], 1),
                    safety_status=leg.get('safety_status'),
                    roll_deg=round(leg['roll_deg'], 1) if leg.get('roll_deg') else None,
                    pitch_deg=round(leg['pitch_deg'], 1) if leg.get('pitch_deg') else None,
                    swell_hs_m=round(leg['swell_hs_m'], 2) if leg.get('swell_hs_m') is not None else None,
                    windsea_hs_m=round(leg['windsea_hs_m'], 2) if leg.get('windsea_hs_m') is not None else None,
                    current_effect_kts=round(leg['current_effect_kts'], 2) if leg.get('current_effect_kts') is not None else None,
                    visibility_m=round(leg['visibility_m'], 0) if leg.get('visibility_m') is not None else None,
                    sst_celsius=round(leg['sst_celsius'], 1) if leg.get('sst_celsius') is not None else None,
                    ice_concentration=round(leg['ice_concentration'], 3) if leg.get('ice_concentration') is not None else None,
                ))
            scenario_models.append(SpeedScenarioModel(
                strategy=sc.strategy,
                label=sc.label,
                total_fuel_mt=round(sc.total_fuel_mt, 2),
                total_time_hours=round(sc.total_time_hours, 2),
                total_distance_nm=round(sc.total_distance_nm, 1),
                avg_speed_kts=round(sc.avg_speed_kts, 1),
                speed_profile=[round(s, 1) for s in sc.speed_profile],
                legs=sc_legs,
                fuel_savings_pct=round(sc.fuel_savings_pct, 1),
                time_savings_pct=round(sc.time_savings_pct, 1),
            ))

        return OptimizationResponse(
            waypoints=waypoints,
            total_fuel_mt=round(result.total_fuel_mt, 2),
            total_time_hours=round(result.total_time_hours, 2),
            total_distance_nm=round(result.total_distance_nm, 1),
            direct_fuel_mt=round(result.direct_fuel_mt, 2),
            direct_time_hours=round(result.direct_time_hours, 2),
            fuel_savings_pct=round(result.fuel_savings_pct, 1),
            time_savings_pct=round(result.time_savings_pct, 1),
            legs=legs,
            speed_profile=[round(s, 1) for s in result.speed_profile],
            avg_speed_kts=round(result.avg_speed_kts, 1),
            variable_speed_enabled=result.variable_speed_enabled,
            engine=engine_name,
            safety=safety_summary,
            scenarios=scenario_models,
            baseline_fuel_mt=round(result.baseline_fuel_mt, 2) if result.baseline_fuel_mt else None,
            baseline_time_hours=round(result.baseline_time_hours, 2) if result.baseline_time_hours else None,
            baseline_distance_nm=round(result.baseline_distance_nm, 1) if result.baseline_distance_nm else None,
            weather_provenance=provenance_models,
            temporal_weather=used_temporal,
            optimization_target=request.optimization_target,
            grid_resolution_deg=active_optimizer.resolution_deg,
            cells_explored=result.cells_explored,
            optimization_time_ms=round(result.optimization_time_ms, 1),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Route optimization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@app.get("/api/optimize/status")
async def get_optimization_status():
    """Get current optimizer configuration."""
    return {
        "status": "ready",
        "default_resolution_deg": RouteOptimizer.DEFAULT_RESOLUTION_DEG,
        "default_max_cells": RouteOptimizer.DEFAULT_MAX_CELLS,
        "optimization_targets": ["fuel", "time"],
        "vessel_model": {
            "dwt": _vs.specs.dwt,
            "service_speed_laden": _vs.specs.service_speed_laden,
        }
    }


# ============================================================================
# API Endpoints - Vessel Configuration
# ============================================================================

@app.get("/api/vessel/specs")
async def get_vessel_specs():
    """Get current vessel specifications."""
    specs = _vs.specs
    return {
        "dwt": specs.dwt,
        "loa": specs.loa,
        "beam": specs.beam,
        "draft_laden": specs.draft_laden,
        "draft_ballast": specs.draft_ballast,
        "mcr_kw": specs.mcr_kw,
        "sfoc_at_mcr": specs.sfoc_at_mcr,
        "service_speed_laden": specs.service_speed_laden,
        "service_speed_ballast": specs.service_speed_ballast,
    }


@app.post("/api/vessel/specs", dependencies=[Depends(require_not_demo("Vessel configuration"))])
@limiter.limit(get_rate_limit_string())
async def update_vessel_specs(
    request: Request,
    config: VesselConfig,
    api_key=Depends(get_api_key),
):
    """
    Update vessel specifications.

    Requires authentication via API key.
    """
    try:
        _vs.update_specs({
            'dwt': config.dwt,
            'loa': config.loa,
            'beam': config.beam,
            'draft_laden': config.draft_laden,
            'draft_ballast': config.draft_ballast,
            'mcr_kw': config.mcr_kw,
            'sfoc_at_mcr': config.sfoc_at_mcr,
            'service_speed_laden': config.service_speed_laden,
            'service_speed_ballast': config.service_speed_ballast,
        })

        # Persist to DB so specs survive container restarts
        try:
            _save_vessel_specs_to_db({
                'dwt': config.dwt, 'loa': config.loa, 'beam': config.beam,
                'draft_laden': config.draft_laden, 'draft_ballast': config.draft_ballast,
                'mcr_kw': config.mcr_kw, 'sfoc_at_mcr': config.sfoc_at_mcr,
                'service_speed_laden': config.service_speed_laden,
                'service_speed_ballast': config.service_speed_ballast,
            })
        except Exception as persist_err:
            logger.warning("Failed to persist vessel specs to DB: %s", persist_err)

        return {"status": "success", "message": "Vessel specs updated and persisted"}

    except Exception as e:
        logger.error(f"Failed to update vessel specs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# API Endpoints - Vessel Calibration
# ============================================================================

@app.get("/api/vessel/calibration")
async def get_calibration():
    """Get current vessel calibration factors."""
    cal = _vs.calibration

    if cal is None:
        return {
            "calibrated": False,
            "factors": {
                "calm_water": 1.0,
                "wind": 1.0,
                "waves": 1.0,
                "sfoc_factor": 1.0,
            },
            "message": "No calibration data. Using default theoretical model."
        }

    return {
        "calibrated": True,
        "factors": {
            "calm_water": cal.calm_water,
            "wind": cal.wind,
            "waves": cal.waves,
            "sfoc_factor": cal.sfoc_factor,
        },
        "calibrated_at": cal.calibrated_at.isoformat() if cal.calibrated_at else None,
        "num_reports_used": cal.num_reports_used,
        "calibration_error_mt": cal.calibration_error,
        "days_since_drydock": cal.days_since_drydock,
    }


@app.post("/api/vessel/calibration/set", dependencies=[Depends(require_not_demo("Vessel calibration"))])
@limiter.limit(get_rate_limit_string())
async def set_calibration_factors(
    request: Request,
    factors: CalibrationFactorsModel,
    api_key=Depends(get_api_key),
):
    """
    Manually set calibration factors.

    Requires authentication via API key.
    """
    _vs.update_calibration(CalibrationFactors(
        calm_water=factors.calm_water,
        wind=factors.wind,
        waves=factors.waves,
        sfoc_factor=factors.sfoc_factor,
        calibrated_at=datetime.utcnow(),
        num_reports_used=0,
        days_since_drydock=factors.days_since_drydock,
    ))

    return {"status": "success", "message": "Calibration factors updated"}


@app.get("/api/vessel/noon-reports")
async def get_noon_reports():
    """Get list of uploaded noon reports."""
    return {
        "count": len(_vs.calibrator.noon_reports),
        "reports": [
            {
                "timestamp": r.timestamp.isoformat(),
                "latitude": r.latitude,
                "longitude": r.longitude,
                "speed_kts": r.speed_over_ground_kts,
                "fuel_mt": r.fuel_consumption_mt,
                "period_hours": r.period_hours,
                "is_laden": r.is_laden,
            }
            for r in _vs.calibrator.noon_reports
        ]
    }


@app.post("/api/vessel/noon-reports", dependencies=[Depends(require_not_demo("Noon report upload"))])
@limiter.limit(get_rate_limit_string())
async def add_noon_report(
    request: Request,
    report: NoonReportModel,
    api_key=Depends(get_api_key),
):
    """
    Add a single noon report for calibration.

    Requires authentication via API key.
    """
    nr = NoonReport(
        timestamp=report.timestamp,
        latitude=report.latitude,
        longitude=report.longitude,
        speed_over_ground_kts=report.speed_over_ground_kts,
        speed_through_water_kts=report.speed_through_water_kts,
        fuel_consumption_mt=report.fuel_consumption_mt,
        period_hours=report.period_hours,
        is_laden=report.is_laden,
        heading_deg=report.heading_deg,
        wind_speed_kts=report.wind_speed_kts,
        wind_direction_deg=report.wind_direction_deg,
        wave_height_m=report.wave_height_m,
        wave_direction_deg=report.wave_direction_deg,
        engine_power_kw=report.engine_power_kw,
    )

    _vs.calibrator.add_noon_report(nr)

    return {
        "status": "success",
        "total_reports": len(_vs.calibrator.noon_reports),
    }


@app.post("/api/vessel/noon-reports/upload-csv", dependencies=[Depends(require_not_demo("Noon report upload"))])
@limiter.limit("10/minute")  # Lower rate limit for file uploads
async def upload_noon_reports_csv(
    request: Request,
    file: UploadFile = File(...),
    api_key=Depends(get_api_key),
):
    """
    Upload noon reports from CSV file.

    Requires authentication via API key.
    Maximum file size: 50 MB.

    Expected columns:
    - timestamp (ISO format or common date format)
    - latitude, longitude
    - speed_over_ground_kts
    - fuel_consumption_mt
    - period_hours (optional, default 24)
    - is_laden (optional, default true)
    - wind_speed_kts, wind_direction_deg (optional)
    - wave_height_m, wave_direction_deg (optional)
    - heading_deg (optional)
    """
    try:
        # Read and validate file size
        content = await file.read()
        if len(content) > MAX_CSV_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {MAX_CSV_SIZE_BYTES // (1024*1024)} MB"
            )

        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        # Save to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            # Import from CSV
            count = _vs.calibrator.add_noon_reports_from_csv(tmp_path)
        finally:
            # Cleanup
            tmp_path.unlink()

        return {
            "status": "success",
            "imported": count,
            "total_reports": len(_vs.calibrator.noon_reports),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to import CSV: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")


@app.post("/api/vessel/noon-reports/upload-excel", dependencies=[Depends(require_not_demo("Noon report upload"))])
@limiter.limit("10/minute")
async def upload_noon_reports_excel(
    request: Request,
    file: UploadFile = File(...),
    api_key=Depends(get_api_key),
):
    """
    Upload noon reports from an Excel file (.xlsx/.xls).

    Uses ExcelParser to auto-detect column mappings.
    """
    try:
        content = await file.read()
        if len(content) > MAX_CSV_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {MAX_CSV_SIZE_BYTES // (1024*1024)} MB"
            )

        # Determine suffix from filename
        suffix = ".xlsx"
        if file.filename:
            suffix = Path(file.filename).suffix or ".xlsx"

        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            count = _vs.calibrator.add_noon_reports_from_excel(tmp_path)
        finally:
            tmp_path.unlink()

        return {
            "status": "success",
            "imported": count,
            "total_reports": len(_vs.calibrator.noon_reports),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to import Excel: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to parse Excel: {str(e)}")


@app.delete("/api/vessel/noon-reports", dependencies=[Depends(require_not_demo("Noon report deletion"))])
@limiter.limit(get_rate_limit_string())
async def clear_noon_reports(
    request: Request,
    api_key=Depends(get_api_key),
):
    """
    Clear all uploaded noon reports.

    Requires authentication via API key.
    """
    _vs.calibrator.noon_reports = []
    return {"status": "success", "message": "All noon reports cleared"}


@app.post("/api/vessel/calibrate", response_model=CalibrationResponse, dependencies=[Depends(require_not_demo("Vessel calibration"))])
@limiter.limit("5/minute")  # Lower limit for CPU-intensive operation
async def calibrate_vessel(
    request: Request,
    days_since_drydock: int = Query(0, ge=0, description="Days since last dry dock"),
    api_key=Depends(get_api_key),
):
    """
    Run calibration using uploaded noon reports.

    Requires authentication via API key.

    Finds optimal calibration factors that minimize prediction error
    compared to actual fuel consumption.
    """
    if len(_vs.calibrator.noon_reports) < VesselCalibrator.MIN_REPORTS:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {VesselCalibrator.MIN_REPORTS} noon reports for calibration. "
                   f"Currently have {len(_vs.calibrator.noon_reports)}."
        )

    try:
        result = _vs.calibrator.calibrate(days_since_drydock=days_since_drydock)

        # Apply calibration atomically (rebuilds model, calculators, optimizers)
        _vs.update_calibration(result.factors)

        # Save calibration to file
        _vs.calibrator.save_calibration("default", _vs.calibration)

        return CalibrationResponse(
            factors=CalibrationFactorsModel(
                calm_water=result.factors.calm_water,
                wind=result.factors.wind,
                waves=result.factors.waves,
                sfoc_factor=result.factors.sfoc_factor,
                calibrated_at=result.factors.calibrated_at,
                num_reports_used=result.factors.num_reports_used,
                calibration_error=result.factors.calibration_error,
                days_since_drydock=result.factors.days_since_drydock,
            ),
            reports_used=result.reports_used,
            reports_skipped=result.reports_skipped,
            mean_error_before_mt=result.mean_error_before,
            mean_error_after_mt=result.mean_error_after,
            improvement_pct=result.improvement_pct,
            residuals=result.residuals,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Calibration failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Calibration failed: {str(e)}")


@app.post("/api/vessel/calibration/estimate-fouling", dependencies=[Depends(require_not_demo("Vessel calibration"))])
async def estimate_hull_fouling(
    days_since_drydock: int = Query(..., ge=0),
    operating_regions: List[str] = Query(default=[], description="Operating regions: tropical, warm_temperate, cold, polar"),
):
    """
    Estimate hull fouling factor without calibration data.

    Useful when no noon reports are available but you know
    the vessel's operating history.
    """
    fouling = _vs.calibrator.estimate_hull_fouling(
        days_since_drydock=days_since_drydock,
        operating_regions=operating_regions,
    )

    return {
        "days_since_drydock": days_since_drydock,
        "operating_regions": operating_regions,
        "estimated_fouling_factor": round(fouling, 3),
        "resistance_increase_pct": round((fouling - 1) * 100, 1),
        "note": "This is an estimate. Calibration with actual noon reports is more accurate.",
    }


@app.get("/api/vessel/model-status", tags=["Vessel"])
async def get_vessel_model_status():
    """
    Full vessel model status: all specs, calibration state, computed values.

    Returns every VesselSpecs field grouped by category, current calibration
    factors with timestamp, and derived performance values (optimal speeds,
    daily fuel at service speed).
    """
    specs = _vs.specs
    cal = _vs.calibration
    model = _vs.model

    # Compute optimal speeds and daily fuel at service speed
    optimal_laden = model.get_optimal_speed(is_laden=True)
    optimal_ballast = model.get_optimal_speed(is_laden=False)

    fuel_at_service_laden = model.calculate_fuel_consumption(
        speed_kts=specs.service_speed_laden, is_laden=True, distance_nm=specs.service_speed_laden * 24,
    )
    fuel_at_service_ballast = model.calculate_fuel_consumption(
        speed_kts=specs.service_speed_ballast, is_laden=False, distance_nm=specs.service_speed_ballast * 24,
    )

    return {
        "specifications": {
            "dimensions": {
                "loa": specs.loa,
                "lpp": specs.lpp,
                "beam": specs.beam,
                "draft_laden": specs.draft_laden,
                "draft_ballast": specs.draft_ballast,
                "dwt": specs.dwt,
                "displacement_laden": specs.displacement_laden,
                "displacement_ballast": specs.displacement_ballast,
            },
            "hull_form": {
                "cb_laden": specs.cb_laden,
                "cb_ballast": specs.cb_ballast,
                "wetted_surface_laden": specs.wetted_surface_laden,
                "wetted_surface_ballast": specs.wetted_surface_ballast,
            },
            "engine": {
                "mcr_kw": specs.mcr_kw,
                "sfoc_at_mcr": specs.sfoc_at_mcr,
                "service_speed_laden": specs.service_speed_laden,
                "service_speed_ballast": specs.service_speed_ballast,
            },
            "areas": {
                "frontal_area_laden": specs.frontal_area_laden,
                "frontal_area_ballast": specs.frontal_area_ballast,
                "lateral_area_laden": specs.lateral_area_laden,
                "lateral_area_ballast": specs.lateral_area_ballast,
            },
        },
        "calibration": {
            "calibrated": cal is not None,
            "factors": {
                "calm_water": cal.calm_water if cal else 1.0,
                "wind": cal.wind if cal else 1.0,
                "waves": cal.waves if cal else 1.0,
                "sfoc_factor": cal.sfoc_factor if cal else 1.0,
            },
            "calibrated_at": cal.calibrated_at.isoformat() if cal and cal.calibrated_at else None,
            "num_reports_used": cal.num_reports_used if cal else 0,
            "calibration_error_mt": cal.calibration_error if cal else 0.0,
            "days_since_drydock": cal.days_since_drydock if cal else 0,
        },
        "wave_method": model.wave_method,
        "computed": {
            "optimal_speed_laden_kts": round(optimal_laden, 1),
            "optimal_speed_ballast_kts": round(optimal_ballast, 1),
            "daily_fuel_service_laden_mt": round(fuel_at_service_laden["fuel_mt"], 2),
            "daily_fuel_service_ballast_mt": round(fuel_at_service_ballast["fuel_mt"], 2),
        },
    }


@app.get("/api/vessel/model-curves", tags=["Vessel"])
async def get_vessel_model_curves():
    """
    Pre-computed model curves for frontend charting.

    Returns speed-indexed arrays for resistance, power, SFOC, and fuel
    consumption — both theoretical (calibration=1.0) and with current
    calibration factors applied.
    """
    import numpy as np

    specs = _vs.specs
    model = _vs.model
    cal = _vs.calibration

    # Speed range: 5-16 kts in 0.5 kts steps
    speeds = list(np.arange(5.0, 16.5, 0.5))

    resistance_theoretical = []
    resistance_calibrated = []
    power_kw_list = []
    sfoc_gkwh_list = []
    fuel_mt_per_day_list = []

    for spd in speeds:
        # Theoretical (no calibration)
        speed_ms = spd * 0.51444
        draft = specs.draft_laden
        displacement = specs.displacement_laden
        cb = specs.cb_laden
        ws = specs.wetted_surface_laden
        r_theo = model._holtrop_mennen_resistance(speed_ms, draft, displacement, cb, ws)
        resistance_theoretical.append(round(r_theo / 1000.0, 2))  # kN

        # Calibrated
        r_cal = r_theo * model.calibration_factors.get("calm_water", 1.0)
        resistance_calibrated.append(round(r_cal / 1000.0, 2))  # kN

        # Power and fuel from the full model (uses calibration)
        result = model.calculate_fuel_consumption(
            speed_kts=spd, is_laden=True, distance_nm=spd * 24,
        )
        power_kw_list.append(round(result["power_kw"], 0))

        load = min(result["power_kw"] / specs.mcr_kw, 1.0)
        sfoc = model._sfoc_curve(load)
        sfoc_gkwh_list.append(round(sfoc, 1))

        fuel_mt_per_day_list.append(round(result["fuel_mt"], 2))

    # SFOC vs engine load (15-100%)
    sfoc_loads = list(range(15, 105, 5))
    sfoc_at_loads = []
    sfoc_at_loads_theoretical = []
    for load_pct in sfoc_loads:
        lf = load_pct / 100.0
        # Theoretical (sfoc_factor=1.0)
        if lf < 0.75:
            theo = specs.sfoc_at_mcr * (1.0 + 0.15 * (0.75 - lf))
        else:
            theo = specs.sfoc_at_mcr * (1.0 + 0.05 * (lf - 0.75))
        sfoc_at_loads_theoretical.append(round(theo, 1))
        sfoc_at_loads.append(round(theo * model.calibration_factors.get("sfoc_factor", 1.0), 1))

    return {
        "speed_range_kts": [round(s, 1) for s in speeds],
        "resistance_theoretical_kn": resistance_theoretical,
        "resistance_calibrated_kn": resistance_calibrated,
        "power_kw": power_kw_list,
        "sfoc_gkwh": sfoc_gkwh_list,
        "fuel_mt_per_day": fuel_mt_per_day_list,
        "sfoc_curve": {
            "load_pct": sfoc_loads,
            "sfoc_theoretical_gkwh": sfoc_at_loads_theoretical,
            "sfoc_calibrated_gkwh": sfoc_at_loads,
        },
        "calibration": {
            "calibrated": cal is not None,
            "factors": {
                "calm_water": cal.calm_water if cal else 1.0,
                "wind": cal.wind if cal else 1.0,
                "waves": cal.waves if cal else 1.0,
                "sfoc_factor": cal.sfoc_factor if cal else 1.0,
            },
            "calibrated_at": cal.calibrated_at.isoformat() if cal and cal.calibrated_at else None,
            "num_reports_used": cal.num_reports_used if cal else 0,
            "calibration_error_mt": cal.calibration_error if cal else 0.0,
        },
    }


@app.get("/api/vessel/fuel-scenarios", tags=["Vessel"])
async def get_fuel_scenarios():
    """
    Compute fuel scenarios using the real physics model with current calibration.

    Returns 4 daily fuel scenarios: calm laden, head wind laden,
    rough seas laden, and calm ballast.
    """
    specs = _vs.specs
    model = _vs.model

    # Scenario 1: Calm water laden (24h at service speed)
    distance_calm_laden = specs.service_speed_laden * 24
    calm_laden = model.calculate_fuel_consumption(
        speed_kts=specs.service_speed_laden, is_laden=True, distance_nm=distance_calm_laden,
    )

    # Scenario 2: Head wind laden (20 kt = 10.3 m/s head wind)
    headwind_wx = {
        "wind_speed_ms": 10.3, "wind_dir_deg": 0, "heading_deg": 0,
        "sig_wave_height_m": 0.5, "wave_dir_deg": 0,
    }
    headwind_laden = model.calculate_fuel_consumption(
        speed_kts=specs.service_speed_laden, is_laden=True,
        weather=headwind_wx, distance_nm=distance_calm_laden,
    )

    # Scenario 3: Rough seas laden (3m waves head seas)
    roughsea_wx = {
        "wind_speed_ms": 8.0, "wind_dir_deg": 0, "heading_deg": 0,
        "sig_wave_height_m": 3.0, "wave_dir_deg": 0,
    }
    roughsea_laden = model.calculate_fuel_consumption(
        speed_kts=specs.service_speed_laden, is_laden=True,
        weather=roughsea_wx, distance_nm=distance_calm_laden,
    )

    # Scenario 4: Calm water ballast
    distance_calm_ballast = specs.service_speed_ballast * 24
    calm_ballast = model.calculate_fuel_consumption(
        speed_kts=specs.service_speed_ballast, is_laden=False, distance_nm=distance_calm_ballast,
    )

    scenarios = [
        {
            "name": "Calm Water (Laden)",
            "conditions": f"{specs.service_speed_laden} kts, no wind/waves",
            "fuel_mt": round(calm_laden["fuel_mt"], 2),
            "power_kw": round(calm_laden["power_kw"], 0),
        },
        {
            "name": "Head Wind (Laden)",
            "conditions": f"{specs.service_speed_laden} kts, 20 kt head wind",
            "fuel_mt": round(headwind_laden["fuel_mt"], 2),
            "power_kw": round(headwind_laden["power_kw"], 0),
        },
        {
            "name": "Rough Seas (Laden)",
            "conditions": f"{specs.service_speed_laden} kts, 3m waves",
            "fuel_mt": round(roughsea_laden["fuel_mt"], 2),
            "power_kw": round(roughsea_laden["power_kw"], 0),
        },
        {
            "name": "Calm Water (Ballast)",
            "conditions": f"{specs.service_speed_ballast} kts, no wind/waves",
            "fuel_mt": round(calm_ballast["fuel_mt"], 2),
            "power_kw": round(calm_ballast["power_kw"], 0),
        },
    ]

    return {"scenarios": scenarios}


@app.post("/api/vessel/predict", tags=["Vessel"])
async def predict_vessel_performance(req: PerformancePredictionRequest):
    """
    Predict vessel speed and fuel consumption under given conditions.

    Two modes:
    - **engine_load_pct**: Find achievable speed at given power + weather
    - **calm_speed_kts**: Find what happens to a target speed in weather
      (power required, actual STW if MCR exceeded, fuel burn)

    All directions are relative to bow (0=ahead, 90=beam, 180=astern).
    """
    model = _vs.model

    # Convert relative directions to absolute with heading=0
    heading = 0.0
    wind_abs = req.wind_relative_deg
    wave_abs = req.wave_relative_deg

    weather = None
    if req.wind_speed_kts > 0 or req.wave_height_m > 0:
        weather = {
            "wind_speed_ms": req.wind_speed_kts * 0.51444,
            "wind_dir_deg": wind_abs,
            "sig_wave_height_m": req.wave_height_m,
            "wave_dir_deg": wave_abs,
        }

    current_ms = req.current_speed_kts * 0.51444
    # Current: relative 0° = head current (opposing) → flowing toward 180°
    current_abs = (180.0 + req.current_relative_deg) % 360

    # Mode 2: calm_speed_kts — calculate fuel at this speed in given weather
    if req.calm_speed_kts is not None:
        stw = req.calm_speed_kts
        distance_24h = stw * 24
        r = model.calculate_fuel_consumption(stw, req.is_laden, weather, distance_nm=distance_24h)

        # Check if MCR is exceeded
        mcr_exceeded = bool(r["required_power_kw"] > model.specs.mcr_kw)
        required_power_raw = float(r["required_power_kw"])
        actual_load_pct = float(min(r["required_power_kw"], model.specs.mcr_kw) / model.specs.mcr_kw * 100)
        sfoc = float(model._sfoc_curve(actual_load_pct / 100))

        # If MCR exceeded, find achievable speed at 100% MCR
        if mcr_exceeded:
            capped = model.predict_performance(
                is_laden=req.is_laden, weather=weather, engine_load_pct=100.0,
                current_speed_ms=current_ms, current_dir_deg=current_abs, heading_deg=heading,
            )
            stw = capped["stw_kts"]
            # Recalculate at capped speed
            r = model.calculate_fuel_consumption(stw, req.is_laden, weather, distance_nm=stw * 24)

        # Current effect
        import math as _math
        current_effect_kts = 0.0
        if current_ms > 0:
            rel_angle = _math.radians(current_abs - heading)
            current_effect_kts = float((current_ms / 0.51444) * _math.cos(rel_angle))
        sog = max(0.0, stw + current_effect_kts)

        # Speed loss from weather
        speed_loss_pct = float((req.calm_speed_kts - stw) / req.calm_speed_kts * 100) if req.calm_speed_kts > 0 else 0.0

        # Sanitise resistance_breakdown_kn (numpy → native float)
        rb = r["resistance_breakdown_kn"]
        rb_clean = {k: round(float(v), 4) for k, v in rb.items()}

        result = {
            "stw_kts": round(float(stw), 2),
            "sog_kts": round(float(sog), 2),
            "fuel_per_day_mt": round(float(r["fuel_mt"]), 3),
            "fuel_per_nm_mt": round(float(r["fuel_mt"]) / (sog * 24), 4) if sog > 0 else 0.0,
            "power_kw": round(float(min(r["required_power_kw"], model.specs.mcr_kw)), 0),
            "required_power_kw": round(float(required_power_raw), 0),
            "load_pct": round(actual_load_pct, 1),
            "sfoc_gkwh": round(sfoc, 1),
            "mcr_exceeded": mcr_exceeded,
            "resistance_breakdown_kn": rb_clean,
            "speed_loss_from_weather_pct": round(max(0.0, speed_loss_pct), 1),
            "calm_water_speed_kts": req.calm_speed_kts,
            "current_effect_kts": round(current_effect_kts, 2),
            "service_speed_kts": float(model.specs.service_speed_laden if req.is_laden else model.specs.service_speed_ballast),
            "mode": "calm_speed",
            "inputs": {
                "calm_speed_kts": req.calm_speed_kts,
                "wind_relative_deg": req.wind_relative_deg,
                "wave_relative_deg": req.wave_relative_deg,
                "current_relative_deg": req.current_relative_deg,
            },
        }
        return result

    # Mode 1: engine_load_pct (default 85%)
    load = req.engine_load_pct if req.engine_load_pct is not None else 85.0

    result = model.predict_performance(
        is_laden=req.is_laden,
        weather=weather,
        engine_load_pct=load,
        current_speed_ms=current_ms,
        current_dir_deg=current_abs,
        heading_deg=heading,
    )

    result["mode"] = "engine_load"
    result["inputs"] = {
        "engine_load_pct": load,
        "wind_relative_deg": req.wind_relative_deg,
        "wave_relative_deg": req.wave_relative_deg,
        "current_relative_deg": req.current_relative_deg,
    }

    return result


# ============================================================================
# API Endpoints - Regulatory Zones
# ============================================================================

@app.get("/api/zones")
async def get_all_zones():
    """
    Get all regulatory zones (built-in and custom).

    Returns GeoJSON FeatureCollection for map display.
    """
    zone_checker = get_zone_checker()
    return zone_checker.export_geojson()


@app.get("/api/zones/list")
async def list_zones():
    """Get zones as a simple list."""
    zone_checker = get_zone_checker()
    zones = []
    for zone in zone_checker.get_all_zones():
        zones.append({
            "id": zone.id,
            "name": zone.properties.name,
            "zone_type": zone.properties.zone_type.value,
            "interaction": zone.properties.interaction.value,
            "penalty_factor": zone.properties.penalty_factor,
            "is_builtin": zone.is_builtin,
        })
    return {"zones": zones, "count": len(zones)}


@app.get("/api/zones/{zone_id}")
async def get_zone(zone_id: str):
    """Get a specific zone by ID."""
    zone_checker = get_zone_checker()
    zone = zone_checker.get_zone(zone_id)

    if zone is None:
        raise HTTPException(status_code=404, detail=f"Zone not found: {zone_id}")

    return zone.to_geojson()


@app.post("/api/zones", response_model=ZoneResponse, dependencies=[Depends(require_not_demo("Zone creation"))])
@limiter.limit(get_rate_limit_string())
async def create_zone(
    http_request: Request,
    request: CreateZoneRequest,
    api_key=Depends(get_api_key),
):
    """
    Create a custom zone.

    Requires authentication via API key.

    Coordinates should be provided as a list of {lat, lon} objects
    forming a closed polygon (first and last point should match).
    """
    import uuid

    zone_checker = get_zone_checker()

    # Validate zone type
    try:
        zone_type = ZoneType(request.zone_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid zone_type. Valid values: {[t.value for t in ZoneType]}"
        )

    # Validate interaction
    try:
        interaction = ZoneInteraction(request.interaction)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid interaction. Valid values: {[i.value for i in ZoneInteraction]}"
        )

    # Convert coordinates
    coords = [(c.lat, c.lon) for c in request.coordinates]

    # Ensure polygon is closed
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    # Create zone
    zone_id = f"custom_{uuid.uuid4().hex[:8]}"
    zone = Zone(
        id=zone_id,
        properties=ZoneProperties(
            name=request.name,
            zone_type=zone_type,
            interaction=interaction,
            penalty_factor=request.penalty_factor,
            notes=request.notes,
        ),
        coordinates=coords,
        is_builtin=False,
    )

    zone_checker.add_zone(zone)

    return ZoneResponse(
        id=zone.id,
        name=zone.properties.name,
        zone_type=zone.properties.zone_type.value,
        interaction=zone.properties.interaction.value,
        penalty_factor=zone.properties.penalty_factor,
        is_builtin=zone.is_builtin,
        coordinates=[ZoneCoordinate(lat=c[0], lon=c[1]) for c in zone.coordinates],
        notes=zone.properties.notes,
    )


@app.delete("/api/zones/{zone_id}", dependencies=[Depends(require_not_demo("Zone deletion"))])
@limiter.limit(get_rate_limit_string())
async def delete_zone(
    request: Request,
    zone_id: str,
    api_key=Depends(get_api_key),
):
    """
    Delete a custom zone.

    Requires authentication via API key.
    Built-in zones cannot be deleted.
    """
    zone_checker = get_zone_checker()

    zone = zone_checker.get_zone(zone_id)
    if zone is None:
        raise HTTPException(status_code=404, detail=f"Zone not found: {zone_id}")

    if zone.is_builtin:
        raise HTTPException(status_code=400, detail="Cannot delete built-in zones")

    zone_checker.remove_zone(zone_id)
    return {"status": "deleted", "zone_id": zone_id}


@app.get("/api/zones/at-point")
async def get_zones_at_point(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
):
    """Get all zones that contain a specific point."""
    zone_checker = get_zone_checker()
    zones = zone_checker.get_zones_at_point(lat, lon)

    return {
        "position": {"lat": lat, "lon": lon},
        "zones": [
            {
                "id": z.id,
                "name": z.properties.name,
                "zone_type": z.properties.zone_type.value,
                "interaction": z.properties.interaction.value,
                "penalty_factor": z.properties.penalty_factor,
            }
            for z in zones
        ]
    }


@app.get("/api/zones/check-path")
async def check_path_zones(
    lat1: float = Query(..., ge=-90, le=90),
    lon1: float = Query(..., ge=-180, le=180),
    lat2: float = Query(..., ge=-90, le=90),
    lon2: float = Query(..., ge=-180, le=180),
):
    """Check which zones a path segment crosses."""
    zone_checker = get_zone_checker()
    zones_by_type = zone_checker.check_path_zones(lat1, lon1, lat2, lon2)
    penalty, warnings = zone_checker.get_path_penalty(lat1, lon1, lat2, lon2)

    return {
        "path": {
            "from": {"lat": lat1, "lon": lon1},
            "to": {"lat": lat2, "lon": lon2},
        },
        "zones": {
            interaction: [
                {"id": z.id, "name": z.properties.name}
                for z in zones
            ]
            for interaction, zones in zones_by_type.items()
        },
        "penalty_factor": penalty if penalty != float('inf') else None,
        "is_forbidden": penalty == float('inf'),
        "warnings": warnings,
    }


# ============================================================================
# Weather Startup & Cache Cleanup
# ============================================================================

# Set of background asyncio tasks — prevents garbage collection (used by forecast prefetch)
_background_tasks: set = set()


@app.on_event("startup")
async def startup_event():
    """Run migrations, load persisted vessel specs, and start background weather ingestion."""
    _run_weather_migrations()

    # Ensure cache dirs exist (volume mounts may lack subdirectories)
    for sub in ("wind", "wave", "current", "ice", "sst", "vis"):
        Path(f"/tmp/windmar_cache/{sub}").mkdir(parents=True, exist_ok=True)

    # Load persisted vessel specs from DB (survives container restarts)
    try:
        saved_specs = _load_vessel_specs_from_db()
        if saved_specs is not None:
            _vs.update_specs(saved_specs)
            logger.info("Vessel specs loaded from DB: %s kW / %s kts",
                        saved_specs.get("mcr_kw"), saved_specs.get("service_speed_laden"))
    except Exception as e:
        logger.warning("Could not load vessel specs from DB (using defaults): %s", e)

    # Auto-load saved calibration from disk (survives container restarts)
    try:
        _saved_cal = _vs.calibrator.load_calibration("default")
        if _saved_cal is not None:
            _vs.update_calibration(_saved_cal)
            logger.info(
                "Auto-loaded calibration: calm_water=%.4f, sfoc_factor=%.4f, reports=%d",
                _saved_cal.calm_water, _saved_cal.sfoc_factor, _saved_cal.num_reports_used,
            )
    except Exception as e:
        logger.warning("Could not auto-load calibration (using theoretical): %s", e)

    logger.info("Startup complete")


def _cleanup_stale_caches():
    """Delete stale CMEMS/GFS cache files to reclaim disk space."""
    import time as _time

    now = _time.time()
    cleaned = 0

    # CMEMS .nc files older than 24h
    cache_dir = Path("data/copernicus_cache")
    if cache_dir.exists():
        for f in cache_dir.glob("*.nc"):
            try:
                if now - f.stat().st_mtime > 24 * 3600:
                    f.unlink()
                    cleaned += 1
            except OSError:
                pass

    # GFS .grib2 files older than 48h
    if cache_dir.exists():
        for f in cache_dir.glob("*.grib2"):
            try:
                if now - f.stat().st_mtime > 48 * 3600:
                    f.unlink()
                    cleaned += 1
            except OSError:
                pass

    # JSON frame cache in /tmp/windmar_cache/ older than 12h
    tmp_cache = Path("/tmp/windmar_cache")
    if tmp_cache.exists():
        for f in tmp_cache.rglob("*.json"):
            try:
                if now - f.stat().st_mtime > 12 * 3600:
                    f.unlink()
                    cleaned += 1
            except OSError:
                pass

    if cleaned > 0:
        logger.info(f"Cache cleanup: removed {cleaned} stale files")


# ============================================================================
# CII Compliance API
# ============================================================================


def _resolve_vessel_type(name: str) -> CIIVesselType:
    """Resolve vessel type string to enum."""
    mapping = {vt.value: vt for vt in CIIVesselType}
    if name in mapping:
        return mapping[name]
    raise HTTPException(status_code=400, detail=f"Unknown vessel type: {name}. Valid: {list(mapping.keys())}")


def _resolve_target_rating(name: str) -> CIIRating:
    """Resolve rating string to enum."""
    mapping = {r.value: r for r in CIIRating}
    if name.upper() in mapping:
        return mapping[name.upper()]
    raise HTTPException(status_code=400, detail=f"Unknown rating: {name}. Valid: A, B, C, D, E")


def _compliance_status(rating: CIIRating) -> str:
    if rating in (CIIRating.A, CIIRating.B):
        return "Compliant"
    elif rating == CIIRating.C:
        return "At Risk"
    else:
        return "Non-Compliant"


@app.get("/api/cii/vessel-types", tags=["CII Compliance"])
async def get_cii_vessel_types():
    """List available IMO vessel type categories for CII calculations."""
    vessel_types = [
        {"id": vt.value, "name": vt.value.replace("_", " ").title()}
        for vt in CIIVesselType
    ]
    return {"vessel_types": vessel_types}


@app.get("/api/cii/fuel-types", tags=["CII Compliance"])
async def get_cii_fuel_types():
    """List available fuel types and their CO2 emission factors."""
    fuel_types = [
        {"id": fuel, "name": fuel.upper().replace("_", " "), "co2_factor": factor}
        for fuel, factor in CIICalculator.CO2_FACTORS.items()
    ]
    return {"fuel_types": fuel_types}


@app.post("/api/cii/calculate", tags=["CII Compliance"])
async def calculate_cii(request: CIICalculateRequest):
    """Calculate CII rating for given fuel consumption and distance."""
    vtype = _resolve_vessel_type(request.vessel_type)
    gt = request.gt if vtype in (CIIVesselType.CRUISE_PASSENGER, CIIVesselType.RO_RO_PASSENGER) else None

    try:
        calc = CIICalculator(vessel_type=vtype, dwt=request.dwt, year=request.year, gt=gt)
        result = calc.calculate(request.fuel_consumption_mt.to_dict(), request.total_distance_nm)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "year": result.year,
        "rating": result.rating.value,
        "compliance_status": _compliance_status(result.rating),
        "attained_cii": result.attained_cii,
        "required_cii": result.required_cii,
        "rating_boundaries": result.rating_boundaries,
        "reduction_factor": result.reduction_factor,
        "total_co2_mt": result.total_co2_mt,
        "total_distance_nm": result.total_distance_nm,
        "capacity": result.capacity,
        "vessel_type": result.vessel_type.value,
        "margin_to_downgrade": result.margin_to_downgrade,
        "margin_to_upgrade": result.margin_to_upgrade,
    }


@app.post("/api/cii/project", tags=["CII Compliance"])
async def project_cii(request: CIIProjectRequest):
    """Project CII rating across multiple years with optional efficiency improvements."""
    vtype = _resolve_vessel_type(request.vessel_type)
    gt = request.gt if vtype in (CIIVesselType.CRUISE_PASSENGER, CIIVesselType.RO_RO_PASSENGER) else None

    try:
        calc = CIICalculator(vessel_type=vtype, dwt=request.dwt, year=request.start_year, gt=gt)
        years = list(range(request.start_year, request.end_year + 1))
        projections = calc.project_rating(
            request.annual_fuel_mt.to_dict(),
            request.annual_distance_nm,
            years=years,
            fuel_reduction_rate=request.fuel_efficiency_improvement_pct,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    proj_list = [
        {
            "year": p.year,
            "rating": p.rating.value,
            "attained_cii": p.attained_cii,
            "required_cii": p.required_cii,
            "reduction_factor": p.reduction_factor,
            "status": p.status,
        }
        for p in projections
    ]

    # Build summary
    current_rating = projections[0].rating.value if projections else "?"
    final_rating = projections[-1].rating.value if projections else "?"
    years_until_d = next(
        (p.year - projections[0].year for p in projections if p.rating in (CIIRating.D, CIIRating.E)),
        "N/A"
    )
    years_until_e = next(
        (p.year - projections[0].year for p in projections if p.rating == CIIRating.E),
        "N/A"
    )

    if final_rating in ("D", "E"):
        recommendation = f"Action required: rating degrades to {final_rating} by {projections[-1].year}."
    elif final_rating == "C":
        recommendation = "Borderline: rating reaches C. Consider efficiency improvements."
    else:
        recommendation = f"On track: rating remains {final_rating} through {projections[-1].year}."

    return {
        "projections": proj_list,
        "summary": {
            "current_rating": current_rating,
            "final_rating": final_rating,
            "years_until_d_rating": years_until_d,
            "years_until_e_rating": years_until_e,
            "recommendation": recommendation,
        },
    }


@app.post("/api/cii/reduction", tags=["CII Compliance"])
async def calculate_cii_reduction(request: CIIReductionRequest):
    """Calculate fuel reduction needed to achieve a target CII rating."""
    vtype = _resolve_vessel_type(request.vessel_type)
    target = _resolve_target_rating(request.target_rating)
    gt = request.gt if vtype in (CIIVesselType.CRUISE_PASSENGER, CIIVesselType.RO_RO_PASSENGER) else None

    try:
        calc = CIICalculator(vessel_type=vtype, dwt=request.dwt, year=request.target_year, gt=gt)
        result = calc.calculate_required_reduction(
            request.current_fuel_mt.to_dict(),
            request.current_distance_nm,
            target_rating=target,
            target_year=request.target_year,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return result


# ============================================================================
# Weather Freshness
# ============================================================================


@app.get("/api/weather/freshness", tags=["Weather"])
async def get_weather_freshness():
    """Get weather data freshness indicator (age of most recent data)."""
    if is_demo():
        return demo_mode_response("Weather freshness")

    if db_weather is None:
        return {
            "status": "unavailable",
            "message": "Weather database not configured",
            "age_hours": None,
            "color": "red",
        }

    freshness = db_weather.get_freshness()
    if freshness is None:
        return {
            "status": "no_data",
            "message": "No weather data ingested yet",
            "age_hours": None,
            "color": "red",
        }

    age_hours = freshness.get("age_hours", None) if isinstance(freshness, dict) else None
    if age_hours is not None:
        if age_hours < 4:
            color = "green"
        elif age_hours < 12:
            color = "yellow"
        else:
            color = "red"
    else:
        color = "red"

    return {
        "status": "ok",
        "age_hours": age_hours,
        "color": color,
        **(freshness if isinstance(freshness, dict) else {"raw": freshness}),
    }


# ============================================================================
# Engine Log Ingestion
# ============================================================================

from api.database import get_db, get_db_context
from api.models import EngineLogEntry, VesselSpec
from src.database.engine_log_parser import EngineLogParser


# ============================================================================
# Vessel Specs DB Persistence
# ============================================================================

def _save_vessel_specs_to_db(specs_dict: dict) -> None:
    """Persist vessel specs to the vessel_specs table (upsert by name='default')."""
    _d = VesselSpecs  # canonical defaults
    with get_db_context() as db:
        row = db.query(VesselSpec).filter(VesselSpec.name == "default").first()
        dwt = specs_dict.get("dwt", _d.dwt)
        speed = specs_dict.get("service_speed_laden", _d.service_speed_laden)
        vals = {
            "name": "default",
            "length": specs_dict.get("loa", _d.loa),
            "beam": specs_dict.get("beam", _d.beam),
            "draft": specs_dict.get("draft_laden", _d.draft_laden),
            "deadweight": dwt,
            "displacement": dwt * 1.33,
            "block_coefficient": _d.cb_laden,
            "engine_power": specs_dict.get("mcr_kw", _d.mcr_kw),
            "service_speed": speed,
            "max_speed": speed + 2.0,
        }
        extra = {
            "draft_ballast": specs_dict.get("draft_ballast"),
            "sfoc_at_mcr": specs_dict.get("sfoc_at_mcr"),
            "service_speed_ballast": specs_dict.get("service_speed_ballast"),
        }
        if row is None:
            row = VesselSpec(**vals, extra_metadata=extra)
            db.add(row)
        else:
            for k, v in vals.items():
                if k != "name":
                    setattr(row, k, v)
            row.extra_metadata = extra
            row.updated_at = datetime.utcnow()
        logger.info("Vessel specs persisted to DB (name='default')")


def _load_vessel_specs_from_db() -> Optional[dict]:
    """Load vessel specs from DB. Returns dict for VesselSpecs() or None."""
    with get_db_context() as db:
        row = db.query(VesselSpec).filter(VesselSpec.name == "default").first()
        if row is None:
            return None
        extra = row.extra_metadata or {}
        return {
            "loa": row.length,
            "beam": row.beam,
            "draft_laden": row.draft,
            "dwt": row.deadweight,
            "mcr_kw": row.engine_power,
            "service_speed_laden": row.service_speed,
            "draft_ballast": extra.get("draft_ballast", 6.5),
            "sfoc_at_mcr": extra.get("sfoc_at_mcr", 171.0),
            "service_speed_ballast": extra.get("service_speed_ballast", 13.0),
        }


MAX_EXCEL_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


@app.post("/api/engine-log/upload", response_model=EngineLogUploadResponse, tags=["Engine Log"], dependencies=[Depends(require_not_demo("Engine log upload"))])
@limiter.limit("10/minute")
async def upload_engine_log(
    request: Request,
    file: UploadFile = File(...),
    vessel_id: Optional[str] = Query(None, description="Vessel UUID to link entries"),
    sheet_name: Optional[str] = Query(None, description="Sheet name (default: E log)"),
    api_key=Depends(get_api_key),
    db=Depends(get_db),
):
    """Upload and parse an engine log Excel workbook."""
    import tempfile
    import uuid as uuid_mod

    content = await file.read()
    if len(content) > MAX_EXCEL_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum: {MAX_EXCEL_UPLOAD_BYTES // (1024 * 1024)} MB")
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    suffix = ".xlsx"
    if file.filename:
        suffix = Path(file.filename).suffix or ".xlsx"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        parser = EngineLogParser(tmp_path)
        entries = parser.parse(sheet_name=sheet_name)
    except (ValueError, FileNotFoundError) as e:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        logger.error(f"Engine log parse error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Parse error: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    if not entries:
        raise HTTPException(status_code=400, detail="No valid entries found in file")

    batch_id = uuid_mod.uuid4()
    vessel_uuid = uuid_mod.UUID(vessel_id) if vessel_id else None

    db_entries = []
    for entry in entries:
        db_entry = EngineLogEntry(
            vessel_id=vessel_uuid,
            timestamp=entry["timestamp"],
            lapse_hours=entry.get("lapse_hours"),
            place=entry.get("place"),
            event=entry.get("event"),
            rpm=entry.get("rpm"),
            engine_distance=entry.get("engine_distance"),
            speed_stw=entry.get("speed_stw"),
            me_power_kw=entry.get("me_power_kw"),
            me_load_pct=entry.get("me_load_pct"),
            me_fuel_index_pct=entry.get("me_fuel_index_pct"),
            shaft_power=entry.get("shaft_power"),
            shaft_torque_knm=entry.get("shaft_torque_knm"),
            slip_pct=entry.get("slip_pct"),
            hfo_me_mt=entry.get("hfo_me_mt"),
            hfo_ae_mt=entry.get("hfo_ae_mt"),
            hfo_boiler_mt=entry.get("hfo_boiler_mt"),
            hfo_total_mt=entry.get("hfo_total_mt"),
            mgo_me_mt=entry.get("mgo_me_mt"),
            mgo_ae_mt=entry.get("mgo_ae_mt"),
            mgo_total_mt=entry.get("mgo_total_mt"),
            methanol_me_mt=entry.get("methanol_me_mt"),
            rob_vlsfo_mt=entry.get("rob_vlsfo_mt"),
            rob_mgo_mt=entry.get("rob_mgo_mt"),
            rob_methanol_mt=entry.get("rob_methanol_mt"),
            rh_me=entry.get("rh_me"),
            rh_ae_total=entry.get("rh_ae_total"),
            tc_rpm=entry.get("tc_rpm"),
            scav_air_press_bar=entry.get("scav_air_press_bar"),
            fuel_temp_c=entry.get("fuel_temp_c"),
            sw_temp_c=entry.get("sw_temp_c"),
            upload_batch_id=batch_id,
            source_sheet=entry.get("source_sheet"),
            source_file=file.filename or entry.get("source_file"),
            extended_data=entry.get("extended_data"),
        )
        db_entries.append(db_entry)

    db.add_all(db_entries)
    db.commit()

    stats = parser.get_statistics()

    return EngineLogUploadResponse(
        status="success",
        batch_id=str(batch_id),
        imported=len(db_entries),
        skipped=0,
        date_range=stats.get("date_range"),
        events_summary=stats.get("events_breakdown"),
    )


@app.get("/api/engine-log/entries", response_model=List[EngineLogEntryResponse], tags=["Engine Log"])
async def get_engine_log_entries(
    event: Optional[str] = Query(None, description="Filter by event type"),
    date_from: Optional[datetime] = Query(None, description="Start date"),
    date_to: Optional[datetime] = Query(None, description="End date"),
    min_rpm: Optional[float] = Query(None, ge=0, description="Minimum RPM"),
    batch_id: Optional[str] = Query(None, description="Filter by batch UUID"),
    limit: int = Query(100, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    db=Depends(get_db),
):
    """Query engine log entries with optional filters and pagination."""
    import uuid as uuid_mod

    query = db.query(EngineLogEntry)
    if event:
        query = query.filter(EngineLogEntry.event == event.upper())
    if date_from:
        query = query.filter(EngineLogEntry.timestamp >= date_from)
    if date_to:
        query = query.filter(EngineLogEntry.timestamp <= date_to)
    if min_rpm is not None:
        query = query.filter(EngineLogEntry.rpm >= min_rpm)
    if batch_id:
        query = query.filter(EngineLogEntry.upload_batch_id == uuid_mod.UUID(batch_id))

    entries = query.order_by(EngineLogEntry.timestamp).offset(offset).limit(limit).all()

    return [
        EngineLogEntryResponse(
            id=str(e.id), timestamp=e.timestamp, lapse_hours=e.lapse_hours,
            place=e.place, event=e.event, rpm=e.rpm, engine_distance=e.engine_distance,
            speed_stw=e.speed_stw, me_power_kw=e.me_power_kw, me_load_pct=e.me_load_pct,
            me_fuel_index_pct=e.me_fuel_index_pct, shaft_power=e.shaft_power,
            shaft_torque_knm=e.shaft_torque_knm, slip_pct=e.slip_pct,
            hfo_me_mt=e.hfo_me_mt, hfo_ae_mt=e.hfo_ae_mt, hfo_boiler_mt=e.hfo_boiler_mt,
            hfo_total_mt=e.hfo_total_mt, mgo_me_mt=e.mgo_me_mt, mgo_ae_mt=e.mgo_ae_mt,
            mgo_total_mt=e.mgo_total_mt, methanol_me_mt=e.methanol_me_mt,
            rob_vlsfo_mt=e.rob_vlsfo_mt, rob_mgo_mt=e.rob_mgo_mt,
            rob_methanol_mt=e.rob_methanol_mt, rh_me=e.rh_me, rh_ae_total=e.rh_ae_total,
            tc_rpm=e.tc_rpm, scav_air_press_bar=e.scav_air_press_bar,
            fuel_temp_c=e.fuel_temp_c, sw_temp_c=e.sw_temp_c,
            upload_batch_id=str(e.upload_batch_id), source_sheet=e.source_sheet,
            source_file=e.source_file, extended_data=e.extended_data,
        )
        for e in entries
    ]


@app.get("/api/engine-log/summary", response_model=EngineLogSummaryResponse, tags=["Engine Log"])
async def get_engine_log_summary(
    batch_id: Optional[str] = Query(None, description="Filter by batch UUID"),
    db=Depends(get_db),
):
    """Get aggregated summary statistics from engine log entries."""
    import uuid as uuid_mod
    from sqlalchemy import func

    query = db.query(EngineLogEntry)
    if batch_id:
        query = query.filter(EngineLogEntry.upload_batch_id == uuid_mod.UUID(batch_id))

    total = query.count()
    if total == 0:
        return EngineLogSummaryResponse(total_entries=0)

    date_q = db.query(func.min(EngineLogEntry.timestamp), func.max(EngineLogEntry.timestamp))
    if batch_id:
        date_q = date_q.filter(EngineLogEntry.upload_batch_id == uuid_mod.UUID(batch_id))
    min_ts, max_ts = date_q.one()

    event_q = db.query(EngineLogEntry.event, func.count(EngineLogEntry.id)).group_by(EngineLogEntry.event)
    if batch_id:
        event_q = event_q.filter(EngineLogEntry.upload_batch_id == uuid_mod.UUID(batch_id))
    events_breakdown = {ev or "UNKNOWN": cnt for ev, cnt in event_q.all()}

    fuel_q = db.query(func.sum(EngineLogEntry.hfo_total_mt), func.sum(EngineLogEntry.mgo_total_mt), func.sum(EngineLogEntry.methanol_me_mt))
    if batch_id:
        fuel_q = fuel_q.filter(EngineLogEntry.upload_batch_id == uuid_mod.UUID(batch_id))
    hfo_sum, mgo_sum, meth_sum = fuel_q.one()

    rpm_q = db.query(func.avg(EngineLogEntry.rpm)).filter(EngineLogEntry.event == "NOON", EngineLogEntry.rpm > 0)
    if batch_id:
        rpm_q = rpm_q.filter(EngineLogEntry.upload_batch_id == uuid_mod.UUID(batch_id))
    avg_rpm = rpm_q.scalar()

    spd_q = db.query(func.avg(EngineLogEntry.speed_stw)).filter(EngineLogEntry.event == "NOON", EngineLogEntry.speed_stw > 0)
    if batch_id:
        spd_q = spd_q.filter(EngineLogEntry.upload_batch_id == uuid_mod.UUID(batch_id))
    avg_speed = spd_q.scalar()

    batch_q = db.query(
        EngineLogEntry.upload_batch_id, func.count(EngineLogEntry.id),
        func.min(EngineLogEntry.timestamp), func.max(EngineLogEntry.timestamp),
        func.min(EngineLogEntry.source_file),
    ).group_by(EngineLogEntry.upload_batch_id)
    batches = [
        {"batch_id": str(bid), "count": cnt,
         "date_start": ds.isoformat() if ds else None,
         "date_end": de.isoformat() if de else None, "source_file": sf}
        for bid, cnt, ds, de, sf in batch_q.all()
    ]

    return EngineLogSummaryResponse(
        total_entries=total,
        date_range={"start": min_ts.isoformat() if min_ts else None, "end": max_ts.isoformat() if max_ts else None},
        events_breakdown=events_breakdown,
        fuel_summary={"hfo_mt": round(float(hfo_sum or 0), 3), "mgo_mt": round(float(mgo_sum or 0), 3), "methanol_mt": round(float(meth_sum or 0), 3)},
        avg_rpm_at_sea=round(float(avg_rpm), 1) if avg_rpm else None,
        avg_speed_stw=round(float(avg_speed), 2) if avg_speed else None,
        batches=batches,
    )


@app.delete("/api/engine-log/batch/{batch_id}", tags=["Engine Log"], dependencies=[Depends(require_not_demo("Engine log deletion"))])
@limiter.limit(get_rate_limit_string())
async def delete_engine_log_batch(
    request: Request,
    batch_id: str,
    api_key=Depends(get_api_key),
    db=Depends(get_db),
):
    """Delete all engine log entries for a given upload batch."""
    import uuid as uuid_mod

    try:
        bid = uuid_mod.UUID(batch_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid batch_id UUID format")

    count = db.query(EngineLogEntry).filter(EngineLogEntry.upload_batch_id == bid).delete()
    db.commit()

    if count == 0:
        raise HTTPException(status_code=404, detail=f"No entries found for batch {batch_id}")

    return {"status": "deleted", "batch_id": batch_id, "deleted_count": count}


# ============================================================================
# Engine Log → Calibration Bridge
# ============================================================================

@app.post("/api/engine-log/calibrate", response_model=EngineLogCalibrateResponse, tags=["Engine Log"], dependencies=[Depends(require_not_demo("Engine log calibration"))])
@limiter.limit("5/minute")
async def calibrate_from_engine_log(
    request: Request,
    batch_id: Optional[str] = Query(None, description="Filter to specific upload batch"),
    days_since_drydock: int = Query(0, ge=0, description="Days since last dry dock"),
    api_key=Depends(get_api_key),
    db=Depends(get_db),
):
    """
    Calibrate vessel model from engine log NOON entries.

    Converts NOON entries (speed, fuel, power, RPM) into NoonReport objects,
    feeds them to VesselCalibrator.calibrate(), and applies the resulting
    factors to the vessel model.
    """
    import uuid as uuid_mod

    # Query NOON entries
    query = db.query(EngineLogEntry).filter(EngineLogEntry.event == "NOON")
    if batch_id:
        try:
            bid = uuid_mod.UUID(batch_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid batch_id UUID format")
        query = query.filter(EngineLogEntry.upload_batch_id == bid)

    noon_rows = query.order_by(EngineLogEntry.timestamp).all()

    # Convert to NoonReport objects, skipping invalid entries
    noon_reports: List[NoonReport] = []
    skipped = 0
    for row in noon_rows:
        speed = row.speed_stw
        hfo = row.hfo_total_mt or 0.0
        mgo = row.mgo_total_mt or 0.0
        fuel = hfo + mgo

        # Skip entries with no speed or no fuel
        if not speed or speed <= 0 or fuel <= 0:
            skipped += 1
            continue

        noon_reports.append(NoonReport(
            timestamp=row.timestamp,
            latitude=0.0,
            longitude=0.0,
            speed_over_ground_kts=speed,
            speed_through_water_kts=speed,
            fuel_consumption_mt=fuel,
            period_hours=row.lapse_hours if row.lapse_hours and row.lapse_hours > 0 else 24.0,
            is_laden=True,
            engine_power_kw=row.me_power_kw,
            engine_rpm=row.rpm,
        ))

    if len(noon_reports) < VesselCalibrator.MIN_REPORTS:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {VesselCalibrator.MIN_REPORTS} valid NOON entries for calibration. "
                   f"Found {len(noon_reports)} valid, {skipped} skipped."
        )

    try:
        # Feed reports to calibrator and run
        _vs.calibrator.noon_reports = noon_reports
        result = _vs.calibrator.calibrate(days_since_drydock=days_since_drydock)

        # Apply calibration atomically (rebuilds model, calculators, optimizers)
        _vs.update_calibration(result.factors)

        _vs.calibrator.save_calibration("default", _vs.calibration)

        return EngineLogCalibrateResponse(
            status="calibrated",
            factors=CalibrationFactorsModel(
                calm_water=result.factors.calm_water,
                wind=result.factors.wind,
                waves=result.factors.waves,
                sfoc_factor=result.factors.sfoc_factor,
                calibrated_at=result.factors.calibrated_at,
                num_reports_used=result.factors.num_reports_used,
                calibration_error=result.factors.calibration_error,
                days_since_drydock=result.factors.days_since_drydock,
            ),
            entries_used=result.reports_used,
            entries_skipped=skipped,
            mean_error_before_mt=result.mean_error_before,
            mean_error_after_mt=result.mean_error_after,
            improvement_pct=result.improvement_pct,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Engine log calibration failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Calibration failed: {str(e)}")


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )

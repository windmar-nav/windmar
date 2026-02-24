"""
FastAPI Backend for WINDMAR — Weather Routing & Performance Analytics.

Provides REST API endpoints for:
- Weather data visualization (wind/wave fields)
- Route optimization (A*/Dijkstra weather routing)
- Voyage calculation (per-leg SOG, ETA, fuel)
- Vessel configuration and calibration
- Engine log analytics
- Regulatory compliance (CII, ECA, TSS)

Version: 0.1.0
License: Apache 2.0 - See LICENSE file
"""

import logging
import os
from pathlib import Path

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from slowapi.errors import RateLimitExceeded
import uvicorn

# Import WINDMAR modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.config import settings
from api.middleware import setup_middleware
from api.rate_limit import limiter
from api.state import get_app_state, get_vessel_state

# Configure structured logging for production
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format='%(message)s',  # JSON logs are self-contained
)
logger = logging.getLogger(__name__)


# =============================================================================
# Application Factory
# =============================================================================

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Run migrations, load persisted vessel specs on startup."""
    _run_weather_migrations()
    _run_voyage_migrations()

    # Ensure cache dirs exist (volume mounts may lack subdirectories)
    for sub in ("wind", "wave", "current", "ice", "sst", "vis"):
        Path(f"/tmp/windmar_cache/{sub}").mkdir(parents=True, exist_ok=True)

    # Load persisted vessel specs from DB (survives container restarts)
    from api.routers.vessel import load_vessel_specs_from_db
    _vs = get_vessel_state()
    try:
        saved_specs = load_vessel_specs_from_db()
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
    yield


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
        lifespan=lifespan,
        description="""
## Weather Routing & Performance Analytics API

Professional-grade API for weather routing, vessel performance analytics,
and voyage planning.

### Features
- Real-time weather data integration (GFS, Copernicus CMEMS)
- A*/Dijkstra weather routing optimization
- Vessel performance modeling with calibration
- Engine log ingestion and analytics
- Regulatory zone management (ECA, HRA, TSS)
- CII compliance calculations and projections

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
        allow_headers=["Content-Type", "X-API-Key", "Authorization", "Accept"],
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


def _run_voyage_migrations():
    """Apply voyage table migrations if they don't exist yet."""
    db_url = os.environ.get("DATABASE_URL", settings.database_url)
    if not db_url.startswith("postgresql"):
        logger.info("Skipping voyage migrations (non-PostgreSQL database)")
        return
    try:
        import psycopg2
    except ImportError:
        logger.warning("psycopg2 not installed, skipping voyage migrations")
        return

    try:
        conn = psycopg2.connect(db_url)
        conn.autocommit = True
        cur = conn.cursor()

        # Use a different advisory lock ID from weather migrations
        cur.execute("SELECT pg_try_advisory_lock(20260221)")
        got_lock = cur.fetchone()[0]
        if not got_lock:
            logger.info("Another worker is running voyage migrations, skipping")
            conn.close()
            return

        try:
            cur.execute(
                "SELECT 1 FROM information_schema.tables WHERE table_name = 'voyages'"
            )
            if cur.fetchone():
                logger.info("Voyage tables already exist")
                return

            migration_path = Path(__file__).parent.parent / "docker" / "migrations" / "002_voyage_tables.sql"
            if migration_path.exists():
                sql = migration_path.read_text()
                cur.execute(sql)
                logger.info("Voyage database migration applied successfully")
            else:
                logger.warning(f"Migration file not found: {migration_path}")
        finally:
            cur.execute("SELECT pg_advisory_unlock(20260221)")
            conn.close()
    except Exception as e:
        logger.error(f"Failed to run voyage migrations: {e}")


# =============================================================================
# Create Application & Include Routers
# =============================================================================

app = create_app()

# Include live sensor API router
try:
    from api.live import include_in_app as include_live_routes
    include_live_routes(app)
except ImportError:
    logging.getLogger(__name__).info("Live sensor module not available, skipping live routes")

# Include domain routers
from api.routers.zones import router as zones_router
from api.routers.cii import router as cii_router
from api.routers.fueleu import router as fueleu_router
from api.routers.routes import router as routes_router
from api.routers.system import router as system_router
from api.routers.engine_log import router as engine_log_router
from api.routers.vessel import router as vessel_router
from api.routers.voyage import router as voyage_router
from api.routers.optimization import router as optimization_router
from api.routers.weather import router as weather_router
from api.routers.voyage_history import router as voyage_history_router
from api.routers.charter_party import router as charter_party_router
app.include_router(zones_router)
app.include_router(cii_router)
app.include_router(fueleu_router)
app.include_router(routes_router)
app.include_router(system_router)
app.include_router(engine_log_router)
app.include_router(vessel_router)
app.include_router(voyage_router)
app.include_router(voyage_history_router)
app.include_router(optimization_router)
app.include_router(weather_router)
app.include_router(charter_party_router)

# Initialize application state (thread-safe singleton)
_ = get_app_state()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    logging.error(f"Validation error on {request.method} {request.url.path}: {exc.errors()}")
    logging.error(f"Request body: {body[:500]}")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )

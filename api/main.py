"""
FastAPI Backend for WINDMAR Maritime Route Optimizer.

Provides REST API endpoints for:
- Weather data visualization (wind/wave fields)
- Route management (waypoints, RTZ import)
- Voyage calculation (per-leg SOG, ETA, fuel)
- Vessel configuration
"""

import io
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import WINDMAR modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimization.vessel_model import VesselModel, VesselSpecs
from src.optimization.voyage import VoyageCalculator, LegWeather
from src.optimization.route_optimizer import RouteOptimizer, OptimizedRoute
from src.optimization.vessel_calibration import (
    VesselCalibrator, NoonReport, CalibrationFactors, create_calibrated_model
)
from src.routes.rtz_parser import (
    Route, Waypoint, parse_rtz_string, create_route_from_waypoints,
    haversine_distance, calculate_bearing
)
from src.data.copernicus import (
    CopernicusDataProvider, SyntheticDataProvider, WeatherData,
    ClimatologyProvider, UnifiedWeatherProvider, WeatherDataSource
)
from src.data.regulatory_zones import (
    get_zone_checker, Zone, ZoneProperties, ZoneType, ZoneInteraction
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize FastAPI app
app = FastAPI(
    title="WINDMAR API",
    description="Maritime Route Optimization API - Weather, Routes, Voyage Planning",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Pydantic Models
# ============================================================================

class Position(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)


class WaypointModel(BaseModel):
    id: int
    name: str
    lat: float
    lon: float


class RouteModel(BaseModel):
    name: str
    waypoints: List[WaypointModel]


class VoyageRequest(BaseModel):
    """Request for voyage calculation."""
    waypoints: List[Position]
    calm_speed_kts: float = Field(..., gt=0, lt=30, description="Calm water speed in knots")
    is_laden: bool = True
    departure_time: Optional[datetime] = None
    use_weather: bool = True


class OptimizationRequest(BaseModel):
    """Request for route optimization."""
    origin: Position
    destination: Position
    calm_speed_kts: float = Field(..., gt=0, lt=30, description="Calm water speed in knots")
    is_laden: bool = True
    departure_time: Optional[datetime] = None
    optimization_target: str = Field("fuel", description="Minimize 'fuel' or 'time'")
    grid_resolution_deg: float = Field(0.5, ge=0.1, le=2.0, description="Grid resolution in degrees")


class OptimizationLegModel(BaseModel):
    """Optimized route leg details."""
    from_lat: float
    from_lon: float
    to_lat: float
    to_lon: float
    distance_nm: float
    bearing_deg: float
    fuel_mt: float
    time_hours: float
    sog_kts: float
    stw_kts: float  # Speed through water (optimized per leg)
    wind_speed_ms: float
    wave_height_m: float
    # Safety metrics per leg
    safety_status: Optional[str] = None
    roll_deg: Optional[float] = None
    pitch_deg: Optional[float] = None


class SafetySummary(BaseModel):
    """Safety assessment summary for optimized route."""
    status: str  # "safe", "marginal", "dangerous"
    warnings: List[str]
    max_roll_deg: float
    max_pitch_deg: float
    max_accel_ms2: float


class OptimizationResponse(BaseModel):
    """Route optimization result."""
    waypoints: List[Position]
    total_fuel_mt: float
    total_time_hours: float
    total_distance_nm: float

    # Comparison with direct route
    direct_fuel_mt: float
    direct_time_hours: float
    fuel_savings_pct: float
    time_savings_pct: float

    # Per-leg details
    legs: List[OptimizationLegModel]

    # Speed profile (variable speed optimization)
    speed_profile: List[float]  # Optimal speed per leg (kts)
    avg_speed_kts: float
    variable_speed_enabled: bool

    # Safety assessment
    safety: Optional[SafetySummary] = None

    # Metadata
    optimization_target: str
    grid_resolution_deg: float
    cells_explored: int
    optimization_time_ms: float


class LegResultModel(BaseModel):
    """Result for a single leg."""
    leg_index: int
    from_wp: WaypointModel
    to_wp: WaypointModel
    distance_nm: float
    bearing_deg: float

    # Weather
    wind_speed_kts: float
    wind_dir_deg: float
    wave_height_m: float
    wave_dir_deg: float

    # Speeds
    calm_speed_kts: float
    stw_kts: float
    sog_kts: float
    speed_loss_pct: float

    # Time
    time_hours: float
    departure_time: datetime
    arrival_time: datetime

    # Fuel
    fuel_mt: float
    power_kw: float

    # Data source info (forecast, climatology, blended)
    data_source: Optional[str] = None
    forecast_weight: Optional[float] = None


class DataSourceSummary(BaseModel):
    """Summary of data sources used in voyage calculation."""
    forecast_legs: int
    blended_legs: int
    climatology_legs: int
    forecast_horizon_days: float
    warning: Optional[str] = None


class VoyageResponse(BaseModel):
    """Complete voyage calculation response."""
    route_name: str
    departure_time: datetime
    arrival_time: datetime

    total_distance_nm: float
    total_time_hours: float
    total_fuel_mt: float
    avg_sog_kts: float
    avg_stw_kts: float

    legs: List[LegResultModel]

    calm_speed_kts: float
    is_laden: bool

    # Data source summary
    data_sources: Optional[DataSourceSummary] = None


class WindDataPoint(BaseModel):
    """Wind data at a point."""
    lat: float
    lon: float
    u: float  # U component (m/s)
    v: float  # V component (m/s)
    speed_kts: float
    dir_deg: float


class WeatherGridResponse(BaseModel):
    """Weather grid data for visualization."""
    parameter: str
    time: datetime
    bbox: Dict[str, float]
    resolution: float
    nx: int
    ny: int
    lats: List[float]
    lons: List[float]
    data: List[List[float]]  # 2D grid


class VelocityDataResponse(BaseModel):
    """Wind velocity data in leaflet-velocity format."""
    header: Dict
    data_u: List[float]
    data_v: List[float]


class VesselConfig(BaseModel):
    """Vessel configuration."""
    dwt: float = 49000.0
    loa: float = 183.0
    beam: float = 32.0
    draft_laden: float = 11.8
    draft_ballast: float = 6.5
    mcr_kw: float = 8840.0
    sfoc_at_mcr: float = 171.0
    service_speed_laden: float = 14.5
    service_speed_ballast: float = 15.0


class NoonReportModel(BaseModel):
    """Noon report data for calibration."""
    timestamp: datetime
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    speed_over_ground_kts: float = Field(..., gt=0)
    speed_through_water_kts: Optional[float] = None
    fuel_consumption_mt: float = Field(..., gt=0)
    period_hours: float = Field(24.0, gt=0)
    is_laden: bool = True
    heading_deg: float = Field(0.0, ge=0, le=360)
    wind_speed_kts: Optional[float] = None
    wind_direction_deg: Optional[float] = None
    wave_height_m: Optional[float] = None
    wave_direction_deg: Optional[float] = None
    engine_power_kw: Optional[float] = None


class CalibrationFactorsModel(BaseModel):
    """Calibration factors for vessel model."""
    calm_water: float = Field(1.0, description="Hull fouling factor")
    wind: float = Field(1.0, description="Wind coefficient adjustment")
    waves: float = Field(1.0, description="Wave response adjustment")
    sfoc_factor: float = Field(1.0, description="SFOC multiplier")
    calibrated_at: Optional[datetime] = None
    num_reports_used: int = 0
    calibration_error: float = 0.0
    days_since_drydock: int = 0


class CalibrationResponse(BaseModel):
    """Calibration result response."""
    factors: CalibrationFactorsModel
    reports_used: int
    reports_skipped: int
    mean_error_before_mt: float
    mean_error_after_mt: float
    improvement_pct: float
    residuals: List[Dict]


# ============================================================================
# Global State
# ============================================================================

current_vessel_specs = VesselSpecs()
current_vessel_model = VesselModel(specs=current_vessel_specs)
voyage_calculator = VoyageCalculator(vessel_model=current_vessel_model)
route_optimizer = RouteOptimizer(vessel_model=current_vessel_model)

# Vessel calibration state
vessel_calibrator = VesselCalibrator(vessel_specs=current_vessel_specs)
current_calibration: Optional[CalibrationFactors] = None

# Initialize data providers
# Copernicus provider (attempts real API if configured)
copernicus_provider = CopernicusDataProvider(cache_dir="data/copernicus_cache")

# Climatology provider (for beyond-forecast-horizon)
climatology_provider = ClimatologyProvider(cache_dir="data/climatology_cache")

# Unified provider (blends forecast + climatology)
unified_weather_provider = UnifiedWeatherProvider(
    copernicus=copernicus_provider,
    climatology=climatology_provider,
    cache_dir="data/weather_cache",
)

# Synthetic fallback provider (always works)
synthetic_provider = SyntheticDataProvider()

# Cached weather data
_weather_cache: Dict[str, WeatherData] = {}
_cache_expiry: Dict[str, datetime] = {}
CACHE_TTL_MINUTES = 60


# ============================================================================
# Helper Functions
# ============================================================================

def _get_cache_key(data_type: str, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> str:
    """Generate cache key for weather data."""
    return f"{data_type}_{lat_min:.1f}_{lat_max:.1f}_{lon_min:.1f}_{lon_max:.1f}"


def _is_cache_valid(key: str) -> bool:
    """Check if cached data is still valid."""
    if key not in _cache_expiry:
        return False
    return datetime.utcnow() < _cache_expiry[key]


def get_wind_field(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    resolution: float = 1.0,
    time: datetime = None
) -> WeatherData:
    """
    Get wind field from Copernicus or synthetic fallback.

    Tries Copernicus CDS first, falls back to synthetic data.
    """
    if time is None:
        time = datetime.utcnow()

    cache_key = _get_cache_key("wind", lat_min, lat_max, lon_min, lon_max)

    # Check cache
    if cache_key in _weather_cache and _is_cache_valid(cache_key):
        logger.debug(f"Using cached wind data for {cache_key}")
        return _weather_cache[cache_key]

    # Try Copernicus first
    wind_data = copernicus_provider.fetch_wind_data(lat_min, lat_max, lon_min, lon_max, time)

    if wind_data is None:
        logger.info("Copernicus wind data unavailable, using synthetic data")
        wind_data = synthetic_provider.generate_wind_field(
            lat_min, lat_max, lon_min, lon_max, resolution, time
        )

    # Cache the result
    _weather_cache[cache_key] = wind_data
    _cache_expiry[cache_key] = datetime.utcnow() + timedelta(minutes=CACHE_TTL_MINUTES)

    return wind_data


def get_wave_field(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    resolution: float = 1.0,
    wind_data: WeatherData = None,
) -> WeatherData:
    """
    Get wave field from Copernicus or synthetic fallback.

    Tries CMEMS first, falls back to synthetic data based on wind.
    """
    cache_key = _get_cache_key("wave", lat_min, lat_max, lon_min, lon_max)

    # Check cache
    if cache_key in _weather_cache and _is_cache_valid(cache_key):
        logger.debug(f"Using cached wave data for {cache_key}")
        return _weather_cache[cache_key]

    # Try Copernicus first
    wave_data = copernicus_provider.fetch_wave_data(lat_min, lat_max, lon_min, lon_max)

    if wave_data is None:
        logger.info("Copernicus wave data unavailable, using synthetic data")
        wave_data = synthetic_provider.generate_wave_field(
            lat_min, lat_max, lon_min, lon_max, resolution, wind_data
        )

    # Cache the result
    _weather_cache[cache_key] = wave_data
    _cache_expiry[cache_key] = datetime.utcnow() + timedelta(minutes=CACHE_TTL_MINUTES)

    return wave_data


def get_current_field(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
) -> Optional[WeatherData]:
    """
    Get ocean current field from CMEMS.

    Returns None if unavailable (currents are optional).
    """
    cache_key = _get_cache_key("current", lat_min, lat_max, lon_min, lon_max)

    if cache_key in _weather_cache and _is_cache_valid(cache_key):
        return _weather_cache[cache_key]

    current_data = copernicus_provider.fetch_current_data(lat_min, lat_max, lon_min, lon_max)

    if current_data is not None:
        _weather_cache[cache_key] = current_data
        _cache_expiry[cache_key] = datetime.utcnow() + timedelta(minutes=CACHE_TTL_MINUTES)

    return current_data


def get_weather_at_point(lat: float, lon: float, time: datetime) -> Tuple[Dict, Optional[WeatherDataSource]]:
    """
    Get weather at a specific point.

    Uses unified provider that blends forecast and climatology.

    Returns:
        Tuple of (weather_dict, data_source) where data_source indicates
        whether data is from forecast, climatology, or blended.
    """
    try:
        # Use unified provider - handles forecast/climatology transition
        point_wx, source = unified_weather_provider.get_weather_at_point(lat, lon, time)

        return {
            'wind_speed_ms': point_wx.wind_speed_ms,
            'wind_dir_deg': point_wx.wind_dir_deg,
            'sig_wave_height_m': point_wx.wave_height_m,
            'wave_period_s': point_wx.wave_period_s,
            'wave_dir_deg': point_wx.wave_dir_deg,
            'current_speed_ms': point_wx.current_speed_ms,
            'current_dir_deg': point_wx.current_dir_deg,
        }, source

    except Exception as e:
        logger.warning(f"Unified provider failed, falling back to grid method: {e}")

        # Fallback to direct grid method
        margin = 2.0
        lat_min, lat_max = lat - margin, lat + margin
        lon_min, lon_max = lon - margin, lon + margin

        wind_data = get_wind_field(lat_min, lat_max, lon_min, lon_max, 0.5, time)
        wave_data = get_wave_field(lat_min, lat_max, lon_min, lon_max, 0.5, wind_data)
        current_data = get_current_field(lat_min, lat_max, lon_min, lon_max)

        point_wx = copernicus_provider.get_weather_at_point(
            lat, lon, time, wind_data, wave_data, current_data
        )

        return {
            'wind_speed_ms': point_wx.wind_speed_ms,
            'wind_dir_deg': point_wx.wind_dir_deg,
            'sig_wave_height_m': point_wx.wave_height_m,
            'wave_period_s': point_wx.wave_period_s,
            'wave_dir_deg': point_wx.wave_dir_deg,
            'current_speed_ms': point_wx.current_speed_ms,
            'current_dir_deg': point_wx.current_dir_deg,
        }, None


# Track data sources for each leg during voyage calculation
_voyage_data_sources: List[Dict] = []


def weather_provider(lat: float, lon: float, time: datetime) -> LegWeather:
    """Weather provider function for voyage calculator."""
    global _voyage_data_sources

    wx, source = get_weather_at_point(lat, lon, time)

    # Track data source for this leg
    if source:
        _voyage_data_sources.append({
            'lat': lat,
            'lon': lon,
            'time': time.isoformat(),
            'source': source.source,
            'forecast_weight': source.forecast_weight,
            'message': source.message,
        })

    return LegWeather(
        wind_speed_ms=wx['wind_speed_ms'],
        wind_dir_deg=wx['wind_dir_deg'],
        sig_wave_height_m=wx['sig_wave_height_m'],
        wave_period_s=wx.get('wave_period_s', 5.0 + wx['sig_wave_height_m']),
        wave_dir_deg=wx['wave_dir_deg'],
    )


# ============================================================================
# API Endpoints - Core
# ============================================================================

@app.get("/")
async def root():
    """API root."""
    return {
        "name": "WINDMAR API",
        "version": "2.0.0",
        "status": "operational",
        "docs": "/api/docs",
        "endpoints": {
            "weather": "/api/weather/...",
            "routes": "/api/routes/...",
            "voyage": "/api/voyage/...",
            "vessel": "/api/vessel/...",
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/data-sources")
async def get_data_sources():
    """
    Get status of available data sources.

    Shows which Copernicus APIs are configured and available.
    """
    return {
        "copernicus": {
            "cds": {
                "available": copernicus_provider._has_cdsapi,
                "description": "Climate Data Store (ERA5 wind data)",
                "setup": "pip install cdsapi && create ~/.cdsapirc with API key",
            },
            "cmems": {
                "available": copernicus_provider._has_copernicusmarine,
                "description": "Copernicus Marine Service (waves, currents)",
                "setup": "pip install copernicusmarine && configure credentials",
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
        "active_source": "copernicus" if (
            copernicus_provider._has_cdsapi and copernicus_provider._has_copernicusmarine
        ) else "synthetic",
    }


# ============================================================================
# API Endpoints - Weather (Layer 1)
# ============================================================================

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
    Uses Copernicus CDS when available, falls back to synthetic data.
    """
    if time is None:
        time = datetime.utcnow()

    wind_data = get_wind_field(lat_min, lat_max, lon_min, lon_max, resolution, time)

    return {
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
        "source": "copernicus" if copernicus_provider._has_cdsapi else "synthetic",
    }


@app.get("/api/weather/wind/velocity")
async def api_get_wind_velocity_format(
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
    resolution: float = Query(1.0),
    time: Optional[datetime] = None,
):
    """
    Get wind data in leaflet-velocity compatible format.

    Returns array of [U-component, V-component] data with headers.
    Uses Copernicus CDS when available.
    """
    if time is None:
        time = datetime.utcnow()

    wind_data = get_wind_field(lat_min, lat_max, lon_min, lon_max, resolution, time)

    # leaflet-velocity format
    header = {
        "parameterCategory": 2,
        "parameterNumber": 2,
        "lo1": lon_min,
        "la1": lat_max,  # Note: lat goes from top to bottom
        "lo2": lon_max,
        "la2": lat_min,
        "dx": resolution,
        "dy": resolution,
        "nx": len(wind_data.lons),
        "ny": len(wind_data.lats),
        "refTime": time.isoformat(),
    }

    # Flatten data (row-major, from top-left)
    u_flat = wind_data.u_component[::-1].flatten().tolist()  # Flip lat axis
    v_flat = wind_data.v_component[::-1].flatten().tolist()

    return [
        {"header": {**header, "parameterNumber": 2}, "data": u_flat},
        {"header": {**header, "parameterNumber": 3}, "data": v_flat},
    ]


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

    Uses Copernicus CMEMS when available, falls back to synthetic data.
    """
    if time is None:
        time = datetime.utcnow()

    # Get wind first for synthetic fallback
    wind_data = get_wind_field(lat_min, lat_max, lon_min, lon_max, resolution, time)

    # Get waves (CMEMS or synthetic)
    wave_data = get_wave_field(lat_min, lat_max, lon_min, lon_max, resolution, wind_data)

    return {
        "parameter": "wave_height",
        "time": time.isoformat(),
        "bbox": {
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
        },
        "resolution": resolution,
        "nx": len(wave_data.lons),
        "ny": len(wave_data.lats),
        "lats": wave_data.lats.tolist(),
        "lons": wave_data.lons.tolist(),
        "data": wave_data.values.tolist(),
        "unit": "m",
        "source": "copernicus" if copernicus_provider._has_copernicusmarine else "synthetic",
        "colorscale": {
            "min": 0,
            "max": 6,
            "colors": ["#00ff00", "#ffff00", "#ff8800", "#ff0000", "#800000"],
        }
    }


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

    Uses Copernicus CMEMS when available. Returns empty if unavailable.
    """
    if time is None:
        time = datetime.utcnow()

    current_data = get_current_field(lat_min, lat_max, lon_min, lon_max)

    if current_data is None:
        return {
            "parameter": "current",
            "time": time.isoformat(),
            "available": False,
            "message": "Current data requires CMEMS credentials",
        }

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
        "nx": len(current_data.lons),
        "ny": len(current_data.lats),
        "lats": current_data.lats.tolist(),
        "lons": current_data.lons.tolist(),
        "u": current_data.u_component.tolist() if current_data.u_component is not None else [],
        "v": current_data.v_component.tolist() if current_data.v_component is not None else [],
        "unit": "m/s",
    }


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
# API Endpoints - Routes (Layer 2)
# ============================================================================

@app.post("/api/routes/parse-rtz")
async def parse_rtz(file: UploadFile = File(...)):
    """
    Parse an uploaded RTZ route file.

    Returns waypoints in standard format.
    """
    try:
        content = await file.read()
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
    global _voyage_data_sources

    if len(request.waypoints) < 2:
        raise HTTPException(status_code=400, detail="At least 2 waypoints required")

    departure = request.departure_time or datetime.utcnow()

    # Create route from waypoints
    wps = [(wp.lat, wp.lon) for wp in request.waypoints]
    route = create_route_from_waypoints(wps, "Voyage Route")

    # Reset data source tracking
    _voyage_data_sources = []

    # Calculate voyage
    wp_func = weather_provider if request.use_weather else None

    result = voyage_calculator.calculate_voyage(
        route=route,
        calm_speed_kts=request.calm_speed_kts,
        is_laden=request.is_laden,
        departure_time=departure,
        weather_provider=wp_func,
    )

    # Build data source summary
    forecast_legs = sum(1 for ds in _voyage_data_sources if ds['source'] == 'forecast')
    blended_legs = sum(1 for ds in _voyage_data_sources if ds['source'] == 'blended')
    climatology_legs = sum(1 for ds in _voyage_data_sources if ds['source'] == 'climatology')

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
        leg_source = _voyage_data_sources[i] if i < len(_voyage_data_sources) else None

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


@app.get("/api/voyage/weather-along-route")
async def get_weather_along_route(
    waypoints: str = Query(..., description="Comma-separated lat,lon pairs: lat1,lon1;lat2,lon2;..."),
    time: Optional[datetime] = None,
):
    """
    Get weather conditions at each waypoint and leg midpoint.
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

    # Get weather at each point
    result = []
    for i, (lat, lon) in enumerate(wps):
        wx = get_weather_at_point(lat, lon, time)
        result.append({
            "waypoint_index": i,
            "position": {"lat": lat, "lon": lon},
            "wind_speed_kts": round(wx['wind_speed_ms'] * 1.94384, 1),
            "wind_dir_deg": round(wx['wind_dir_deg'], 0),
            "wave_height_m": round(wx['sig_wave_height_m'], 1),
            "wave_dir_deg": round(wx['wave_dir_deg'], 0),
        })

    return {"time": time.isoformat(), "waypoints": result}


# ============================================================================
# API Endpoints - Route Optimization (Layer 4)
# ============================================================================

@app.post("/api/optimize/route", response_model=OptimizationResponse)
async def optimize_route(request: OptimizationRequest):
    """
    Find optimal route through weather using A* search.

    Minimizes fuel consumption (or time) by routing around adverse weather.

    The algorithm:
    1. Builds a grid around the origin-destination corridor
    2. Uses A* search with weather-aware cost function
    3. Smooths the resulting path to create navigable waypoints
    4. Returns optimized route with fuel/time savings comparison

    Grid resolution affects accuracy vs computation time:
    - 0.25° = ~15nm cells, high accuracy, slower
    - 0.5° = ~30nm cells, good balance (default)
    - 1.0° = ~60nm cells, fast, less precise
    """
    global route_optimizer

    departure = request.departure_time or datetime.utcnow()

    # Configure optimizer
    route_optimizer.resolution_deg = request.grid_resolution_deg
    route_optimizer.optimization_target = request.optimization_target

    try:
        result = route_optimizer.optimize_route(
            origin=(request.origin.lat, request.origin.lon),
            destination=(request.destination.lat, request.destination.lon),
            departure_time=departure,
            calm_speed_kts=request.calm_speed_kts,
            is_laden=request.is_laden,
            weather_provider=weather_provider,
        )

        # Format response
        waypoints = [Position(lat=wp[0], lon=wp[1]) for wp in result.waypoints]

        legs = []
        for leg in result.leg_details:
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
                stw_kts=round(leg.get('stw_kts', leg['sog_kts']), 1),  # Optimized speed through water
                wind_speed_ms=round(leg['wind_speed_ms'], 1),
                wave_height_m=round(leg['wave_height_m'], 1),
                safety_status=leg.get('safety_status'),
                roll_deg=round(leg['roll_deg'], 1) if leg.get('roll_deg') else None,
                pitch_deg=round(leg['pitch_deg'], 1) if leg.get('pitch_deg') else None,
            ))

        # Build safety summary
        safety_summary = SafetySummary(
            status=result.safety_status,
            warnings=result.safety_warnings,
            max_roll_deg=round(result.max_roll_deg, 1),
            max_pitch_deg=round(result.max_pitch_deg, 1),
            max_accel_ms2=round(result.max_accel_ms2, 2),
        )

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
            safety=safety_summary,
            optimization_target=request.optimization_target,
            grid_resolution_deg=request.grid_resolution_deg,
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
            "dwt": current_vessel_specs.dwt,
            "service_speed_laden": current_vessel_specs.service_speed_laden,
        }
    }


# ============================================================================
# API Endpoints - Vessel Configuration
# ============================================================================

@app.get("/api/vessel/specs")
async def get_vessel_specs():
    """Get current vessel specifications."""
    specs = current_vessel_specs
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


@app.post("/api/vessel/specs")
async def update_vessel_specs(config: VesselConfig):
    """Update vessel specifications."""
    global current_vessel_specs, current_vessel_model, voyage_calculator

    try:
        current_vessel_specs = VesselSpecs(
            dwt=config.dwt,
            loa=config.loa,
            beam=config.beam,
            draft_laden=config.draft_laden,
            draft_ballast=config.draft_ballast,
            mcr_kw=config.mcr_kw,
            sfoc_at_mcr=config.sfoc_at_mcr,
            service_speed_laden=config.service_speed_laden,
            service_speed_ballast=config.service_speed_ballast,
        )
        current_vessel_model = VesselModel(specs=current_vessel_specs)
        voyage_calculator = VoyageCalculator(vessel_model=current_vessel_model)

        return {"status": "success", "message": "Vessel specs updated"}

    except Exception as e:
        logger.error(f"Failed to update vessel specs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# API Endpoints - Vessel Calibration
# ============================================================================

@app.get("/api/vessel/calibration")
async def get_calibration():
    """Get current vessel calibration factors."""
    global current_calibration

    if current_calibration is None:
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
            "calm_water": current_calibration.calm_water,
            "wind": current_calibration.wind,
            "waves": current_calibration.waves,
            "sfoc_factor": current_calibration.sfoc_factor,
        },
        "calibrated_at": current_calibration.calibrated_at.isoformat() if current_calibration.calibrated_at else None,
        "num_reports_used": current_calibration.num_reports_used,
        "calibration_error_mt": current_calibration.calibration_error,
        "days_since_drydock": current_calibration.days_since_drydock,
    }


@app.post("/api/vessel/calibration/set")
async def set_calibration_factors(factors: CalibrationFactorsModel):
    """Manually set calibration factors."""
    global current_calibration, current_vessel_model, voyage_calculator, route_optimizer

    current_calibration = CalibrationFactors(
        calm_water=factors.calm_water,
        wind=factors.wind,
        waves=factors.waves,
        sfoc_factor=factors.sfoc_factor,
        calibrated_at=datetime.utcnow(),
        num_reports_used=0,
        days_since_drydock=factors.days_since_drydock,
    )

    # Update vessel model with new calibration
    current_vessel_model = VesselModel(
        specs=current_vessel_specs,
        calibration_factors={
            'calm_water': current_calibration.calm_water,
            'wind': current_calibration.wind,
            'waves': current_calibration.waves,
        }
    )
    voyage_calculator = VoyageCalculator(vessel_model=current_vessel_model)
    route_optimizer = RouteOptimizer(vessel_model=current_vessel_model)

    return {"status": "success", "message": "Calibration factors updated"}


@app.get("/api/vessel/noon-reports")
async def get_noon_reports():
    """Get list of uploaded noon reports."""
    global vessel_calibrator

    return {
        "count": len(vessel_calibrator.noon_reports),
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
            for r in vessel_calibrator.noon_reports
        ]
    }


@app.post("/api/vessel/noon-reports")
async def add_noon_report(report: NoonReportModel):
    """Add a single noon report for calibration."""
    global vessel_calibrator

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

    vessel_calibrator.add_noon_report(nr)

    return {
        "status": "success",
        "total_reports": len(vessel_calibrator.noon_reports),
    }


@app.post("/api/vessel/noon-reports/upload-csv")
async def upload_noon_reports_csv(file: UploadFile = File(...)):
    """
    Upload noon reports from CSV file.

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
    global vessel_calibrator

    try:
        # Save to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        # Import from CSV
        count = vessel_calibrator.add_noon_reports_from_csv(tmp_path)

        # Cleanup
        tmp_path.unlink()

        return {
            "status": "success",
            "imported": count,
            "total_reports": len(vessel_calibrator.noon_reports),
        }

    except Exception as e:
        logger.error(f"Failed to import CSV: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")


@app.delete("/api/vessel/noon-reports")
async def clear_noon_reports():
    """Clear all uploaded noon reports."""
    global vessel_calibrator

    vessel_calibrator.noon_reports = []
    return {"status": "success", "message": "All noon reports cleared"}


@app.post("/api/vessel/calibrate", response_model=CalibrationResponse)
async def calibrate_vessel(
    days_since_drydock: int = Query(0, ge=0, description="Days since last dry dock"),
):
    """
    Run calibration using uploaded noon reports.

    Finds optimal calibration factors that minimize prediction error
    compared to actual fuel consumption.
    """
    global vessel_calibrator, current_calibration
    global current_vessel_model, voyage_calculator, route_optimizer

    if len(vessel_calibrator.noon_reports) < VesselCalibrator.MIN_REPORTS:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {VesselCalibrator.MIN_REPORTS} noon reports for calibration. "
                   f"Currently have {len(vessel_calibrator.noon_reports)}."
        )

    try:
        result = vessel_calibrator.calibrate(days_since_drydock=days_since_drydock)

        # Store calibration
        current_calibration = result.factors

        # Update vessel model with calibration
        current_vessel_model = VesselModel(
            specs=current_vessel_specs,
            calibration_factors={
                'calm_water': current_calibration.calm_water,
                'wind': current_calibration.wind,
                'waves': current_calibration.waves,
            }
        )
        voyage_calculator = VoyageCalculator(vessel_model=current_vessel_model)
        route_optimizer = RouteOptimizer(vessel_model=current_vessel_model)

        # Save calibration to file
        vessel_calibrator.save_calibration("default", current_calibration)

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


@app.post("/api/vessel/calibration/estimate-fouling")
async def estimate_hull_fouling(
    days_since_drydock: int = Query(..., ge=0),
    operating_regions: List[str] = Query(default=[], description="Operating regions: tropical, warm_temperate, cold, polar"),
):
    """
    Estimate hull fouling factor without calibration data.

    Useful when no noon reports are available but you know
    the vessel's operating history.
    """
    global vessel_calibrator

    fouling = vessel_calibrator.estimate_hull_fouling(
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


# ============================================================================
# API Endpoints - Regulatory Zones
# ============================================================================

class ZoneCoordinate(BaseModel):
    """A coordinate in a zone polygon."""
    lat: float
    lon: float


class CreateZoneRequest(BaseModel):
    """Request to create a custom zone."""
    name: str
    zone_type: str = Field(..., description="eca, hra, tss, exclusion, custom, etc.")
    interaction: str = Field(..., description="mandatory, exclusion, penalty, advisory")
    coordinates: List[ZoneCoordinate]
    penalty_factor: float = Field(1.0, ge=1.0, le=10.0)
    notes: Optional[str] = None


class ZoneResponse(BaseModel):
    """Zone information response."""
    id: str
    name: str
    zone_type: str
    interaction: str
    penalty_factor: float
    is_builtin: bool
    coordinates: List[ZoneCoordinate]
    notes: Optional[str] = None


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


@app.post("/api/zones", response_model=ZoneResponse)
async def create_zone(request: CreateZoneRequest):
    """
    Create a custom zone.

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


@app.delete("/api/zones/{zone_id}")
async def delete_zone(zone_id: str):
    """
    Delete a custom zone.

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

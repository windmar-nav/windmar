"""
Weather data service for WINDMAR API.

Centralizes all weather field fetching, caching (Redis + in-memory fallback),
ocean masking, and point-weather queries.  Functions obtain data providers
via ``get_app_state().weather_providers`` so they stay decoupled from the
main FastAPI module.
"""

import base64
import json
import logging
import os
import zlib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.data.copernicus import WeatherData, WeatherDataSource
from src.data.land_mask import is_ocean
from src.optimization.voyage import LegWeather

try:
    import redis as redis_lib
except ImportError:
    redis_lib = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safe serialization (replaces pickle for Redis cache)
# ---------------------------------------------------------------------------
_WEATHER_DATA_ARRAY_FIELDS = [
    "lats", "lons", "values", "u_component", "v_component",
    "wave_period", "wave_direction",
    "windwave_height", "windwave_period", "windwave_direction",
    "swell_height", "swell_period", "swell_direction",
    "sst", "visibility", "ice_concentration",
]


def _serialize_ndarray(arr: np.ndarray) -> Dict[str, Any]:
    """Serialize numpy array to JSON-safe dict with zlib compression."""
    raw = arr.astype(np.float64).tobytes()
    compressed = zlib.compress(raw, level=1)
    return {
        "d": base64.b64encode(compressed).decode("ascii"),
        "s": list(arr.shape),
    }


def _deserialize_ndarray(obj: Dict[str, Any]) -> np.ndarray:
    """Deserialize numpy array from JSON-safe dict."""
    raw = zlib.decompress(base64.b64decode(obj["d"]))
    return np.frombuffer(raw, dtype=np.float64).reshape(obj["s"])


def _serialize_weather_data(wd: WeatherData) -> bytes:
    """Serialize WeatherData to compressed JSON bytes (safe, no pickle)."""
    doc: Dict[str, Any] = {
        "parameter": wd.parameter,
        "time": wd.time.isoformat(),
        "unit": wd.unit,
    }
    for field in _WEATHER_DATA_ARRAY_FIELDS:
        val = getattr(wd, field, None)
        if val is not None:
            doc[field] = _serialize_ndarray(val)
    return zlib.compress(json.dumps(doc).encode(), level=1)


def _deserialize_weather_data(data: bytes) -> WeatherData:
    """Deserialize WeatherData from compressed JSON bytes (safe, no pickle)."""
    doc = json.loads(zlib.decompress(data))
    kwargs: Dict[str, Any] = {
        "parameter": doc["parameter"],
        "time": datetime.fromisoformat(doc["time"]),
        "unit": doc["unit"],
    }
    for field in _WEATHER_DATA_ARRAY_FIELDS:
        if field in doc:
            kwargs[field] = _deserialize_ndarray(doc[field])
        elif field in ("lats", "lons", "values"):
            raise ValueError(f"Missing required field '{field}' in cached weather data")
    return WeatherData(**kwargs)


# ---------------------------------------------------------------------------
# Redis / in-memory cache layer
# ---------------------------------------------------------------------------
CACHE_TTL_MINUTES = 60

_redis_client = None  # Optional[redis.Redis]
_weather_cache: Dict[str, WeatherData] = {}
_cache_expiry: Dict[str, datetime] = {}


def _get_redis():
    """Lazy-init Redis client. Returns None if unavailable."""
    global _redis_client
    if redis_lib is None:
        return None
    if _redis_client is not None:
        return _redis_client
    try:
        from api.config import settings
        redis_url = os.environ.get("REDIS_URL", settings.redis_url)
        _redis_client = redis_lib.Redis.from_url(redis_url, decode_responses=False)
        _redis_client.ping()
        logger.info("Redis connected for weather cache")
        return _redis_client
    except Exception as e:
        logger.warning(f"Redis unavailable, falling back to in-memory cache: {e}")
        return None


def _redis_cache_get(key: str) -> Optional[WeatherData]:
    """Get from Redis, falling back to in-memory."""
    r = _get_redis()
    if r is not None:
        try:
            data = r.get(f"wx:{key}")
            if data:
                return _deserialize_weather_data(data)
        except Exception:
            logger.debug("Redis cache get failed for key %s", key, exc_info=True)
    # Fallback to in-memory
    if key in _weather_cache and key in _cache_expiry and datetime.now(timezone.utc) < _cache_expiry[key]:
        return _weather_cache[key]
    return None


def _redis_cache_put(key: str, data: WeatherData, ttl_seconds: int = 900):
    """Put to Redis, falling back to in-memory."""
    r = _get_redis()
    if r is not None:
        try:
            r.setex(f"wx:{key}", ttl_seconds, _serialize_weather_data(data))
            return
        except Exception:
            logger.debug("Redis cache put failed for key %s", key, exc_info=True)
    # Fallback to in-memory
    _weather_cache[key] = data
    _cache_expiry[key] = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)


# ---------------------------------------------------------------------------
# Cache key helper
# ---------------------------------------------------------------------------

def _get_cache_key(data_type: str, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> str:
    """Generate cache key for weather data."""
    return f"{data_type}_{lat_min:.1f}_{lat_max:.1f}_{lon_min:.1f}_{lon_max:.1f}"


# ---------------------------------------------------------------------------
# Ocean mask utilities
# ---------------------------------------------------------------------------

def build_ocean_mask(lat_min, lat_max, lon_min, lon_max, step=0.05):
    """Build high-res ocean mask using vectorized numpy calls (fast)."""
    lat_min = max(-89.99, lat_min)
    lat_max = min(89.99, lat_max)
    lon_min = max(-180.0, lon_min)
    lon_max = min(180.0, lon_max)
    mask_lats = np.arange(lat_min, lat_max + step / 2, step)
    mask_lons = np.arange(lon_min, lon_max + step / 2, step)
    lon_grid, lat_grid = np.meshgrid(mask_lons, mask_lats)
    try:
        from global_land_mask import globe
        mask = globe.is_ocean(lat_grid, lon_grid)
        return mask_lats.tolist(), mask_lons.tolist(), mask.tolist()
    except ImportError:
        # Fallback to point-by-point (slow)
        mask = [
            [is_ocean(round(float(lat), 2), round(float(lon), 2)) for lon in mask_lons]
            for lat in mask_lats
        ]
        return mask_lats.tolist(), mask_lons.tolist(), mask


def build_ocean_mask_at_coords(lats, lons):
    """Build an ocean mask at exactly the provided lat/lon arrays.

    Returns (mask_lats_list, mask_lons_list, mask_2d_list) where the mask
    dimensions match len(lats) x len(lons) exactly — no interpolation mismatch.
    """
    lats_arr = np.asarray(lats, dtype=np.float64)
    lons_arr = np.asarray(lons, dtype=np.float64)
    lon_grid, lat_grid = np.meshgrid(lons_arr, lats_arr)
    try:
        from global_land_mask import globe
        mask = globe.is_ocean(lat_grid, lon_grid)
    except ImportError:
        mask = np.array([
            [is_ocean(round(float(lat), 2), round(float(lon), 2)) for lon in lons_arr]
            for lat in lats_arr
        ])
    return lats_arr.tolist(), lons_arr.tolist(), mask.tolist()


def apply_ocean_mask_velocity(u: np.ndarray, v: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> tuple:
    """Zero out U/V components over land so leaflet-velocity skips land areas.

    Applies 1-cell erosion so coastal ocean cells adjacent to land are also
    zeroed — prevents particles from drifting over land during animation.
    """
    try:
        from global_land_mask import globe
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        ocean = globe.is_ocean(lat_grid, lon_grid)
        # Erode 3 cells: prevents animated particles from drifting onto land.
        # At 0.25° grid resolution this creates ~83 km coastal buffer.
        eroded = ocean.copy()
        for _ in range(3):
            buf = eroded.copy()
            buf[:-1, :] &= eroded[1:, :]
            buf[1:, :]  &= eroded[:-1, :]
            buf[:, :-1] &= eroded[:, 1:]
            buf[:, 1:]  &= eroded[:, :-1]
            eroded = buf
        u_masked = np.where(eroded, u, 0.0)
        v_masked = np.where(eroded, v, 0.0)
        return u_masked, v_masked
    except ImportError:
        return u, v


# ---------------------------------------------------------------------------
# Provider accessor (gets providers from ApplicationState)
# ---------------------------------------------------------------------------

def _providers() -> Dict:
    """Get weather providers dict from application state."""
    from api.state import get_app_state
    return get_app_state().weather_providers


# ---------------------------------------------------------------------------
# GFS wind supplement for temporal providers
# ---------------------------------------------------------------------------

def supplement_temporal_wind(
    temporal_wx,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    departure: datetime,
) -> bool:
    """Inject GFS wind snapshot into a temporal provider that lacks wind grids.

    Fetches a single GFS forecast-hour-0 wind field from NOAA (fast, ~2-5s)
    and injects wind_u / wind_v into the temporal provider's grids so that
    wind resistance is included in calculations.

    Returns True if wind was successfully injected.
    """
    import time as _time
    t0 = _time.monotonic()
    try:
        gfs = _providers()['gfs']
        wind_data = gfs.fetch_wind_data(
            lat_min, lat_max, lon_min, lon_max, departure, forecast_hour=0
        )
        if wind_data is None or wind_data.u_component is None:
            logger.warning("GFS wind supplement: fetch returned None — wind resistance unavailable")
            return False

        # Convert WeatherData to GridDict format: {forecast_hour: (lats_1d, lons_1d, data_2d)}
        lats = wind_data.lats
        lons = wind_data.lons
        temporal_wx.grids["wind_u"] = {0: (lats, lons, wind_data.u_component)}
        temporal_wx.grids["wind_v"] = {0: (lats, lons, wind_data.v_component)}
        temporal_wx._sorted_hours["wind_u"] = [0]
        temporal_wx._sorted_hours["wind_v"] = [0]

        logger.info(
            f"GFS wind supplement injected: {len(lats)}x{len(lons)} grid in {_time.monotonic()-t0:.1f}s"
        )
        return True

    except Exception as e:
        logger.warning(f"GFS wind supplement failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Weather field fetchers (provider chain pattern)
# ---------------------------------------------------------------------------

def get_wind_field(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    resolution: float = 1.0,
    time: datetime = None
) -> WeatherData:
    """Get wind field data.

    Provider chain: Redis cache -> DB (pre-ingested) -> GFS live -> ERA5 -> Synthetic.
    """
    if time is None:
        time = datetime.now(timezone.utc)

    cache_key = _get_cache_key("wind", lat_min, lat_max, lon_min, lon_max)

    # Check Redis/in-memory cache
    cached = _redis_cache_get(cache_key)
    if cached is not None:
        logger.debug(f"Using cached wind data for {cache_key}")
        return cached

    p = _providers()
    db_weather = p.get('db_weather')
    gfs = p['gfs']
    copernicus = p['copernicus']
    synthetic = p['synthetic']

    # Try DB first (pre-ingested grids — sub-second)
    if db_weather is not None:
        wind_data, _ = db_weather.get_wind_from_db(lat_min, lat_max, lon_min, lon_max, time)
        if wind_data is not None:
            logger.info("Using DB pre-ingested wind data")
            _redis_cache_put(cache_key, wind_data, CACHE_TTL_MINUTES * 60)
            return wind_data

    # Try GFS live (near-real-time, ~3.5h lag)
    wind_data = gfs.fetch_wind_data(lat_min, lat_max, lon_min, lon_max, time)
    if wind_data is not None:
        logger.info("Using GFS near-real-time wind data")
    else:
        # Fall back to ERA5 (reanalysis, ~5-day lag)
        wind_data = copernicus.fetch_wind_data(lat_min, lat_max, lon_min, lon_max, time)
        if wind_data is not None:
            logger.info("GFS unavailable, using ERA5 reanalysis wind data")
        else:
            # Final fallback: synthetic
            logger.info("GFS and ERA5 unavailable, using synthetic wind data")
            wind_data = synthetic.generate_wind_field(
                lat_min, lat_max, lon_min, lon_max, resolution, time
            )

    _redis_cache_put(cache_key, wind_data, CACHE_TTL_MINUTES * 60)
    return wind_data


def get_wave_field(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    resolution: float = 1.0,
    wind_data: WeatherData = None,
) -> WeatherData:
    """Get wave field data.

    Provider chain: Redis cache -> DB (pre-ingested) -> CMEMS live -> Synthetic.
    """
    cache_key = _get_cache_key("wave", lat_min, lat_max, lon_min, lon_max)

    cached = _redis_cache_get(cache_key)
    if cached is not None:
        logger.debug(f"Using cached wave data for {cache_key}")
        return cached

    p = _providers()
    db_weather = p.get('db_weather')
    copernicus = p['copernicus']
    synthetic = p['synthetic']

    # Try DB first
    if db_weather is not None:
        wave_data, _ = db_weather.get_wave_from_db(lat_min, lat_max, lon_min, lon_max)
        if wave_data is not None:
            logger.info("Using DB pre-ingested wave data")
            _redis_cache_put(cache_key, wave_data, CACHE_TTL_MINUTES * 60)
            return wave_data

    # Try CMEMS live
    wave_data = copernicus.fetch_wave_data(lat_min, lat_max, lon_min, lon_max)

    if wave_data is None:
        logger.info("Copernicus wave data unavailable, using synthetic data")
        wave_data = synthetic.generate_wave_field(
            lat_min, lat_max, lon_min, lon_max, resolution, wind_data
        )

    _redis_cache_put(cache_key, wave_data, CACHE_TTL_MINUTES * 60)
    return wave_data


def get_current_field(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    resolution: float = 1.0,
) -> WeatherData:
    """Get ocean current field.

    Provider chain: Redis cache -> DB (pre-ingested) -> CMEMS live -> Synthetic.
    """
    cache_key = _get_cache_key("current", lat_min, lat_max, lon_min, lon_max)

    cached = _redis_cache_get(cache_key)
    if cached is not None:
        return cached

    p = _providers()
    db_weather = p.get('db_weather')
    copernicus = p['copernicus']
    synthetic = p['synthetic']

    # Try DB first
    if db_weather is not None:
        current_data, _ = db_weather.get_current_from_db(lat_min, lat_max, lon_min, lon_max)
        if current_data is not None:
            logger.info("Using DB pre-ingested current data")
            _redis_cache_put(cache_key, current_data, CACHE_TTL_MINUTES * 60)
            return current_data

    # Try CMEMS live
    current_data = copernicus.fetch_current_data(lat_min, lat_max, lon_min, lon_max)

    if current_data is None:
        logger.info("CMEMS current data unavailable, using synthetic data")
        current_data = synthetic.generate_current_field(
            lat_min, lat_max, lon_min, lon_max, resolution
        )

    _redis_cache_put(cache_key, current_data, CACHE_TTL_MINUTES * 60)
    return current_data


def get_sst_field(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    resolution: float = 1.0,
    time: datetime = None,
) -> WeatherData:
    """Get SST field data.

    Provider chain: CMEMS live -> Synthetic.
    """
    if time is None:
        time = datetime.now(timezone.utc)

    cache_key = _get_cache_key("sst", lat_min, lat_max, lon_min, lon_max)
    cached = _redis_cache_get(cache_key)
    if cached is not None:
        return cached

    p = _providers()
    copernicus = p['copernicus']
    synthetic = p['synthetic']

    sst_data = copernicus.fetch_sst_data(lat_min, lat_max, lon_min, lon_max, time)
    if sst_data is None:
        logger.info("CMEMS SST unavailable, using synthetic data")
        sst_data = synthetic.generate_sst_field(
            lat_min, lat_max, lon_min, lon_max, resolution, time
        )

    _redis_cache_put(cache_key, sst_data, CACHE_TTL_MINUTES * 60)
    return sst_data


def get_visibility_field(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    resolution: float = 1.0,
    time: datetime = None,
) -> WeatherData:
    """Get visibility field data.

    Provider chain: GFS live -> Synthetic.
    """
    if time is None:
        time = datetime.now(timezone.utc)

    cache_key = _get_cache_key("visibility", lat_min, lat_max, lon_min, lon_max)
    cached = _redis_cache_get(cache_key)
    if cached is not None:
        return cached

    p = _providers()
    gfs = p['gfs']
    synthetic = p['synthetic']

    vis_data = gfs.fetch_visibility_data(lat_min, lat_max, lon_min, lon_max, time)
    if vis_data is None:
        logger.info("GFS visibility unavailable, using synthetic data")
        vis_data = synthetic.generate_visibility_field(
            lat_min, lat_max, lon_min, lon_max, resolution, time
        )

    _redis_cache_put(cache_key, vis_data, CACHE_TTL_MINUTES * 60)
    return vis_data


def get_ice_field(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    resolution: float = 1.0,
    time: datetime = None,
) -> WeatherData:
    """Get sea ice concentration field.

    Provider chain: Redis -> DB -> CMEMS live -> Synthetic.
    """
    if time is None:
        time = datetime.now(timezone.utc)

    cache_key = _get_cache_key("ice", lat_min, lat_max, lon_min, lon_max)
    cached = _redis_cache_get(cache_key)
    if cached is not None:
        return cached

    p = _providers()
    db_weather = p.get('db_weather')
    copernicus = p['copernicus']
    synthetic = p['synthetic']

    # Try PostgreSQL (from previous ice forecast ingestion)
    if db_weather is not None:
        ice_data, _ = db_weather.get_ice_from_db(lat_min, lat_max, lon_min, lon_max, time)
        if ice_data is not None:
            logger.info("Ice data served from DB")
            _redis_cache_put(cache_key, ice_data, CACHE_TTL_MINUTES * 60)
            return ice_data

    ice_data = copernicus.fetch_ice_data(lat_min, lat_max, lon_min, lon_max, time)
    if ice_data is None:
        logger.info("CMEMS ice data unavailable, using synthetic data")
        ice_data = synthetic.generate_ice_field(
            lat_min, lat_max, lon_min, lon_max, resolution, time
        )

    _redis_cache_put(cache_key, ice_data, CACHE_TTL_MINUTES * 60)
    return ice_data


# ---------------------------------------------------------------------------
# Point weather query + voyage weather provider
# ---------------------------------------------------------------------------

def get_weather_at_point(lat: float, lon: float, time: datetime) -> Tuple[Dict, Optional[WeatherDataSource]]:
    """Get weather at a specific point.

    Uses unified provider that blends forecast and climatology.

    Returns:
        Tuple of (weather_dict, data_source) where data_source indicates
        whether data is from forecast, climatology, or blended.
    """
    p = _providers()
    unified = p['unified']
    copernicus = p['copernicus']

    try:
        # Use unified provider - handles forecast/climatology transition
        point_wx, source = unified.get_weather_at_point(lat, lon, time)

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

        point_wx = copernicus.get_weather_at_point(
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


def reset_voyage_data_sources():
    """Reset voyage data source tracking (call before each voyage calculation)."""
    global _voyage_data_sources
    _voyage_data_sources = []


def get_voyage_data_sources() -> List[Dict]:
    """Get tracked data sources from the last voyage calculation."""
    return _voyage_data_sources

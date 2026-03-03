"""
Weather data service for WINDMAR API.

Centralizes weather field fetching (provider chain) and point-weather queries.
Functions obtain data providers via ``get_app_state().weather_providers``.

Ocean masking has moved to ``api.weather.ocean_mask``.
Redis weather data cache has been removed (file cache is sufficient).
"""

import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from src.data.copernicus import WeatherData, WeatherDataSource
from src.optimization.voyage import LegWeather

try:
    import redis as redis_lib
except ImportError:
    redis_lib = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Redis client accessor (used by ForecastLayerManager for distributed locks)
# ---------------------------------------------------------------------------

_redis_client = None


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
        logger.info("Redis connected for distributed locks")
        return _redis_client
    except Exception as e:
        logger.warning(f"Redis unavailable: {e}")
        return None


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

    Provider chain: DB (pre-ingested) -> GFS live -> ERA5 -> Synthetic.
    """
    if time is None:
        time = datetime.now(timezone.utc)

    p = _providers()
    db_weather = p.get('db_weather')
    gfs = p['gfs']
    copernicus = p['copernicus']
    synthetic = p['synthetic']

    if db_weather is not None:
        wind_data, _ = db_weather.get_wind_from_db(lat_min, lat_max, lon_min, lon_max, time)
        if wind_data is not None:
            logger.info("Using DB pre-ingested wind data")
            return wind_data

    wind_data = gfs.fetch_wind_data(lat_min, lat_max, lon_min, lon_max, time)
    if wind_data is not None:
        logger.info("Using GFS near-real-time wind data")
    else:
        wind_data = copernicus.fetch_wind_data(lat_min, lat_max, lon_min, lon_max, time)
        if wind_data is not None:
            logger.info("GFS unavailable, using ERA5 reanalysis wind data")
        else:
            logger.info("GFS and ERA5 unavailable, using synthetic wind data")
            wind_data = synthetic.generate_wind_field(
                lat_min, lat_max, lon_min, lon_max, resolution, time
            )

    return wind_data


def get_wave_field(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    resolution: float = 1.0,
    wind_data: WeatherData = None,
) -> WeatherData:
    """Get wave field data.

    Provider chain: DB (pre-ingested) -> CMEMS live -> Synthetic.
    """
    p = _providers()
    db_weather = p.get('db_weather')
    copernicus = p['copernicus']
    synthetic = p['synthetic']

    if db_weather is not None:
        wave_data, _ = db_weather.get_wave_from_db(lat_min, lat_max, lon_min, lon_max)
        if wave_data is not None:
            logger.info("Using DB pre-ingested wave data")
            return wave_data

    wave_data = copernicus.fetch_wave_data(lat_min, lat_max, lon_min, lon_max)

    if wave_data is None:
        logger.info("Copernicus wave data unavailable, using synthetic data")
        wave_data = synthetic.generate_wave_field(
            lat_min, lat_max, lon_min, lon_max, resolution, wind_data
        )

    return wave_data


def get_current_field(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    resolution: float = 1.0,
) -> WeatherData:
    """Get ocean current field.

    Provider chain: DB (pre-ingested) -> CMEMS live -> Synthetic.
    """
    p = _providers()
    db_weather = p.get('db_weather')
    copernicus = p['copernicus']
    synthetic = p['synthetic']

    if db_weather is not None:
        current_data, _ = db_weather.get_current_from_db(lat_min, lat_max, lon_min, lon_max)
        if current_data is not None:
            logger.info("Using DB pre-ingested current data")
            return current_data

    current_data = copernicus.fetch_current_data(lat_min, lat_max, lon_min, lon_max)

    if current_data is None:
        logger.info("CMEMS current data unavailable, using synthetic data")
        current_data = synthetic.generate_current_field(
            lat_min, lat_max, lon_min, lon_max, resolution
        )

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

    p = _providers()
    copernicus = p['copernicus']
    synthetic = p['synthetic']

    sst_data = copernicus.fetch_sst_data(lat_min, lat_max, lon_min, lon_max, time)
    if sst_data is None:
        logger.info("CMEMS SST unavailable, using synthetic data")
        sst_data = synthetic.generate_sst_field(
            lat_min, lat_max, lon_min, lon_max, resolution, time
        )

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

    p = _providers()
    gfs = p['gfs']
    synthetic = p['synthetic']

    vis_data = gfs.fetch_visibility_data(lat_min, lat_max, lon_min, lon_max, time)
    if vis_data is None:
        logger.info("GFS visibility unavailable, using synthetic data")
        vis_data = synthetic.generate_visibility_field(
            lat_min, lat_max, lon_min, lon_max, resolution, time
        )

    return vis_data


def get_ice_field(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    resolution: float = 1.0,
    time: datetime = None,
) -> WeatherData:
    """Get sea ice concentration field.

    Provider chain: DB -> CMEMS live -> Synthetic.
    """
    if time is None:
        time = datetime.now(timezone.utc)

    p = _providers()
    db_weather = p.get('db_weather')
    copernicus = p['copernicus']
    synthetic = p['synthetic']

    if db_weather is not None:
        ice_data, _ = db_weather.get_ice_from_db(lat_min, lat_max, lon_min, lon_max, time)
        if ice_data is not None:
            logger.info("Ice data served from DB")
            return ice_data

    ice_data = copernicus.fetch_ice_data(lat_min, lat_max, lon_min, lon_max, time)
    if ice_data is None:
        logger.info("CMEMS ice data unavailable, using synthetic data")
        ice_data = synthetic.generate_ice_field(
            lat_min, lat_max, lon_min, lon_max, resolution, time
        )

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

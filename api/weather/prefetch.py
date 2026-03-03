"""
Background prefetch orchestration for weather layers.

Coordinates downloading forecast data from providers (CMEMS, GFS),
building frame caches, and persisting to PostgreSQL.
"""

import logging
import time as _time
from datetime import datetime
from pathlib import Path

from api.weather_fields import get_field, WEATHER_FIELDS, FIELD_NAMES
from api.weather.grid_processor import clamp_bbox
from api.weather.frame_builder import (
    build_frames_from_db,
    build_frames_from_provider,
    build_wind_frames_from_grib,
)
from api.forecast_layer_manager import ForecastLayerManager, cache_covers_bounds

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer manager instances — one per field (module-level singletons)
# ---------------------------------------------------------------------------

_layer_managers: dict[str, ForecastLayerManager] = {}


def get_layer_manager(field_name: str) -> ForecastLayerManager:
    """Get or create a ForecastLayerManager for a field."""
    if field_name not in _layer_managers:
        cfg = get_field(field_name)
        _layer_managers[field_name] = ForecastLayerManager(
            cfg.name,
            cache_subdir=cfg.cache_subdir or cfg.name,
            use_redis=cfg.use_redis,
        )
    return _layer_managers[field_name]


# Eagerly create managers for all fields at import time
for _fn in FIELD_NAMES:
    get_layer_manager(_fn)


# ---------------------------------------------------------------------------
# Lazy provider resolution
# ---------------------------------------------------------------------------

def _get_providers():
    from api.state import get_app_state
    return get_app_state().weather_providers


def _db_weather():
    from api.state import get_app_state
    return get_app_state().weather_providers.get('db_weather')


def _weather_ingestion():
    from api.state import get_app_state
    return get_app_state().weather_providers.get('weather_ingestion')


# ---------------------------------------------------------------------------
# Generic prefetch
# ---------------------------------------------------------------------------

def do_generic_prefetch(
    mgr: ForecastLayerManager,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    *,
    _skip_clamp: bool = False,
):
    """Generic prefetch that works for any CMEMS/GFS field.

    For wind, delegates to GFS-specific logic (GRIB file cache).
    For everything else, calls the provider's forecast method and builds frames.
    """
    field_name = mgr.name
    cfg = get_field(field_name)

    if not _skip_clamp:
        lat_min, lat_max, lon_min, lon_max = clamp_bbox(lat_min, lat_max, lon_min, lon_max)

    cache_key = mgr.make_cache_key(lat_min, lat_max, lon_min, lon_max)

    # Check if cache is already complete
    existing = mgr.cache_get(cache_key)
    min_frames = cfg.expected_frames
    if existing and len(existing.get("frames", {})) >= min_frames:
        if cache_covers_bounds(existing, lat_min, lat_max, lon_min, lon_max):
            logger.info(f"{field_name} forecast file cache already complete, skipping download")
            return

    # Try rebuild from DB first
    db_weather = _db_weather()
    if db_weather is not None:
        rebuilt = build_frames_from_db(field_name, db_weather, lat_min, lat_max, lon_min, lon_max)
        if rebuilt and len(rebuilt.get("frames", {})) >= min_frames:
            mgr.cache_put(cache_key, rebuilt)
            if cache_covers_bounds(rebuilt, lat_min, lat_max, lon_min, lon_max):
                logger.info(f"{field_name} forecast rebuilt from DB, skipping provider download")
                return

    # Clear stale cache
    stale_path = mgr.cache_path(cache_key)
    if stale_path.exists():
        stale_path.unlink(missing_ok=True)

    # Wind has special GFS logic
    if field_name == "wind":
        _do_wind_prefetch(mgr, lat_min, lat_max, lon_min, lon_max)
        return

    # Fetch from provider
    providers = _get_providers()
    weather_ingestion = _weather_ingestion()

    if cfg.source.startswith("gfs"):
        provider = providers['gfs']
    else:
        provider = providers['copernicus']

    logger.info(f"{field_name} forecast prefetch started")
    fetch_fn = getattr(provider, cfg.fetch_method)
    result = fetch_fn(lat_min, lat_max, lon_min, lon_max)

    if not result:
        if field_name == "ice":
            synthetic = providers['synthetic']
            result = synthetic.generate_ice_forecast(lat_min, lat_max, lon_min, lon_max)
        if not result:
            logger.error(f"{field_name} forecast fetch returned empty")
            return

    envelope = build_frames_from_provider(field_name, result, cfg)
    if envelope:
        mgr.cache_put(cache_key, envelope)

    # Persist to DB
    if weather_ingestion is not None:
        _INGEST_FRAMES_METHOD = {
            "waves": "ingest_wave_forecast_frames",
            "swell": "ingest_wave_forecast_frames",
            "currents": "ingest_current_forecast_frames",
            "ice": "ingest_ice_forecast_frames",
            "sst": "ingest_sst_forecast_frames",
            "visibility": "ingest_visibility_forecast_frames",
        }
        method_name = _INGEST_FRAMES_METHOD.get(field_name)
        if method_name:
            try:
                logger.info(f"Ingesting {field_name} forecast frames into PostgreSQL...")
                getattr(weather_ingestion, method_name)(result)
            except Exception as db_e:
                logger.error(f"{field_name} forecast DB ingestion failed: {db_e}")


# ---------------------------------------------------------------------------
# Wind-specific prefetch (GFS GRIB files)
# ---------------------------------------------------------------------------

def _do_wind_prefetch(mgr, lat_min, lat_max, lon_min, lon_max):
    """Download all GFS forecast hours and build wind frames cache."""
    gfs_provider = _get_providers()['gfs']
    run_date, run_hour = gfs_provider._get_latest_run()
    mgr.last_run = (run_date, run_hour)
    logger.info(f"GFS forecast prefetch started (run {run_date}/{run_hour}z)")
    gfs_provider.prefetch_forecast_hours(lat_min, lat_max, lon_min, lon_max)
    logger.info("GFS forecast prefetch completed, building frames cache...")

    result = build_wind_frames_from_grib(
        gfs_provider, lat_min, lat_max, lon_min, lon_max, run_date, run_hour,
    )
    cache_key = mgr.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    mgr.cache_put(cache_key, result)
    logger.info("Wind frames cache ready")


# ---------------------------------------------------------------------------
# Stale cache cleanup
# ---------------------------------------------------------------------------

def cleanup_stale_caches():
    """Delete stale CMEMS/GFS cache files to reclaim disk space."""
    now = _time.time()
    cleaned = 0

    cache_dir = Path("data/copernicus_cache")
    if cache_dir.exists():
        for f in cache_dir.glob("*.nc"):
            try:
                if now - f.stat().st_mtime > 24 * 3600:
                    f.unlink()
                    cleaned += 1
            except OSError:
                pass
        for f in cache_dir.glob("*.grib2"):
            try:
                if now - f.stat().st_mtime > 48 * 3600:
                    f.unlink()
                    cleaned += 1
            except OSError:
                pass

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

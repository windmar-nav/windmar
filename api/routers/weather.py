"""
Weather router for WINDMAR API — Generic Pipeline v2.

One set of generic endpoints replaces 31 bespoke per-field endpoints.
All field-specific behaviour is driven by the field registry in
``api.weather_fields``.

Endpoints:
    GET  /api/weather/{field}              → single-frame overlay
    GET  /api/weather/{field}/frames       → all forecast frames (timeline)
    GET  /api/weather/{field}/status       → forecast status / completeness
    POST /api/weather/{field}/prefetch     → trigger background download
    POST /api/weather/{field}/resync       → re-ingest from provider (sync)
    GET  /api/weather/health               → per-source health (kept)
    GET  /api/weather/freshness            → data age indicator (kept)
    GET  /api/weather/point                → point weather (kept)
"""

import asyncio
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from starlette.responses import Response

from api.demo import require_not_demo, is_demo, is_demo_user, demo_mode_response, limit_demo_frames
from api.state import get_app_state
from api.weather_fields import (
    WEATHER_FIELDS, FIELD_NAMES, LAYER_TO_SOURCE,
    get_field, validate_field_name, FieldConfig,
)
from api.weather_service import (
    get_wind_field, get_wave_field, get_current_field,
    get_sst_field, get_visibility_field, get_ice_field,
    get_weather_at_point,
    build_ocean_mask as _build_ocean_mask,
    apply_ocean_mask_velocity as _apply_ocean_mask_velocity,
)
from api.forecast_layer_manager import (
    ForecastLayerManager,
    cache_covers_bounds,
)
from src.data.copernicus import GFSDataProvider

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Weather"])


# ============================================================================
# Lazy provider resolution helpers
# ============================================================================

def _get_providers():
    return get_app_state().weather_providers


def _db_weather():
    return get_app_state().weather_providers.get('db_weather')


def _weather_ingestion():
    return get_app_state().weather_providers.get('weather_ingestion')


# ============================================================================
# Layer manager instances — one per field (module-level singletons)
# ============================================================================

_layer_managers: dict[str, ForecastLayerManager] = {}


def _get_layer_manager(field_name: str) -> ForecastLayerManager:
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
    _get_layer_manager(_fn)


# ============================================================================
# Constants & helpers
# ============================================================================

_OVERLAY_MAX_DIM = 500

# Maximum bbox span for CMEMS/GFS downloads — prevents OOM.
# Wave forecast at 30×70° ≈ 400 MB (8 params × 41 steps × ~304K grid points).
_MAX_LAT_SPAN = 30.0
_MAX_LON_SPAN = 70.0


def _clamp_bbox(lat_min, lat_max, lon_min, lon_max):
    """Clamp bbox to _MAX_LAT/LON_SPAN, centered on the original midpoint."""
    lat_span = lat_max - lat_min
    lon_span = lon_max - lon_min
    if lat_span > _MAX_LAT_SPAN or lon_span > _MAX_LON_SPAN:
        lat_center = (lat_min + lat_max) / 2
        lon_center = (lon_min + lon_max) / 2
        lat_half = min(lat_span / 2, _MAX_LAT_SPAN / 2)
        lon_half = min(lon_span / 2, _MAX_LON_SPAN / 2)
        lat_min = max(-89.9, lat_center - lat_half)
        lat_max = min(89.9, lat_center + lat_half)
        lon_min = max(-180.0, lon_center - lon_half)
        lon_max = min(180.0, lon_center + lon_half)
    return lat_min, lat_max, lon_min, lon_max


def _overlay_step(lats, lons):
    """Compute subsample step for overlay grids exceeding target size."""
    return max(1, math.ceil(max(len(lats), len(lons)) / _OVERLAY_MAX_DIM))


def _sub2d(arr, step, decimals=2, nan_fill=0.0):
    """Subsample, sanitize NaN/Inf, round a 2D numpy array. Returns list or None."""
    if arr is None:
        return None
    sub = arr[::step, ::step]
    clean = np.where(np.isfinite(sub), sub, nan_fill)
    return np.round(clean, decimals).tolist()


def _dynamic_mask_step(lat_min, lat_max, lon_min, lon_max):
    """Compute ocean mask step that keeps grid under 500 points per axis."""
    span = max(lat_max - lat_min, lon_max - lon_min)
    return round(max(0.05, span / 500), 3)


def _cleanup_stale_caches():
    """Delete stale CMEMS/GFS cache files to reclaim disk space."""
    import time as _time
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


def _sanitize_nan(val, fill=None):
    """Replace NaN/Infinity with fill value in a numpy array."""
    if val is None:
        return None
    return np.where(np.isfinite(val), val, fill if fill is not None else 0.0)


# ============================================================================
# Generic: build forecast frames from DB for any field
# ============================================================================

def _rebuild_frames_from_db(field_name: str, cache_key: str,
                            lat_min: float, lat_max: float,
                            lon_min: float, lon_max: float):
    """Rebuild forecast frame cache from PostgreSQL for any field.

    This single function replaces _rebuild_wind_cache_from_db,
    _rebuild_wave_cache_from_db, _rebuild_current_cache_from_db,
    _rebuild_ice_cache_from_db, _rebuild_sst_cache_from_db, and
    _rebuild_vis_cache_from_db.
    """
    db_weather = _db_weather()
    if db_weather is None:
        return None

    cfg = get_field(field_name)
    run_time, hours = db_weather.get_available_hours_by_source(cfg.source)
    if not hours:
        return None

    logger.info(f"Rebuilding {field_name} cache from DB: {len(hours)} hours")

    grids = db_weather.get_grids_for_timeline(
        cfg.source, list(cfg.parameters),
        lat_min, lat_max, lon_min, lon_max, hours,
    )

    if not grids:
        return None

    # Find a representative parameter to get grid dimensions
    primary_param = cfg.parameters[0]
    if primary_param not in grids or not grids[primary_param]:
        return None

    first_fh = min(grids[primary_param].keys())
    lats_full, lons_full, _ = grids[primary_param][first_fh]
    STEP = _overlay_step(lats_full, lons_full)
    shared_lats = lats_full[::STEP]
    shared_lons = lons_full[::STEP]

    # Build ocean mask at data resolution
    ocean_mask_data = None
    if cfg.needs_ocean_mask:
        mask_step = max(STEP, 1)
        mask_lats = lats_full[::mask_step]
        mask_lons = lons_full[::mask_step]
        try:
            from global_land_mask import globe
            lon_grid, lat_grid = np.meshgrid(mask_lons, mask_lats)
            ocean_mask_data = globe.is_ocean(lat_grid, lon_grid).tolist()
        except ImportError:
            ocean_mask_data = None

    def _sub(arr, dec=cfg.decimals):
        if arr is None:
            return None
        return np.round(arr[::STEP, ::STEP], dec).tolist()

    frames = {}

    if cfg.components == "vector":
        u_param, v_param = cfg.parameters[0], cfg.parameters[1]
        for fh in sorted(hours):
            if fh not in grids.get(u_param, {}) or fh not in grids.get(v_param, {}):
                continue
            _, _, u_data = grids[u_param][fh]
            _, _, v_data = grids[v_param][fh]

            if field_name == "wind":
                # Wind uses leaflet-velocity format
                u_masked, v_masked = _apply_ocean_mask_velocity(u_data, v_data, lats_full, lons_full)
                actual_dx = abs(float(lons_full[1] - lons_full[0])) if len(lons_full) > 1 else 0.25
                actual_dy = abs(float(lats_full[1] - lats_full[0])) if len(lats_full) > 1 else 0.25
                if len(lats_full) > 1 and lats_full[1] > lats_full[0]:
                    u_ordered = u_masked[::-1]
                    v_ordered = v_masked[::-1]
                    lat_north = float(lats_full[-1])
                    lat_south = float(lats_full[0])
                else:
                    u_ordered = u_masked
                    v_ordered = v_masked
                    lat_north = float(lats_full[0])
                    lat_south = float(lats_full[-1])
                header = {
                    "parameterCategory": 2, "parameterNumber": 2,
                    "lo1": float(lons_full[0]), "la1": lat_north,
                    "lo2": float(lons_full[-1]), "la2": lat_south,
                    "dx": actual_dx, "dy": actual_dy,
                    "nx": len(lons_full), "ny": len(lats_full),
                    "refTime": run_time.isoformat() if run_time else "",
                    "forecastHour": fh,
                }
                frames[str(fh)] = [
                    {"header": {**header, "parameterNumber": 2}, "data": u_ordered.flatten().tolist()},
                    {"header": {**header, "parameterNumber": 3}, "data": v_ordered.flatten().tolist()},
                ]
            else:
                # Currents: subsample + ocean mask
                u_sub = u_data[::STEP, ::STEP]
                v_sub = v_data[::STEP, ::STEP]
                u_m, v_m = _apply_ocean_mask_velocity(u_sub, v_sub, shared_lats, shared_lons)
                frames[str(fh)] = {
                    "u": np.round(u_m[::-1], cfg.decimals).tolist(),
                    "v": np.round(v_m[::-1], cfg.decimals).tolist(),
                }

    elif cfg.components == "wave_decomp":
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

    elif cfg.components == "scalar":
        param = cfg.parameters[0]
        for fh in sorted(hours):
            if fh not in grids.get(param, {}):
                continue
            _, _, d = grids[param][fh]
            clean = np.nan_to_num(d[::STEP, ::STEP], nan=cfg.nan_fill)
            frames[str(fh)] = {"data": np.round(clean, cfg.decimals).tolist()}

    # Build cache data envelope
    cache_data = {
        "run_time": run_time.isoformat() if run_time else "",
        "total_hours": len(frames),
        "cached_hours": len(frames),
        "source": cfg.source.split("_")[0],  # "gfs" or "cmems"
        "field": field_name,
        "frames": frames,
    }

    # Wind uses leaflet-velocity format — different envelope
    if field_name == "wind":
        run_date_str = run_time.strftime("%Y%m%d") if run_time else ""
        run_hour_str = run_time.strftime("%H") if run_time else "00"
        cache_data["run_date"] = run_date_str
        cache_data["run_hour"] = run_hour_str
        cache_data["total_hours"] = len(GFSDataProvider.FORECAST_HOURS)
    else:
        cache_data["lats"] = shared_lats.tolist() if hasattr(shared_lats, 'tolist') else list(shared_lats)
        cache_data["lons"] = shared_lons.tolist() if hasattr(shared_lons, 'tolist') else list(shared_lons)
        cache_data["ny"] = len(shared_lats)
        cache_data["nx"] = len(shared_lons)

        if ocean_mask_data is not None:
            cache_data["ocean_mask"] = ocean_mask_data
            cache_data["ocean_mask_lats"] = mask_lats.tolist()
            cache_data["ocean_mask_lons"] = mask_lons.tolist()

        if cfg.colorscale_colors:
            cs = {
                "min": cfg.colorscale_min,
                "max": cfg.colorscale_max,
                "colors": list(cfg.colorscale_colors),
            }
            # SST: compute data range for dynamic colorscale
            if field_name == "sst":
                all_vals = []
                for f in frames.values():
                    if isinstance(f, dict) and "data" in f:
                        flat = [v for row in f["data"] for v in row if v > -100]
                        all_vals.extend(flat)
                if all_vals:
                    cs["data_min"] = round(min(all_vals), 2)
                    cs["data_max"] = round(max(all_vals), 2)
            cache_data["colorscale"] = cs

    mgr = _get_layer_manager(field_name)
    mgr.cache_put(cache_key, cache_data)
    logger.info(f"{field_name} cache rebuilt from DB: {len(frames)} frames")
    return cache_data


# ============================================================================
# Generic: prefetch from provider (background task)
# ============================================================================

def _do_generic_prefetch(mgr: ForecastLayerManager, lat_min: float, lat_max: float,
                         lon_min: float, lon_max: float):
    """Generic prefetch that works for any CMEMS/GFS field.

    For wind, delegates to GFS-specific logic (GRIB file cache).
    For everything else, calls the provider's forecast method and builds frames.
    """
    field_name = mgr.name
    cfg = get_field(field_name)

    # Cap bbox to prevent OOM on large viewports
    lat_min, lat_max, lon_min, lon_max = _clamp_bbox(lat_min, lat_max, lon_min, lon_max)

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
    if db_weather is not None and field_name != "wind":
        rebuilt = _rebuild_frames_from_db(field_name, cache_key, lat_min, lat_max, lon_min, lon_max)
        if rebuilt and len(rebuilt.get("frames", {})) >= min_frames:
            if cache_covers_bounds(rebuilt, lat_min, lat_max, lon_min, lon_max):
                logger.info(f"{field_name} forecast rebuilt from DB, skipping provider download")
                return

    # Clear stale cache
    stale_path = mgr.cache_path(cache_key)
    if stale_path.exists():
        stale_path.unlink(missing_ok=True)

    # Wind has special GFS logic
    if field_name == "wind":
        _do_wind_prefetch_impl(mgr, lat_min, lat_max, lon_min, lon_max)
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
        # Ice fallback: synthetic
        if field_name == "ice":
            synthetic = providers['synthetic']
            result = synthetic.generate_ice_forecast(lat_min, lat_max, lon_min, lon_max)
        if not result:
            logger.error(f"{field_name} forecast fetch returned empty")
            return

    first_wd = next(iter(result.values()))
    STEP = _overlay_step(first_wd.lats, first_wd.lons)
    logger.info(f"{field_name} forecast: grid {len(first_wd.lats)}x{len(first_wd.lons)}, STEP={STEP}")
    sub_lats = first_wd.lats[::STEP]
    sub_lons = first_wd.lons[::STEP]

    # Build ocean mask at DATA resolution (not a separate high-res mask)
    ocean_mask_data = None
    mask_lats_list = None
    mask_lons_list = None
    if cfg.needs_ocean_mask:
        try:
            from global_land_mask import globe
            lon_grid, lat_grid = np.meshgrid(sub_lons, sub_lats)
            ocean_mask_data = globe.is_ocean(lat_grid, lon_grid).tolist()
            mask_lats_list = sub_lats.tolist() if hasattr(sub_lats, 'tolist') else list(sub_lats)
            mask_lons_list = sub_lons.tolist() if hasattr(sub_lons, 'tolist') else list(sub_lons)
        except ImportError:
            pass

    def _subsample_round(arr, dec=cfg.decimals):
        if arr is None:
            return None
        return np.round(arr[::STEP, ::STEP], dec).tolist()

    frames = {}

    if cfg.components == "vector":
        # Currents
        for fh, wd in sorted(result.items()):
            if wd.u_component is not None and wd.v_component is not None:
                u_sub = wd.u_component[::STEP, ::STEP]
                v_sub = wd.v_component[::STEP, ::STEP]
                u_m, v_m = _apply_ocean_mask_velocity(u_sub, v_sub, sub_lats, sub_lons)
                frames[str(fh)] = {
                    "u": np.round(u_m[::-1], cfg.decimals).tolist(),
                    "v": np.round(v_m[::-1], cfg.decimals).tolist(),
                }

    elif cfg.components == "wave_decomp":
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

    elif cfg.components == "scalar":
        global_min, global_max = float('inf'), float('-inf')
        for fh, wd in sorted(result.items()):
            vals = getattr(wd, field_name, None) or wd.values
            if field_name == "ice":
                vals = wd.ice_concentration if wd.ice_concentration is not None else wd.values
            if field_name == "visibility":
                vals = wd.visibility if wd.visibility is not None else wd.values
            if field_name == "sst":
                vals = wd.sst if wd.sst is not None else wd.values
            if vals is not None:
                clean = np.nan_to_num(vals[::STEP, ::STEP], nan=cfg.nan_fill)
                if field_name == "sst":
                    valid = clean[clean > -100]
                    if valid.size > 0:
                        global_min = min(global_min, float(np.min(valid)))
                        global_max = max(global_max, float(np.max(valid)))
                frames[str(fh)] = {"data": np.round(clean, cfg.decimals).tolist()}

    # Build cache envelope
    cache_data = {
        "run_time": first_wd.time.isoformat() if first_wd.time else "",
        "total_hours": len(frames),
        "cached_hours": len(frames),
        "source": cfg.source.split("_")[0],
        "field": field_name,
        "lats": sub_lats.tolist() if hasattr(sub_lats, 'tolist') else list(sub_lats),
        "lons": sub_lons.tolist() if hasattr(sub_lons, 'tolist') else list(sub_lons),
        "ny": len(sub_lats),
        "nx": len(sub_lons),
        "frames": frames,
    }

    if ocean_mask_data is not None:
        cache_data["ocean_mask"] = ocean_mask_data
        cache_data["ocean_mask_lats"] = mask_lats_list
        cache_data["ocean_mask_lons"] = mask_lons_list

    if cfg.colorscale_colors:
        cs = {
            "min": cfg.colorscale_min,
            "max": cfg.colorscale_max,
            "colors": list(cfg.colorscale_colors),
        }
        if field_name == "sst" and np.isfinite(global_min):
            cs["data_min"] = round(global_min, 2)
            cs["data_max"] = round(global_max, 2)
        cache_data["colorscale"] = cs

    mgr.cache_put(cache_key, cache_data)
    logger.info(f"{field_name} forecast cached: {len(frames)} frames")

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


# ============================================================================
# Wind-specific prefetch (GFS GRIB files)
# ============================================================================

def _do_wind_prefetch_impl(mgr, lat_min, lat_max, lon_min, lon_max):
    """Download all GFS forecast hours and build wind frames cache."""
    gfs_provider = _get_providers()['gfs']
    run_date, run_hour = gfs_provider._get_latest_run()
    mgr.last_run = (run_date, run_hour)
    logger.info(f"GFS forecast prefetch started (run {run_date}/{run_hour}z)")
    gfs_provider.prefetch_forecast_hours(lat_min, lat_max, lon_min, lon_max)
    logger.info("GFS forecast prefetch completed, building frames cache...")
    _build_wind_frames(lat_min, lat_max, lon_min, lon_max, run_date, run_hour)
    logger.info("Wind frames cache ready")


def _build_wind_frames(lat_min, lat_max, lon_min, lon_max, run_date, run_hour):
    """Process all cached GRIB files into leaflet-velocity frames dict."""
    gfs_provider = _get_providers()['gfs']
    run_time = datetime.strptime(f"{run_date}{run_hour}", "%Y%m%d%H")
    hours_status = gfs_provider.get_cached_forecast_hours(lat_min, lat_max, lon_min, lon_max, run_date, run_hour)

    mgr = _get_layer_manager("wind")
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
            "parameterCategory": 2, "parameterNumber": 2,
            "lo1": float(actual_lons[0]), "la1": lat_north,
            "lo2": float(actual_lons[-1]), "la2": lat_south,
            "dx": actual_dx, "dy": actual_dy,
            "nx": len(actual_lons), "ny": len(actual_lats),
            "refTime": run_time.isoformat() if run_time else valid_time.isoformat(),
            "forecastHour": fh,
        }
        frames[str(fh)] = [
            {"header": {**header, "parameterNumber": 2}, "data": u_ordered.flatten().tolist()},
            {"header": {**header, "parameterNumber": 3}, "data": v_ordered.flatten().tolist()},
        ]

    result = {
        "run_date": run_date,
        "run_hour": run_hour,
        "run_time": run_time.isoformat(),
        "total_hours": len(GFSDataProvider.FORECAST_HOURS),
        "cached_hours": len(frames),
        "source": "gfs",
        "frames": frames,
    }

    cache_key = mgr.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    mgr.cache_put(cache_key, result)
    logger.info(f"Wind frames cache saved: {len(frames)} frames")
    return result


# ============================================================================
# Single-frame field fetcher map
# ============================================================================

_SINGLE_FRAME_FETCHER = {
    "wind": lambda params, time: get_wind_field(
        params["lat_min"], params["lat_max"], params["lon_min"], params["lon_max"],
        params["resolution"], time),
    "waves": lambda params, time: get_wave_field(
        params["lat_min"], params["lat_max"], params["lon_min"], params["lon_max"],
        params["resolution"],
        get_wind_field(params["lat_min"], params["lat_max"], params["lon_min"], params["lon_max"],
                       params["resolution"], time)),
    "swell": lambda params, time: get_wave_field(
        params["lat_min"], params["lat_max"], params["lon_min"], params["lon_max"],
        params["resolution"]),
    "currents": lambda params, time: get_current_field(
        params["lat_min"], params["lat_max"], params["lon_min"], params["lon_max"],
        params["resolution"]),
    "sst": lambda params, time: get_sst_field(
        params["lat_min"], params["lat_max"], params["lon_min"], params["lon_max"],
        params["resolution"], time),
    "visibility": lambda params, time: get_visibility_field(
        params["lat_min"], params["lat_max"], params["lon_min"], params["lon_max"],
        params["resolution"], time),
    "ice": lambda params, time: get_ice_field(
        params["lat_min"], params["lat_max"], params["lon_min"], params["lon_max"],
        params["resolution"], time),
}

# DB-first fetcher methods on db_weather provider
_DB_FIRST_METHODS = {
    "wind": "get_wind_from_db",
    "waves": "get_wave_from_db",
    "sst": "get_sst_from_db",
    "visibility": "get_visibility_from_db",
    "ice": "get_ice_from_db",
}


# ============================================================================
# Static endpoints — MUST be before {field} parameterized routes
# ============================================================================

@router.get("/api/weather/health")
async def api_weather_health():
    """Return per-source health status for all weather sources."""
    db_weather = _db_weather()
    if db_weather is None:
        raise HTTPException(status_code=503, detail="Database weather provider not configured")
    health = await asyncio.to_thread(db_weather.get_health)
    return health


@router.get("/api/weather/point")
async def api_get_weather_point(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    time: Optional[datetime] = None,
):
    """Get weather at a specific point (wind, waves, currents)."""
    if time is None:
        time = datetime.now(timezone.utc)

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


@router.get("/api/weather/freshness")
async def get_weather_freshness(request: Request):
    """Get weather data freshness indicator (age of most recent data)."""
    if is_demo() and is_demo_user(request):
        return demo_mode_response("Weather freshness")

    db_weather = _db_weather()
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
# Backward-compatible route aliases (old forecast/* URLs)
# ============================================================================
# The frontend uses /api/weather/forecast/{layer}/status etc.
# The generic endpoints use /api/weather/{field}/status.
# These wrappers MUST be registered BEFORE the generic {field} routes
# so FastAPI matches the literal "forecast" path segment first.

# --- Wind forecast (old: /api/weather/forecast/...) ---

@router.get("/api/weather/forecast/status")
async def _compat_wind_status(
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    return await api_get_field_status(field="wind", lat_min=lat_min,
                                      lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)


@router.post("/api/weather/forecast/prefetch",
             dependencies=[Depends(require_not_demo("Weather prefetch"))])
async def _compat_wind_prefetch(
    background_tasks: BackgroundTasks,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    return await api_trigger_field_prefetch(field="wind", background_tasks=background_tasks,
                                            lat_min=lat_min, lat_max=lat_max,
                                            lon_min=lon_min, lon_max=lon_max)


@router.get("/api/weather/forecast/frames")
async def _compat_wind_frames(
    request: Request,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    return await api_get_field_frames(field="wind", request=request,
                                      lat_min=lat_min, lat_max=lat_max,
                                      lon_min=lon_min, lon_max=lon_max)


# --- Wave forecast (old: /api/weather/forecast/wave/...) ---

@router.get("/api/weather/forecast/wave/status")
async def _compat_wave_status(
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    return await api_get_field_status(field="waves", lat_min=lat_min,
                                      lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)


@router.post("/api/weather/forecast/wave/prefetch",
             dependencies=[Depends(require_not_demo("Weather prefetch"))])
async def _compat_wave_prefetch(
    background_tasks: BackgroundTasks,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    return await api_trigger_field_prefetch(field="waves", background_tasks=background_tasks,
                                            lat_min=lat_min, lat_max=lat_max,
                                            lon_min=lon_min, lon_max=lon_max)


@router.get("/api/weather/forecast/wave/frames")
async def _compat_wave_frames(
    request: Request,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    return await api_get_field_frames(field="waves", request=request,
                                      lat_min=lat_min, lat_max=lat_max,
                                      lon_min=lon_min, lon_max=lon_max)


# --- Current forecast (old: /api/weather/forecast/current/...) ---

@router.get("/api/weather/forecast/current/status")
async def _compat_current_status(
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    return await api_get_field_status(field="currents", lat_min=lat_min,
                                      lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)


@router.post("/api/weather/forecast/current/prefetch",
             dependencies=[Depends(require_not_demo("Weather prefetch"))])
async def _compat_current_prefetch(
    background_tasks: BackgroundTasks,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    return await api_trigger_field_prefetch(field="currents", background_tasks=background_tasks,
                                            lat_min=lat_min, lat_max=lat_max,
                                            lon_min=lon_min, lon_max=lon_max)


@router.get("/api/weather/forecast/current/frames")
async def _compat_current_frames(
    request: Request,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    return await api_get_field_frames(field="currents", request=request,
                                      lat_min=lat_min, lat_max=lat_max,
                                      lon_min=lon_min, lon_max=lon_max)


# --- Ice forecast (old: /api/weather/forecast/ice/...) ---

@router.get("/api/weather/forecast/ice/status")
async def _compat_ice_status(
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    return await api_get_field_status(field="ice", lat_min=lat_min,
                                      lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)


@router.post("/api/weather/forecast/ice/prefetch",
             dependencies=[Depends(require_not_demo("Weather prefetch"))])
async def _compat_ice_prefetch(
    background_tasks: BackgroundTasks,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    return await api_trigger_field_prefetch(field="ice", background_tasks=background_tasks,
                                            lat_min=lat_min, lat_max=lat_max,
                                            lon_min=lon_min, lon_max=lon_max)


@router.get("/api/weather/forecast/ice/frames")
async def _compat_ice_frames(
    request: Request,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    return await api_get_field_frames(field="ice", request=request,
                                      lat_min=lat_min, lat_max=lat_max,
                                      lon_min=lon_min, lon_max=lon_max)


# --- SST forecast (old: /api/weather/forecast/sst/...) ---

@router.get("/api/weather/forecast/sst/status")
async def _compat_sst_status(
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    return await api_get_field_status(field="sst", lat_min=lat_min,
                                      lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)


@router.post("/api/weather/forecast/sst/prefetch",
             dependencies=[Depends(require_not_demo("Weather prefetch"))])
async def _compat_sst_prefetch(
    background_tasks: BackgroundTasks,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    return await api_trigger_field_prefetch(field="sst", background_tasks=background_tasks,
                                            lat_min=lat_min, lat_max=lat_max,
                                            lon_min=lon_min, lon_max=lon_max)


@router.get("/api/weather/forecast/sst/frames")
async def _compat_sst_frames(
    request: Request,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    return await api_get_field_frames(field="sst", request=request,
                                      lat_min=lat_min, lat_max=lat_max,
                                      lon_min=lon_min, lon_max=lon_max)


# --- Visibility forecast (old: /api/weather/forecast/visibility/...) ---

@router.get("/api/weather/forecast/visibility/status")
async def _compat_vis_status(
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    return await api_get_field_status(field="visibility", lat_min=lat_min,
                                      lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)


@router.post("/api/weather/forecast/visibility/prefetch",
             dependencies=[Depends(require_not_demo("Weather prefetch"))])
async def _compat_vis_prefetch(
    background_tasks: BackgroundTasks,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    return await api_trigger_field_prefetch(field="visibility", background_tasks=background_tasks,
                                            lat_min=lat_min, lat_max=lat_max,
                                            lon_min=lon_min, lon_max=lon_max)


@router.get("/api/weather/forecast/visibility/frames")
async def _compat_vis_frames(
    request: Request,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    return await api_get_field_frames(field="visibility", request=request,
                                      lat_min=lat_min, lat_max=lat_max,
                                      lon_min=lon_min, lon_max=lon_max)


# ============================================================================
# Generic API Endpoints (parameterized by {field})
# ============================================================================

@router.get("/api/weather/{field}")
async def api_get_weather_field(
    field: str,
    lat_min: float = Query(30.0, ge=-90, le=90),
    lat_max: float = Query(60.0, ge=-90, le=90),
    lon_min: float = Query(-15.0, ge=-180, le=180),
    lon_max: float = Query(40.0, ge=-180, le=180),
    resolution: float = Query(1.0, ge=0.25, le=5.0),
    time: Optional[datetime] = None,
):
    """Get single-frame weather field data for visualization.

    Generic endpoint for all 7 fields. Returns grid data with
    ocean mask, colorscale, and field-specific properties.
    """
    # Validate field name — reject early for unknown fields
    if field not in WEATHER_FIELDS:
        raise HTTPException(status_code=400,
                            detail=f"Unknown field: {field}. Valid: {list(FIELD_NAMES)}")

    cfg = get_field(field)

    if time is None:
        time = datetime.now(timezone.utc)

    # Cap bbox to prevent OOM on large viewports
    lat_min, lat_max, lon_min, lon_max = _clamp_bbox(lat_min, lat_max, lon_min, lon_max)

    params = dict(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min,
                  lon_max=lon_max, resolution=resolution)

    ingested_at = None
    db_weather = _db_weather()

    # DB-first for fields that support it
    data = None
    db_method = _DB_FIRST_METHODS.get(field)
    if db_weather is not None and db_method is not None:
        fetch = getattr(db_weather, db_method)
        if field in ("wind", "sst", "visibility"):
            data, ingested_at = fetch(lat_min, lat_max, lon_min, lon_max, time)
        else:
            data, ingested_at = fetch(lat_min, lat_max, lon_min, lon_max)

    if data is None:
        data = _SINGLE_FRAME_FETCHER[field](params, time)
        if db_weather is not None:
            ingested_at = datetime.now(timezone.utc)

    if data is None or not hasattr(data, 'lats') or data.lats is None:
        raise HTTPException(status_code=503,
                            detail=f"No {field} data available. Try resyncing.")

    # Subsample
    step = _overlay_step(data.lats, data.lons)
    sub_lats = data.lats[::step].tolist()
    sub_lons = data.lons[::step].tolist()

    # Build response
    response = {
        "parameter": cfg.parameters[0] if cfg.components == "scalar" else cfg.name,
        "field": field,
        "time": time.isoformat(),
        "bbox": {
            "lat_min": float(data.lats.min()),
            "lat_max": float(data.lats.max()),
            "lon_min": float(data.lons.min()),
            "lon_max": float(data.lons.max()),
        },
        "resolution": resolution,
        "nx": len(sub_lons),
        "ny": len(sub_lats),
        "lats": sub_lats,
        "lons": sub_lons,
        "unit": cfg.unit,
        "source": cfg.source.split("_")[0],
    }

    # Ocean mask at data resolution
    if cfg.needs_ocean_mask:
        mask_step = _dynamic_mask_step(lat_min, lat_max, lon_min, lon_max)
        mask_lats, mask_lons, ocean_mask = _build_ocean_mask(
            lat_min, lat_max, lon_min, lon_max, step=mask_step,
        )
        response["ocean_mask"] = ocean_mask
        response["ocean_mask_lats"] = mask_lats
        response["ocean_mask_lons"] = mask_lons

    # Colorscale
    if cfg.colorscale_colors:
        response["colorscale"] = {
            "min": cfg.colorscale_min,
            "max": cfg.colorscale_max,
            "colors": list(cfg.colorscale_colors),
        }

    # Field-specific data
    if cfg.components == "vector":
        response["u"] = _sub2d(data.u_component, step) if data.u_component is not None else []
        response["v"] = _sub2d(data.v_component, step) if data.v_component is not None else []

    elif cfg.components == "wave_decomp":
        response["data"] = _sub2d(data.values, step)
        if data.wave_direction is not None:
            response["direction"] = _sub2d(data.wave_direction, step, 1)

        has_decomp = data.windwave_height is not None and data.swell_height is not None
        response["has_decomposition"] = has_decomp

        if has_decomp:
            response["windwave"] = {
                "height": _sub2d(data.windwave_height, step),
                "period": _sub2d(data.windwave_period, step, 1),
                "direction": _sub2d(data.windwave_direction, step, 1),
            }
            response["swell"] = {
                "height": _sub2d(data.swell_height, step),
                "period": _sub2d(data.swell_period, step, 1),
                "direction": _sub2d(data.swell_direction, step, 1),
            }

        # Swell-specific extra fields
        if field == "swell":
            response["total_hs"] = _sub2d(data.values, step)
            response["swell_hs"] = _sub2d(data.swell_height, step)
            response["swell_tp"] = _sub2d(data.swell_period, step, 1)
            response["swell_dir"] = _sub2d(data.swell_direction, step, 1)
            response["windsea_hs"] = _sub2d(data.windwave_height, step)
            response["windsea_tp"] = _sub2d(data.windwave_period, step, 1)
            response["windsea_dir"] = _sub2d(data.windwave_direction, step, 1)

    elif cfg.components == "scalar":
        clean = np.nan_to_num(data.values[::step, ::step], nan=cfg.nan_fill)
        response["data"] = np.round(clean, cfg.decimals).tolist()

        # Ice: prefer ice_concentration field if available
        if field == "ice" and data.ice_concentration is not None:
            clean_ice = np.nan_to_num(data.ice_concentration[::step, ::step], nan=0.0)
            response["data"] = np.round(clean_ice, cfg.decimals).tolist()

    if ingested_at is not None:
        response["ingested_at"] = ingested_at.isoformat()

    return response


@router.get("/api/weather/{field}/velocity")
async def api_get_velocity_format(
    field: str,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
    resolution: float = Query(1.0),
    time: Optional[datetime] = None,
    forecast_hour: int = Query(0, ge=0, le=120),
):
    """Get vector field data in leaflet-velocity compatible format.

    Works for wind and currents.
    """
    if field not in ("wind", "currents"):
        raise HTTPException(status_code=400, detail=f"Velocity format only for wind/currents, not {field}")

    if time is None:
        time = datetime.now(timezone.utc)

    if field == "wind":
        gfs_provider = _get_providers()['gfs']
        if forecast_hour > 0:
            data = gfs_provider.fetch_wind_data(lat_min, lat_max, lon_min, lon_max, time, forecast_hour)
            if data is None:
                raise HTTPException(status_code=404, detail=f"Forecast hour f{forecast_hour:03d} not available")
        else:
            data = get_wind_field(lat_min, lat_max, lon_min, lon_max, resolution, time)
    else:
        data = get_current_field(lat_min, lat_max, lon_min, lon_max, resolution)

    u_masked, v_masked = _apply_ocean_mask_velocity(
        data.u_component, data.v_component, data.lats, data.lons,
    )

    step = _overlay_step(data.lats, data.lons)
    if step > 1:
        u_masked = u_masked[::step, ::step]
        v_masked = v_masked[::step, ::step]
        actual_lats = data.lats[::step]
        actual_lons = data.lons[::step]
    else:
        actual_lats = data.lats
        actual_lons = data.lons

    actual_dx = abs(float(actual_lons[1] - actual_lons[0])) if len(actual_lons) > 1 else resolution
    actual_dy = abs(float(actual_lats[1] - actual_lats[0])) if len(actual_lats) > 1 else resolution

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
        "parameterCategory": 2, "parameterNumber": 2,
        "lo1": float(actual_lons[0]), "la1": lat_north,
        "lo2": float(actual_lons[-1]), "la2": lat_south,
        "dx": actual_dx, "dy": actual_dy,
        "nx": len(actual_lons), "ny": len(actual_lats),
        "refTime": (data.time.isoformat() if isinstance(data.time, datetime) else time.isoformat()),
    }

    u_flat = np.nan_to_num(u_ordered.flatten(), nan=0.0, posinf=0.0, neginf=0.0)
    v_flat = np.nan_to_num(v_ordered.flatten(), nan=0.0, posinf=0.0, neginf=0.0)
    return [
        {"header": {**header, "parameterNumber": 2}, "data": u_flat.tolist()},
        {"header": {**header, "parameterNumber": 3}, "data": v_flat.tolist()},
    ]


@router.get("/api/weather/{field}/status")
async def api_get_field_status(
    field: str,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    """Get forecast status for any weather field."""
    if field not in WEATHER_FIELDS:
        raise HTTPException(status_code=400,
                            detail=f"Unknown field: {field}. Valid: {list(FIELD_NAMES)}")

    cfg = get_field(field)
    mgr = _get_layer_manager(field)
    db_weather = _db_weather()

    cache_key = mgr.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    cached = mgr.cache_get(cache_key)
    prefetch_running = mgr.is_running
    total_hours = cfg.expected_frames

    if cached and not prefetch_running:
        cached_hours = len(cached.get("frames", {}))
        result = {
            "total_hours": total_hours,
            "cached_hours": cached_hours,
            "complete": cached_hours >= total_hours,
            "prefetch_running": False,
        }
        # Wind: include run_date/run_hour
        if field == "wind":
            result["run_date"] = cached.get("run_date", "")
            result["run_hour"] = cached.get("run_hour", "")
        return result

    # Wind: fall back to scanning GRIB files
    if field == "wind":
        gfs_provider = _get_providers()['gfs']
        if mgr.last_run:
            run_date, run_hour = mgr.last_run
        else:
            run_date, run_hour = gfs_provider._get_latest_run()
        hours = gfs_provider.get_cached_forecast_hours(lat_min, lat_max, lon_min, lon_max, run_date, run_hour)
        cached_count = sum(1 for h in hours if h["cached"])

        if cached_count == 0 and not prefetch_running:
            best = gfs_provider.find_best_cached_run(lat_min, lat_max, lon_min, lon_max)
            if best:
                run_date, run_hour = best
                hours = gfs_provider.get_cached_forecast_hours(lat_min, lat_max, lon_min, lon_max, run_date, run_hour)
        cached_count = sum(1 for h in hours if h["cached"])

        if cached_count == 0 and db_weather is not None:
            db_run_time, db_hours = db_weather.get_available_hours_by_source("gfs")
            if db_hours:
                return {
                    "run_date": db_run_time.strftime("%Y%m%d") if db_run_time else run_date,
                    "run_hour": db_run_time.strftime("%H") if db_run_time else run_hour,
                    "total_hours": len(GFSDataProvider.FORECAST_HOURS),
                    "cached_hours": len(db_hours),
                    "complete": True,
                    "prefetch_running": False,
                }

        return {
            "run_date": run_date,
            "run_hour": run_hour,
            "total_hours": len(hours),
            "cached_hours": cached_count,
            "complete": cached_count == len(hours) and not prefetch_running,
            "prefetch_running": prefetch_running,
        }

    # Other fields: check DB
    if db_weather is not None:
        try:
            run_time, hours = db_weather.get_available_hours_by_source(cfg.source)
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


@router.post("/api/weather/{field}/prefetch",
             dependencies=[Depends(require_not_demo("Weather prefetch"))])
async def api_trigger_field_prefetch(
    field: str,
    background_tasks: BackgroundTasks,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    """Trigger background download of forecast data for any field."""
    if field not in WEATHER_FIELDS:
        raise HTTPException(status_code=400,
                            detail=f"Unknown field: {field}. Valid: {list(FIELD_NAMES)}")

    mgr = _get_layer_manager(field)
    return mgr.trigger_response(
        background_tasks, _do_generic_prefetch,
        lat_min, lat_max, lon_min, lon_max,
    )


@router.get("/api/weather/{field}/frames")
async def api_get_field_frames(
    field: str,
    request: Request,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    """Return all forecast frames for any field.

    Serves from file cache (instant) with DB fallback.
    Demo users get filtered frames.
    """
    if field not in WEATHER_FIELDS:
        raise HTTPException(status_code=400,
                            detail=f"Unknown field: {field}. Valid: {list(FIELD_NAMES)}")

    cfg = get_field(field)
    mgr = _get_layer_manager(field)
    cache_key = mgr.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    _is_demo_user = is_demo() and is_demo_user(request)

    # Try file cache
    if _is_demo_user:
        cached = mgr.cache_get(cache_key)
        if cached:
            return limit_demo_frames(cached)
    else:
        use_covering = field in ("sst", "visibility")
        raw = mgr.serve_frames_file(
            cache_key, lat_min, lat_max, lon_min, lon_max,
            use_covering=use_covering,
        )
        if raw is not None:
            return raw

    # Fallback: rebuild from PostgreSQL
    cached = await asyncio.to_thread(
        _rebuild_frames_from_db, field, cache_key,
        lat_min, lat_max, lon_min, lon_max,
    )

    if cached:
        if _is_demo_user:
            return limit_demo_frames(cached)
        return cached

    # No data at all
    empty = {
        "run_time": "",
        "total_hours": cfg.expected_frames,
        "cached_hours": 0,
        "source": "none",
        "field": field,
        "frames": {},
    }
    if field == "wind":
        gfs_provider = _get_providers()['gfs']
        run_date, run_hour = gfs_provider._get_latest_run()
        run_time = datetime.strptime(f"{run_date}{run_hour}", "%Y%m%d%H")
        empty.update(run_date=run_date, run_hour=run_hour,
                     run_time=run_time.isoformat())
    else:
        empty.update(lats=[], lons=[], ny=0, nx=0)
    return empty


# ============================================================================
# Resync endpoint (uses {field} path param — after static routes)
# ============================================================================

@router.post("/api/weather/{field}/resync")
async def api_weather_layer_resync(
    field: str,
    lat_min: Optional[float] = Query(None, ge=-90, le=90),
    lat_max: Optional[float] = Query(None, ge=-90, le=90),
    lon_min: Optional[float] = Query(None, ge=-180, le=180),
    lon_max: Optional[float] = Query(None, ge=-180, le=180),
):
    """Re-ingest a single weather layer and return fresh ingested_at.

    Synchronous — blocks until ingestion completes (30-120s for CMEMS layers).
    """
    weather_ingestion = _weather_ingestion()
    db_weather = _db_weather()

    if field not in WEATHER_FIELDS:
        raise HTTPException(status_code=400,
                            detail=f"Unknown field: {field}. Valid: {list(FIELD_NAMES)}")
    if weather_ingestion is None:
        raise HTTPException(status_code=503, detail="Weather ingestion not configured")

    cfg = get_field(field)

    # Cap CMEMS bbox to prevent OOM
    has_bbox = all(v is not None for v in (lat_min, lat_max, lon_min, lon_max))
    if has_bbox:
        lat_min, lat_max, lon_min, lon_max = _clamp_bbox(lat_min, lat_max, lon_min, lon_max)

    logger.info(f"Per-layer resync starting: {field}" +
                (f" bbox=[{lat_min:.1f},{lat_max:.1f}]x[{lon_min:.1f},{lon_max:.1f}]" if has_bbox else ""))

    try:
        ingest_fn = getattr(weather_ingestion, cfg.ingest_method)
        cmems_layers = {"waves", "currents", "swell", "ice", "sst"}
        if has_bbox and field in cmems_layers:
            await asyncio.to_thread(ingest_fn, True, lat_min, lat_max, lon_min, lon_max)
        else:
            await asyncio.to_thread(ingest_fn, True)

        # Supersede old runs and clean orphans
        weather_ingestion._supersede_old_runs(cfg.source)
        weather_ingestion.cleanup_orphaned_grid_data(cfg.source)

        # Clear layer-specific frame cache
        mgr = _get_layer_manager(field)
        if mgr.cache_dir.exists():
            for f in mgr.cache_dir.iterdir():
                f.unlink(missing_ok=True)

        _cleanup_stale_caches()

        _, db_ingested_at = db_weather._find_latest_run(cfg.source) if db_weather else (None, None)
        ingested_at = db_ingested_at or datetime.now(timezone.utc)
        logger.info(f"Per-layer resync complete: {field}, ingested_at={ingested_at.isoformat()}")
        return {"status": "complete", "ingested_at": ingested_at.isoformat()}

    except Exception as e:
        logger.error(f"Resync failed for {field}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Resync failed: {e}")



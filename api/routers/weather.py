"""
Weather router for WINDMAR API.

Extracted from main.py (Phase 6 refactoring). Contains all 31 weather
endpoints, forecast layer helpers, and cache management utilities.

All weather providers are resolved lazily via get_app_state().weather_providers
to avoid module-level initialization ordering issues.
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
from api.weather_service import (
    get_wind_field, get_wave_field, get_current_field,
    get_sst_field, get_visibility_field, get_ice_field,
    get_weather_at_point,
    build_ocean_mask as _build_ocean_mask,
    apply_ocean_mask_velocity as _apply_ocean_mask_velocity,
)
from api.forecast_layer_manager import (
    wind_layer, wave_layer, current_layer, ice_layer, sst_layer, vis_layer,
    cache_covers_bounds,
)
from src.data.copernicus import GFSDataProvider

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Weather"])


# ============================================================================
# Lazy provider resolution helpers
# ============================================================================

def _get_providers():
    """Lazy-resolve weather providers from app state."""
    return get_app_state().weather_providers


def _db_weather():
    return get_app_state().weather_providers.get('db_weather')


def _weather_ingestion():
    return get_app_state().weather_providers.get('weather_ingestion')


# ============================================================================
# Constants
# ============================================================================

# Layer name -> ingestion source mapping for per-layer resync
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

# --- Overlay grid subsampling (prevents browser OOM on large viewports) ---
_OVERLAY_MAX_DIM = 500  # max grid points per axis for single-frame overlays


# ============================================================================
# Helper functions
# ============================================================================

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
# Wind frame helpers
# ============================================================================

def _build_wind_frames(lat_min, lat_max, lon_min, lon_max, run_date, run_hour):
    """Process all cached GRIB files into leaflet-velocity frames dict.

    Called once after prefetch completes. Result is saved to file cache.
    """
    gfs_provider = _get_providers()['gfs']
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
    pattern: read complete DB run -> build leaflet-velocity frames -> save.
    """
    db_weather = _db_weather()
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

        valid_time = run_time + timedelta(hours=fh) if run_time else datetime.now(timezone.utc)
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


def _do_wind_prefetch(mgr, lat_min, lat_max, lon_min, lon_max):
    """Download all GFS forecast hours and build wind frames cache."""
    gfs_provider = _get_providers()['gfs']
    run_date, run_hour = gfs_provider._get_latest_run()
    mgr.last_run = (run_date, run_hour)
    logger.info(f"GFS forecast prefetch started (run {run_date}/{run_hour}z)")
    gfs_provider.prefetch_forecast_hours(lat_min, lat_max, lon_min, lon_max)
    logger.info("GFS forecast prefetch completed, building frames cache...")
    _build_wind_frames(lat_min, lat_max, lon_min, lon_max, run_date, run_hour)
    logger.info("Wind frames cache ready")


# ============================================================================
# Wave cache helpers
# ============================================================================

def _rebuild_wave_cache_from_db(cache_key, lat_min, lat_max, lon_min, lon_max):
    """Rebuild wave forecast file cache from PostgreSQL data."""
    db_weather = _db_weather()
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
    STEP = _overlay_step(lats_full, lons_full)
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


def _do_wave_prefetch(mgr, lat_min, lat_max, lon_min, lon_max):
    """Download CMEMS wave forecast and build frames cache."""
    copernicus_provider = _get_providers()['copernicus']
    weather_ingestion = _weather_ingestion()

    cache_key_chk = mgr.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    existing = mgr.cache_get(cache_key_chk)
    if existing and len(existing.get("frames", {})) >= 41 and cache_covers_bounds(existing, lat_min, lat_max, lon_min, lon_max):
        logger.info("Wave forecast file cache already complete, skipping CMEMS download")
        return

    db_weather = _db_weather()
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
    STEP = _overlay_step(first_wd.lats, first_wd.lons)
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


# ============================================================================
# Current cache helpers
# ============================================================================

def _rebuild_current_cache_from_db(cache_key, lat_min, lat_max, lon_min, lon_max):
    """Rebuild current forecast file cache from PostgreSQL data."""
    db_weather = _db_weather()
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
    STEP = _overlay_step(lats_full, lons_full)
    sub_lats = lats_full[::STEP]  # numpy, S->N order
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
            # Flip N->S for leaflet-velocity (lats stay S->N for header)
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


def _do_current_prefetch(mgr, lat_min, lat_max, lon_min, lon_max):
    """Download CMEMS current forecast and build frames cache."""
    copernicus_provider = _get_providers()['copernicus']
    weather_ingestion = _weather_ingestion()

    cache_key_chk = mgr.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    existing = mgr.cache_get(cache_key_chk)
    if existing and len(existing.get("frames", {})) >= 41 and cache_covers_bounds(existing, lat_min, lat_max, lon_min, lon_max):
        logger.info("Current forecast file cache already complete, skipping CMEMS download")
        return

    db_weather = _db_weather()
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
    STEP = _overlay_step(first_wd.lats, first_wd.lons)
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


# ============================================================================
# Ice cache helpers
# ============================================================================

def _rebuild_ice_cache_from_db(cache_key, lat_min, lat_max, lon_min, lon_max):
    """Rebuild ice forecast file cache from PostgreSQL data."""
    db_weather = _db_weather()
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
    STEP = _overlay_step(lats_full, lons_full)
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


def _do_ice_prefetch(mgr, lat_min, lat_max, lon_min, lon_max):
    """Download CMEMS ice forecast and build frames cache."""
    wx = _get_providers()
    copernicus_provider = wx['copernicus']
    synthetic_provider = wx['synthetic']
    weather_ingestion = _weather_ingestion()

    cache_key_chk = mgr.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    existing = mgr.cache_get(cache_key_chk)
    if existing and len(existing.get("frames", {})) >= 10 and cache_covers_bounds(existing, lat_min, lat_max, lon_min, lon_max):
        logger.info("Ice forecast file cache already complete, skipping CMEMS download")
        return

    db_weather = _db_weather()
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
    STEP = _overlay_step(first_wd.lats, first_wd.lons)
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


# ============================================================================
# SST cache helpers
# ============================================================================

def _do_sst_prefetch(mgr, lat_min, lat_max, lon_min, lon_max):
    """Download CMEMS SST forecast and build frames cache."""
    copernicus_provider = _get_providers()['copernicus']
    weather_ingestion = _weather_ingestion()

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
    STEP = _overlay_step(first_wd.lats, first_wd.lons)
    logger.info(f"SST forecast: grid {len(first_wd.lats)}x{len(first_wd.lons)}, STEP={STEP}")
    sub_lats = first_wd.lats[::STEP]
    sub_lons = first_wd.lons[::STEP]

    mask_lats_arr, mask_lons_arr, ocean_mask_arr = _build_ocean_mask(
        lat_min, lat_max, lon_min, lon_max
    )

    frames = {}
    global_min, global_max = float('inf'), float('-inf')
    for fh, wd in sorted(result.items()):
        sst_vals = wd.sst if wd.sst is not None else wd.values
        if sst_vals is not None:
            clean = np.nan_to_num(sst_vals[::STEP, ::STEP], nan=-999.0)
            valid = clean[clean > -100]
            if valid.size > 0:
                global_min = min(global_min, float(np.min(valid)))
                global_max = max(global_max, float(np.max(valid)))
            frames[str(fh)] = {
                "data": np.round(clean, 2).tolist(),
            }

    if not np.isfinite(global_min):
        global_min, global_max = -2.0, 32.0

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
        "colorscale": {
            "min": -2, "max": 32,
            "data_min": round(global_min, 2), "data_max": round(global_max, 2),
            "colors": ["#0000ff", "#00ccff", "#00ff88", "#ffff00", "#ff8800", "#ff0000"],
        },
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


def _rebuild_sst_cache_from_db(cache_key, lat_min, lat_max, lon_min, lon_max):
    """Rebuild SST forecast file cache from PostgreSQL data."""
    db_weather = _db_weather()
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
    STEP = _overlay_step(lats_full, lons_full)
    shared_lats = lats_full[::STEP].tolist()
    shared_lons = lons_full[::STEP].tolist()

    mask_lats_arr, mask_lons_arr, ocean_mask_arr = _build_ocean_mask(
        lat_min, lat_max, lon_min, lon_max
    )

    frames = {}
    global_min, global_max = float('inf'), float('-inf')
    for fh in sorted(hours):
        if fh in grids["sst"]:
            _, _, d = grids["sst"][fh]
            clean = np.nan_to_num(d[::STEP, ::STEP], nan=-999.0)
            valid = clean[clean > -100]
            if valid.size > 0:
                global_min = min(global_min, float(np.min(valid)))
                global_max = max(global_max, float(np.max(valid)))
            frames[str(fh)] = {
                "data": np.round(clean, 2).tolist(),
            }

    if not np.isfinite(global_min):
        global_min, global_max = -2.0, 32.0

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
        "colorscale": {
            "min": -2, "max": 32,
            "data_min": round(global_min, 2), "data_max": round(global_max, 2),
            "colors": ["#0000ff", "#00ccff", "#00ff88", "#ffff00", "#ff8800", "#ff0000"],
        },
        "frames": frames,
    }

    sst_layer.cache_put(cache_key, cache_data)
    logger.info(f"SST cache rebuilt from DB: {len(frames)} frames")
    return cache_data


# ============================================================================
# Visibility cache helpers
# ============================================================================

def _rebuild_vis_cache_from_db(cache_key, lat_min, lat_max, lon_min, lon_max):
    """Rebuild visibility forecast file cache from PostgreSQL data."""
    db_weather = _db_weather()
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
    STEP = _overlay_step(lats_full, lons_full)
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


def _do_vis_prefetch(mgr, lat_min, lat_max, lon_min, lon_max):
    """Download GFS visibility forecast and build frames cache."""
    gfs_provider = _get_providers()['gfs']
    weather_ingestion = _weather_ingestion()

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
    STEP = _overlay_step(first_wd.lats, first_wd.lons)
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


# ============================================================================
# API Endpoints - Weather Health & Resync
# ============================================================================

@router.get("/api/weather/health")
async def api_weather_health():
    """Return per-source health status for all weather sources.

    Purely informational -- returns per-layer presence, completeness,
    ingested_at, and age_hours. No longer drives automation.
    """
    db_weather = _db_weather()
    if db_weather is None:
        raise HTTPException(status_code=503, detail="Database weather provider not configured")

    health = await asyncio.to_thread(db_weather.get_health)
    return health


@router.post("/api/weather/{layer}/resync")
async def api_weather_layer_resync(
    layer: str,
    lat_min: Optional[float] = Query(None, ge=-90, le=90),
    lat_max: Optional[float] = Query(None, ge=-90, le=90),
    lon_min: Optional[float] = Query(None, ge=-180, le=180),
    lon_max: Optional[float] = Query(None, ge=-180, le=180),
):
    """Re-ingest a single weather layer and return fresh ingested_at.

    Synchronous -- blocks until ingestion completes (30-120s for CMEMS layers).
    Clears the layer's frame cache so the next timeline request rebuilds it.
    When bbox params are provided, CMEMS layers use them instead of defaults.
    """
    weather_ingestion = _weather_ingestion()
    db_weather = _db_weather()

    if layer not in _LAYER_INGEST_FN:
        raise HTTPException(status_code=400, detail=f"Unknown layer: {layer}. Valid: {list(_LAYER_INGEST_FN.keys())}")
    if weather_ingestion is None:
        raise HTTPException(status_code=503, detail="Weather ingestion not configured")

    # Cap CMEMS bbox — copernicusmarine downloads at native 0.083° resolution
    # regardless of any post-download subsampling, so we must limit the area.
    # 55×130 covers the NE Atlantic + Med demo viewport (~700 MB download).
    _CMEMS_MAX_LAT_SPAN = 55.0
    _CMEMS_MAX_LON_SPAN = 130.0
    has_bbox = all(v is not None for v in (lat_min, lat_max, lon_min, lon_max))
    if has_bbox:
        lat_span = lat_max - lat_min
        lon_span = lon_max - lon_min
        lat_center = (lat_min + lat_max) / 2
        lon_center = (lon_min + lon_max) / 2
        lat_half = min(lat_span / 2, _CMEMS_MAX_LAT_SPAN / 2)
        lon_half = min(lon_span / 2, _CMEMS_MAX_LON_SPAN / 2)
        lat_min = max(-89.9, lat_center - lat_half)
        lat_max = min(89.9, lat_center + lat_half)
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

        # Supersede old runs and clean orphans -- scoped to this source only
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


# ============================================================================
# API Endpoints - Wind
# ============================================================================

@router.get("/api/weather/wind")
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
    db_weather = _db_weather()
    copernicus_provider = _get_providers()['copernicus']

    if time is None:
        time = datetime.now(timezone.utc)

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

    # High-resolution ocean mask (0.05 deg ~ 5.5km) via vectorized numpy
    mask_lats, mask_lons, ocean_mask = _build_ocean_mask(lat_min, lat_max, lon_min, lon_max, step=0.05)

    # SPEC-P1: Piggyback SST on wind endpoint (same bounding box)
    sst_data = get_sst_field(lat_min, lat_max, lon_min, lon_max, resolution, time)

    response = {
        "parameter": "wind",
        "time": time.isoformat(),
        "bbox": {
            "lat_min": float(wind_data.lats.min()),
            "lat_max": float(wind_data.lats.max()),
            "lon_min": float(wind_data.lons.min()),
            "lon_max": float(wind_data.lons.max()),
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
            "gfs" if (wind_data.time and (datetime.now(timezone.utc) - wind_data.time).total_seconds() < 43200)
            else "copernicus" if copernicus_provider._has_cdsapi
            else "synthetic"
        ),
    }
    if sst_data is not None and sst_data.values is not None:
        response["sst"] = {
            "lats": sst_data.lats.tolist(),
            "lons": sst_data.lons.tolist(),
            "data": sst_data.values.tolist(),
            "unit": "\u00b0C",
        }
    if ingested_at is not None:
        response["ingested_at"] = ingested_at.isoformat()
    return response


@router.get("/api/weather/wind/velocity")
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
    gfs_provider = _get_providers()['gfs']

    if time is None:
        time = datetime.now(timezone.utc)

    if forecast_hour > 0:
        # Direct GFS fetch for specific forecast hour
        wind_data = gfs_provider.fetch_wind_data(lat_min, lat_max, lon_min, lon_max, time, forecast_hour)
        if wind_data is None:
            raise HTTPException(status_code=404, detail=f"Forecast hour f{forecast_hour:03d} not available")
    else:
        wind_data = get_wind_field(lat_min, lat_max, lon_min, lon_max, resolution, time)

    # Apply ocean mask -- zero out land so particles don't render there
    u_masked, v_masked = _apply_ocean_mask_velocity(
        wind_data.u_component, wind_data.v_component,
        wind_data.lats, wind_data.lons,
    )

    # leaflet-velocity format -- derive header from actual data arrays
    # (data may come at native resolution, e.g. 0.25 deg from ERA5)
    actual_lats = wind_data.lats
    actual_lons = wind_data.lons
    actual_dx = abs(float(actual_lons[1] - actual_lons[0])) if len(actual_lons) > 1 else resolution
    actual_dy = abs(float(actual_lats[1] - actual_lats[0])) if len(actual_lats) > 1 else resolution

    # leaflet-velocity expects data ordered N->S (descending lat), W->E (ascending lon)
    if len(actual_lats) > 1 and actual_lats[1] > actual_lats[0]:
        # Ascending (S->N from np.arange) -- flip to N->S
        u_ordered = u_masked[::-1]
        v_ordered = v_masked[::-1]
        lat_north = float(actual_lats[-1])
        lat_south = float(actual_lats[0])
    else:
        # Already descending (N->S, typical for ERA5 netCDF)
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

@router.get("/api/weather/forecast/status")
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
    gfs_provider = _get_providers()['gfs']
    db_weather = _db_weather()

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

    # No file cache -- fall back to scanning GRIB files
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


@router.post("/api/weather/forecast/prefetch", dependencies=[Depends(require_not_demo("Weather prefetch"))])
async def api_trigger_forecast_prefetch(
    background_tasks: BackgroundTasks,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    """Trigger background download of all GFS forecast hours (f000-f120)."""
    return wind_layer.trigger_response(background_tasks, _do_wind_prefetch, lat_min, lat_max, lon_min, lon_max)


@router.get("/api/weather/forecast/frames")
async def api_get_forecast_frames(
    request: Request,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    """
    Return all wind forecast frames from file cache (instant).

    The cache is built once during prefetch. No GRIB parsing happens here.
    Serves the raw JSON file to avoid parse+re-serialize overhead.
    For demo-tier users, frames are filtered to every Nth hour.
    """
    gfs_provider = _get_providers()['gfs']

    cache_key = wind_layer.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    cache_file = wind_layer.cache_path(cache_key)

    _is_demo_user = is_demo() and is_demo_user(request)

    if cache_file.exists():
        if _is_demo_user:
            cached = wind_layer.cache_get(cache_key)
            if cached:
                return limit_demo_frames(cached)
        return Response(content=cache_file.read_bytes(), media_type="application/json")

    # No file cache -- fallback: rebuild from PostgreSQL
    cached = await asyncio.to_thread(
        _rebuild_wind_cache_from_db, cache_key, lat_min, lat_max, lon_min, lon_max
    )

    if cached:
        if _is_demo_user:
            return limit_demo_frames(cached)
        return cached

    # No DB data either -- return empty
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

@router.get("/api/weather/forecast/wave/status")
async def api_get_wave_forecast_status(
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    """Get wave forecast prefetch status."""
    db_weather = _db_weather()

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

    # File cache miss -- check DB for data from the auto-prefetch
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


@router.post("/api/weather/forecast/wave/prefetch", dependencies=[Depends(require_not_demo("Weather prefetch"))])
async def api_trigger_wave_forecast_prefetch(
    background_tasks: BackgroundTasks,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    """Trigger background download of CMEMS wave forecast (0-120h)."""
    return wave_layer.trigger_response(background_tasks, _do_wave_prefetch, lat_min, lat_max, lon_min, lon_max)


@router.get("/api/weather/forecast/wave/frames")
async def api_get_wave_forecast_frames(
    request: Request,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    """Return all cached CMEMS wave forecast frames.

    Serves the raw JSON file to avoid parse+re-serialize overhead.
    For demo-tier users, frames are filtered to every Nth hour.
    """
    cache_key = wave_layer.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    _is_demo_user = is_demo() and is_demo_user(request)

    if _is_demo_user:
        cached = wave_layer.cache_get(cache_key)
        if cached:
            return limit_demo_frames(cached)
    else:
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

    if _is_demo_user:
        return limit_demo_frames(cached)
    return cached


# =========================================================================
# Current Forecast Endpoints (CMEMS)
# =========================================================================

@router.get("/api/weather/forecast/current/status")
async def api_get_current_forecast_status(
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    """Get current forecast prefetch status."""
    db_weather = _db_weather()

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

    # File cache miss -- check DB for data from the auto-prefetch
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


@router.post("/api/weather/forecast/current/prefetch", dependencies=[Depends(require_not_demo("Weather prefetch"))])
async def api_trigger_current_forecast_prefetch(
    background_tasks: BackgroundTasks,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    """Trigger background download of CMEMS current forecast (0-120h)."""
    return current_layer.trigger_response(background_tasks, _do_current_prefetch, lat_min, lat_max, lon_min, lon_max)


@router.get("/api/weather/forecast/current/frames")
async def api_get_current_forecast_frames(
    request: Request,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    """Return all cached CMEMS current forecast frames.

    Serves the raw JSON file to avoid parse+re-serialize overhead.
    For demo-tier users, frames are filtered to every Nth hour.
    """
    cache_key = current_layer.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    _is_demo_user = is_demo() and is_demo_user(request)

    if _is_demo_user:
        cached = current_layer.cache_get(cache_key)
        if cached:
            return limit_demo_frames(cached)
    else:
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

    if _is_demo_user:
        return limit_demo_frames(cached)
    return cached


# =========================================================================
# Ice Forecast Endpoints (CMEMS, 10-day daily)
# =========================================================================

@router.get("/api/weather/forecast/ice/status")
async def api_get_ice_forecast_status(
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    """Get ice forecast prefetch status."""
    db_weather = _db_weather()

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

    # File cache miss -- check DB for data from the auto-prefetch
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


@router.post("/api/weather/forecast/ice/prefetch", dependencies=[Depends(require_not_demo("Weather prefetch"))])
async def api_trigger_ice_forecast_prefetch(
    background_tasks: BackgroundTasks,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    """Trigger background download of CMEMS ice forecast (10-day daily)."""
    return ice_layer.trigger_response(background_tasks, _do_ice_prefetch, lat_min, lat_max, lon_min, lon_max)


@router.get("/api/weather/forecast/ice/frames")
async def api_get_ice_forecast_frames(
    request: Request,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    """Return all cached CMEMS ice forecast frames.

    Serves the raw JSON file to avoid parse+re-serialize overhead.
    For demo-tier users, frames are filtered to every Nth hour.
    """
    cache_key = ice_layer.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    _is_demo_user = is_demo() and is_demo_user(request)

    if _is_demo_user:
        cached = ice_layer.cache_get(cache_key)
        if cached:
            return limit_demo_frames(cached)
    else:
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

    if _is_demo_user:
        return limit_demo_frames(cached)
    return cached


# ======================================================================
# SST Forecast Prefetch Pipeline
# ======================================================================

@router.get("/api/weather/forecast/sst/status")
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


@router.post("/api/weather/forecast/sst/prefetch", dependencies=[Depends(require_not_demo("Weather prefetch"))])
async def api_trigger_sst_forecast_prefetch(
    background_tasks: BackgroundTasks,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    """Trigger background download of CMEMS SST forecast (0-120h, 3h steps)."""
    return sst_layer.trigger_response(background_tasks, _do_sst_prefetch, lat_min, lat_max, lon_min, lon_max)


@router.get("/api/weather/forecast/sst/frames")
async def api_get_sst_forecast_frames(
    request: Request,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    """Return all cached CMEMS SST forecast frames.

    Serves the raw JSON file to avoid parse+re-serialize overhead.
    Falls back to any cache file whose bounds cover the requested viewport,
    then to PostgreSQL if no file cache exists.
    For demo-tier users, frames are filtered to every Nth hour.
    """
    cache_key = sst_layer.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    _is_demo_user = is_demo() and is_demo_user(request)

    if _is_demo_user:
        cached = sst_layer.cache_get(cache_key)
        if cached:
            return limit_demo_frames(cached)
    else:
        raw = sst_layer.serve_frames_file(cache_key, lat_min, lat_max, lon_min, lon_max, use_covering=True)
        if raw is not None:
            return raw

    # Fallback: rebuild from PostgreSQL
    cached = await asyncio.to_thread(
        _rebuild_sst_cache_from_db, cache_key, lat_min, lat_max, lon_min, lon_max
    )

    if cached:
        if _is_demo_user:
            return limit_demo_frames(cached)
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

@router.get("/api/weather/forecast/visibility/status")
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


@router.post("/api/weather/forecast/visibility/prefetch", dependencies=[Depends(require_not_demo("Weather prefetch"))])
async def api_trigger_vis_forecast_prefetch(
    background_tasks: BackgroundTasks,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    """Trigger background download of GFS visibility forecast (0-120h, 3h steps)."""
    return vis_layer.trigger_response(background_tasks, _do_vis_prefetch, lat_min, lat_max, lon_min, lon_max)


@router.get("/api/weather/forecast/visibility/frames")
async def api_get_vis_forecast_frames(
    request: Request,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-15.0),
    lon_max: float = Query(40.0),
):
    """Return all cached GFS visibility forecast frames.

    Serves the raw JSON file to avoid parse+re-serialize overhead.
    Falls back to any cache file whose bounds cover the requested viewport.
    For demo-tier users, frames are filtered to every Nth hour.
    """
    cache_key = vis_layer.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    _is_demo_user = is_demo() and is_demo_user(request)

    if _is_demo_user:
        cached = vis_layer.cache_get(cache_key)
        if cached:
            return limit_demo_frames(cached)
    else:
        raw = vis_layer.serve_frames_file(cache_key, lat_min, lat_max, lon_min, lon_max, use_covering=True)
        if raw is not None:
            return raw

    # Fallback: rebuild from PostgreSQL
    cached = await asyncio.to_thread(
        _rebuild_vis_cache_from_db, cache_key, lat_min, lat_max, lon_min, lon_max
    )

    if cached:
        if _is_demo_user:
            return limit_demo_frames(cached)
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


# ============================================================================
# API Endpoints - Single-frame Overlays
# ============================================================================

@router.get("/api/weather/waves")
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
    db_weather = _db_weather()
    copernicus_provider = _get_providers()['copernicus']

    if time is None:
        time = datetime.now(timezone.utc)

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
            "lat_min": float(wave_data.lats.min()),
            "lat_max": float(wave_data.lats.max()),
            "lon_min": float(wave_data.lons.min()),
            "lon_max": float(wave_data.lons.max()),
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


@router.get("/api/weather/currents")
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
    copernicus_provider = _get_providers()['copernicus']

    if time is None:
        time = datetime.now(timezone.utc)

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
            "lat_min": float(current_data.lats.min()),
            "lat_max": float(current_data.lats.max()),
            "lon_min": float(current_data.lons.min()),
            "lon_max": float(current_data.lons.max()),
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


@router.get("/api/weather/currents/velocity")
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
        time = datetime.now(timezone.utc)

    current_data = get_current_field(lat_min, lat_max, lon_min, lon_max, resolution)

    # Apply ocean mask -- zero out land so particles don't render there
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

    # leaflet-velocity expects data ordered N->S, W->E
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


@router.get("/api/weather/point")
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


# ============================================================================
# API Endpoints - Extended Weather Fields (SPEC-P1)
# ============================================================================

@router.get("/api/weather/sst")
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
    db_weather = _db_weather()
    copernicus_provider = _get_providers()['copernicus']

    if time is None:
        time = datetime.now(timezone.utc)

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
            "lat_min": float(sst_data.lats.min()),
            "lat_max": float(sst_data.lats.max()),
            "lon_min": float(sst_data.lons.min()),
            "lon_max": float(sst_data.lons.max()),
        },
        "resolution": resolution,
        "nx": len(sub_lons),
        "ny": len(sub_lats),
        "lats": sub_lats,
        "lons": sub_lons,
        "data": np.round(np.nan_to_num(sst_data.values[::step, ::step], nan=-999.0), 2).tolist(),
        "unit": "\u00b0C",
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


@router.get("/api/weather/visibility")
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
    db_weather = _db_weather()

    if time is None:
        time = datetime.now(timezone.utc)

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
            "lat_min": float(vis_data.lats.min()),
            "lat_max": float(vis_data.lats.max()),
            "lon_min": float(vis_data.lons.min()),
            "lon_max": float(vis_data.lons.max()),
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


@router.get("/api/weather/ice")
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
    db_weather = _db_weather()
    copernicus_provider = _get_providers()['copernicus']

    if time is None:
        time = datetime.now(timezone.utc)

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
            "lat_min": float(ice_data.lats.min()),
            "lat_max": float(ice_data.lats.max()),
            "lon_min": float(ice_data.lons.min()),
            "lon_max": float(ice_data.lons.max()),
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


@router.get("/api/weather/swell")
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
        time = datetime.now(timezone.utc)

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
            "lat_min": float(wave_data.lats.min()),
            "lat_max": float(wave_data.lats.max()),
            "lon_min": float(wave_data.lons.min()),
            "lon_max": float(wave_data.lons.max()),
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
# Weather Freshness
# ============================================================================

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

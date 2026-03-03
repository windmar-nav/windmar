"""
Weather API router — thin FastAPI endpoints.

All processing logic lives in sibling modules (frame_builder, formatters,
prefetch, grid_processor, ocean_mask).  This file only wires HTTP to logic.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from starlette.responses import Response

from api.demo import require_not_demo, is_demo, is_demo_user, demo_mode_response, limit_demo_frames
from api.state import get_app_state
from api.weather_fields import (
    WEATHER_FIELDS, FIELD_NAMES, LAYER_TO_SOURCE,
    get_field, validate_field_name, FieldConfig,
    CACHE_SCHEMA_VERSION,
)
from api.weather_service import (
    get_wind_field, get_wave_field, get_current_field,
    get_sst_field, get_visibility_field, get_ice_field,
    get_weather_at_point,
)
from api.weather.grid_processor import clamp_bbox, compute_step
from api.weather.ocean_mask import build_ice_ocean_mask
from api.weather.frame_builder import build_frames_from_db
from api.weather.formatters import format_single_frame, format_velocity_response
from api.weather.prefetch import (
    get_layer_manager, do_generic_prefetch, cleanup_stale_caches,
    _get_providers, _db_weather, _weather_ingestion,
)

from src.data.copernicus import GFSDataProvider

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Weather"])


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

_DB_FIRST_METHODS = {
    "wind": "get_wind_from_db",
    "waves": "get_wave_from_db",
    "sst": "get_sst_from_db",
    "visibility": "get_visibility_from_db",
    "ice": "get_ice_from_db",
}


# ============================================================================
# Ocean mask callback for single-frame formatter
# ============================================================================

def _make_ocean_mask_fn(field_name):
    """Return an ocean mask builder callback for single-frame formatting."""
    if field_name == "ice":
        def _ice_mask(grid):
            mask_list, _ = build_ice_ocean_mask(grid)
            return grid.lats.tolist(), grid.lons.tolist(), mask_list
        return _ice_mask
    else:
        def _nan_mask(grid):
            # For single-frame overlay, use global_land_mask at the subsampled coords
            # (NaN-union requires multiple frames; single frame uses coordinates)
            from api.weather.ocean_mask import build_ice_ocean_mask as _glm
            try:
                from global_land_mask import globe
                lon_grid, lat_grid = np.meshgrid(grid.lons, grid.lats)
                mask = globe.is_ocean(lat_grid, lon_grid)
                return grid.lats.tolist(), grid.lons.tolist(), mask.tolist()
            except ImportError:
                from src.data.land_mask import is_ocean
                mask = [
                    [is_ocean(round(float(lat), 2), round(float(lon), 2)) for lon in grid.lons]
                    for lat in grid.lats
                ]
                return grid.lats.tolist(), grid.lons.tolist(), mask
        return _nan_mask


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

@router.get("/api/weather/forecast/status")
async def _compat_wind_status(
    lat_min: float = Query(30.0), lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0), lon_max: float = Query(40.0),
):
    return await api_get_field_status(field="wind", lat_min=lat_min, lat_max=lat_max,
                                      lon_min=lon_min, lon_max=lon_max)


@router.post("/api/weather/forecast/prefetch",
             dependencies=[Depends(require_not_demo("Weather prefetch"))])
async def _compat_wind_prefetch(
    background_tasks: BackgroundTasks,
    lat_min: float = Query(30.0), lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0), lon_max: float = Query(40.0),
):
    return await api_trigger_field_prefetch(field="wind", background_tasks=background_tasks,
                                            lat_min=lat_min, lat_max=lat_max,
                                            lon_min=lon_min, lon_max=lon_max)


@router.get("/api/weather/forecast/frames")
async def _compat_wind_frames(
    request: Request,
    lat_min: float = Query(30.0), lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0), lon_max: float = Query(40.0),
):
    return await api_get_field_frames(field="wind", request=request,
                                      lat_min=lat_min, lat_max=lat_max,
                                      lon_min=lon_min, lon_max=lon_max)


@router.get("/api/weather/forecast/wave/status")
async def _compat_wave_status(
    lat_min: float = Query(30.0), lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0), lon_max: float = Query(40.0),
):
    return await api_get_field_status(field="waves", lat_min=lat_min, lat_max=lat_max,
                                      lon_min=lon_min, lon_max=lon_max)


@router.post("/api/weather/forecast/wave/prefetch",
             dependencies=[Depends(require_not_demo("Weather prefetch"))])
async def _compat_wave_prefetch(
    background_tasks: BackgroundTasks,
    lat_min: float = Query(30.0), lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0), lon_max: float = Query(40.0),
):
    return await api_trigger_field_prefetch(field="waves", background_tasks=background_tasks,
                                            lat_min=lat_min, lat_max=lat_max,
                                            lon_min=lon_min, lon_max=lon_max)


@router.get("/api/weather/forecast/wave/frames")
async def _compat_wave_frames(
    request: Request,
    lat_min: float = Query(30.0), lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0), lon_max: float = Query(40.0),
):
    return await api_get_field_frames(field="waves", request=request,
                                      lat_min=lat_min, lat_max=lat_max,
                                      lon_min=lon_min, lon_max=lon_max)


@router.get("/api/weather/forecast/current/status")
async def _compat_current_status(
    lat_min: float = Query(30.0), lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0), lon_max: float = Query(40.0),
):
    return await api_get_field_status(field="currents", lat_min=lat_min, lat_max=lat_max,
                                      lon_min=lon_min, lon_max=lon_max)


@router.post("/api/weather/forecast/current/prefetch",
             dependencies=[Depends(require_not_demo("Weather prefetch"))])
async def _compat_current_prefetch(
    background_tasks: BackgroundTasks,
    lat_min: float = Query(30.0), lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0), lon_max: float = Query(40.0),
):
    return await api_trigger_field_prefetch(field="currents", background_tasks=background_tasks,
                                            lat_min=lat_min, lat_max=lat_max,
                                            lon_min=lon_min, lon_max=lon_max)


@router.get("/api/weather/forecast/current/frames")
async def _compat_current_frames(
    request: Request,
    lat_min: float = Query(30.0), lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0), lon_max: float = Query(40.0),
):
    return await api_get_field_frames(field="currents", request=request,
                                      lat_min=lat_min, lat_max=lat_max,
                                      lon_min=lon_min, lon_max=lon_max)


@router.get("/api/weather/forecast/ice/status")
async def _compat_ice_status(
    lat_min: float = Query(30.0), lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0), lon_max: float = Query(40.0),
):
    return await api_get_field_status(field="ice", lat_min=lat_min, lat_max=lat_max,
                                      lon_min=lon_min, lon_max=lon_max)


@router.post("/api/weather/forecast/ice/prefetch",
             dependencies=[Depends(require_not_demo("Weather prefetch"))])
async def _compat_ice_prefetch(
    background_tasks: BackgroundTasks,
    lat_min: float = Query(30.0), lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0), lon_max: float = Query(40.0),
):
    return await api_trigger_field_prefetch(field="ice", background_tasks=background_tasks,
                                            lat_min=lat_min, lat_max=lat_max,
                                            lon_min=lon_min, lon_max=lon_max)


@router.get("/api/weather/forecast/ice/frames")
async def _compat_ice_frames(
    request: Request,
    lat_min: float = Query(30.0), lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0), lon_max: float = Query(40.0),
):
    return await api_get_field_frames(field="ice", request=request,
                                      lat_min=lat_min, lat_max=lat_max,
                                      lon_min=lon_min, lon_max=lon_max)


@router.get("/api/weather/forecast/sst/status")
async def _compat_sst_status(
    lat_min: float = Query(30.0), lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0), lon_max: float = Query(40.0),
):
    return await api_get_field_status(field="sst", lat_min=lat_min, lat_max=lat_max,
                                      lon_min=lon_min, lon_max=lon_max)


@router.post("/api/weather/forecast/sst/prefetch",
             dependencies=[Depends(require_not_demo("Weather prefetch"))])
async def _compat_sst_prefetch(
    background_tasks: BackgroundTasks,
    lat_min: float = Query(30.0), lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0), lon_max: float = Query(40.0),
):
    return await api_trigger_field_prefetch(field="sst", background_tasks=background_tasks,
                                            lat_min=lat_min, lat_max=lat_max,
                                            lon_min=lon_min, lon_max=lon_max)


@router.get("/api/weather/forecast/sst/frames")
async def _compat_sst_frames(
    request: Request,
    lat_min: float = Query(30.0), lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0), lon_max: float = Query(40.0),
):
    return await api_get_field_frames(field="sst", request=request,
                                      lat_min=lat_min, lat_max=lat_max,
                                      lon_min=lon_min, lon_max=lon_max)


@router.get("/api/weather/forecast/visibility/status")
async def _compat_vis_status(
    lat_min: float = Query(30.0), lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0), lon_max: float = Query(40.0),
):
    return await api_get_field_status(field="visibility", lat_min=lat_min, lat_max=lat_max,
                                      lon_min=lon_min, lon_max=lon_max)


@router.post("/api/weather/forecast/visibility/prefetch",
             dependencies=[Depends(require_not_demo("Weather prefetch"))])
async def _compat_vis_prefetch(
    background_tasks: BackgroundTasks,
    lat_min: float = Query(30.0), lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0), lon_max: float = Query(40.0),
):
    return await api_trigger_field_prefetch(field="visibility", background_tasks=background_tasks,
                                            lat_min=lat_min, lat_max=lat_max,
                                            lon_min=lon_min, lon_max=lon_max)


@router.get("/api/weather/forecast/visibility/frames")
async def _compat_vis_frames(
    request: Request,
    lat_min: float = Query(30.0), lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0), lon_max: float = Query(40.0),
):
    return await api_get_field_frames(field="visibility", request=request,
                                      lat_min=lat_min, lat_max=lat_max,
                                      lon_min=lon_min, lon_max=lon_max)


# ============================================================================
# Debug endpoint
# ============================================================================

@router.get("/api/weather/{field}/debug")
async def api_get_field_debug(
    field: str,
    lat_min: float = Query(30.0), lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0), lon_max: float = Query(40.0),
):
    """Return lightweight diagnostics for a layer's cached data."""
    if field not in WEATHER_FIELDS:
        raise HTTPException(status_code=400,
                            detail=f"Unknown field: {field}. Valid: {list(FIELD_NAMES)}")

    cfg = get_field(field)
    mgr = get_layer_manager(field)
    lat_min, lat_max, lon_min, lon_max = clamp_bbox(lat_min, lat_max, lon_min, lon_max)
    cache_key = mgr.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    cached = mgr.cache_get(cache_key)

    if not cached:
        return {"field": field, "status": "no_cache", "cache_key": cache_key}

    frames = cached.get("frames", {})
    frame_hours = sorted(frames.keys(), key=lambda h: int(h) if h.lstrip('-').isdigit() else 0)

    lats = cached.get("lats", [])
    lons = cached.get("lons", [])
    ny = cached.get("ny", 0)
    nx = cached.get("nx", 0)

    sample_frame = frames.get(frame_hours[0]) if frame_hours else None
    sample_rows = 0
    sample_cols = 0
    if sample_frame is not None:
        if isinstance(sample_frame, dict) and "data" in sample_frame:
            data_arr = sample_frame["data"]
            if isinstance(data_arr, list):
                sample_rows = len(data_arr)
                sample_cols = len(data_arr[0]) if data_arr else 0
        elif isinstance(sample_frame, list) and sample_frame:
            header = sample_frame[0].get("header", {})
            sample_rows = header.get("ny", 0)
            sample_cols = header.get("nx", 0)
        elif isinstance(sample_frame, dict):
            u_arr = sample_frame.get("u")
            if isinstance(u_arr, list):
                sample_rows = len(u_arr)
                sample_cols = len(u_arr[0]) if u_arr else 0

    ocean_mask = cached.get("ocean_mask")
    ocean_mask_lats = cached.get("ocean_mask_lats", [])
    ocean_mask_lons = cached.get("ocean_mask_lons", [])
    colorscale = cached.get("colorscale")

    checks = []
    if field != "wind":
        checks.append({"check": "ny == len(lats)", "pass": ny == len(lats),
                        "detail": f"ny={ny}, len(lats)={len(lats)}"})
        checks.append({"check": "nx == len(lons)", "pass": nx == len(lons),
                        "detail": f"nx={nx}, len(lons)={len(lons)}"})
        if sample_rows > 0:
            checks.append({"check": "frame_data_rows == ny", "pass": sample_rows == ny,
                            "detail": f"sample_rows={sample_rows}, ny={ny}"})
            checks.append({"check": "frame_data_cols == nx", "pass": sample_cols == nx,
                            "detail": f"sample_cols={sample_cols}, nx={nx}"})
    else:
        if sample_rows > 0:
            checks.append({"check": "wind header ny/nx consistent",
                            "pass": sample_rows > 0 and sample_cols > 0,
                            "detail": f"header ny={sample_rows}, nx={sample_cols}"})

    if ocean_mask is not None:
        mask_rows = len(ocean_mask) if isinstance(ocean_mask, list) else 0
        mask_cols = len(ocean_mask[0]) if mask_rows > 0 else 0
        checks.append({"check": "ocean_mask rows == mask_lats",
                        "pass": mask_rows == len(ocean_mask_lats),
                        "detail": f"mask_rows={mask_rows}, mask_lats={len(ocean_mask_lats)}"})
        checks.append({"check": "ocean_mask cols == mask_lons",
                        "pass": mask_cols == len(ocean_mask_lons),
                        "detail": f"mask_cols={mask_cols}, mask_lons={len(ocean_mask_lons)}"})

    schema_version = cached.get("_schema_version", 0)
    checks.append({"check": "schema_version current",
                    "pass": schema_version == CACHE_SCHEMA_VERSION,
                    "detail": f"cached={schema_version}, current={CACHE_SCHEMA_VERSION}"})

    all_pass = all(c["pass"] for c in checks)

    return {
        "field": field,
        "source": cached.get("source", ""),
        "run_time": cached.get("run_time", ""),
        "schema_version": schema_version,
        "frame_count": len(frames),
        "frame_hours": frame_hours[:5] + (["..."] if len(frame_hours) > 5 else []),
        "lats_len": len(lats),
        "lats_first": round(lats[0], 4) if lats else None,
        "lats_last": round(lats[-1], 4) if lats else None,
        "lons_len": len(lons),
        "lons_first": round(lons[0], 4) if lons else None,
        "lons_last": round(lons[-1], 4) if lons else None,
        "ny": ny,
        "nx": nx,
        "sample_frame_data_rows": sample_rows,
        "sample_frame_data_cols": sample_cols,
        "has_ocean_mask": ocean_mask is not None,
        "ocean_mask_shape": f"{len(ocean_mask)}x{len(ocean_mask[0]) if ocean_mask else 0}" if ocean_mask else None,
        "has_colorscale": colorscale is not None,
        "colorscale_keys": list(colorscale.keys()) if isinstance(colorscale, dict) else None,
        "checks": checks,
        "all_checks_pass": all_pass,
    }


# ============================================================================
# Generic API Endpoints (parameterized by {field})
# ============================================================================

@router.get("/api/weather/{field}")
async def api_get_weather_field(
    field: str,
    lat_min: float = Query(30.0, ge=-90, le=90),
    lat_max: float = Query(60.0, ge=-90, le=90),
    lon_min: float = Query(-30.0, ge=-180, le=180),
    lon_max: float = Query(40.0, ge=-180, le=180),
    resolution: float = Query(1.0, ge=0.25, le=5.0),
    time: Optional[datetime] = None,
):
    """Get single-frame weather field data for visualization."""
    if field not in WEATHER_FIELDS:
        raise HTTPException(status_code=400,
                            detail=f"Unknown field: {field}. Valid: {list(FIELD_NAMES)}")

    cfg = get_field(field)
    if time is None:
        time = datetime.now(timezone.utc)

    lat_min, lat_max, lon_min, lon_max = clamp_bbox(lat_min, lat_max, lon_min, lon_max)

    params = dict(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min,
                  lon_max=lon_max, resolution=resolution)

    ingested_at = None
    db_weather = _db_weather()

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

    return format_single_frame(
        field, cfg, data, time,
        ocean_mask_fn=_make_ocean_mask_fn(field),
        ingested_at=ingested_at,
    )


@router.get("/api/weather/{field}/velocity")
async def api_get_velocity_format(
    field: str,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0),
    lon_max: float = Query(40.0),
    resolution: float = Query(1.0),
    time: Optional[datetime] = None,
    forecast_hour: int = Query(0, ge=0, le=120),
):
    """Get vector field data in leaflet-velocity compatible format."""
    if field not in ("wind", "currents"):
        raise HTTPException(status_code=400, detail=f"Velocity format only for wind/currents, not {field}")

    if time is None:
        time = datetime.now(timezone.utc)

    lat_min, lat_max, lon_min, lon_max = clamp_bbox(lat_min, lat_max, lon_min, lon_max)

    if field == "wind":
        providers = _get_providers()
        gfs_provider = providers['gfs']
        if forecast_hour > 0:
            data = gfs_provider.fetch_wind_data(lat_min, lat_max, lon_min, lon_max, time, forecast_hour)
            if data is None:
                raise HTTPException(status_code=404, detail=f"Forecast hour f{forecast_hour:03d} not available")
        else:
            data = get_wind_field(lat_min, lat_max, lon_min, lon_max, resolution, time)
    else:
        data = get_current_field(lat_min, lat_max, lon_min, lon_max, resolution)

    cfg = get_field(field)
    step = compute_step(data.lats, data.lons, cfg.subsample_target)

    from api.weather.ocean_mask import mask_velocity_with_nan
    from api.weather.grid_processor import SubsampledGrid

    def _mask_fn(u, v):
        grid = SubsampledGrid(
            lats=data.lats[::step] if step > 1 else data.lats,
            lons=data.lons[::step] if step > 1 else data.lons,
            step=step,
            ny=len(data.lats[::step]) if step > 1 else len(data.lats),
            nx=len(data.lons[::step]) if step > 1 else len(data.lons),
        )
        return mask_velocity_with_nan(u, v, grid)

    return format_velocity_response(data, data.lats, data.lons, time, step, _mask_fn)


@router.get("/api/weather/{field}/status")
async def api_get_field_status(
    field: str,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0),
    lon_max: float = Query(40.0),
):
    """Get forecast status for any weather field."""
    if field not in WEATHER_FIELDS:
        raise HTTPException(status_code=400,
                            detail=f"Unknown field: {field}. Valid: {list(FIELD_NAMES)}")

    cfg = get_field(field)
    mgr = get_layer_manager(field)
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
        if field == "wind":
            result["run_date"] = cached.get("run_date", "")
            result["run_hour"] = cached.get("run_hour", "")
        return result

    if field == "wind":
        providers = _get_providers()
        gfs_provider = providers['gfs']
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
    lon_min: float = Query(-30.0),
    lon_max: float = Query(40.0),
):
    """Trigger background download of forecast data for any field."""
    if field not in WEATHER_FIELDS:
        raise HTTPException(status_code=400,
                            detail=f"Unknown field: {field}. Valid: {list(FIELD_NAMES)}")

    mgr = get_layer_manager(field)
    return mgr.trigger_response(
        background_tasks, do_generic_prefetch,
        lat_min, lat_max, lon_min, lon_max,
    )


@router.get("/api/weather/{field}/frames")
async def api_get_field_frames(
    field: str,
    request: Request,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0),
    lon_max: float = Query(40.0),
):
    """Return all forecast frames for any field."""
    if field not in WEATHER_FIELDS:
        raise HTTPException(status_code=400,
                            detail=f"Unknown field: {field}. Valid: {list(FIELD_NAMES)}")

    lat_min, lat_max, lon_min, lon_max = clamp_bbox(lat_min, lat_max, lon_min, lon_max)

    cfg = get_field(field)
    mgr = get_layer_manager(field)
    cache_key = mgr.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    _is_demo_user = is_demo() and is_demo_user(request)

    cached = mgr.cache_get(cache_key)
    if cached is not None and cached.get("_schema_version") != CACHE_SCHEMA_VERSION:
        logger.info(f"{field} cache stale (schema {cached.get('_schema_version')} != {CACHE_SCHEMA_VERSION}), rebuilding")
        cached = None
    if cached is not None:
        if _is_demo_user:
            return limit_demo_frames(cached)
        raw = mgr.serve_frames_file(
            cache_key, lat_min, lat_max, lon_min, lon_max,
            use_covering=True,
        )
        if raw is not None:
            return raw
        return cached

    covering_raw = mgr.serve_frames_file(
        cache_key, lat_min, lat_max, lon_min, lon_max,
        use_covering=True,
    )
    if covering_raw is not None:
        logger.info(f"{field} frames: serving covering cache for [{lat_min:.0f},{lat_max:.0f}]x[{lon_min:.0f},{lon_max:.0f}]")
        if _is_demo_user:
            import json as _json
            covering_data = _json.loads(covering_raw.body)
            return limit_demo_frames(covering_data)
        return covering_raw

    db_weather = _db_weather()
    if db_weather is not None:
        cached = await asyncio.to_thread(
            build_frames_from_db, field, db_weather,
            lat_min, lat_max, lon_min, lon_max,
        )
        if cached:
            mgr.cache_put(cache_key, cached)
            if _is_demo_user:
                return limit_demo_frames(cached)
            return cached

    empty = {
        "run_time": "",
        "total_hours": cfg.expected_frames,
        "cached_hours": 0,
        "source": "none",
        "field": field,
        "frames": {},
    }
    if field == "wind":
        providers = _get_providers()
        gfs_provider = providers['gfs']
        run_date, run_hour = gfs_provider._get_latest_run()
        from datetime import datetime as dt
        run_time = dt.strptime(f"{run_date}{run_hour}", "%Y%m%d%H")
        empty.update(run_date=run_date, run_hour=run_hour,
                     run_time=run_time.isoformat())
    else:
        empty.update(lats=[], lons=[], ny=0, nx=0)
    return empty


@router.post("/api/weather/{field}/resync")
async def api_weather_layer_resync(
    field: str,
    lat_min: Optional[float] = Query(None, ge=-90, le=90),
    lat_max: Optional[float] = Query(None, ge=-90, le=90),
    lon_min: Optional[float] = Query(None, ge=-180, le=180),
    lon_max: Optional[float] = Query(None, ge=-180, le=180),
):
    """Re-ingest a single weather layer and return fresh ingested_at."""
    weather_ingestion = _weather_ingestion()
    db_weather = _db_weather()

    if field not in WEATHER_FIELDS:
        raise HTTPException(status_code=400,
                            detail=f"Unknown field: {field}. Valid: {list(FIELD_NAMES)}")
    if weather_ingestion is None:
        raise HTTPException(status_code=503, detail="Weather ingestion not configured")

    cfg = get_field(field)

    has_bbox = all(v is not None for v in (lat_min, lat_max, lon_min, lon_max))
    if has_bbox:
        lat_min, lat_max, lon_min, lon_max = clamp_bbox(lat_min, lat_max, lon_min, lon_max)

    logger.info(f"Per-layer resync starting: {field}" +
                (f" bbox=[{lat_min:.1f},{lat_max:.1f}]x[{lon_min:.1f},{lon_max:.1f}]" if has_bbox else ""))

    try:
        ingest_fn = getattr(weather_ingestion, cfg.ingest_method)
        cmems_layers = {"waves", "currents", "swell", "ice", "sst"}
        if has_bbox and field in cmems_layers:
            await asyncio.to_thread(ingest_fn, True, lat_min, lat_max, lon_min, lon_max)
        else:
            await asyncio.to_thread(ingest_fn, True)

        weather_ingestion._supersede_old_runs(cfg.source)
        weather_ingestion.cleanup_orphaned_grid_data(cfg.source)

        mgr = get_layer_manager(field)
        if mgr.cache_dir.exists():
            for f in mgr.cache_dir.iterdir():
                f.unlink(missing_ok=True)

        cleanup_stale_caches()

        _, db_ingested_at = db_weather._find_latest_run(cfg.source) if db_weather else (None, None)
        ingested_at = db_ingested_at or datetime.now(timezone.utc)
        logger.info(f"Per-layer resync complete: {field}, ingested_at={ingested_at.isoformat()}")
        return {"status": "complete", "ingested_at": ingested_at.isoformat()}

    except Exception as e:
        logger.error(f"Resync failed for {field}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Resync failed: {e}")

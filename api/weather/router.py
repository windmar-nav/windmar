"""
Weather API router — thin FastAPI endpoints.

All processing logic lives in sibling modules (frame_builder, formatters,
prefetch, grid_processor, ocean_mask).  This file only wires HTTP to logic.
"""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from starlette.responses import Response

from api.demo import (
    require_not_demo,
    is_demo,
    is_demo_user,
    demo_mode_response,
    limit_demo_frames,
)
from api.state import get_app_state
from api.weather_fields import (
    WEATHER_FIELDS,
    FIELD_NAMES,
    LAYER_TO_SOURCE,
    get_field,
    validate_field_name,
    FieldConfig,
    CACHE_SCHEMA_VERSION,
)
from api.weather_service import (
    get_wind_field,
    get_wave_field,
    get_current_field,
    get_sst_field,
    get_visibility_field,
    get_ice_field,
    get_weather_at_point,
)
from api.weather.grid_processor import clamp_bbox, compute_step
from api.weather.ocean_mask import build_ice_ocean_mask
from api.weather.frame_builder import build_frames_from_db
from api.weather.formatters import format_single_frame, format_velocity_response
from api.weather.prefetch import (
    get_layer_manager,
    do_generic_prefetch,
    cleanup_stale_caches,
    _get_providers,
    _db_weather,
    _weather_ingestion,
    acquire_resync,
    release_resync,
    get_resync_status,
    set_resync_progress,
    get_resync_progress,
    clear_resync_progress,
)
from api.forecast_layer_manager import cache_covers_bounds, find_covering_cache
from api.weather.adrs_areas import (
    ADRS_AREAS,
    AREA_SPECIFIC_FIELDS,
    GLOBAL_FIELDS,
    get_adrs_area,
    compute_union_bbox,
    compute_union_ice_bbox,
)
from api.weather.area_config import get_selected_areas, set_selected_areas

from src.data.copernicus import GFSDataProvider

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Weather"])


# ============================================================================
# Single-frame field fetcher map
# ============================================================================

_SINGLE_FRAME_FETCHER = {
    "wind": lambda params, time: get_wind_field(
        params["lat_min"],
        params["lat_max"],
        params["lon_min"],
        params["lon_max"],
        params["resolution"],
        time,
    ),
    "waves": lambda params, time: get_wave_field(
        params["lat_min"],
        params["lat_max"],
        params["lon_min"],
        params["lon_max"],
        params["resolution"],
        get_wind_field(
            params["lat_min"],
            params["lat_max"],
            params["lon_min"],
            params["lon_max"],
            params["resolution"],
            time,
        ),
    ),
    "swell": lambda params, time: get_wave_field(
        params["lat_min"],
        params["lat_max"],
        params["lon_min"],
        params["lon_max"],
        params["resolution"],
    ),
    "currents": lambda params, time: get_current_field(
        params["lat_min"],
        params["lat_max"],
        params["lon_min"],
        params["lon_max"],
        params["resolution"],
    ),
    "sst": lambda params, time: get_sst_field(
        params["lat_min"],
        params["lat_max"],
        params["lon_min"],
        params["lon_max"],
        params["resolution"],
        time,
    ),
    "visibility": lambda params, time: get_visibility_field(
        params["lat_min"],
        params["lat_max"],
        params["lon_min"],
        params["lon_max"],
        params["resolution"],
        time,
    ),
    "ice": lambda params, time: get_ice_field(
        params["lat_min"],
        params["lat_max"],
        params["lon_min"],
        params["lon_max"],
        params["resolution"],
        time,
    ),
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
                    [
                        is_ocean(round(float(lat), 2), round(float(lon), 2))
                        for lon in grid.lons
                    ]
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
        raise HTTPException(
            status_code=503, detail="Database weather provider not configured"
        )
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
            "speed_ms": wx["wind_speed_ms"],
            "speed_kts": wx["wind_speed_ms"] * 1.94384,
            "dir_deg": wx["wind_dir_deg"],
        },
        "waves": {
            "height_m": wx["sig_wave_height_m"],
            "dir_deg": wx["wave_dir_deg"],
        },
        "current": {
            "speed_ms": wx["current_speed_ms"],
            "speed_kts": wx["current_speed_ms"] * 1.94384,
            "dir_deg": wx["current_dir_deg"],
        },
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

    age_hours = (
        freshness.get("age_hours", None) if isinstance(freshness, dict) else None
    )
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
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0),
    lon_max: float = Query(40.0),
):
    return await api_get_field_status(
        field="wind", lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max
    )


@router.post(
    "/api/weather/forecast/prefetch",
    dependencies=[Depends(require_not_demo("Weather prefetch"))],
)
async def _compat_wind_prefetch(
    background_tasks: BackgroundTasks,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0),
    lon_max: float = Query(40.0),
):
    return await api_trigger_field_prefetch(
        field="wind",
        background_tasks=background_tasks,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )


@router.get("/api/weather/forecast/frames")
async def _compat_wind_frames(
    request: Request,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0),
    lon_max: float = Query(40.0),
):
    return await api_get_field_frames(
        field="wind",
        request=request,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )


@router.get("/api/weather/forecast/wave/status")
async def _compat_wave_status(
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0),
    lon_max: float = Query(40.0),
):
    return await api_get_field_status(
        field="waves",
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )


@router.post(
    "/api/weather/forecast/wave/prefetch",
    dependencies=[Depends(require_not_demo("Weather prefetch"))],
)
async def _compat_wave_prefetch(
    background_tasks: BackgroundTasks,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0),
    lon_max: float = Query(40.0),
):
    return await api_trigger_field_prefetch(
        field="waves",
        background_tasks=background_tasks,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )


@router.get("/api/weather/forecast/wave/frames")
async def _compat_wave_frames(
    request: Request,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0),
    lon_max: float = Query(40.0),
):
    return await api_get_field_frames(
        field="waves",
        request=request,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )


@router.get("/api/weather/forecast/current/status")
async def _compat_current_status(
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0),
    lon_max: float = Query(40.0),
):
    return await api_get_field_status(
        field="currents",
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )


@router.post(
    "/api/weather/forecast/current/prefetch",
    dependencies=[Depends(require_not_demo("Weather prefetch"))],
)
async def _compat_current_prefetch(
    background_tasks: BackgroundTasks,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0),
    lon_max: float = Query(40.0),
):
    return await api_trigger_field_prefetch(
        field="currents",
        background_tasks=background_tasks,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )


@router.get("/api/weather/forecast/current/frames")
async def _compat_current_frames(
    request: Request,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0),
    lon_max: float = Query(40.0),
):
    return await api_get_field_frames(
        field="currents",
        request=request,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )


@router.get("/api/weather/forecast/ice/status")
async def _compat_ice_status(
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0),
    lon_max: float = Query(40.0),
):
    return await api_get_field_status(
        field="ice", lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max
    )


@router.post(
    "/api/weather/forecast/ice/prefetch",
    dependencies=[Depends(require_not_demo("Weather prefetch"))],
)
async def _compat_ice_prefetch(
    background_tasks: BackgroundTasks,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0),
    lon_max: float = Query(40.0),
):
    return await api_trigger_field_prefetch(
        field="ice",
        background_tasks=background_tasks,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )


@router.get("/api/weather/forecast/ice/frames")
async def _compat_ice_frames(
    request: Request,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0),
    lon_max: float = Query(40.0),
):
    return await api_get_field_frames(
        field="ice",
        request=request,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )


@router.get("/api/weather/forecast/sst/status")
async def _compat_sst_status(
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0),
    lon_max: float = Query(40.0),
):
    return await api_get_field_status(
        field="sst", lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max
    )


@router.post(
    "/api/weather/forecast/sst/prefetch",
    dependencies=[Depends(require_not_demo("Weather prefetch"))],
)
async def _compat_sst_prefetch(
    background_tasks: BackgroundTasks,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0),
    lon_max: float = Query(40.0),
):
    return await api_trigger_field_prefetch(
        field="sst",
        background_tasks=background_tasks,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )


@router.get("/api/weather/forecast/sst/frames")
async def _compat_sst_frames(
    request: Request,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0),
    lon_max: float = Query(40.0),
):
    return await api_get_field_frames(
        field="sst",
        request=request,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )


@router.get("/api/weather/forecast/visibility/status")
async def _compat_vis_status(
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0),
    lon_max: float = Query(40.0),
):
    return await api_get_field_status(
        field="visibility",
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )


@router.post(
    "/api/weather/forecast/visibility/prefetch",
    dependencies=[Depends(require_not_demo("Weather prefetch"))],
)
async def _compat_vis_prefetch(
    background_tasks: BackgroundTasks,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0),
    lon_max: float = Query(40.0),
):
    return await api_trigger_field_prefetch(
        field="visibility",
        background_tasks=background_tasks,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )


@router.get("/api/weather/forecast/visibility/frames")
async def _compat_vis_frames(
    request: Request,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0),
    lon_max: float = Query(40.0),
):
    return await api_get_field_frames(
        field="visibility",
        request=request,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )


# ============================================================================
# Resync status + resync-all + ocean area config
# (registered before {field} catch-all routes)
# ============================================================================


@router.get("/api/weather/readiness")
async def api_weather_readiness():
    """Per-area weather data readiness for startup area selector.

    Inspects file cache only — no DB queries, sub-millisecond response.
    Returns global fields (wind/visibility) once, then per-area status for
    CMEMS fields (waves, currents, sst, ice).
    """
    from api.main import is_prefetch_running

    selected = get_selected_areas()

    # Global fields — checked once against their default bbox
    global_fields = {}
    for name in ("wind", "visibility"):
        cfg = get_field(name)
        mgr = get_layer_manager(name)
        lat_min, lat_max, lon_min, lon_max = cfg.default_bbox
        cache_key = mgr.make_cache_key(lat_min, lat_max, lon_min, lon_max)
        envelope = mgr.cache_get(cache_key)
        frame_count = len(envelope.get("frames", {})) if envelope else 0
        global_fields[name] = {
            "status": "ready" if frame_count >= cfg.expected_frames else "missing",
            "frames": frame_count,
            "expected": cfg.expected_frames,
        }

    # Per-area fields — check cache for each selected ADRS area
    areas = {}
    for area_id in selected:
        try:
            area = get_adrs_area(area_id)
        except KeyError:
            continue

        area_fields = {}
        for name in ("waves", "currents", "sst", "ice"):
            cfg = get_field(name)

            # Ice is not applicable for areas without ice_bbox
            if name == "ice" and area.ice_bbox is None:
                area_fields[name] = {
                    "status": "not_applicable",
                    "frames": 0,
                    "expected": 0,
                }
                continue

            bbox = area.ice_bbox if name == "ice" else area.bbox
            lat_min, lat_max, lon_min, lon_max = bbox
            mgr = get_layer_manager(name)
            cache_key = mgr.make_cache_key(lat_min, lat_max, lon_min, lon_max)
            envelope = mgr.cache_get(cache_key)

            # Fallback: check if a covering cache (e.g. union bbox) exists
            if envelope is None:
                covering = find_covering_cache(
                    mgr.cache_dir,
                    mgr.name,
                    lat_min,
                    lat_max,
                    lon_min,
                    lon_max,
                )
                if covering is not None:
                    try:
                        import json as _json

                        envelope = _json.loads(covering.read_text())
                    except Exception:
                        envelope = None

            frame_count = len(envelope.get("frames", {})) if envelope else 0

            # Validate that cached data actually covers the requested bbox
            if envelope and frame_count >= cfg.expected_frames:
                if not cache_covers_bounds(
                    envelope, lat_min, lat_max, lon_min, lon_max, min_coverage=0.8
                ):
                    frame_count = 0  # data is from wrong area

            area_fields[name] = {
                "status": "ready" if frame_count >= cfg.expected_frames else "missing",
                "frames": frame_count,
                "expected": cfg.expected_frames,
            }

        area_all_ready = all(
            f["status"] in ("ready", "not_applicable") for f in area_fields.values()
        )
        areas[area_id] = {
            "label": area.label,
            "fields": area_fields,
            "all_ready": area_all_ready,
        }

    global_ok = all(f["status"] == "ready" for f in global_fields.values())
    areas_ok = all(a["all_ready"] for a in areas.values()) if areas else True
    all_ready = global_ok and areas_ok

    # Available areas list (for area selector UI)
    available = [
        {
            "id": a.id,
            "label": a.label,
            "description": a.description,
            "bbox": list(a.bbox),
            "ice_bbox": list(a.ice_bbox) if a.ice_bbox else None,
            "disabled": a.disabled,
        }
        for a in ADRS_AREAS.values()
    ]

    resync_active = get_resync_status()
    progress = get_resync_progress() if resync_active else {}

    return {
        "global_fields": global_fields,
        "areas": areas,
        "all_ready": all_ready,
        "prefetch_running": is_prefetch_running(),
        "resync_active": resync_active,
        "resync_progress": progress,
        "selected_areas": selected,
        "available_areas": available,
    }


@router.get("/api/weather/resync-status")
async def api_weather_resync_status():
    """Return the currently running resync field, or null."""
    active = get_resync_status()
    return {"active": active}


@router.post(
    "/api/weather/resync-all",
    dependencies=[Depends(require_not_demo("Weather resync"))],
)
async def api_weather_resync_all():
    """Re-ingest all weather fields using union bbox of selected ADRS areas.

    Uses a single CMEMS download per field covering all selected areas
    so one DB run holds the full extent.  This prevents the display
    issue where only the last-ingested area's data is visible.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    weather_ingestion = _weather_ingestion()
    if weather_ingestion is None:
        raise HTTPException(status_code=503, detail="Weather ingestion not configured")

    if not acquire_resync("all"):
        raise HTTPException(
            status_code=409, detail=f"Resync already running: {get_resync_status()}"
        )

    import threading

    def _run_all():
        try:
            clear_resync_progress()
            selected = get_selected_areas()

            union_bbox = compute_union_bbox(selected)
            union_ice = compute_union_ice_bbox(selected)

            if union_bbox is None:
                logger.warning("Resync-all: no valid areas selected")
                return

            def _resync_field(field_name: str, bbox, progress_labels: list):
                for lbl in progress_labels:
                    set_resync_progress(lbl, "downloading")
                try:
                    cfg = get_field(field_name)
                    ingest_fn = getattr(weather_ingestion, cfg.ingest_method)

                    if field_name in AREA_SPECIFIC_FIELDS:
                        ingest_fn(True, *bbox)
                    else:
                        ingest_fn(True)

                    # Clear cache files that overlap with this bbox
                    mgr = get_layer_manager(field_name)
                    lat_min, lat_max, lon_min, lon_max = bbox
                    if mgr.cache_dir.exists():
                        for f in mgr.cache_dir.iterdir():
                            if f.suffix == ".json":
                                f.unlink(missing_ok=True)
                            elif f.name.endswith(".json.gz"):
                                f.unlink(missing_ok=True)

                    # Rebuild file cache from DB
                    db_w = _db_weather()
                    if db_w is not None:
                        cache_key = mgr.make_cache_key(
                            lat_min, lat_max, lon_min, lon_max
                        )
                        rebuilt = build_frames_from_db(
                            field_name, db_w, lat_min, lat_max, lon_min, lon_max
                        )
                        if rebuilt and len(rebuilt.get("frames", {})) > 0:
                            mgr.cache_put(cache_key, rebuilt)

                    for lbl in progress_labels:
                        set_resync_progress(lbl, "done")
                    return field_name
                except Exception:
                    for lbl in progress_labels:
                        set_resync_progress(lbl, "failed")
                    raise

            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = {}

                # Global fields (wind, visibility) — once
                for field_name in GLOBAL_FIELDS:
                    cfg = get_field(field_name)
                    labels = [f"{field_name}:global"]
                    fut = pool.submit(
                        _resync_field, field_name, cfg.default_bbox, labels
                    )
                    futures[fut] = labels[0]

                # Area-specific fields — single download with union bbox
                for field_name in ("waves", "currents", "sst"):
                    labels = [f"{field_name}:{a}" for a in selected]
                    fut = pool.submit(_resync_field, field_name, union_bbox, labels)
                    futures[fut] = labels[0]

                # Ice — union ice bbox (skip if no areas have ice)
                if union_ice is not None:
                    ice_areas = [
                        a
                        for a in selected
                        if ADRS_AREAS.get(a) and ADRS_AREAS[a].ice_bbox
                    ]
                    labels = [f"ice:{a}" for a in ice_areas]
                    fut = pool.submit(_resync_field, "ice", union_ice, labels)
                    futures[fut] = labels[0]

                for future in as_completed(futures):
                    label = futures[future]
                    try:
                        future.result()
                        logger.info(f"Resync-all: {label} complete")
                    except Exception as e:
                        logger.error(f"Resync-all: {label} failed: {e}")

            # Supersede old runs and clean up AFTER all fields are done
            seen_sources = set()
            for field_name in list(GLOBAL_FIELDS) + ["waves", "currents", "sst", "ice"]:
                cfg = get_field(field_name)
                if cfg.source not in seen_sources:
                    seen_sources.add(cfg.source)
                    weather_ingestion._supersede_old_runs(cfg.source)
                    weather_ingestion.cleanup_orphaned_grid_data(cfg.source)

            cleanup_stale_caches()
            logger.info("Resync-all complete")
        except Exception as e:
            logger.error(f"Resync-all failed: {e}", exc_info=True)
        finally:
            release_resync()

    threading.Thread(target=_run_all, daemon=True).start()

    return {"status": "started", "areas": get_selected_areas()}


@router.get("/api/weather/ocean-areas")
async def api_weather_ocean_areas():
    """Return ADRS Volume 6 area definitions."""
    selected = get_selected_areas()
    areas = []
    for area in ADRS_AREAS.values():
        areas.append(
            {
                "id": area.id,
                "label": area.label,
                "description": area.description,
                "bbox": list(area.bbox),
                "ice_bbox": list(area.ice_bbox) if area.ice_bbox else None,
                "disabled": area.disabled,
            }
        )
    return {"areas": areas, "selected": selected}


@router.get("/api/weather/selected-areas")
async def api_get_selected_areas():
    """Return the currently selected ADRS area IDs."""
    return {"selected": get_selected_areas()}


@router.post(
    "/api/weather/selected-areas",
    dependencies=[Depends(require_not_demo("Area configuration"))],
)
async def api_set_selected_areas(areas: list[str]):
    """Update the selected ADRS area IDs."""
    try:
        set_selected_areas(areas)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"selected": get_selected_areas()}


@router.post(
    "/api/weather/resync-area",
    dependencies=[Depends(require_not_demo("Weather resync"))],
)
async def api_weather_resync_area(area: str = Query(...)):
    """Re-ingest CMEMS fields for a single ADRS area (background)."""
    try:
        adrs_area = get_adrs_area(area)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unknown ADRS area: {area}")

    weather_ingestion = _weather_ingestion()
    if weather_ingestion is None:
        raise HTTPException(status_code=503, detail="Weather ingestion not configured")

    if not acquire_resync(f"area:{area}"):
        raise HTTPException(
            status_code=409, detail=f"Resync already running: {get_resync_status()}"
        )

    import threading

    def _run_resync():
        try:
            clear_resync_progress()
            seen_sources = set()
            for field_name in ("waves", "currents", "sst", "ice"):
                if field_name == "ice" and adrs_area.ice_bbox is None:
                    continue

                label = f"{field_name}:{area}"
                bbox = adrs_area.ice_bbox if field_name == "ice" else adrs_area.bbox
                cfg = get_field(field_name)
                ingest_fn = getattr(weather_ingestion, cfg.ingest_method)

                set_resync_progress(label, "downloading")
                try:
                    ingest_fn(True, *bbox)
                    seen_sources.add(cfg.source)

                    # Clear ALL caches for this field (per-area and union)
                    mgr = get_layer_manager(field_name)
                    if mgr.cache_dir.exists():
                        for f in mgr.cache_dir.iterdir():
                            if f.suffix == ".json" or f.name.endswith(".json.gz"):
                                f.unlink(missing_ok=True)

                    # Rebuild file cache from DB for this area bbox
                    lat_min, lat_max, lon_min, lon_max = bbox
                    db_w = _db_weather()
                    if db_w is not None:
                        cache_key = mgr.make_cache_key(
                            lat_min, lat_max, lon_min, lon_max
                        )
                        rebuilt = build_frames_from_db(
                            field_name, db_w, lat_min, lat_max, lon_min, lon_max
                        )
                        if rebuilt and len(rebuilt.get("frames", {})) > 0:
                            mgr.cache_put(cache_key, rebuilt)

                    set_resync_progress(label, "done")
                    logger.info(f"Resync-area {area}: {field_name} complete")
                except Exception as e:
                    set_resync_progress(label, "failed")
                    logger.error(f"Resync-area {area}: {field_name} failed: {e}")

            # Supersede/cleanup after all fields done
            for source in seen_sources:
                weather_ingestion._supersede_old_runs(source)
                weather_ingestion.cleanup_orphaned_grid_data(source)

            cleanup_stale_caches()
            logger.info(f"Resync-area {area}: all fields complete")
        except Exception as e:
            logger.error(f"Resync-area {area} failed: {e}", exc_info=True)
        finally:
            release_resync()

    threading.Thread(target=_run_resync, daemon=True).start()

    return {"status": "started", "area": area}


@router.post(
    "/api/weather/purge-all",
    dependencies=[Depends(require_not_demo("Weather purge"))],
)
async def api_weather_purge_all():
    """Delete ALL weather data (DB + file caches) for a clean slate."""
    if get_resync_status() is not None:
        raise HTTPException(
            status_code=409, detail=f"Resync running: {get_resync_status()}"
        )

    db_weather = _db_weather()
    purged = {"db_runs": 0, "db_grids": 0, "cache_files": 0}

    # Purge DB
    if db_weather is not None:
        try:
            conn = db_weather._get_conn()
            with conn.cursor() as cur:
                cur.execute("DELETE FROM weather_grid_data")
                purged["db_grids"] = cur.rowcount
                cur.execute("DELETE FROM weather_forecast_runs")
                purged["db_runs"] = cur.rowcount
            conn.commit()
            conn.close()
            logger.info(
                f"Purged DB: {purged['db_runs']} runs, {purged['db_grids']} grids"
            )
        except Exception as e:
            logger.error(f"DB purge failed: {e}")
            raise HTTPException(status_code=500, detail=f"DB purge failed: {e}")

    # Purge file caches
    for cache_dir in [Path("/tmp/windmar_cache"), Path("/tmp/windmar_tiles")]:
        if cache_dir.exists():
            for item in cache_dir.rglob("*"):
                if item.is_file():
                    item.unlink(missing_ok=True)
                    purged["cache_files"] += 1

    copernicus_cache = Path("data/copernicus_cache")
    if copernicus_cache.exists():
        for item in copernicus_cache.rglob("*"):
            if item.is_file():
                item.unlink(missing_ok=True)
                purged["cache_files"] += 1

    logger.info(f"Purge-all complete: {purged}")
    return {"status": "complete", "purged": purged}


# ============================================================================
# Debug endpoint
# ============================================================================


@router.get("/api/weather/{field}/debug")
async def api_get_field_debug(
    field: str,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0),
    lon_max: float = Query(40.0),
):
    """Return lightweight diagnostics for a layer's cached data."""
    if field not in WEATHER_FIELDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown field: {field}. Valid: {list(FIELD_NAMES)}",
        )

    cfg = get_field(field)
    mgr = get_layer_manager(field)
    lat_min, lat_max, lon_min, lon_max = clamp_bbox(lat_min, lat_max, lon_min, lon_max)
    cache_key = mgr.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    cached = mgr.cache_get(cache_key)

    if not cached:
        return {"field": field, "status": "no_cache", "cache_key": cache_key}

    frames = cached.get("frames", {})
    frame_hours = sorted(
        frames.keys(), key=lambda h: int(h) if h.lstrip("-").isdigit() else 0
    )

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
        checks.append(
            {
                "check": "ny == len(lats)",
                "pass": ny == len(lats),
                "detail": f"ny={ny}, len(lats)={len(lats)}",
            }
        )
        checks.append(
            {
                "check": "nx == len(lons)",
                "pass": nx == len(lons),
                "detail": f"nx={nx}, len(lons)={len(lons)}",
            }
        )
        if sample_rows > 0:
            checks.append(
                {
                    "check": "frame_data_rows == ny",
                    "pass": sample_rows == ny,
                    "detail": f"sample_rows={sample_rows}, ny={ny}",
                }
            )
            checks.append(
                {
                    "check": "frame_data_cols == nx",
                    "pass": sample_cols == nx,
                    "detail": f"sample_cols={sample_cols}, nx={nx}",
                }
            )
    else:
        if sample_rows > 0:
            checks.append(
                {
                    "check": "wind header ny/nx consistent",
                    "pass": sample_rows > 0 and sample_cols > 0,
                    "detail": f"header ny={sample_rows}, nx={sample_cols}",
                }
            )

    if ocean_mask is not None:
        mask_rows = len(ocean_mask) if isinstance(ocean_mask, list) else 0
        mask_cols = len(ocean_mask[0]) if mask_rows > 0 else 0
        checks.append(
            {
                "check": "ocean_mask rows == mask_lats",
                "pass": mask_rows == len(ocean_mask_lats),
                "detail": f"mask_rows={mask_rows}, mask_lats={len(ocean_mask_lats)}",
            }
        )
        checks.append(
            {
                "check": "ocean_mask cols == mask_lons",
                "pass": mask_cols == len(ocean_mask_lons),
                "detail": f"mask_cols={mask_cols}, mask_lons={len(ocean_mask_lons)}",
            }
        )

    schema_version = cached.get("_schema_version", 0)
    checks.append(
        {
            "check": "schema_version current",
            "pass": schema_version == CACHE_SCHEMA_VERSION,
            "detail": f"cached={schema_version}, current={CACHE_SCHEMA_VERSION}",
        }
    )

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
        "ocean_mask_shape": (
            f"{len(ocean_mask)}x{len(ocean_mask[0]) if ocean_mask else 0}"
            if ocean_mask
            else None
        ),
        "has_colorscale": colorscale is not None,
        "colorscale_keys": (
            list(colorscale.keys()) if isinstance(colorscale, dict) else None
        ),
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
        raise HTTPException(
            status_code=400,
            detail=f"Unknown field: {field}. Valid: {list(FIELD_NAMES)}",
        )

    cfg = get_field(field)
    if time is None:
        time = datetime.now(timezone.utc)

    lat_min, lat_max, lon_min, lon_max = clamp_bbox(lat_min, lat_max, lon_min, lon_max)

    params = dict(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        resolution=resolution,
    )

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

    # Only fall through to live provider for GFS-backed layers (fast).
    # CMEMS layers (waves, swell, currents, sst, ice) must come from DB —
    # live CMEMS fetches take minutes and corrupt concurrent prefetch downloads.
    _CMEMS_FIELDS = {"waves", "swell", "currents", "sst", "ice"}
    if data is None and field not in _CMEMS_FIELDS:
        data = _SINGLE_FRAME_FETCHER[field](params, time)
        if db_weather is not None:
            ingested_at = datetime.now(timezone.utc)

    if data is None or not hasattr(data, "lats") or data.lats is None:
        raise HTTPException(
            status_code=503, detail=f"No {field} data available. Try resyncing."
        )

    return format_single_frame(
        field,
        cfg,
        data,
        time,
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
        raise HTTPException(
            status_code=400,
            detail=f"Velocity format only for wind/currents, not {field}",
        )

    if time is None:
        time = datetime.now(timezone.utc)

    lat_min, lat_max, lon_min, lon_max = clamp_bbox(lat_min, lat_max, lon_min, lon_max)

    if field == "wind":
        providers = _get_providers()
        gfs_provider = providers["gfs"]
        if forecast_hour > 0:
            data = gfs_provider.fetch_wind_data(
                lat_min, lat_max, lon_min, lon_max, time, forecast_hour
            )
            if data is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Forecast hour f{forecast_hour:03d} not available",
                )
        else:
            data = get_wind_field(lat_min, lat_max, lon_min, lon_max, resolution, time)
    else:
        # Currents: DB-first, no live CMEMS fallback
        db_weather = _db_weather()
        data = None
        if db_weather is not None:
            data, _ = db_weather.get_current_from_db(lat_min, lat_max, lon_min, lon_max)
        if data is None:
            raise HTTPException(
                status_code=503, detail="No currents data available. Try resyncing."
            )

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
        raise HTTPException(
            status_code=400,
            detail=f"Unknown field: {field}. Valid: {list(FIELD_NAMES)}",
        )

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
        gfs_provider = providers["gfs"]
        if mgr.last_run:
            run_date, run_hour = mgr.last_run
        else:
            run_date, run_hour = gfs_provider._get_latest_run()
        hours = gfs_provider.get_cached_forecast_hours(
            lat_min, lat_max, lon_min, lon_max, run_date, run_hour
        )
        cached_count = sum(1 for h in hours if h["cached"])

        if cached_count == 0 and not prefetch_running:
            best = gfs_provider.find_best_cached_run(lat_min, lat_max, lon_min, lon_max)
            if best:
                run_date, run_hour = best
                hours = gfs_provider.get_cached_forecast_hours(
                    lat_min, lat_max, lon_min, lon_max, run_date, run_hour
                )
        cached_count = sum(1 for h in hours if h["cached"])

        if cached_count == 0 and db_weather is not None:
            db_run_time, db_hours = db_weather.get_available_hours_by_source("gfs")
            if db_hours:
                return {
                    "run_date": (
                        db_run_time.strftime("%Y%m%d") if db_run_time else run_date
                    ),
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


@router.post(
    "/api/weather/{field}/prefetch",
    dependencies=[Depends(require_not_demo("Weather prefetch"))],
)
async def api_trigger_field_prefetch(
    field: str,
    background_tasks: BackgroundTasks,
    lat_min: float = Query(30.0),
    lat_max: float = Query(60.0),
    lon_min: float = Query(-30.0),
    lon_max: float = Query(40.0),
):
    """Trigger background rebuild of forecast cache for any field.

    Rebuilds file cache from DB only.  Provider downloads (CMEMS/GFS) are
    triggered exclusively by the ``/resync`` endpoint.
    """
    if field not in WEATHER_FIELDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown field: {field}. Valid: {list(FIELD_NAMES)}",
        )

    mgr = get_layer_manager(field)

    def _db_only_prefetch(mgr, la1, la2, lo1, lo2, **kw):
        do_generic_prefetch(mgr, la1, la2, lo1, lo2, db_only=True, **kw)

    return mgr.trigger_response(
        background_tasks,
        _db_only_prefetch,
        lat_min,
        lat_max,
        lon_min,
        lon_max,
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
        raise HTTPException(
            status_code=400,
            detail=f"Unknown field: {field}. Valid: {list(FIELD_NAMES)}",
        )

    lat_min, lat_max, lon_min, lon_max = clamp_bbox(lat_min, lat_max, lon_min, lon_max)

    cfg = get_field(field)
    mgr = get_layer_manager(field)
    cache_key = mgr.make_cache_key(lat_min, lat_max, lon_min, lon_max)
    _is_demo_user = is_demo() and is_demo_user(request)

    cached = mgr.cache_get(cache_key)
    if cached is not None and cached.get("_schema_version") != CACHE_SCHEMA_VERSION:
        logger.info(
            f"{field} cache stale (schema {cached.get('_schema_version')} != {CACHE_SCHEMA_VERSION}), rebuilding"
        )
        cached = None
    if cached is not None:
        if _is_demo_user:
            return limit_demo_frames(cached)
        raw = mgr.serve_frames_file(
            cache_key,
            lat_min,
            lat_max,
            lon_min,
            lon_max,
            use_covering=True,
        )
        if raw is not None:
            return raw
        return cached

    covering_raw = mgr.serve_frames_file(
        cache_key,
        lat_min,
        lat_max,
        lon_min,
        lon_max,
        use_covering=True,
    )
    if covering_raw is not None:
        logger.info(
            f"{field} frames: serving covering cache for [{lat_min:.0f},{lat_max:.0f}]x[{lon_min:.0f},{lon_max:.0f}]"
        )
        if _is_demo_user:
            import json as _json

            covering_data = _json.loads(covering_raw.body)
            return limit_demo_frames(covering_data)
        return covering_raw

    db_weather = _db_weather()
    if db_weather is not None:
        cached = await asyncio.to_thread(
            build_frames_from_db,
            field,
            db_weather,
            lat_min,
            lat_max,
            lon_min,
            lon_max,
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
        gfs_provider = providers["gfs"]
        run_date, run_hour = gfs_provider._get_latest_run()
        from datetime import datetime as dt

        run_time = dt.strptime(f"{run_date}{run_hour}", "%Y%m%d%H")
        empty.update(
            run_date=run_date, run_hour=run_hour, run_time=run_time.isoformat()
        )
    else:
        empty.update(lats=[], lons=[], ny=0, nx=0)
    return empty


@router.post(
    "/api/weather/{field}/resync",
    dependencies=[Depends(require_not_demo("Weather resync"))],
)
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
        raise HTTPException(
            status_code=400,
            detail=f"Unknown field: {field}. Valid: {list(FIELD_NAMES)}",
        )
    if weather_ingestion is None:
        raise HTTPException(status_code=503, detail="Weather ingestion not configured")

    cfg = get_field(field)

    has_bbox = all(v is not None for v in (lat_min, lat_max, lon_min, lon_max))
    if has_bbox:
        lat_min, lat_max, lon_min, lon_max = clamp_bbox(
            lat_min, lat_max, lon_min, lon_max
        )

    logger.info(
        f"Per-layer resync starting: {field}"
        + (
            f" bbox=[{lat_min:.1f},{lat_max:.1f}]x[{lon_min:.1f},{lon_max:.1f}]"
            if has_bbox
            else ""
        )
    )

    if not acquire_resync(field):
        raise HTTPException(
            status_code=409, detail=f"Resync already running: {get_resync_status()}"
        )

    try:
        ingest_fn = getattr(weather_ingestion, cfg.ingest_method)
        cmems_layers = {"waves", "currents", "swell", "ice", "sst"}
        if has_bbox and field in cmems_layers:
            await asyncio.to_thread(ingest_fn, True, lat_min, lat_max, lon_min, lon_max)
        else:
            await asyncio.to_thread(ingest_fn, True)

        weather_ingestion._supersede_old_runs(cfg.source)
        weather_ingestion.cleanup_orphaned_grid_data(cfg.source)

        # Clear only the cache file for the requested bbox
        mgr = get_layer_manager(field)
        if has_bbox:
            ck = mgr.make_cache_key(lat_min, lat_max, lon_min, lon_max)
            cf = mgr.cache_path(ck)
            if cf.exists():
                cf.unlink(missing_ok=True)
        elif mgr.cache_dir.exists():
            for f in mgr.cache_dir.iterdir():
                f.unlink(missing_ok=True)

        cleanup_stale_caches()

        _, db_ingested_at = (
            db_weather._find_latest_run(cfg.source) if db_weather else (None, None)
        )
        ingested_at = db_ingested_at or datetime.now(timezone.utc)
        logger.info(
            f"Per-layer resync complete: {field}, ingested_at={ingested_at.isoformat()}"
        )
        return {"status": "complete", "ingested_at": ingested_at.isoformat()}

    except Exception as e:
        logger.error(f"Resync failed for {field}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Resync failed: {e}")
    finally:
        release_resync()

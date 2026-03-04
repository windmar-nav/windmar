"""
Frame builder — constructs forecast-frame cache envelopes.

Two entry points:
- ``build_frames_from_db``    — from PostgreSQL grid data
- ``build_frames_from_provider`` — from live provider WeatherData objects

Both produce the same cache envelope dict that gets written to the file
cache by ``ForecastLayerManager.cache_put``.
"""

import logging
from datetime import timedelta

import numpy as np

from api.weather_fields import FieldConfig, CACHE_SCHEMA_VERSION, get_field
from api.weather.grid_processor import SubsampledGrid, make_grid
from api.weather.ocean_mask import (
    build_ocean_mask_from_data,
    build_ocean_mask_from_weather_data,
    build_ice_ocean_mask,
    mask_velocity_with_nan,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Leaflet-velocity helpers
# ---------------------------------------------------------------------------


def _velocity_header(lats, lons, run_time, fh):
    """Build a leaflet-velocity header dict."""
    dx = abs(float(lons[1] - lons[0])) if len(lons) > 1 else 0.25
    dy = abs(float(lats[1] - lats[0])) if len(lats) > 1 else 0.25
    if len(lats) > 1 and lats[1] > lats[0]:
        lat_north = float(lats[-1])
        lat_south = float(lats[0])
    else:
        lat_north = float(lats[0])
        lat_south = float(lats[-1])
    return {
        "parameterCategory": 2,
        "parameterNumber": 2,
        "lo1": float(lons[0]),
        "la1": lat_north,
        "lo2": float(lons[-1]),
        "la2": lat_south,
        "dx": dx,
        "dy": dy,
        "nx": len(lons),
        "ny": len(lats),
        "refTime": run_time.isoformat() if run_time else "",
        "forecastHour": fh,
    }


def _order_north_to_south(u, v, lats):
    """Flip arrays to north-to-south order if needed for leaflet-velocity."""
    if len(lats) > 1 and lats[1] > lats[0]:
        return u[::-1], v[::-1]
    return u, v


# ---------------------------------------------------------------------------
# Build frames from DB grids
# ---------------------------------------------------------------------------


def build_frames_from_db(
    field_name: str,
    db_weather,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
):
    """Rebuild forecast frame cache from PostgreSQL for any field.

    Returns a cache envelope dict, or None if no data.
    """
    cfg = get_field(field_name)
    run_time, hours = db_weather.get_available_hours_by_source(cfg.source)
    if not hours:
        return None

    logger.info(f"Rebuilding {field_name} cache from DB: {len(hours)} hours")

    grids = db_weather.get_grids_for_timeline(
        cfg.source,
        list(cfg.parameters),
        lat_min,
        lat_max,
        lon_min,
        lon_max,
        hours,
    )

    if not grids:
        return None

    primary_param = cfg.parameters[0]
    if primary_param not in grids or not grids[primary_param]:
        return None

    first_fh = min(grids[primary_param].keys())
    lats_full, lons_full, _ = grids[primary_param][first_fh]

    # ONE step for the entire field — the root-cause fix
    grid = make_grid(lats_full, lons_full, cfg)

    # Ocean mask — same grid, guaranteed shape match
    ocean_mask_data = None
    ice_ocean_arr = None
    if cfg.needs_ocean_mask:
        if field_name == "ice":
            ocean_mask_data, ice_ocean_arr = build_ice_ocean_mask(grid)
        else:
            ocean_mask_data = build_ocean_mask_from_data(
                grids,
                primary_param,
                sorted(hours),
                grid,
                lats_full,
                lons_full,
            )

    frames = {}

    if cfg.components == "vector":
        frames = _build_vector_frames_db(
            cfg,
            field_name,
            grids,
            hours,
            grid,
            lats_full,
            lons_full,
            run_time,
        )
    elif cfg.components == "wave_decomp":
        frames = _build_wave_frames_db(cfg, grids, hours, grid)
    elif cfg.components == "scalar":
        frames = _build_scalar_frames_db(
            cfg,
            field_name,
            grids,
            hours,
            grid,
            ice_ocean_arr,
        )

    return _assemble_envelope(
        cfg,
        field_name,
        grid,
        frames,
        run_time,
        ocean_mask_data=ocean_mask_data,
    )


def _build_vector_frames_db(
    cfg, field_name, grids, hours, grid, lats_full, lons_full, run_time
):
    """Build vector frames (wind/currents) from DB grids."""
    u_param, v_param = cfg.parameters[0], cfg.parameters[1]
    frames = {}
    for fh in sorted(hours):
        if fh not in grids.get(u_param, {}) or fh not in grids.get(v_param, {}):
            continue
        _, _, u_data = grids[u_param][fh]
        _, _, v_data = grids[v_param][fh]
        u_sub = u_data[:: grid.step, :: grid.step]
        v_sub = v_data[:: grid.step, :: grid.step]

        if field_name == "wind":
            u_m, v_m = mask_velocity_with_nan(
                u_sub,
                v_sub,
                grid,
                grids=grids,
                primary_param=u_param,
                hours=sorted(hours),
                full_lats=lats_full,
                full_lons=lons_full,
            )
            u_ord, v_ord = _order_north_to_south(u_m, v_m, grid.lats)
            header = _velocity_header(grid.lats, grid.lons, run_time, fh)
            frames[str(fh)] = [
                {
                    "header": {**header, "parameterNumber": 2},
                    "data": u_ord.flatten().tolist(),
                },
                {
                    "header": {**header, "parameterNumber": 3},
                    "data": v_ord.flatten().tolist(),
                },
            ]
        else:
            u_m, v_m = mask_velocity_with_nan(
                u_sub,
                v_sub,
                grid,
                grids=grids,
                primary_param=u_param,
                hours=sorted(hours),
                full_lats=lats_full,
                full_lons=lons_full,
            )
            frames[str(fh)] = {
                "u": np.round(u_m[::-1], cfg.decimals).tolist(),
                "v": np.round(v_m[::-1], cfg.decimals).tolist(),
            }
    return frames


def _build_wave_frames_db(cfg, grids, hours, grid):
    """Build wave decomposition frames from DB grids."""
    frames = {}
    for fh in sorted(hours):
        frame = {}
        if "wave_hs" in grids and fh in grids["wave_hs"]:
            _, _, d = grids["wave_hs"][fh]
            frame["data"] = np.round(
                d[:: grid.step, :: grid.step], cfg.decimals
            ).tolist()
        if "wave_dir" in grids and fh in grids["wave_dir"]:
            _, _, d = grids["wave_dir"][fh]
            frame["direction"] = np.round(
                d[:: grid.step, :: grid.step], cfg.decimals
            ).tolist()
        has_decomp = fh in grids.get("windwave_hs", {}) and fh in grids.get(
            "swell_hs", {}
        )
        if has_decomp:
            frame["windwave"] = {}
            for p, k in [
                ("windwave_hs", "height"),
                ("windwave_tp", "period"),
                ("windwave_dir", "direction"),
            ]:
                if fh in grids.get(p, {}):
                    _, _, d = grids[p][fh]
                    frame["windwave"][k] = np.round(
                        d[:: grid.step, :: grid.step], cfg.decimals
                    ).tolist()
            frame["swell"] = {}
            for p, k in [
                ("swell_hs", "height"),
                ("swell_tp", "period"),
                ("swell_dir", "direction"),
            ]:
                if fh in grids.get(p, {}):
                    _, _, d = grids[p][fh]
                    frame["swell"][k] = np.round(
                        d[:: grid.step, :: grid.step], cfg.decimals
                    ).tolist()
        if frame:
            frames[str(fh)] = frame
    return frames


def _build_scalar_frames_db(cfg, field_name, grids, hours, grid, ice_ocean_arr):
    """Build scalar frames from DB grids."""
    param = cfg.parameters[0]
    frames = {}
    for fh in sorted(hours):
        if fh not in grids.get(param, {}):
            continue
        _, _, d = grids[param][fh]
        clean = np.nan_to_num(d[:: grid.step, :: grid.step], nan=cfg.nan_fill)
        if clean.shape != (grid.ny, grid.nx):
            logger.error(
                f"{field_name} fh={fh} shape mismatch: data={clean.shape} "
                f"expected=({grid.ny},{grid.nx})"
            )
        if ice_ocean_arr is not None and clean.shape == ice_ocean_arr.shape:
            clean = np.where(ice_ocean_arr, clean, cfg.nan_fill)
        frames[str(fh)] = {"data": np.round(clean, cfg.decimals).tolist()}
    return frames


# ---------------------------------------------------------------------------
# Build frames from provider WeatherData
# ---------------------------------------------------------------------------


def build_frames_from_provider(
    field_name: str,
    result: dict,
    cfg: FieldConfig,
):
    """Build forecast frames from provider ``{fh: WeatherData}`` dict.

    Returns a cache envelope dict.
    """
    first_wd = next(iter(result.values()))
    grid = make_grid(first_wd.lats, first_wd.lons, cfg)

    ocean_mask_data = None
    ice_ocean_arr = None
    if cfg.needs_ocean_mask:
        if field_name == "ice":
            ocean_mask_data, ice_ocean_arr = build_ice_ocean_mask(grid)
        else:
            ocean_mask_data = build_ocean_mask_from_weather_data(
                result,
                cfg,
                grid,
                first_wd.lats,
                first_wd.lons,
            )

    frames = {}

    if cfg.components == "vector":
        frames = _build_vector_frames_provider(cfg, result, grid)
    elif cfg.components == "wave_decomp":
        frames = _build_wave_frames_provider(cfg, result, grid)
    elif cfg.components == "scalar":
        frames = _build_scalar_frames_provider(
            cfg,
            field_name,
            result,
            grid,
            ice_ocean_arr,
        )

    run_time = first_wd.time
    return _assemble_envelope(
        cfg,
        field_name,
        grid,
        frames,
        run_time,
        ocean_mask_data=ocean_mask_data,
    )


def _build_vector_frames_provider(cfg, result, grid):
    """Build vector frames from provider data (currents only — wind uses GRIB path)."""
    frames = {}
    for fh, wd in sorted(result.items()):
        if wd.u_component is not None and wd.v_component is not None:
            u_sub = wd.u_component[:: grid.step, :: grid.step]
            v_sub = wd.v_component[:: grid.step, :: grid.step]
            u_m, v_m = mask_velocity_with_nan(u_sub, v_sub, grid)
            frames[str(fh)] = {
                "u": np.round(u_m[::-1], cfg.decimals).tolist(),
                "v": np.round(v_m[::-1], cfg.decimals).tolist(),
            }
    return frames


def _build_wave_frames_provider(cfg, result, grid):
    """Build wave decomposition frames from provider data."""
    frames = {}
    for fh, wd in sorted(result.items()):
        frame = {}
        if wd.values is not None:
            frame["data"] = np.round(
                wd.values[:: grid.step, :: grid.step], cfg.decimals
            ).tolist()
        if wd.wave_direction is not None:
            frame["direction"] = np.round(
                wd.wave_direction[:: grid.step, :: grid.step], cfg.decimals
            ).tolist()
        has_decomp = wd.windwave_height is not None and wd.swell_height is not None
        if has_decomp:
            frame["windwave"] = {
                "height": (
                    np.round(
                        wd.windwave_height[:: grid.step, :: grid.step], cfg.decimals
                    ).tolist()
                    if wd.windwave_height is not None
                    else None
                ),
                "period": (
                    np.round(
                        wd.windwave_period[:: grid.step, :: grid.step], cfg.decimals
                    ).tolist()
                    if wd.windwave_period is not None
                    else None
                ),
                "direction": (
                    np.round(
                        wd.windwave_direction[:: grid.step, :: grid.step], cfg.decimals
                    ).tolist()
                    if wd.windwave_direction is not None
                    else None
                ),
            }
            frame["swell"] = {
                "height": (
                    np.round(
                        wd.swell_height[:: grid.step, :: grid.step], cfg.decimals
                    ).tolist()
                    if wd.swell_height is not None
                    else None
                ),
                "period": (
                    np.round(
                        wd.swell_period[:: grid.step, :: grid.step], cfg.decimals
                    ).tolist()
                    if wd.swell_period is not None
                    else None
                ),
                "direction": (
                    np.round(
                        wd.swell_direction[:: grid.step, :: grid.step], cfg.decimals
                    ).tolist()
                    if wd.swell_direction is not None
                    else None
                ),
            }
        frames[str(fh)] = frame
    return frames


def _build_scalar_frames_provider(cfg, field_name, result, grid, ice_ocean_arr):
    """Build scalar frames from provider data."""
    frames = {}
    for fh, wd in sorted(result.items()):
        vals = _pick_scalar_values(wd, field_name)
        if vals is not None:
            clean = np.nan_to_num(vals[:: grid.step, :: grid.step], nan=cfg.nan_fill)
            if ice_ocean_arr is not None and clean.shape == ice_ocean_arr.shape:
                clean = np.where(ice_ocean_arr, clean, cfg.nan_fill)
            frames[str(fh)] = {"data": np.round(clean, cfg.decimals).tolist()}
    return frames


def _pick_scalar_values(wd, field_name):
    """Pick the scalar data array from a WeatherData object."""
    if field_name == "ice" and wd.ice_concentration is not None:
        return wd.ice_concentration
    if field_name == "visibility" and wd.visibility is not None:
        return wd.visibility
    if field_name == "sst" and wd.sst is not None:
        return wd.sst
    val = getattr(wd, field_name, None)
    if val is not None:
        return val
    return wd.values


# ---------------------------------------------------------------------------
# Build wind frames from GFS GRIB cache
# ---------------------------------------------------------------------------


def build_wind_frames_from_grib(
    gfs_provider,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    run_date: str,
    run_hour: str,
):
    """Process all cached GRIB files into leaflet-velocity frame cache.

    Returns a cache envelope dict.
    """
    from datetime import datetime
    from src.data.copernicus import GFSDataProvider

    run_time = datetime.strptime(f"{run_date}{run_hour}", "%Y%m%d%H")
    hours_status = gfs_provider.get_cached_forecast_hours(
        lat_min,
        lat_max,
        lon_min,
        lon_max,
        run_date,
        run_hour,
    )

    cfg = get_field("wind")
    frames = {}
    grid = None

    for h_info in hours_status:
        if not h_info["cached"]:
            continue
        fh = h_info["forecast_hour"]
        wind_data = gfs_provider.fetch_wind_data(
            lat_min,
            lat_max,
            lon_min,
            lon_max,
            forecast_hour=fh,
            run_date=run_date,
            run_hour=run_hour,
        )
        if wind_data is None:
            continue

        # Build grid once from first frame
        if grid is None:
            grid = make_grid(wind_data.lats, wind_data.lons, cfg)

        u_sub = wind_data.u_component[:: grid.step, :: grid.step]
        v_sub = wind_data.v_component[:: grid.step, :: grid.step]
        u_m, v_m = mask_velocity_with_nan(u_sub, v_sub, grid)
        u_ord, v_ord = _order_north_to_south(u_m, v_m, grid.lats)

        header = _velocity_header(grid.lats, grid.lons, run_time, fh)
        frames[str(fh)] = [
            {
                "header": {**header, "parameterNumber": 2},
                "data": u_ord.flatten().tolist(),
            },
            {
                "header": {**header, "parameterNumber": 3},
                "data": v_ord.flatten().tolist(),
            },
        ]

    return {
        "_schema_version": CACHE_SCHEMA_VERSION,
        "run_date": run_date,
        "run_hour": run_hour,
        "run_time": run_time.isoformat(),
        "total_hours": len(GFSDataProvider.FORECAST_HOURS),
        "cached_hours": len(frames),
        "source": "gfs",
        "field": "wind",
        "frames": frames,
    }


# ---------------------------------------------------------------------------
# Cache envelope assembly
# ---------------------------------------------------------------------------


def _assemble_envelope(
    cfg: FieldConfig,
    field_name: str,
    grid: SubsampledGrid,
    frames: dict,
    run_time,
    *,
    ocean_mask_data=None,
):
    """Assemble the cache envelope dict from computed frames."""
    if not frames:
        return None

    envelope = {
        "_schema_version": CACHE_SCHEMA_VERSION,
        "run_time": run_time.isoformat() if run_time else "",
        "total_hours": len(frames),
        "cached_hours": len(frames),
        "source": cfg.source.split("_")[0],
        "field": field_name,
        "frames": frames,
    }

    # Wind uses leaflet-velocity format — different envelope
    if field_name == "wind":
        from src.data.copernicus import GFSDataProvider

        run_date_str = run_time.strftime("%Y%m%d") if run_time else ""
        run_hour_str = run_time.strftime("%H") if run_time else "00"
        envelope["run_date"] = run_date_str
        envelope["run_hour"] = run_hour_str
        envelope["total_hours"] = len(GFSDataProvider.FORECAST_HOURS)
    else:
        envelope["lats"] = grid.lats.tolist()
        envelope["lons"] = grid.lons.tolist()
        envelope["ny"] = grid.ny
        envelope["nx"] = grid.nx

        if ocean_mask_data is not None:
            envelope["ocean_mask"] = ocean_mask_data
            envelope["ocean_mask_lats"] = grid.lats.tolist()
            envelope["ocean_mask_lons"] = grid.lons.tolist()

        if cfg.colorscale_colors:
            cs = {
                "min": cfg.colorscale_min,
                "max": cfg.colorscale_max,
                "colors": list(cfg.colorscale_colors),
            }
            if field_name == "sst":
                all_vals = []
                for f in frames.values():
                    if isinstance(f, dict) and "data" in f:
                        flat = [v for row in f["data"] for v in row if v > -100]
                        all_vals.extend(flat)
                if all_vals:
                    cs["data_min"] = round(min(all_vals), 2)
                    cs["data_max"] = round(max(all_vals), 2)
            envelope["colorscale"] = cs

    logger.info(f"{field_name} cache built: {len(frames)} frames")
    return envelope

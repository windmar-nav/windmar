"""
Single-frame response formatters for weather overlay endpoints.

These format the response for ``GET /api/weather/{field}`` — a single
forecast snapshot for canvas-based overlay rendering.
"""

import numpy as np

from api.weather_fields import FieldConfig
from api.weather.grid_processor import SubsampledGrid, make_grid, subsample_2d


def format_single_frame(
    field_name: str,
    cfg: FieldConfig,
    data,
    time,
    ocean_mask_fn,
    ingested_at=None,
):
    """Build a single-frame overlay response dict.

    Parameters
    ----------
    field_name : str
        e.g. "wind", "waves", "sst"
    cfg : FieldConfig
        Field configuration.
    data : WeatherData
        Single forecast snapshot from DB or provider.
    time : datetime
        Request time.
    ocean_mask_fn : callable
        ``fn(grid) -> (mask_lats, mask_lons, mask_2d)`` for building ocean mask.
    ingested_at : datetime | None
        When this data was ingested into the DB.

    Returns
    -------
    dict
        API response.
    """
    grid = make_grid(data.lats, data.lons, cfg)

    # Compute grid resolution from subsampled lats
    if grid.ny > 1:
        resolution = round(abs(float(grid.lats[1] - grid.lats[0])), 4)
    elif grid.nx > 1:
        resolution = round(abs(float(grid.lons[1] - grid.lons[0])), 4)
    else:
        resolution = 0.25

    response = {
        "parameter": cfg.parameters[0] if cfg.components == "scalar" else cfg.name,
        "field": field_name,
        "time": time.isoformat(),
        "bbox": {
            "lat_min": float(data.lats.min()),
            "lat_max": float(data.lats.max()),
            "lon_min": float(data.lons.min()),
            "lon_max": float(data.lons.max()),
        },
        "resolution": resolution,
        "nx": grid.nx,
        "ny": grid.ny,
        "lats": grid.lats.tolist(),
        "lons": grid.lons.tolist(),
        "unit": cfg.unit,
        "source": cfg.source.split("_")[0],
    }

    if cfg.needs_ocean_mask:
        mask_lats, mask_lons, ocean_mask = ocean_mask_fn(grid)
        response["ocean_mask"] = ocean_mask
        response["ocean_mask_lats"] = mask_lats
        response["ocean_mask_lons"] = mask_lons

    if cfg.colorscale_colors:
        response["colorscale"] = {
            "min": cfg.colorscale_min,
            "max": cfg.colorscale_max,
            "colors": list(cfg.colorscale_colors),
        }

    if cfg.components == "vector":
        response["u"] = (
            subsample_2d(data.u_component, grid.step)
            if data.u_component is not None
            else []
        )
        response["v"] = (
            subsample_2d(data.v_component, grid.step)
            if data.v_component is not None
            else []
        )

    elif cfg.components == "wave_decomp":
        response["data"] = subsample_2d(data.values, grid.step, cfg.decimals)
        if data.wave_direction is not None:
            response["direction"] = subsample_2d(data.wave_direction, grid.step, 1)

        has_decomp = data.windwave_height is not None and data.swell_height is not None
        response["has_decomposition"] = has_decomp

        if has_decomp:
            response["windwave"] = {
                "height": subsample_2d(data.windwave_height, grid.step, cfg.decimals),
                "period": subsample_2d(data.windwave_period, grid.step, 1),
                "direction": subsample_2d(data.windwave_direction, grid.step, 1),
            }
            response["swell"] = {
                "height": subsample_2d(data.swell_height, grid.step, cfg.decimals),
                "period": subsample_2d(data.swell_period, grid.step, 1),
                "direction": subsample_2d(data.swell_direction, grid.step, 1),
            }

        if field_name == "swell":
            response["total_hs"] = subsample_2d(data.values, grid.step, cfg.decimals)
            response["swell_hs"] = subsample_2d(
                data.swell_height, grid.step, cfg.decimals
            )
            response["swell_tp"] = subsample_2d(data.swell_period, grid.step, 1)
            response["swell_dir"] = subsample_2d(data.swell_direction, grid.step, 1)
            response["windsea_hs"] = subsample_2d(
                data.windwave_height, grid.step, cfg.decimals
            )
            response["windsea_tp"] = subsample_2d(data.windwave_period, grid.step, 1)
            response["windsea_dir"] = subsample_2d(
                data.windwave_direction, grid.step, 1
            )

    elif cfg.components == "scalar":
        clean = np.nan_to_num(data.values[:: grid.step, :: grid.step], nan=cfg.nan_fill)
        response["data"] = np.round(clean, cfg.decimals).tolist()

        if field_name == "ice" and data.ice_concentration is not None:
            clean_ice = np.nan_to_num(
                data.ice_concentration[:: grid.step, :: grid.step], nan=-999.0
            )
            if "ocean_mask" in response:
                ocean_arr = np.array(response["ocean_mask"], dtype=bool)
                if clean_ice.shape == ocean_arr.shape:
                    clean_ice = np.where(ocean_arr, clean_ice, -999.0)
            response["data"] = np.round(clean_ice, cfg.decimals).tolist()

    if ingested_at is not None:
        response["ingested_at"] = ingested_at.isoformat()

    return response


def format_velocity_response(
    data,
    lats,
    lons,
    time,
    step: int,
    mask_fn,
):
    """Build leaflet-velocity response for wind or currents.

    Parameters
    ----------
    data : WeatherData
        Single snapshot with u_component / v_component.
    lats, lons : ndarray
        Already subsampled coordinate arrays.
    time : datetime
        Reference time.
    step : int
        Subsample step (for applying to u/v if not already subsampled).
    mask_fn : callable
        ``fn(u, v) -> (u_masked, v_masked)``
    """
    from datetime import datetime

    u_sub = data.u_component[::step, ::step] if step > 1 else data.u_component
    v_sub = data.v_component[::step, ::step] if step > 1 else data.v_component
    u_m, v_m = mask_fn(u_sub, v_sub)

    actual_lats = data.lats[::step] if step > 1 else data.lats
    actual_lons = data.lons[::step] if step > 1 else data.lons

    dx = abs(float(actual_lons[1] - actual_lons[0])) if len(actual_lons) > 1 else 0.25
    dy = abs(float(actual_lats[1] - actual_lats[0])) if len(actual_lats) > 1 else 0.25

    if len(actual_lats) > 1 and actual_lats[1] > actual_lats[0]:
        u_ordered = u_m[::-1]
        v_ordered = v_m[::-1]
        lat_north = float(actual_lats[-1])
        lat_south = float(actual_lats[0])
    else:
        u_ordered = u_m
        v_ordered = v_m
        lat_north = float(actual_lats[0])
        lat_south = float(actual_lats[-1])

    ref_time = (
        data.time.isoformat() if isinstance(data.time, datetime) else time.isoformat()
    )
    header = {
        "parameterCategory": 2,
        "parameterNumber": 2,
        "lo1": float(actual_lons[0]),
        "la1": lat_north,
        "lo2": float(actual_lons[-1]),
        "la2": lat_south,
        "dx": dx,
        "dy": dy,
        "nx": len(actual_lons),
        "ny": len(actual_lats),
        "refTime": ref_time,
    }

    u_flat = np.nan_to_num(u_ordered.flatten(), nan=0.0, posinf=0.0, neginf=0.0)
    v_flat = np.nan_to_num(v_ordered.flatten(), nan=0.0, posinf=0.0, neginf=0.0)
    return [
        {"header": {**header, "parameterNumber": 2}, "data": u_flat.tolist()},
        {"header": {**header, "parameterNumber": 3}, "data": v_flat.tolist()},
    ]

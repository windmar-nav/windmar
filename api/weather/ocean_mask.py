"""
Ocean mask utilities for weather layers.

Strategy:
- **All fields except ice**: derive the ocean mask from CMEMS/GFS data itself.
  A cell is "ocean" if ANY forecast frame has a finite (non-NaN) value at that
  location.  This is more accurate than ``global_land_mask`` at coastlines
  where the library incorrectly marks valid CMEMS ocean cells as land.
- **Ice only**: CMEMS ice products have valid (non-NaN) values over land, so
  NaN-based detection does not work.  Ice falls back to ``global_land_mask``.

The mask is always built at the SAME subsampled step as the data grid,
preventing the shape-mismatch bugs that plagued the old monolithic router.
"""

import logging

import numpy as np

from api.weather.grid_processor import SubsampledGrid

logger = logging.getLogger(__name__)


def build_ocean_mask_from_data(
    grids: dict,
    primary_param: str,
    hours: list[int],
    grid: SubsampledGrid,
    full_lats: np.ndarray,
    full_lons: np.ndarray,
) -> list[list[bool]]:
    """Build ocean mask by OR-ing ``np.isfinite(data)`` across all forecast hours.

    The union is computed on the FULL-resolution grid, then subsampled at
    ``grid.step`` — guaranteeing the mask shape matches ``(grid.ny, grid.nx)``.

    Parameters
    ----------
    grids : dict
        ``{param: {fh: (lats, lons, data_2d)}}`` as returned by
        ``DbWeatherProvider.get_grids_for_timeline``.
    primary_param : str
        The parameter name used to detect finite values (e.g. ``"wave_hs"``).
    hours : list[int]
        Forecast hours to scan.
    grid : SubsampledGrid
        Target geometry (step, ny, nx).
    full_lats, full_lons : np.ndarray
        Full-resolution coordinate arrays (before subsampling).

    Returns
    -------
    list[list[bool]]
        2-D boolean mask with shape ``(grid.ny, grid.nx)``.
    """
    ocean = np.zeros((len(full_lats), len(full_lons)), dtype=bool)
    for fh in hours:
        if primary_param in grids and fh in grids[primary_param]:
            _, _, d = grids[primary_param][fh]
            ocean |= np.isfinite(d)
    return ocean[::grid.step, ::grid.step].tolist()


def build_ocean_mask_from_weather_data(
    result: dict,
    cfg,
    grid: SubsampledGrid,
    full_lats: np.ndarray,
    full_lons: np.ndarray,
) -> list[list[bool]]:
    """Build ocean mask from provider ``WeatherData`` objects.

    Same NaN-union logic as ``build_ocean_mask_from_data`` but works with
    the ``{fh: WeatherData}`` dict returned by provider fetch methods.

    Parameters
    ----------
    result : dict
        ``{forecast_hour: WeatherData}`` from a provider.
    cfg : FieldConfig
        Field configuration (used to pick the right data attribute).
    grid : SubsampledGrid
        Target geometry.
    full_lats, full_lons : np.ndarray
        Full-resolution coordinate arrays.
    """
    ocean = np.zeros((len(full_lats), len(full_lons)), dtype=bool)
    for _fh, wd in result.items():
        raw = _pick_raw_data(wd, cfg)
        if raw is not None:
            ocean |= np.isfinite(raw)
    return ocean[::grid.step, ::grid.step].tolist()


def build_ice_ocean_mask(grid: SubsampledGrid) -> tuple[list[list[bool]], np.ndarray]:
    """Build ocean mask for ice using ``global_land_mask``.

    Returns both the serializable list and the numpy bool array (for masking
    ice frame data in the frame builder).
    """
    lon_grid, lat_grid = np.meshgrid(grid.lons, grid.lats)
    try:
        from global_land_mask import globe
        mask = globe.is_ocean(lat_grid, lon_grid)
    except ImportError:
        from src.data.land_mask import is_ocean
        mask = np.array([
            [is_ocean(round(float(lat), 2), round(float(lon), 2)) for lon in grid.lons]
            for lat in grid.lats
        ])
    return mask.tolist(), mask


def mask_velocity_with_nan(
    u: np.ndarray,
    v: np.ndarray,
    grid: SubsampledGrid,
    grids: dict | None = None,
    primary_param: str | None = None,
    hours: list[int] | None = None,
    full_lats: np.ndarray | None = None,
    full_lons: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero out U/V components over land for velocity animations.

    Uses NaN-derived ocean mask from the data itself with 1-cell erosion so
    coastal ocean cells adjacent to land are also zeroed — prevents particles
    from drifting over land during leaflet-velocity animation.

    If no ``grids`` dict is provided, falls back to NaN detection on u/v
    directly (already subsampled).
    """
    if grids is not None and primary_param and hours and full_lats is not None:
        ocean_full = np.zeros((len(full_lats), len(full_lons)), dtype=bool)
        for fh in hours:
            if primary_param in grids and fh in grids[primary_param]:
                _, _, d = grids[primary_param][fh]
                ocean_full |= np.isfinite(d)
        ocean = ocean_full[::grid.step, ::grid.step]
    else:
        # Fallback: derive from the u/v arrays themselves
        ocean = np.isfinite(u) & np.isfinite(v)

    # 1-cell erosion: ocean cell must have all 4-connected neighbors also ocean
    eroded = ocean.copy()
    eroded[:-1, :] &= ocean[1:, :]
    eroded[1:, :] &= ocean[:-1, :]
    eroded[:, :-1] &= ocean[:, 1:]
    eroded[:, 1:] &= ocean[:, :-1]

    u_masked = np.where(eroded, u, 0.0)
    v_masked = np.where(eroded, v, 0.0)
    return u_masked, v_masked


def _pick_raw_data(wd, cfg) -> np.ndarray | None:
    """Pick the primary data array from a WeatherData object for NaN detection."""
    if cfg.components == "vector":
        return wd.u_component
    if cfg.components == "wave_decomp":
        return wd.values
    # scalar — try field-specific attribute first
    for attr in (cfg.name, "values"):
        raw = getattr(wd, attr, None)
        if raw is not None:
            return raw
    # Explicit fallbacks for specific fields
    if cfg.name == "visibility" and wd.visibility is not None:
        return wd.visibility
    if cfg.name == "sst" and wd.sst is not None:
        return wd.sst
    return None

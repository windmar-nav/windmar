"""
Grid processing utilities for weather data.

Guarantees that data grids and ocean masks always share the same
subsampled coordinate arrays.  Every field invocation produces exactly
ONE ``SubsampledGrid`` and all downstream consumers (frame builder,
formatters, ocean mask) use the same step — eliminating shape mismatches.
"""

import math
from dataclasses import dataclass

import numpy as np

from api.weather_fields import FieldConfig


@dataclass(frozen=True)
class SubsampledGrid:
    """Immutable grid geometry after subsampling.

    Guarantees ``ny == len(lats)`` and ``nx == len(lons)`` — any consumer
    that receives this object can trust the dimensions are consistent.
    """

    lats: np.ndarray
    lons: np.ndarray
    step: int
    ny: int  # = len(lats)
    nx: int  # = len(lons)

    def __post_init__(self):
        if len(self.lats) != self.ny or len(self.lons) != self.nx:
            raise ValueError(
                f"Grid dimension mismatch: lats={len(self.lats)} vs ny={self.ny}, "
                f"lons={len(self.lons)} vs nx={self.nx}"
            )


def compute_step(lats: np.ndarray, lons: np.ndarray, target: int) -> int:
    """Compute subsample step to keep the largest axis under *target* points."""
    max_dim = max(len(lats), len(lons))
    return max(1, math.ceil(max_dim / target))


def make_grid(lats: np.ndarray, lons: np.ndarray, cfg: FieldConfig) -> SubsampledGrid:
    """Build a ``SubsampledGrid`` for *cfg* using its ``subsample_target``.

    This is the SINGLE entry point that all frame builders must use to
    obtain grid geometry — guaranteeing one step per field.
    """
    step = compute_step(lats, lons, cfg.subsample_target)
    sub_lats = lats[::step]
    sub_lons = lons[::step]
    return SubsampledGrid(
        lats=sub_lats,
        lons=sub_lons,
        step=step,
        ny=len(sub_lats),
        nx=len(sub_lons),
    )


def subsample_2d(
    arr: np.ndarray | None,
    step: int,
    decimals: int = 2,
    nan_fill: float = 0.0,
) -> list | None:
    """Subsample a 2-D array, sanitize NaN/Inf, round, and return nested list.

    Returns ``None`` if *arr* is ``None``.
    """
    if arr is None:
        return None
    sub = arr[::step, ::step]
    clean = np.where(np.isfinite(sub), sub, nan_fill)
    return np.round(clean, decimals).tolist()


def subsample_2d_raw(
    arr: np.ndarray | None,
    step: int,
) -> np.ndarray | None:
    """Subsample a 2-D array without sanitizing or converting to list.

    Used when further processing (masking, velocity zeroing) is needed
    before the final ``.tolist()`` call.
    """
    if arr is None:
        return None
    return arr[::step, ::step]


def clamp_bbox(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    max_lat_span: float = 180.0,
    max_lon_span: float = 360.0,
) -> tuple[float, float, float, float]:
    """Clamp a bounding box to the given maximum spans, centered on the midpoint."""
    lat_span = lat_max - lat_min
    lon_span = lon_max - lon_min
    if lat_span > max_lat_span or lon_span > max_lon_span:
        lat_center = (lat_min + lat_max) / 2
        lon_center = (lon_min + lon_max) / 2
        lat_half = min(lat_span / 2, max_lat_span / 2)
        lon_half = min(lon_span / 2, max_lon_span / 2)
        lat_min = max(-89.9, lat_center - lat_half)
        lat_max = min(89.9, lat_center + lat_half)
        lon_min = max(-180.0, lon_center - lon_half)
        lon_max = min(180.0, lon_center + lon_half)
    return lat_min, lat_max, lon_min, lon_max

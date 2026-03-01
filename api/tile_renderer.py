"""
Raster tile renderer — converts DB weather grids to 256x256 PNG tiles.

Each tile is addressed by standard slippy-map coordinates (z/x/y) and a
forecast hour.  The renderer loads the full grid from the DB (via the
existing ForecastLayerManager file cache), crops to the tile bbox,
applies the same color ramps used by the frontend canvas painter, and
returns PNG bytes ready for HTTP streaming.

Colour ramps are ported 1:1 from frontend/components/WeatherGridLayer.tsx
so that tiles look identical to the legacy canvas path.
"""

import io
import logging
import math
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from api.weather_fields import WEATHER_FIELDS, FieldConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Slippy-map math
# ---------------------------------------------------------------------------

def _tile_bbox(z: int, x: int, y: int) -> Tuple[float, float, float, float]:
    """Return (lat_min, lat_max, lon_min, lon_max) for a tile."""
    n = 2.0 ** z
    lon_min = x / n * 360.0 - 180.0
    lon_max = (x + 1) / n * 360.0 - 180.0
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return lat_min, lat_max, lon_min, lon_max


def _mercator_pixel_lats(lat_min: float, lat_max: float, n: int = 256) -> np.ndarray:
    """Generate Mercator-projected latitude values for *n* pixel rows.

    In Web Mercator (EPSG:3857) tile pixels are evenly spaced in
    Mercator Y, not in geographic latitude.  Returns latitudes from
    *lat_max* (top row) to *lat_min* (bottom row).
    """
    lat_max_r = np.radians(np.clip(lat_max, -85, 85))
    lat_min_r = np.radians(np.clip(lat_min, -85, 85))
    merc_top = np.log(np.tan(np.pi / 4 + lat_max_r / 2))
    merc_bot = np.log(np.tan(np.pi / 4 + lat_min_r / 2))
    merc_y = np.linspace(merc_top, merc_bot, n)
    return np.degrees(2 * np.arctan(np.exp(merc_y)) - np.pi / 2)


# ---------------------------------------------------------------------------
# Land mask (per-tile, using global_land_mask)
# ---------------------------------------------------------------------------

try:
    from global_land_mask import globe as _globe
except ImportError:
    _globe = None


def _build_tile_land_mask(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float,
) -> Optional[np.ndarray]:
    """Build a 256x256 boolean land mask for the tile bbox.

    Returns True where land, False where ocean.  Returns None if
    global_land_mask is not available.
    """
    if _globe is None:
        return None
    pixel_lats = np.linspace(lat_max, lat_min, TILE_SIZE)   # top→bottom
    pixel_lons = np.linspace(lon_min, lon_max, TILE_SIZE)
    lon_grid, lat_grid = np.meshgrid(pixel_lons, pixel_lats)
    is_ocean = _globe.is_ocean(lat_grid, lon_grid)
    return ~is_ocean  # True = land


# ---------------------------------------------------------------------------
# Colour ramps  (ported from WeatherGridLayer.tsx)
# ---------------------------------------------------------------------------

# Each ramp is an Nx4 float32 array: [[threshold, R, G, B], ...]
# Alpha is applied per-field after ramp lookup.

_WIND_RAMP = np.array([
    [0,  30,  80, 220],
    [5,   0, 200, 220],
    [10,  0, 200,  50],
    [15, 240, 220,   0],
    [20, 240, 130,   0],
    [25, 220,  30,  30],
], dtype=np.float32)

_WAVE_RAMP = np.array([
    [0,    60, 110, 220],
    [0.5,  30, 160, 240],
    [1,     0, 200, 170],
    [1.5, 120, 220,  40],
    [2,   240, 220,   0],
    [3,   240, 130,   0],
    [4,   220,  30,  80],
    [6,   160,   0, 180],
], dtype=np.float32)

_ICE_RAMP = np.array([
    [0.00,   0, 100, 255],
    [0.10, 150, 200, 255],
    [0.30, 140, 255, 160],
    [0.60, 255, 255,   0],
    [0.80, 255, 125,   7],
    [1.00, 255,   0,   0],
], dtype=np.float32)

_SST_RAMP = np.array([
    [-2,  20,  30, 140],
    [ 2,  40,  80, 200],
    [ 8,   0, 180, 220],
    [14,   0, 200,  80],
    [20, 220, 220,   0],
    [26, 240, 130,   0],
    [32, 220,  30,  30],
], dtype=np.float32)

_SWELL_RAMP = np.array([
    [0,  60, 120, 200],
    [1,   0, 200, 180],
    [2, 100, 200,  50],
    [3, 240, 200,   0],
    [5, 240, 100,   0],
    [8, 200,  30,  30],
], dtype=np.float32)

_VIS_RAMP = np.array([
    [ 0,  20,  80,  10],
    [ 1,  40, 120,  20],
    [ 4,  80, 170,  40],
    [10, 130, 210,  70],
    [20, 180, 240, 120],
], dtype=np.float32)

_CURRENT_RAMP = np.array([
    [0.0,  34, 211, 238],   # cyan
    [0.5,  59, 130, 246],   # blue
    [1.0, 139,  92, 246],   # purple
    [2.0, 217,  70, 239],   # magenta
], dtype=np.float32)

FIELD_RAMP = {
    "wind":       (_WIND_RAMP,    180, 200, 180),
    "waves":      (_WAVE_RAMP,    170, 190, 175),
    "swell":      (_SWELL_RAMP,   140, 190, 160),
    "ice":        (_ICE_RAMP,     120, 200, 180),
    "sst":        (_SST_RAMP,     160, 180, 170),
    "visibility": (_VIS_RAMP,       0,   0,   0),   # alpha handled specially
    "currents":   (_CURRENT_RAMP,  170, 200, 175),
}


def _apply_ramp(values: np.ndarray, ramp: np.ndarray,
                alpha_low: int, alpha_high: int, alpha_default: int,
                ) -> np.ndarray:
    """Vectorised colour-ramp lookup.

    Parameters
    ----------
    values : 2D float array (H, W)
    ramp   : Nx4 array [[threshold, R, G, B], ...]

    Returns
    -------
    RGBA uint8 array (H, W, 4)
    """
    thresholds = ramp[:, 0]
    colors = ramp[:, 1:4]  # (N, 3)
    n_stops = len(thresholds)

    # Bucket each value into the correct segment
    # indices[i] = j means value[i] falls between stop j-1 and stop j
    indices = np.searchsorted(thresholds, values, side="right")  # (H, W)
    indices = np.clip(indices, 1, n_stops - 1)

    lo = indices - 1
    hi = indices

    t_lo = thresholds[lo]
    t_hi = thresholds[hi]
    span = t_hi - t_lo
    span = np.where(span == 0, 1.0, span)
    t = np.clip((values - t_lo) / span, 0.0, 1.0)

    c_lo = colors[lo]  # (H, W, 3)
    c_hi = colors[hi]
    rgb = c_lo + t[..., np.newaxis] * (c_hi - c_lo)

    # Alpha
    alpha = np.full(values.shape, alpha_default, dtype=np.float32)
    alpha = np.where(values <= thresholds[0], alpha_low, alpha)
    alpha = np.where(values >= thresholds[-1], alpha_high, alpha)

    rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
    rgb_safe = np.nan_to_num(rgb, nan=0.0)
    alpha_safe = np.nan_to_num(alpha, nan=0.0)
    rgba[..., :3] = np.clip(rgb_safe, 0, 255).astype(np.uint8)
    rgba[..., 3] = np.clip(alpha_safe, 0, 255).astype(np.uint8)
    return rgba


def _apply_visibility_ramp(values: np.ndarray) -> np.ndarray:
    """Visibility has special inverse-alpha logic: fog=opaque, clear=transparent."""
    ramp, _, _, _ = FIELD_RAMP["visibility"]
    thresholds = ramp[:, 0]
    colors = ramp[:, 1:4]
    n_stops = len(thresholds)

    indices = np.searchsorted(thresholds, values, side="right")
    indices = np.clip(indices, 1, n_stops - 1)
    lo = indices - 1
    hi = indices

    t_lo = thresholds[lo]
    t_hi = thresholds[hi]
    span = np.where((t_hi - t_lo) == 0, 1.0, t_hi - t_lo)
    t = np.clip((values - t_lo) / span, 0.0, 1.0)

    c_lo = colors[lo]
    c_hi = colors[hi]
    rgb = c_lo + t[..., np.newaxis] * (c_hi - c_lo)

    # Inverse alpha: 0 km → 220, 20 km → 0
    alpha = np.clip(220.0 * (1.0 - values / 20.0), 0, 255)
    # Transparent where vis >= 20 or invalid
    alpha = np.where((values < 0) | (values > 20), 0, alpha)

    rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    rgba[..., 3] = alpha.astype(np.uint8)
    return rgba


def _apply_ice_ramp(values: np.ndarray) -> np.ndarray:
    """Ice has a transparent cutoff below 1%."""
    ramp, alpha_low, alpha_high, alpha_default = FIELD_RAMP["ice"]
    rgba = _apply_ramp(values, ramp, alpha_low, alpha_high, alpha_default)
    rgba[values < 0.005, 3] = 0  # transparent below 0.5% (noise floor)
    return rgba


# ---------------------------------------------------------------------------
# Tile renderer
# ---------------------------------------------------------------------------

# Max zoom per source resolution
MAX_ZOOM = {
    "gfs": 8,            # 0.5 deg
    "gfs_visibility": 8,
    "cmems_wave": 10,    # 0.083 deg
    "cmems_current": 10,
    "cmems_sst": 10,
    "cmems_ice": 10,
}

TILE_SIZE = 256


def get_max_zoom(field_name: str) -> int:
    """Return the max useful zoom level for a field."""
    cfg = WEATHER_FIELDS.get(field_name)
    if cfg is None:
        return 8
    return MAX_ZOOM.get(cfg.source, 8)


def _normalize_velocity_cache(cache_data: dict) -> dict:
    """Convert leaflet-velocity format cache to standard grid format.

    Velocity caches (wind) store frames as:
        [{"header": {..., nx, ny, la1, la2, lo1, lo2, dx, dy}, "data": [flat_u]},
         {"header": {...}, "data": [flat_v]}]

    This normalizes to the standard format expected by render_tile:
        {"lats": [...], "lons": [...], "frames": {"0": {"wind_u": [[]], "wind_v": [[]]}}}
    """
    frames = cache_data.get("frames", {})
    if not frames:
        return cache_data

    # Detect velocity format: first frame is a list of dicts with "header"
    sample_key = next(iter(frames))
    sample = frames[sample_key]
    if not isinstance(sample, list) or len(sample) < 2:
        return cache_data
    if not isinstance(sample[0], dict) or "header" not in sample[0]:
        return cache_data

    # Extract grid info from first frame header
    hdr = sample[0]["header"]
    nx = hdr.get("nx", 0)
    ny = hdr.get("ny", 0)
    la1 = hdr.get("la1", 0)  # top latitude
    la2 = hdr.get("la2", 0)  # bottom latitude
    lo1 = hdr.get("lo1", 0)
    lo2 = hdr.get("lo2", 0)
    dx = hdr.get("dx", 0.5)
    dy = hdr.get("dy", 0.5)

    if nx < 2 or ny < 2:
        return cache_data

    # Build lat/lon arrays (la1 is top → descending; we'll flip in render_tile)
    lats = [la1 - j * dy for j in range(ny)]
    lons = [lo1 + i * dx for i in range(nx)]

    # Convert each frame
    new_frames = {}
    for fh_key, frame_list in frames.items():
        if not isinstance(frame_list, list) or len(frame_list) < 2:
            continue
        flat_u = frame_list[0].get("data", [])
        flat_v = frame_list[1].get("data", [])
        if len(flat_u) < nx * ny or len(flat_v) < nx * ny:
            continue
        u_2d = [flat_u[j * nx: (j + 1) * nx] for j in range(ny)]
        v_2d = [flat_v[j * nx: (j + 1) * nx] for j in range(ny)]
        new_frames[fh_key] = {"wind_u": u_2d, "wind_v": v_2d}

    result = dict(cache_data)
    result["lats"] = lats
    result["lons"] = lons
    result["ny"] = ny
    result["nx"] = nx
    result["frames"] = new_frames
    return result


def render_tile(
    field: str,
    z: int,
    x: int,
    y: int,
    forecast_hour: int = 0,
    cache_data: Optional[dict] = None,
) -> Optional[bytes]:
    """Render a single 256x256 PNG tile for the given field and tile coords.

    Parameters
    ----------
    field : weather field name (wind, waves, sst, ...)
    z, x, y : slippy-map tile coordinates
    forecast_hour : forecast hour offset (0, 3, 6, ...)
    cache_data : pre-loaded ForecastLayerManager cache dict. If None,
                 the caller must provide it.

    Returns
    -------
    PNG bytes, or None if no data covers this tile.
    """
    if cache_data is None:
        return None

    cfg = WEATHER_FIELDS.get(field)
    if cfg is None:
        return None

    # Normalize velocity-format caches (wind) to standard grid format
    cache_data = _normalize_velocity_cache(cache_data)

    lat_min, lat_max, lon_min, lon_max = _tile_bbox(z, x, y)

    # Extract grid coordinates
    lats = np.asarray(cache_data.get("lats", []), dtype=np.float64)
    lons = np.asarray(cache_data.get("lons", []), dtype=np.float64)
    if lats.size < 2 or lons.size < 2:
        return None

    # Ensure ascending latitude
    lat_ascending = lats[0] < lats[-1]
    if not lat_ascending:
        lats = lats[::-1]

    grid_lat_min, grid_lat_max = float(lats[0]), float(lats[-1])
    grid_lon_min, grid_lon_max = float(lons[0]), float(lons[-1])

    # Quick reject: tile completely outside grid coverage
    if (lat_max < grid_lat_min or lat_min > grid_lat_max or
            lon_max < grid_lon_min or lon_min > grid_lon_max):
        return None

    # Load the frame data
    frames = cache_data.get("frames", {})
    frame_key = str(forecast_hour)
    frame = frames.get(frame_key)
    if frame is None:
        # Try hour 0 as fallback
        frame = frames.get("0")
        if frame is None:
            return None

    values_2d = _extract_values(field, cfg, frame, lat_ascending)
    if values_2d is None:
        return None
    ny, nx = values_2d.shape
    if ny != len(lats) or nx != len(lons):
        return None
    pixel_values = _resample_to_tile(
        values_2d, lats, lons, lat_min, lat_max, lon_min, lon_max
    )
    if pixel_values is None:
        return None

    # Build land mask directly from global_land_mask (per-pixel, reliable)
    land_mask = None
    if cfg.needs_ocean_mask:
        land_mask = _build_tile_land_mask(lat_min, lat_max, lon_min, lon_max)

    # Apply colour ramp
    rgba = _colorize(field, pixel_values)

    # Mask NaN / sentinel values
    nan_mask = np.isnan(pixel_values) | (pixel_values < -100)
    rgba[nan_mask, 3] = 0

    # Mask land pixels
    if land_mask is not None:
        rgba[land_mask, 3] = 0

    # Edge fade: soften the boundary at grid coverage edges (5° ramp)
    fade_deg = 5.0
    pixel_lats = _mercator_pixel_lats(lat_min, lat_max, TILE_SIZE)
    pixel_lons = np.linspace(lon_min, lon_max, TILE_SIZE)
    lat_fade = np.clip(
        np.minimum(
            (pixel_lats - grid_lat_min) / fade_deg,
            (grid_lat_max - pixel_lats) / fade_deg,
        ),
        0, 1,
    )
    lon_fade = np.clip(
        np.minimum(
            (pixel_lons - grid_lon_min) / fade_deg,
            (grid_lon_max - pixel_lons) / fade_deg,
        ),
        0, 1,
    )
    edge_fade = np.outer(lat_fade, lon_fade)  # (256, 256)
    rgba[..., 3] = (rgba[..., 3].astype(np.float32) * edge_fade).astype(np.uint8)

    # Check if tile is fully transparent (all land / no data)
    if rgba[..., 3].max() == 0:
        return None

    # Encode to PNG
    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _extract_values(
    field: str, cfg: FieldConfig, frame: dict, lat_ascending: bool
) -> Optional[np.ndarray]:
    """Extract or compute the scalar value grid from a frame dict."""
    if cfg.components == "vector":
        # Wind / currents: compute magnitude from u/v
        u_key = cfg.parameters[0]  # e.g. "wind_u"
        v_key = cfg.parameters[1]  # e.g. "wind_v"
        u = frame.get(u_key)
        v = frame.get(v_key)
        if u is None or v is None:
            # Try generic keys
            u = frame.get("u") or frame.get("data")
            v = frame.get("v")
            if u is None or v is None:
                return None
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        values = np.sqrt(u ** 2 + v ** 2)
    elif cfg.components == "wave_decomp":
        # Waves / swell: primary height field
        # Cache format stores swell either as flat keys (swell_hs) or nested
        # dicts (swell: {height: [[...]]}).  Handle both.
        if field == "swell":
            data = frame.get("swell_hs")
            if data is None:
                swell_obj = frame.get("swell")
                if isinstance(swell_obj, dict):
                    data = swell_obj.get("height")
            if data is None:
                data = frame.get("data")
        else:
            data = frame.get("data") or frame.get("wave_hs")
        if data is None:
            return None
        values = np.asarray(data, dtype=np.float64)
    else:
        # Scalar: data key
        data = frame.get("data")
        if data is None:
            # Try parameter name
            data = frame.get(cfg.parameters[0])
        if data is None:
            return None
        values = np.asarray(data, dtype=np.float64)

    if not lat_ascending:
        values = values[::-1]

    return values


def _resample_to_tile(
    grid: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> Optional[np.ndarray]:
    """Resample a grid to 256x256 pixels covering the tile bbox.

    Uses NumPy vectorised nearest-neighbour + linear interpolation.
    """
    ny, nx = grid.shape
    grid_lat_min, grid_lat_max = float(lats[0]), float(lats[-1])
    grid_lon_min, grid_lon_max = float(lons[0]), float(lons[-1])

    # Pixel centres in geographic coordinates
    # Y axis: top of tile = lat_max, bottom = lat_min
    # Use Mercator-projected latitudes — tile pixels are evenly spaced
    # in Mercator Y, not in geographic latitude.
    pixel_lats = _mercator_pixel_lats(lat_min, lat_max, TILE_SIZE)
    pixel_lons = np.linspace(lon_min, lon_max, TILE_SIZE)

    # Map pixel coords to fractional grid indices
    lat_span = grid_lat_max - grid_lat_min
    lon_span = grid_lon_max - grid_lon_min
    if lat_span == 0 or lon_span == 0:
        return None

    lat_frac = (pixel_lats - grid_lat_min) / lat_span * (ny - 1)
    lon_frac = (pixel_lons - grid_lon_min) / lon_span * (nx - 1)

    # Create 2D coordinate grids
    lat_idx_2d, lon_idx_2d = np.meshgrid(lat_frac, lon_frac, indexing="ij")

    # Bilinear interpolation
    lat_lo = np.floor(lat_idx_2d).astype(np.intp)
    lon_lo = np.floor(lon_idx_2d).astype(np.intp)

    # Clamp to valid range
    lat_lo = np.clip(lat_lo, 0, ny - 2)
    lon_lo = np.clip(lon_lo, 0, nx - 2)
    lat_hi = lat_lo + 1
    lon_hi = lon_lo + 1

    lat_t = np.clip(lat_idx_2d - lat_lo, 0, 1).astype(np.float32)
    lon_t = np.clip(lon_idx_2d - lon_lo, 0, 1).astype(np.float32)

    # Four corners
    v00 = grid[lat_lo, lon_lo]
    v01 = grid[lat_lo, lon_hi]
    v10 = grid[lat_hi, lon_lo]
    v11 = grid[lat_hi, lon_hi]

    # Interpolate
    result = (
        v00 * (1 - lat_t) * (1 - lon_t)
        + v01 * (1 - lat_t) * lon_t
        + v10 * lat_t * (1 - lon_t)
        + v11 * lat_t * lon_t
    )

    # Mask pixels outside grid coverage
    outside = (
        (lat_idx_2d < -0.5) | (lat_idx_2d > ny - 0.5)
        | (lon_idx_2d < -0.5) | (lon_idx_2d > nx - 0.5)
    )
    result[outside] = np.nan

    return result.astype(np.float64)


def _colorize(field: str, values: np.ndarray) -> np.ndarray:
    """Apply the correct colour ramp for the field."""
    if field == "visibility":
        return _apply_visibility_ramp(values)
    if field == "ice":
        return _apply_ice_ramp(values)

    entry = FIELD_RAMP.get(field)
    if entry is None:
        # Fallback: grey
        rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
        rgba[..., :3] = 128
        rgba[..., 3] = 150
        return rgba

    ramp, alpha_low, alpha_high, alpha_default = entry
    return _apply_ramp(values, ramp, alpha_low, alpha_high, alpha_default)

"""
Raster tile endpoint — serves pre-rendered 256×256 PNG weather tiles.

    GET /api/tiles/{field}/{z}/{x}/{y}.png?h={forecast_hour}

Tiles are rendered on-demand from the existing ForecastLayerManager JSON
cache and written to a disk cache under /tmp/windmar_tiles/ for subsequent
instant serving.

This is a parallel code path alongside the JSON+canvas pipeline.  The
existing ``/api/weather/`` endpoints remain fully operational as fallback.
"""

import hashlib
import logging
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query, Response
from fastapi.responses import FileResponse

from api.tile_renderer import render_tile, get_max_zoom
from api.weather_fields import FIELD_NAMES, WEATHER_FIELDS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tiles", tags=["Tiles"])

_TILE_CACHE_ROOT = Path("/tmp/windmar_tiles")
_TILE_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

# Map field name → ForecastLayerManager cache subdir
_CACHE_ROOT = Path("/tmp/windmar_cache")

# Transparent 1x1 PNG (returned for 204-equivalent — avoids browser 404 noise)
_EMPTY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
    b"\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _tile_cache_path(field: str, h: int, z: int, x: int, y: int) -> Path:
    """Deterministic disk path for a cached tile PNG."""
    return _TILE_CACHE_ROOT / field / str(h) / str(z) / str(x) / f"{y}.png"


def _find_cache_data(field: str) -> Optional[dict]:
    """Load the forecast cache file with the best coverage.

    Prefers the file with the most data (largest file size) because
    filename bbox reflects the *requested* region, not the actual data
    coverage — global requests produce small low-resolution files while
    regional requests contain full-resolution grids.
    """
    import json
    import re

    cfg = WEATHER_FIELDS.get(field)
    if cfg is None:
        return None

    cache_subdir = cfg.cache_subdir or cfg.name
    cache_dir = _CACHE_ROOT / cache_subdir
    if not cache_dir.exists():
        return None

    # Collect all candidate cache files
    patterns = [f"{cfg.name}_*.json"]
    if field == "swell":
        patterns.append("waves_*.json")

    candidates: list[Path] = []
    for pat in patterns:
        candidates.extend(cache_dir.glob(pat))

    if not candidates:
        return None

    # Prefer broadest geographic coverage for tile rendering.
    # A global cache (17 MB, 360° lon span) is better than a regional
    # cache (19 MB, 220° lon span) — tiles outside the regional bbox
    # would render as empty.  Sort by bbox area then file size as tie-breaker.
    bbox_pat = re.compile(r"_(-?\d+)_(-?\d+)_(-?\d+)_(-?\d+)\.json$")

    def _bbox_area(fp: Path) -> float:
        m = bbox_pat.search(fp.name)
        if not m:
            return 0.0
        c_lat_min, c_lat_max, c_lon_min, c_lon_max = (
            float(m.group(i)) for i in range(1, 5)
        )
        return (c_lat_max - c_lat_min) * (c_lon_max - c_lon_min)

    candidates.sort(
        key=lambda f: (_bbox_area(f), f.stat().st_size), reverse=True,
    )
    best_file = candidates[0] if candidates else None

    if best_file is None:
        return None

    try:
        data = json.loads(best_file.read_text())
    except Exception as exc:
        logger.warning("Failed to load tile cache %s: %s", best_file, exc)
        return None

    # Sanity check: velocity-format wind needs at least one valid frame
    # with reasonable dimensions to avoid using broken cache files.
    frames = data.get("frames", {})
    if frames:
        sample = next(iter(frames.values()))
        if isinstance(sample, list) and len(sample) >= 2:
            hdr = sample[0].get("header", {})
            if hdr.get("nx", 0) < 2 or hdr.get("ny", 0) < 2:
                logger.warning("Rejecting broken wind cache %s (nx=%s)", best_file, hdr.get("nx"))
                # Try next broadest-coverage file
                candidates.remove(best_file)
                candidates.sort(
                    key=lambda f: (_bbox_area(f), f.stat().st_size),
                    reverse=True,
                )
                for f in candidates:
                    try:
                        data = json.loads(f.read_text())
                        s = next(iter(data.get("frames", {}).values()), None)
                        if isinstance(s, list) and len(s) >= 2:
                            h = s[0].get("header", {})
                            if h.get("nx", 0) >= 2 and h.get("ny", 0) >= 2:
                                return data
                        elif isinstance(s, dict):
                            return data
                    except Exception:
                        continue
                return None

    return data


def _cache_fingerprint(cache_data: dict) -> str:
    """Short fingerprint of the cache for cache-busting tile files."""
    run_time = cache_data.get("run_time", "")
    n_frames = len(cache_data.get("frames", {}))
    raw = f"{run_time}:{n_frames}"
    return hashlib.md5(raw.encode()).hexdigest()[:8]


@router.get("/{field}/{z}/{x}/{y}.png")
async def get_tile(
    field: str,
    z: float,
    x: int,
    y: int,
    h: int = Query(0, description="Forecast hour offset"),
):
    """Serve a raster weather tile as PNG.

    Returns 200 with PNG on hit, or a transparent 1×1 PNG if the tile
    is outside grid coverage (land / no data).
    """
    # Round fractional zoom from maps with zoomSnap < 1
    z = int(round(z))

    # Validate field
    if field not in FIELD_NAMES:
        return Response(content=_EMPTY_PNG, media_type="image/png", status_code=200)

    # Validate zoom
    max_z = get_max_zoom(field)
    if z < 1 or z > max_z:
        return Response(content=_EMPTY_PNG, media_type="image/png", status_code=200)

    # Validate tile coords
    max_tiles = 2 ** z
    if x < 0 or x >= max_tiles or y < 0 or y >= max_tiles:
        return Response(content=_EMPTY_PNG, media_type="image/png", status_code=200)

    # Check disk cache first
    tile_path = _tile_cache_path(field, h, z, x, y)
    if tile_path.exists():
        return FileResponse(
            tile_path,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=1800"},
        )

    # Load forecast data from ForecastLayerManager cache
    cache_data = _find_cache_data(field)
    if cache_data is None:
        return Response(content=_EMPTY_PNG, media_type="image/png", status_code=200)

    # Render
    png_bytes = render_tile(field, z, x, y, forecast_hour=h, cache_data=cache_data)
    if png_bytes is None:
        return Response(content=_EMPTY_PNG, media_type="image/png", status_code=200)

    # Write to disk cache
    tile_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp = tile_path.with_suffix(".tmp")
        tmp.write_bytes(png_bytes)
        tmp.rename(tile_path)
    except Exception as exc:
        logger.warning("Failed to write tile cache %s: %s", tile_path, exc)

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=1800"},
    )


@router.delete("/cache", tags=["Tiles"])
async def clear_tile_cache(field: Optional[str] = Query(None)):
    """Clear the tile cache.  Optionally restricted to a single field."""
    if field:
        target = _TILE_CACHE_ROOT / field
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
            return {"cleared": field}
        return {"cleared": None}

    for child in _TILE_CACHE_ROOT.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
    return {"cleared": "all"}

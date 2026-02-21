"""
Land mask for maritime route optimization.

Provides is_ocean(lat, lon) function to check if a point is navigable water.

Primary method: GSHHS vector polygons via shapely (sub-km coastal accuracy).
Fallback chain:
1. GSHHS + shapely.prepared (5 µs point-in-polygon)
2. global-land-mask package (1km raster)
3. Simplified bounding box heuristics
"""

import logging
import math
import threading
from functools import lru_cache
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GSHHS singleton (lazy-loaded on first call)
# ---------------------------------------------------------------------------
_gshhs_lock = threading.Lock()
_gshhs_land: object = None  # shapely MultiPolygon
_gshhs_prepared: object = None  # shapely PreparedGeometry
_gshhs_loaded = False
_gshhs_failed = False


def _load_gshhs() -> bool:
    """Load GSHHS intermediate-resolution polygons. Thread-safe singleton."""
    global _gshhs_land, _gshhs_prepared, _gshhs_loaded, _gshhs_failed

    if _gshhs_loaded or _gshhs_failed:
        return _gshhs_loaded

    with _gshhs_lock:
        # Double-check after acquiring lock
        if _gshhs_loaded or _gshhs_failed:
            return _gshhs_loaded

        try:
            import cartopy.io.shapereader as shpreader
            from shapely.geometry import MultiPolygon, shape
            from shapely.ops import unary_union
            from shapely.prepared import prep

            # Load GSHHS intermediate resolution, level 1 (continental + major islands)
            shp_path = shpreader.gshhs('i', 1)
            reader = shpreader.Reader(shp_path)

            polygons = []
            for record in reader.records():
                geom = shape(record.geometry)
                if geom.is_valid:
                    polygons.append(geom)
                else:
                    polygons.append(geom.buffer(0))

            if not polygons:
                logger.warning("GSHHS: no polygons loaded")
                _gshhs_failed = True
                return False

            land_union = unary_union(polygons)
            if not isinstance(land_union, MultiPolygon):
                land_union = MultiPolygon([land_union])

            _gshhs_land = land_union
            _gshhs_prepared = prep(land_union)
            _gshhs_loaded = True
            logger.info(f"GSHHS loaded: {len(polygons)} polygons, intermediate resolution")
            return True

        except Exception as e:
            logger.warning(f"GSHHS load failed: {e}. Falling back to alternative methods.")
            _gshhs_failed = True
            return False


def get_land_geometry():
    """Return the GSHHS land MultiPolygon, or None if unavailable.

    Used by routing_graph.py for distance-to-coast calculations.
    """
    if not _gshhs_loaded:
        _load_gshhs()
    return _gshhs_land


# ---------------------------------------------------------------------------
# global-land-mask fallback
# ---------------------------------------------------------------------------
_HAS_LAND_MASK = False
_globe = None

try:
    from global_land_mask import globe
    _globe = globe
    _HAS_LAND_MASK = True
    logger.info("global-land-mask available as fallback")
except ImportError:
    logger.info("global-land-mask not installed — GSHHS primary, bbox fallback")


# ---------------------------------------------------------------------------
# Simplified bounding-box fallback data
# ---------------------------------------------------------------------------
CONTINENTAL_BOUNDS = [
    (25, 72, -170, -50, "North America"),
    (-56, 12, -82, -34, "South America"),
    (36, 71, -10, 40, "Europe"),
    (-35, 37, -18, 52, "Africa"),
    (5, 77, 40, 180, "Asia"),
    (-45, -10, 112, 155, "Australia"),
    (-90, -60, -180, 180, "Antarctica"),
]

INLAND_WATER = [
    (30, 46, -6, 36, "Mediterranean"),
    (40, 47, 27, 42, "Black Sea"),
    (12, 30, 32, 44, "Red Sea"),
    (23, 30, 48, 57, "Persian Gulf"),
    (53, 66, 10, 30, "Baltic Sea"),
    (18, 31, -98, -80, "Gulf of Mexico"),
    (9, 23, -88, -60, "Caribbean"),
    (51, 65, -95, -77, "Hudson Bay"),
    (33, 52, 127, 142, "Sea of Japan"),
    (0, 23, 100, 121, "South China Sea"),
]

SIMPLIFIED_COASTLINES = {
    "western_europe": [
        (36.0, -10.0), (43.0, -10.0), (48.0, -5.0), (51.0, 2.0),
        (54.0, 8.0), (57.0, 8.0), (58.0, 12.0), (56.0, 12.0),
        (54.0, 10.0), (53.0, 7.0), (51.0, 4.0), (49.0, 0.0),
        (46.0, -2.0), (43.0, -2.0), (42.0, 3.0), (41.0, 2.0),
        (37.0, -6.0), (36.0, -6.0), (36.0, -10.0),
    ],
    "uk": [
        (50.0, -6.0), (51.0, -5.0), (52.0, -5.0), (53.5, -5.0),
        (55.0, -6.0), (58.5, -7.0), (59.0, -3.0), (58.0, -1.5),
        (55.0, -1.5), (54.0, 0.0), (53.0, 0.5), (52.5, 1.5),
        (51.0, 1.5), (50.5, 0.0), (50.0, -2.0), (50.0, -6.0),
    ],
    "us_east_coast": [
        (25.0, -80.0), (30.0, -81.0), (32.0, -81.0), (35.0, -76.0),
        (37.0, -76.0), (39.0, -75.0), (40.0, -74.0), (41.0, -72.0),
        (42.0, -71.0), (43.0, -70.0), (45.0, -67.0), (47.0, -68.0),
        (45.0, -66.0), (44.0, -66.0), (43.0, -65.0),
    ],
}


# ---------------------------------------------------------------------------
# Helper: (lat, lon) → shapely Point(lon, lat)
# ---------------------------------------------------------------------------
def _pt(lat: float, lon: float):
    """Create a shapely Point from (lat, lon). Centralizes the coordinate swap."""
    from shapely.geometry import Point
    return Point(lon, lat)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@lru_cache(maxsize=200_000)
def is_ocean(lat: float, lon: float) -> bool:
    """
    Check if a point is in navigable ocean water.

    Args:
        lat: Latitude (-90 to 90)
        lon: Longitude (-180 to 180)

    Returns:
        True if point is ocean/sea, False if land
    """
    # Try GSHHS first (most accurate)
    if _gshhs_loaded or (not _gshhs_failed and _load_gshhs()):
        try:
            return not _gshhs_prepared.contains(_pt(lat, lon))
        except Exception:
            pass

    # Fallback: global-land-mask (1km raster)
    if _HAS_LAND_MASK and _globe is not None:
        try:
            return bool(_globe.is_ocean(lat, lon))
        except Exception:
            pass

    # Last resort: bounding-box heuristics
    return _simplified_is_ocean(lat, lon)


def is_path_clear(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    num_checks: int = 0,
) -> bool:
    """
    Check if a path between two points crosses land.

    Uses geometric intersection with GSHHS when available (exact),
    falls back to point sampling otherwise.

    Args:
        lat1, lon1: Start point
        lat2, lon2: End point
        num_checks: Number of points to sample (0 = auto; ignored when GSHHS available)

    Returns:
        True if path is entirely over water
    """
    # GSHHS: exact geometric intersection
    if _gshhs_loaded or (not _gshhs_failed and _load_gshhs()):
        try:
            from shapely.geometry import LineString
            line = LineString([(lon1, lat1), (lon2, lat2)])
            return not _gshhs_prepared.intersects(line)
        except Exception:
            pass

    # Fallback: point sampling
    if num_checks <= 0:
        dlat = (lat2 - lat1) * 60
        dlon = (lon2 - lon1) * 60 * math.cos(math.radians((lat1 + lat2) / 2))
        dist_nm = math.sqrt(dlat * dlat + dlon * dlon)
        num_checks = max(10, min(200, int(dist_nm / 2)))

    for i in range(num_checks + 1):
        t = i / num_checks
        lat = lat1 + t * (lat2 - lat1)
        lon = lon1 + t * (lon2 - lon1)
        if not is_ocean(lat, lon):
            return False

    return True


def get_land_mask_status() -> dict:
    """Get information about land mask availability and method in use."""
    # Trigger lazy load if not yet attempted
    if not _gshhs_loaded and not _gshhs_failed:
        _load_gshhs()

    if _gshhs_loaded:
        method = "GSHHS intermediate (vector, sub-km)"
    elif _HAS_LAND_MASK:
        method = "global-land-mask (1km raster)"
    else:
        method = "simplified bounding boxes"

    return {
        "high_resolution_available": _gshhs_loaded or _HAS_LAND_MASK,
        "gshhs_loaded": _gshhs_loaded,
        "global_land_mask_available": _HAS_LAND_MASK,
        "method": method,
        "cache_size": is_ocean.cache_info().currsize if hasattr(is_ocean, 'cache_info') else 0,
    }


# ---------------------------------------------------------------------------
# Fallback implementations
# ---------------------------------------------------------------------------

def _simplified_is_ocean(lat: float, lon: float) -> bool:
    """Simplified ocean detection using bounding boxes."""
    for lat_min, lat_max, lon_min, lon_max, name in INLAND_WATER:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return True

    for lat_min, lat_max, lon_min, lon_max, name in CONTINENTAL_BOUNDS:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            if _is_coastal_water(lat, lon):
                return True
            return False

    return True


def _is_coastal_water(lat: float, lon: float) -> bool:
    """Check if a point is in coastal waters within a continental bounding box."""
    if 30 <= lat <= 46 and -6 <= lon <= 36:
        return True
    if 51 <= lat <= 62 and -4 <= lon <= 10:
        return True
    if 48 <= lat <= 52 and -6 <= lon <= 2:
        return True
    if 43 <= lat <= 48 and -10 <= lon <= -1:
        return True
    if 25 <= lat <= 45 and -82 <= lon <= -65:
        if lon < -75:
            return True
    return False


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test():
    """Run quick self-test on known points."""
    test_cases = [
        (45.0, -30.0, True, "Mid-Atlantic"),
        (51.5, -0.1, False, "London"),
        (40.75, -73.97, False, "Manhattan"),
        (35.0, -50.0, True, "Atlantic Ocean"),
        (0.0, 0.0, True, "Gulf of Guinea"),
        (48.8, 2.3, False, "Paris"),
        (35.0, 139.0, False, "Tokyo area"),
        (50.0, -5.0, True, "English Channel"),
        (43.0, 5.0, True, "Mediterranean"),
    ]

    results = []
    for lat, lon, expected, desc in test_cases:
        actual = is_ocean(lat, lon)
        passed = actual == expected
        results.append({
            "point": (lat, lon),
            "description": desc,
            "expected": expected,
            "actual": actual,
            "passed": passed,
        })
        if not passed:
            logger.warning(f"Land mask test failed: {desc} ({lat}, {lon}) - "
                          f"expected {expected}, got {actual}")

    return results

"""
Land mask for maritime route optimization.

Provides is_ocean(lat, lon) function to check if a point is navigable water.

Uses multiple approaches:
1. Try global-land-mask package (accurate, 1km resolution)
2. Fallback to simplified polygon-based coastlines
3. Basic bounding box checks for major land masses
"""

import logging
import math
from functools import lru_cache
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Try to import global-land-mask
_HAS_LAND_MASK = False
_globe = None

try:
    from global_land_mask import globe
    _globe = globe
    _HAS_LAND_MASK = True
    logger.info("Using global-land-mask for land detection")
except ImportError:
    logger.warning("global-land-mask not installed. Using simplified land detection. "
                   "For better accuracy: pip install global-land-mask")


# Simplified continental bounding boxes for fallback
# Format: (lat_min, lat_max, lon_min, lon_max, name)
CONTINENTAL_BOUNDS = [
    # North America
    (25, 72, -170, -50, "North America"),
    # South America
    (-56, 12, -82, -34, "South America"),
    # Europe
    (36, 71, -10, 40, "Europe"),
    # Africa
    (-35, 37, -18, 52, "Africa"),
    # Asia
    (5, 77, 40, 180, "Asia"),
    # Australia
    (-45, -10, 112, 155, "Australia"),
    # Antarctica (simplified)
    (-90, -60, -180, 180, "Antarctica"),
]

# Major inland seas/lakes to exclude (these ARE water)
INLAND_WATER = [
    # Mediterranean
    (30, 46, -6, 36, "Mediterranean"),
    # Black Sea
    (40, 47, 27, 42, "Black Sea"),
    # Red Sea
    (12, 30, 32, 44, "Red Sea"),
    # Persian Gulf
    (23, 30, 48, 57, "Persian Gulf"),
    # Baltic Sea
    (53, 66, 10, 30, "Baltic Sea"),
    # Gulf of Mexico
    (18, 31, -98, -80, "Gulf of Mexico"),
    # Caribbean
    (9, 23, -88, -60, "Caribbean"),
    # Hudson Bay
    (51, 65, -95, -77, "Hudson Bay"),
    # Sea of Japan
    (33, 52, 127, 142, "Sea of Japan"),
    # South China Sea
    (0, 23, 100, 121, "South China Sea"),
]

# Simplified coastline polygons for major shipping areas
# Points are (lat, lon) forming a polygon
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


@lru_cache(maxsize=100000)
def is_ocean(lat: float, lon: float) -> bool:
    """
    Check if a point is in navigable ocean water.

    Args:
        lat: Latitude (-90 to 90)
        lon: Longitude (-180 to 180)

    Returns:
        True if point is ocean/sea, False if land
    """
    # Use global-land-mask if available
    if _HAS_LAND_MASK and _globe is not None:
        try:
            return bool(_globe.is_ocean(lat, lon))
        except Exception:
            pass

    # Fallback to simplified detection
    return _simplified_is_ocean(lat, lon)


def _simplified_is_ocean(lat: float, lon: float) -> bool:
    """Simplified ocean detection using bounding boxes."""

    # Check if in inland water bodies first (these are navigable)
    for lat_min, lat_max, lon_min, lon_max, name in INLAND_WATER:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return True

    # Check continental bounding boxes
    for lat_min, lat_max, lon_min, lon_max, name in CONTINENTAL_BOUNDS:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            # Inside continental bounding box - likely land
            # But need to check if it's actually coastal water
            if _is_coastal_water(lat, lon):
                return True
            return False

    # Not in any continental bounding box - assume ocean
    return True


def _is_coastal_water(lat: float, lon: float) -> bool:
    """Check if a point is in coastal waters within a continental bounding box."""

    # Simple distance-from-coast heuristic for some regions
    # This is very approximate

    # Mediterranean Sea
    if 30 <= lat <= 46 and -6 <= lon <= 36:
        return True

    # North Sea
    if 51 <= lat <= 62 and -4 <= lon <= 10:
        return True

    # English Channel
    if 48 <= lat <= 52 and -6 <= lon <= 2:
        return True

    # Bay of Biscay
    if 43 <= lat <= 48 and -10 <= lon <= -1:
        return True

    # US East Coast (offshore)
    if 25 <= lat <= 45 and -82 <= lon <= -65:
        # Very rough: if far enough east, it's ocean
        if lon < -75:
            return True

    return False


def is_path_clear(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    num_checks: int = 10
) -> bool:
    """
    Check if a path between two points crosses land.

    Args:
        lat1, lon1: Start point
        lat2, lon2: End point
        num_checks: Number of points to sample along path

    Returns:
        True if path is entirely over water
    """
    for i in range(num_checks + 1):
        t = i / num_checks
        lat = lat1 + t * (lat2 - lat1)
        lon = lon1 + t * (lon2 - lon1)

        if not is_ocean(lat, lon):
            return False

    return True


def get_land_mask_status() -> dict:
    """Get information about land mask availability."""
    return {
        "high_resolution_available": _HAS_LAND_MASK,
        "method": "global-land-mask (1km)" if _HAS_LAND_MASK else "simplified bounding boxes",
        "cache_size": is_ocean.cache_info().currsize if hasattr(is_ocean, 'cache_info') else 0,
    }


# Quick test points for validation
def _self_test():
    """Run quick self-test on known points."""
    test_cases = [
        # (lat, lon, expected_is_ocean, description)
        (45.0, -30.0, True, "Mid-Atlantic"),
        (51.5, -0.1, False, "London"),
        (40.7, -74.0, False, "New York City"),
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

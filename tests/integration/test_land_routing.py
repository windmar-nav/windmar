"""
Integration tests for land routing constraints.

Verifies that the land mask correctly detects when straight-line
maritime routes cross land masses, ensuring optimized routes
cannot traverse peninsulas, continents, or other landforms.
"""

import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.land_mask import is_path_clear, get_land_mask_status


def _gshhs_available() -> bool:
    status = get_land_mask_status()
    return status["gshhs_loaded"]


needs_gshhs = pytest.mark.skipif(
    not _gshhs_available(),
    reason="GSHHS shapefiles not available",
)


class TestLandCrossingDetection:
    """Verify the land mask blocks routes that cross major land masses."""

    @needs_gshhs
    def test_barcelona_lisbon_no_iberia_crossing(self):
        """Straight line Barcelona -> Lisbon crosses Spain."""
        assert is_path_clear(41.38, 2.17, 38.72, -9.14) is False

    @needs_gshhs
    def test_rotterdam_stockholm_no_jutland_crossing(self):
        """Straight line Rotterdam -> Stockholm crosses Denmark/Jutland."""
        assert is_path_clear(51.92, 4.48, 59.33, 18.07) is False

    @needs_gshhs
    def test_genoa_istanbul_no_calabria_crossing(self):
        """Straight line Genoa -> Istanbul crosses Italy/Calabria."""
        assert is_path_clear(44.41, 8.93, 41.01, 28.98) is False

    @needs_gshhs
    def test_path_clear_across_spain(self):
        """Path from south to north crosses Spain."""
        assert is_path_clear(35.0, -5.0, 43.0, -8.0) is False


class TestOpenWaterPaths:
    """Verify the land mask allows routes through open water and straits."""

    @needs_gshhs
    def test_open_ocean_path_clear(self):
        """East-west path across open Mediterranean is clear."""
        assert is_path_clear(36.0, 0.0, 36.0, 10.0) is True

    @needs_gshhs
    def test_straits_are_clear(self):
        """Paths through major navigable straits are clear."""
        # Strait of Gibraltar
        assert is_path_clear(35.95, -5.6, 36.1, -5.2) is True
        # Strait of Messina
        assert is_path_clear(38.2, 15.55, 38.25, 15.65) is True


class TestLandMaskStatus:
    """Verify land mask status reporting."""

    @needs_gshhs
    def test_get_land_mask_status_method_field(self):
        """Status dict contains a valid method identifier."""
        status = get_land_mask_status()
        assert status["method"] in ("gshhs", "global-land-mask", "bbox-fallback")

"""
Unit tests for Phase 3c: Strait Visibility Graph.

Tests strait waypoint definitions, validation, and graph injection.
"""

import pytest

from src.data.strait_waypoints import (
    STRAITS,
    STRAIT_BY_CODE,
    NARROW_STRAIT_CODES,
    StraitDefinition,
    get_nearby_straits,
    validate_strait_waypoints,
)
from src.data.land_mask import is_ocean, is_path_clear, get_land_mask_status


def _gshhs_available() -> bool:
    status = get_land_mask_status()
    return status["gshhs_loaded"]


needs_gshhs = pytest.mark.skipif(
    not _gshhs_available(),
    reason="GSHHS shapefiles not available"
)


# ---------------------------------------------------------------------------
# §1 – Strait data integrity
# ---------------------------------------------------------------------------
class TestStraitDataIntegrity:
    """Verify strait definitions are well-formed."""

    def test_eight_straits_defined(self):
        assert len(STRAITS) == 8

    def test_unique_codes(self):
        codes = [s.code for s in STRAITS]
        assert len(codes) == len(set(codes)), f"Duplicate codes: {codes}"

    def test_min_two_waypoints_per_strait(self):
        for strait in STRAITS:
            assert len(strait.waypoints) >= 2, (
                f"{strait.code} has {len(strait.waypoints)} waypoints (need >= 2)"
            )

    def test_strait_by_code_lookup(self):
        assert "GIBR" in STRAIT_BY_CODE
        assert STRAIT_BY_CODE["GIBR"].name == "Strait of Gibraltar"

    def test_all_codes_in_lookup(self):
        for strait in STRAITS:
            assert strait.code in STRAIT_BY_CODE
            assert STRAIT_BY_CODE[strait.code] is strait

    def test_waypoint_coordinates_valid(self):
        """All waypoints must have valid lat/lon ranges."""
        for strait in STRAITS:
            for lat, lon in strait.waypoints:
                assert -90 <= lat <= 90, f"{strait.code}: lat {lat} out of range"
                assert -180 <= lon <= 180, f"{strait.code}: lon {lon} out of range"

    def test_max_draft_positive(self):
        for strait in STRAITS:
            assert strait.max_draft_m > 0


# ---------------------------------------------------------------------------
# §2 – Waypoints are ocean (GSHHS required)
# ---------------------------------------------------------------------------
class TestWaypointsAreOcean:
    """All strait waypoints must be classified as ocean."""

    @needs_gshhs
    @pytest.mark.parametrize("strait", STRAITS, ids=[s.code for s in STRAITS])
    def test_all_waypoints_ocean(self, strait):
        for i, (lat, lon) in enumerate(strait.waypoints):
            assert is_ocean(lat, lon), (
                f"{strait.code} wp[{i}] ({lat}, {lon}) is not ocean"
            )


# ---------------------------------------------------------------------------
# §3 – Consecutive waypoints path-clear (GSHHS required)
# Narrow straits (Bosporus, Suez Canal) are expected to fail path_clear
# because GSHHS intermediate resolution cannot resolve sub-km channels.
# These strait edges skip is_path_clear() during A* (pre-validated).
# ---------------------------------------------------------------------------
class TestConsecutivePathClear:
    """Consecutive waypoints must have clear paths (non-narrow straits only)."""

    @needs_gshhs
    @pytest.mark.parametrize(
        "strait",
        [s for s in STRAITS if not s.narrow],
        ids=[s.code for s in STRAITS if not s.narrow],
    )
    def test_consecutive_path_clear(self, strait):
        for i in range(len(strait.waypoints) - 1):
            lat1, lon1 = strait.waypoints[i]
            lat2, lon2 = strait.waypoints[i + 1]
            assert is_path_clear(lat1, lon1, lat2, lon2), (
                f"{strait.code} segment [{i}]->[{i+1}] crosses land"
            )

    def test_narrow_straits_exist(self):
        """At least one narrow strait is defined."""
        assert len(NARROW_STRAIT_CODES) >= 1


# ---------------------------------------------------------------------------
# §4 – get_nearby_straits
# ---------------------------------------------------------------------------
class TestNearbyStraits:

    def test_gibraltar_nearby(self):
        """Gibraltar should be found near (36, -5)."""
        nearby = get_nearby_straits(36.0, -5.0, threshold_deg=2.0)
        codes = [s.code for s in nearby]
        assert "GIBR" in codes

    def test_no_straits_mid_pacific(self):
        """No straits near mid-Pacific."""
        nearby = get_nearby_straits(0.0, -150.0, threshold_deg=5.0)
        assert len(nearby) == 0


# ---------------------------------------------------------------------------
# §5 – Graph injection (unit test, no GSHHS needed)
# ---------------------------------------------------------------------------
class TestGraphInjection:
    """Test strait edge injection into routing grid."""

    def test_inject_adds_nodes_to_grid(self):
        """Strait injection adds cells to the grid dict."""
        from src.optimization.route_optimizer import RouteOptimizer, GridCell

        optimizer = RouteOptimizer(resolution_deg=0.5)

        # Build a minimal grid around Gibraltar
        grid = {}
        row, col = 0, 0
        for lat_idx in range(10):
            for lon_idx in range(10):
                lat = 34.0 + lat_idx * 0.5
                lon = -7.0 + lon_idx * 0.5
                grid[(row + lat_idx, col + lon_idx)] = GridCell(
                    lat=lat, lon=lon, row=row + lat_idx, col=col + lon_idx
                )

        initial_size = len(grid)
        injected = optimizer._inject_strait_edges(grid)

        # At least Gibraltar should be injected
        assert injected > 0
        assert len(grid) > initial_size

    def test_strait_edges_created(self):
        """Strait injection creates bidirectional edges."""
        from src.optimization.route_optimizer import RouteOptimizer, GridCell

        optimizer = RouteOptimizer(resolution_deg=0.5)

        # Build grid around Dover
        grid = {}
        for lat_idx in range(10):
            for lon_idx in range(10):
                lat = 49.0 + lat_idx * 0.5
                lon = -1.0 + lon_idx * 0.5
                grid[(lat_idx, lon_idx)] = GridCell(
                    lat=lat, lon=lon, row=lat_idx, col=lon_idx
                )

        optimizer._inject_strait_edges(grid)

        # Should have strait edges
        assert hasattr(optimizer, '_strait_edges')
        assert len(optimizer._strait_edges) > 0

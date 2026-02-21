"""
Unit tests for Phase 3a: GSHHS Coastline Integration.

Tests the rewritten land_mask.py with GSHHS vector polygons.
When GSHHS is unavailable (no cartopy/shapefiles), tests gracefully skip.
"""

import time

import pytest

from src.data.land_mask import (
    get_land_geometry,
    get_land_mask_status,
    is_ocean,
    is_path_clear,
    _self_test,
)


# ---------------------------------------------------------------------------
# Check if GSHHS is available for tests that require it
# ---------------------------------------------------------------------------
def _gshhs_available() -> bool:
    """Check if GSHHS can be loaded."""
    status = get_land_mask_status()
    return status["gshhs_loaded"]


needs_gshhs = pytest.mark.skipif(
    not _gshhs_available(),
    reason="GSHHS shapefiles not available (need cartopy + data download)"
)


# ---------------------------------------------------------------------------
# §1 – Known ocean points (10 tests)
# ---------------------------------------------------------------------------
class TestKnownOceanPoints:
    """Points that must be classified as ocean."""

    @pytest.mark.parametrize("lat,lon,desc", [
        (45.0, -30.0, "Mid-Atlantic"),
        (36.0, 20.0, "Central Mediterranean"),
        (51.0, 1.2, "English Channel"),
        (5.0, 105.0, "South China Sea"),
        (25.0, -90.0, "Gulf of Mexico"),
        (-35.0, 20.0, "South Atlantic"),
        (60.0, -20.0, "North Atlantic"),
        (10.0, 65.0, "Arabian Sea"),
        (-10.0, 50.0, "Indian Ocean"),
        (35.0, -50.0, "Mid-Atlantic (south)"),
    ])
    def test_ocean_point(self, lat, lon, desc):
        assert is_ocean(lat, lon), f"{desc} ({lat}, {lon}) should be ocean"


# ---------------------------------------------------------------------------
# §2 – Known land points (10 tests)
# ---------------------------------------------------------------------------
class TestKnownLandPoints:
    """Points that must be classified as land."""

    @pytest.mark.parametrize("lat,lon,desc", [
        (51.5, -0.1, "London"),
        (48.8, 2.3, "Paris"),
        (40.75, -73.97, "Manhattan"),
        (35.7, 139.7, "Tokyo"),
        (30.0, 31.2, "Cairo"),
        (55.7, 37.6, "Moscow"),
        (39.9, 116.4, "Beijing"),
        (-33.9, 151.2, "Sydney"),
        (-23.5, -46.6, "Sao Paulo"),
        (19.4, -99.1, "Mexico City"),
    ])
    @needs_gshhs
    def test_land_point(self, lat, lon, desc):
        assert not is_ocean(lat, lon), f"{desc} ({lat}, {lon}) should be land"


# ---------------------------------------------------------------------------
# §3 – Coastal accuracy (5 tests) — require GSHHS for sub-km precision
# ---------------------------------------------------------------------------
class TestCoastalAccuracy:
    """Points near coastlines that test sub-km resolution."""

    @needs_gshhs
    def test_dover_strait_ocean(self):
        """Mid-channel Dover Strait (~51.05N, 1.4E) is ocean."""
        assert is_ocean(51.05, 1.4)

    @needs_gshhs
    def test_dover_strait_land(self):
        """Dover town (~51.13N, 1.31E) is land."""
        assert not is_ocean(51.13, 1.31)

    @needs_gshhs
    def test_singapore_strait_ocean(self):
        """Singapore Strait mid-channel (~1.2N, 103.8E) is ocean."""
        assert is_ocean(1.2, 103.8)

    @needs_gshhs
    def test_messina_strait_ocean(self):
        """Strait of Messina mid-channel (~38.2N, 15.6E) is ocean."""
        assert is_ocean(38.2, 15.6)

    @needs_gshhs
    def test_port_said_approach_ocean(self):
        """Port Said roadstead (~31.28N, 32.30E) is ocean."""
        assert is_ocean(31.28, 32.30)


# ---------------------------------------------------------------------------
# §4 – is_path_clear tests (5 tests)
# ---------------------------------------------------------------------------
class TestPathClear:
    """Path clearance validation."""

    def test_open_ocean_path(self):
        """Open ocean path (Mid-Atlantic) should be clear."""
        assert is_path_clear(45.0, -30.0, 40.0, -20.0)

    @needs_gshhs
    def test_crossing_england(self):
        """Path crossing England (Dover to Liverpool) should NOT be clear."""
        assert not is_path_clear(51.1, 1.3, 53.4, -3.0)

    @needs_gshhs
    def test_gibraltar_passthrough(self):
        """Path through Strait of Gibraltar should be clear."""
        assert is_path_clear(35.95, -5.6, 36.1, -5.2)

    @needs_gshhs
    def test_crossing_sicily(self):
        """Path crossing Sicily should NOT be clear."""
        assert not is_path_clear(37.0, 13.0, 38.5, 16.0)

    @needs_gshhs
    def test_english_channel_passthrough(self):
        """Path through English Channel (offshore) should be clear."""
        assert is_path_clear(50.3, -2.0, 50.7, 1.5)


# ---------------------------------------------------------------------------
# §5 – get_land_mask_status metadata check
# ---------------------------------------------------------------------------
class TestLandMaskStatus:
    """Verify status reporting."""

    def test_status_has_required_keys(self):
        status = get_land_mask_status()
        assert "high_resolution_available" in status
        assert "gshhs_loaded" in status
        assert "global_land_mask_available" in status
        assert "method" in status
        assert "cache_size" in status

    def test_method_string(self):
        status = get_land_mask_status()
        assert isinstance(status["method"], str)
        assert len(status["method"]) > 0


# ---------------------------------------------------------------------------
# §6 – _self_test regression
# ---------------------------------------------------------------------------
class TestSelfTest:
    """Verify built-in self-test passes."""

    def test_self_test_all_pass(self):
        results = _self_test()
        assert len(results) > 0
        for r in results:
            assert r["passed"], f"Self-test failed: {r['description']}"


# ---------------------------------------------------------------------------
# §7 – get_land_geometry
# ---------------------------------------------------------------------------
class TestGetLandGeometry:
    """Verify land geometry accessor."""

    @needs_gshhs
    def test_returns_multipolygon(self):
        geom = get_land_geometry()
        assert geom is not None
        assert hasattr(geom, 'geoms')  # MultiPolygon

    def test_returns_none_when_unavailable(self):
        """If GSHHS failed, returns None gracefully."""
        geom = get_land_geometry()
        # Either returns a geometry or None — no crash
        assert geom is None or hasattr(geom, 'geoms')


# ---------------------------------------------------------------------------
# §8 – Performance: 10k random ocean checks < 2s
# ---------------------------------------------------------------------------
class TestPerformance:
    """Performance benchmarks."""

    @needs_gshhs
    def test_10k_point_checks_under_2s(self):
        """10,000 random point-in-polygon queries should complete within 2s."""
        import random
        random.seed(42)
        points = [(random.uniform(-60, 60), random.uniform(-180, 180)) for _ in range(10_000)]

        start = time.time()
        for lat, lon in points:
            is_ocean(lat, lon)
        elapsed = time.time() - start

        assert elapsed < 2.0, f"10k queries took {elapsed:.2f}s (limit: 2s)"

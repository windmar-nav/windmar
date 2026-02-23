"""
Unit tests for TSS heading alignment penalty.

Tests that the ZoneChecker correctly applies heading penalties for
wrong-way transit through TSS zones and incentives for aligned transit.
"""

import pytest
from src.data.regulatory_zones import ZoneChecker, ZoneType


class TestBearing:
    """Tests for bearing computation."""

    def test_bearing_east(self):
        """Bearing from (0,0) to (0,1) should be ~90°."""
        brg = ZoneChecker._compute_bearing(0, 0, 0, 1)
        assert 89 < brg < 91

    def test_bearing_north(self):
        """Bearing from (0,0) to (1,0) should be ~0°."""
        brg = ZoneChecker._compute_bearing(0, 0, 1, 0)
        assert brg < 1 or brg > 359

    def test_bearing_south(self):
        """Bearing from (1,0) to (0,0) should be ~180°."""
        brg = ZoneChecker._compute_bearing(1, 0, 0, 0)
        assert 179 < brg < 181

    def test_bearing_west(self):
        """Bearing from (0,1) to (0,0) should be ~270°."""
        brg = ZoneChecker._compute_bearing(0, 1, 0, 0)
        assert 269 < brg < 271


class TestAngularDeviation:
    """Tests for angular deviation calculation."""

    def test_same_bearing(self):
        assert ZoneChecker._angular_deviation(90, 90) == 0

    def test_opposite(self):
        assert ZoneChecker._angular_deviation(0, 180) == 180

    def test_wrap_around(self):
        assert ZoneChecker._angular_deviation(10, 350) == 20

    def test_small_deviation(self):
        assert ZoneChecker._angular_deviation(85, 90) == 5


class TestTSSHeadingPenalty:
    """Tests for TSS heading alignment in get_path_penalty()."""

    @pytest.fixture
    def checker(self):
        return ZoneChecker()

    def test_aligned_transit_through_gibraltar_ew(self, checker):
        """E-W transit through Gibraltar should get incentive (0.4x)."""
        # Gibraltar TSS direction is 90° (E-W), bidirectional
        # Point inside Gibraltar TSS, going east
        # Use midpoints inside the Gibraltar TSS polygon
        zone = checker.get_zone("tss_strait_of_gibraltar")
        if zone is None:
            pytest.skip("Gibraltar TSS not loaded")

        # Compute bearing for east-bound transit
        bearing = ZoneChecker._compute_bearing(35.95, -5.6, 35.95, -5.4)
        penalty, warn = checker._check_tss_heading(zone, bearing)
        assert penalty == 0.4, f"Aligned transit should be 0.4x, got {penalty}"
        assert warn is None

    def test_aligned_transit_through_gibraltar_we(self, checker):
        """W-E transit (reciprocal) should also get incentive."""
        zone = checker.get_zone("tss_strait_of_gibraltar")
        if zone is None:
            pytest.skip("Gibraltar TSS not loaded")

        bearing = ZoneChecker._compute_bearing(35.95, -5.4, 35.95, -5.6)
        penalty, warn = checker._check_tss_heading(zone, bearing)
        assert penalty == 0.4

    def test_wrong_way_gibraltar_ns(self, checker):
        """N-S transit through Gibraltar (perpendicular) should get heavy penalty."""
        zone = checker.get_zone("tss_strait_of_gibraltar")
        if zone is None:
            pytest.skip("Gibraltar TSS not loaded")

        # Going north (0°) — perpendicular to E-W TSS
        bearing = 0.0
        penalty, warn = checker._check_tss_heading(zone, bearing)
        assert penalty == 10.0, f"Perpendicular transit should be 10x, got {penalty}"
        assert warn is not None
        assert "heading violation" in warn.lower()

    def test_dover_strait_ne_aligned(self, checker):
        """NE-bound transit through Dover Strait should be aligned."""
        zone = checker.get_zone("tss_dover_strait")
        if zone is None:
            pytest.skip("Dover Strait TSS not loaded")

        # Dover TSS direction is 40° (NE), tolerance 25°
        bearing = 45.0  # Within tolerance
        penalty, warn = checker._check_tss_heading(zone, bearing)
        assert penalty == 0.4

    def test_dover_strait_sw_aligned(self, checker):
        """SW-bound (reciprocal) through Dover Strait should be aligned."""
        zone = checker.get_zone("tss_dover_strait")
        if zone is None:
            pytest.skip("Dover Strait TSS not loaded")

        bearing = 220.0  # 40 + 180 = 220, SW-bound
        penalty, warn = checker._check_tss_heading(zone, bearing)
        assert penalty == 0.4

    def test_dover_strait_crosswise(self, checker):
        """East-west crossing of Dover Strait should be penalized."""
        zone = checker.get_zone("tss_dover_strait")
        if zone is None:
            pytest.skip("Dover Strait TSS not loaded")

        bearing = 120.0  # ESE — perpendicular to NE/SW lanes
        penalty, warn = checker._check_tss_heading(zone, bearing)
        assert penalty == 10.0

    def test_no_direction_metadata(self, checker):
        """Zone without direction_deg should return 1.0 (no penalty)."""
        # Create a fake TSS zone with no direction metadata
        from src.data.regulatory_zones import Zone, ZoneProperties, ZoneInteraction
        fake = Zone(
            id="tss_fake_no_direction",
            properties=ZoneProperties(
                name="Fake TSS",
                zone_type=ZoneType.TSS,
                interaction=ZoneInteraction.MANDATORY,
            ),
            coordinates=[(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)],
        )
        penalty, warn = checker._check_tss_heading(fake, 45.0)
        assert penalty == 1.0
        assert warn is None

    def test_all_tss_zones_have_direction(self, checker):
        """All TSS zones loaded from tss_zones.py should have direction metadata."""
        from src.data.tss_zones import TSS_METADATA
        for key, meta in TSS_METADATA.items():
            assert "direction_deg" in meta, f"TSS zone '{key}' missing direction_deg"
            assert "tolerance_deg" in meta, f"TSS zone '{key}' missing tolerance_deg"
            assert 0 <= meta["direction_deg"] < 360, f"TSS zone '{key}' invalid direction"
            assert 0 < meta["tolerance_deg"] <= 90, f"TSS zone '{key}' invalid tolerance"


class TestTSSCount:
    """Tests for TSS zone loading."""

    def test_tss_zone_count(self):
        """Verify all 22 TSS zones are loaded."""
        checker = ZoneChecker()
        tss_zones = checker.get_zones_by_type(ZoneType.TSS)
        assert len(tss_zones) == 22

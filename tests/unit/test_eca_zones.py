"""
Unit tests for ECA zone functionality.
"""

import pytest
from src.data.eca_zones import (
    ECAZone,
    ECAManager,
    ECA_ZONES,
    BALTIC_SEA_ECA,
    NORTH_SEA_ECA,
    NORTH_AMERICAN_ECA,
    NORTH_AMERICAN_PACIFIC_ECA,
    US_CARIBBEAN_ECA,
    eca_manager,
)


class TestECAZone:
    """Tests for ECAZone class."""

    def test_zone_creation(self):
        """Test creating an ECA zone."""
        zone = ECAZone(
            name="Test Zone",
            code="TEST",
            polygon=[
                (0, 0),
                (0, 10),
                (10, 10),
                (10, 0),
            ],
        )
        assert zone.name == "Test Zone"
        assert zone.code == "TEST"
        assert zone.sox_limit == 0.1
        assert zone.nox_tier is None

    def test_point_in_zone(self):
        """Test point-in-polygon detection."""
        zone = ECAZone(
            name="Square Zone",
            code="SQUARE",
            polygon=[
                (0, 0),
                (0, 10),
                (10, 10),
                (10, 0),
            ],
        )

        # Point inside
        assert zone.contains_point(5, 5) is True

        # Point outside
        assert zone.contains_point(15, 15) is False
        assert zone.contains_point(-5, 5) is False

        # Point on edge (behavior depends on algorithm)
        # Ray casting typically includes some edges

    def test_point_in_baltic_eca(self):
        """Test points in Baltic Sea ECA."""
        # Stockholm area - should be in Baltic ECA
        assert BALTIC_SEA_ECA.contains_point(59.3, 18.0) is True

        # Copenhagen area - should be in Baltic ECA
        assert BALTIC_SEA_ECA.contains_point(55.7, 12.5) is True

        # London - should NOT be in Baltic ECA
        assert BALTIC_SEA_ECA.contains_point(51.5, -0.1) is False

    def test_point_in_north_sea_eca(self):
        """Test points in North Sea ECA."""
        # Rotterdam area - should be in North Sea ECA
        assert NORTH_SEA_ECA.contains_point(51.9, 4.5) is True

        # London area (Thames estuary) - should be in North Sea ECA
        assert NORTH_SEA_ECA.contains_point(51.5, 0.5) is True

        # Paris - should NOT be in North Sea ECA
        assert NORTH_SEA_ECA.contains_point(48.8, 2.3) is False

    def test_point_in_north_american_eca(self):
        """Test points in North American ECA."""
        # New York Harbor - should be in NA ECA
        assert NORTH_AMERICAN_ECA.contains_point(40.6, -74.0) is True

        # Miami area - should be in NA ECA
        assert NORTH_AMERICAN_ECA.contains_point(26.0, -80.0) is True

        # Mid-Atlantic (Bermuda area) - should NOT be in NA ECA
        assert NORTH_AMERICAN_ECA.contains_point(32.3, -64.7) is False

    def test_point_in_pacific_eca(self):
        """Test points in North American Pacific ECA."""
        # Los Angeles area - should be in Pacific ECA
        assert NORTH_AMERICAN_PACIFIC_ECA.contains_point(33.7, -118.2) is True

        # Seattle area - should be in Pacific ECA
        assert NORTH_AMERICAN_PACIFIC_ECA.contains_point(47.6, -122.3) is True

        # Hawaii - should NOT be in Pacific ECA
        assert NORTH_AMERICAN_PACIFIC_ECA.contains_point(21.3, -157.8) is False

    def test_point_in_caribbean_eca(self):
        """Test points in US Caribbean ECA."""
        # San Juan, Puerto Rico - should be in Caribbean ECA
        assert US_CARIBBEAN_ECA.contains_point(18.4, -66.0) is True

        # Jamaica - should NOT be in Caribbean ECA
        assert US_CARIBBEAN_ECA.contains_point(18.1, -77.3) is False

    def test_to_geojson(self):
        """Test GeoJSON conversion."""
        zone = ECAZone(
            name="Test Zone",
            code="TEST",
            polygon=[
                (0, 0),
                (0, 10),
                (10, 10),
                (10, 0),
            ],
            sox_limit=0.1,
            nox_tier=3,
        )

        geojson = zone.to_geojson()

        assert geojson["type"] == "Feature"
        assert geojson["properties"]["name"] == "Test Zone"
        assert geojson["properties"]["code"] == "TEST"
        assert geojson["properties"]["sox_limit"] == 0.1
        assert geojson["properties"]["nox_tier"] == 3
        assert geojson["geometry"]["type"] == "Polygon"
        assert len(geojson["geometry"]["coordinates"][0]) == 5  # Closed polygon


class TestECAManager:
    """Tests for ECAManager class."""

    def test_manager_initialization(self):
        """Test ECA manager initialization."""
        manager = ECAManager()
        assert len(manager.zones) == 6  # All default zones (incl. Mediterranean)

    def test_manager_custom_zones(self):
        """Test manager with custom zones."""
        custom_zone = ECAZone(
            name="Custom",
            code="CUSTOM",
            polygon=[(0, 0), (0, 1), (1, 1), (1, 0)],
        )
        manager = ECAManager(zones=[custom_zone])
        assert len(manager.zones) == 1
        assert manager.zones[0].code == "CUSTOM"

    def test_get_zone_at_point(self):
        """Test getting zone at a point."""
        manager = ECAManager()

        # Point in Baltic
        zone = manager.get_zone_at_point(59.3, 18.0)
        assert zone is not None
        assert zone.code == "BALTIC"

        # Point in open ocean
        zone = manager.get_zone_at_point(45.0, -30.0)
        assert zone is None

    def test_is_in_eca(self):
        """Test is_in_eca check."""
        manager = ECAManager()

        # In ECA
        assert manager.is_in_eca(51.9, 4.5) is True  # Rotterdam

        # Not in ECA
        assert manager.is_in_eca(45.0, -30.0) is False  # Atlantic

    def test_get_zones_for_route(self):
        """Test getting zones for a route."""
        manager = ECAManager()

        # Route from Rotterdam to Augusta (Mediterranean)
        # Should pass through North Sea ECA
        waypoints = [
            (51.9, 4.5),  # Rotterdam
            (50.0, 0.0),  # English Channel
            (48.0, -5.0),  # Bay of Biscay
            (43.0, -9.0),  # Off Portugal
            (36.0, -6.0),  # Strait of Gibraltar
            (37.2, 15.2),  # Augusta
        ]

        zones = manager.get_zones_for_route(waypoints)

        # Should include North Sea ECA
        zone_codes = [z.code for z in zones]
        assert "NORTHSEA" in zone_codes

    def test_get_eca_distance(self):
        """Test ECA distance calculation."""
        manager = ECAManager()

        # Simple route within and outside ECA
        # Rotterdam (in ECA) to somewhere in North Sea
        waypoints = [
            (51.9, 4.5),  # Rotterdam - in ECA
            (52.5, 4.0),  # Still in ECA
        ]

        eca_dist, non_eca_dist = manager.get_eca_distance(waypoints)

        # Should have some ECA distance
        assert eca_dist > 0

    def test_to_geojson_collection(self):
        """Test GeoJSON collection generation."""
        manager = ECAManager()

        geojson = manager.to_geojson_collection()

        assert geojson["type"] == "FeatureCollection"
        assert len(geojson["features"]) == 6

        # Check all features are valid
        for feature in geojson["features"]:
            assert feature["type"] == "Feature"
            assert "properties" in feature
            assert "geometry" in feature
            assert feature["geometry"]["type"] == "Polygon"


class TestECAZoneProperties:
    """Tests for ECA zone properties."""

    def test_sox_limits(self):
        """Test all zones have SOx limits."""
        for zone in ECA_ZONES:
            assert zone.sox_limit == 0.1  # Current global ECA limit

    def test_nox_requirements(self):
        """Test NOx tier requirements."""
        # North American and Caribbean ECAs have NOx Tier III
        assert NORTH_AMERICAN_ECA.nox_tier == 3
        assert NORTH_AMERICAN_PACIFIC_ECA.nox_tier == 3
        assert US_CARIBBEAN_ECA.nox_tier == 3

        # European ECAs are SOx only
        assert BALTIC_SEA_ECA.nox_tier is None
        assert NORTH_SEA_ECA.nox_tier is None

    def test_zone_colors(self):
        """Test all zones have display colors."""
        for zone in ECA_ZONES:
            assert zone.color is not None
            assert zone.color.startswith("#")

    def test_zone_codes_unique(self):
        """Test all zone codes are unique."""
        codes = [zone.code for zone in ECA_ZONES]
        assert len(codes) == len(set(codes))


class TestGlobalECAManager:
    """Tests for global eca_manager singleton."""

    def test_singleton_exists(self):
        """Test global manager exists."""
        assert eca_manager is not None
        assert isinstance(eca_manager, ECAManager)

    def test_singleton_has_all_zones(self):
        """Test global manager has all zones."""
        assert len(eca_manager.zones) == 6

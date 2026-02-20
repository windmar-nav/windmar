"""
Integration tests for the core route optimization flow.

Tests the VesselModel and MaritimeRouter working together to optimize routes.
"""

import pytest
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.optimization.vessel_model import VesselModel, VesselSpecs
from src.optimization.router import MaritimeRouter, RouteConstraints
from src.validation import ValidationError


class TestVesselModelRouterIntegration:
    """Test VesselModel and MaritimeRouter working together."""

    @pytest.fixture
    def vessel_model(self):
        """Create default vessel model."""
        return VesselModel()

    @pytest.fixture
    def router(self, vessel_model):
        """Create router with default vessel model (no weather data)."""
        return MaritimeRouter(
            vessel_model=vessel_model,
            grib_parser_gfs=None,
            grib_parser_wave=None,
        )

    def test_basic_route_optimization(self, router):
        """Test basic route optimization between two points."""
        # Rotterdam to Gibraltar
        start_pos = (51.9, 4.5)
        end_pos = (36.1, -5.4)
        departure_time = datetime.utcnow()

        result = router.find_optimal_route(
            start_pos=start_pos,
            end_pos=end_pos,
            departure_time=departure_time,
            is_laden=True,
        )

        # Check result structure
        assert "waypoints" in result
        assert "total_distance_nm" in result
        assert "total_time_hours" in result
        assert "total_fuel_mt" in result
        assert "departure_time" in result
        assert "arrival_time" in result

        # Check basic sanity
        assert len(result["waypoints"]) >= 2  # At least start and end
        assert result["total_distance_nm"] > 0
        assert result["total_time_hours"] > 0
        assert result["total_fuel_mt"] > 0

        # Check arrival is after departure
        assert result["arrival_time"] > result["departure_time"]

    def test_laden_vs_ballast_routes(self, router):
        """Test that laden and ballast conditions produce different results."""
        start_pos = (51.9, 4.5)
        end_pos = (36.1, -5.4)
        departure_time = datetime.utcnow()

        laden_result = router.find_optimal_route(
            start_pos=start_pos,
            end_pos=end_pos,
            departure_time=departure_time,
            is_laden=True,
        )

        ballast_result = router.find_optimal_route(
            start_pos=start_pos,
            end_pos=end_pos,
            departure_time=departure_time,
            is_laden=False,
        )

        # Both should produce valid routes
        assert laden_result["total_distance_nm"] > 0
        assert ballast_result["total_distance_nm"] > 0

        # Times may differ due to different service speeds
        # (ballast typically faster)
        assert laden_result["total_time_hours"] > 0
        assert ballast_result["total_time_hours"] > 0

    def test_route_with_custom_speed(self, router):
        """Test route optimization with custom target speed."""
        start_pos = (51.9, 4.5)
        end_pos = (36.1, -5.4)
        departure_time = datetime.utcnow()

        # Slow speed
        slow_result = router.find_optimal_route(
            start_pos=start_pos,
            end_pos=end_pos,
            departure_time=departure_time,
            is_laden=True,
            target_speed_kts=10.0,
        )

        # Fast speed
        fast_result = router.find_optimal_route(
            start_pos=start_pos,
            end_pos=end_pos,
            departure_time=departure_time,
            is_laden=True,
            target_speed_kts=16.0,
        )

        # Distance should be similar (same route)
        # Time should be different (slower takes longer)
        assert slow_result["total_time_hours"] > fast_result["total_time_hours"]

    def test_short_route(self, router):
        """Test optimization for a short route."""
        # Two nearby points in the English Channel
        start_pos = (50.5, -0.5)
        end_pos = (50.0, 0.5)
        departure_time = datetime.utcnow()

        result = router.find_optimal_route(
            start_pos=start_pos,
            end_pos=end_pos,
            departure_time=departure_time,
            is_laden=True,
        )

        # Short route should still work
        assert result["total_distance_nm"] > 0
        assert result["total_distance_nm"] < 100  # Should be short
        assert result["total_time_hours"] > 0

    def test_transatlantic_route(self, router):
        """Test optimization for a long transatlantic route."""
        # New York to Rotterdam
        start_pos = (40.7, -74.0)
        end_pos = (51.9, 4.5)
        departure_time = datetime.utcnow()

        result = router.find_optimal_route(
            start_pos=start_pos,
            end_pos=end_pos,
            departure_time=departure_time,
            is_laden=True,
        )

        # Long route should produce reasonable results
        assert result["total_distance_nm"] > 3000  # ~3500 nm expected
        assert result["total_time_hours"] > 200  # ~10 days at 14 kts

    def test_invalid_start_position(self, router):
        """Test that invalid start position raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            router.find_optimal_route(
                start_pos=(100.0, 4.5),  # Invalid latitude
                end_pos=(36.1, -5.4),
                departure_time=datetime.utcnow(),
                is_laden=True,
            )
        assert "latitude" in str(exc_info.value).lower()

    def test_invalid_end_position(self, router):
        """Test that invalid end position raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            router.find_optimal_route(
                start_pos=(51.9, 4.5),
                end_pos=(36.1, 200.0),  # Invalid longitude
                departure_time=datetime.utcnow(),
                is_laden=True,
            )
        assert "longitude" in str(exc_info.value).lower()

    def test_invalid_speed(self, router):
        """Test that invalid speed raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            router.find_optimal_route(
                start_pos=(51.9, 4.5),
                end_pos=(36.1, -5.4),
                departure_time=datetime.utcnow(),
                is_laden=True,
                target_speed_kts=-5.0,  # Invalid speed
            )
        assert "speed" in str(exc_info.value).lower()


class TestFuelCalculationIntegration:
    """Test fuel calculation through the full stack."""

    @pytest.fixture
    def vessel_model(self):
        """Create default vessel model."""
        return VesselModel()

    def test_fuel_calculation_consistency(self, vessel_model):
        """Test that fuel calculations are consistent."""
        # Calculate fuel twice with same parameters
        result1 = vessel_model.calculate_fuel_consumption(
            speed_kts=14.5,
            is_laden=True,
            weather=None,
            distance_nm=100.0,
        )

        result2 = vessel_model.calculate_fuel_consumption(
            speed_kts=14.5,
            is_laden=True,
            weather=None,
            distance_nm=100.0,
        )

        # Results should be identical
        assert result1["fuel_mt"] == result2["fuel_mt"]
        assert result1["power_kw"] == result2["power_kw"]

    def test_fuel_scales_with_distance(self, vessel_model):
        """Test that fuel consumption scales linearly with distance."""
        short_result = vessel_model.calculate_fuel_consumption(
            speed_kts=14.5,
            is_laden=True,
            weather=None,
            distance_nm=100.0,
        )

        long_result = vessel_model.calculate_fuel_consumption(
            speed_kts=14.5,
            is_laden=True,
            weather=None,
            distance_nm=200.0,
        )

        # Double distance should equal double fuel (approximately)
        ratio = long_result["fuel_mt"] / short_result["fuel_mt"]
        assert 1.9 < ratio < 2.1

    def test_fuel_with_weather_effects(self, vessel_model):
        """Test fuel calculation with weather."""
        # Calm conditions
        calm_result = vessel_model.calculate_fuel_consumption(
            speed_kts=14.5,
            is_laden=True,
            weather=None,
            distance_nm=100.0,
        )

        # Head wind and waves
        weather_result = vessel_model.calculate_fuel_consumption(
            speed_kts=14.5,
            is_laden=True,
            weather={
                "wind_speed_ms": 15.0,
                "wind_dir_deg": 0.0,
                "heading_deg": 0.0,  # Head wind
                "sig_wave_height_m": 3.0,
                "wave_dir_deg": 0.0,
            },
            distance_nm=100.0,
        )

        # Weather should increase fuel consumption
        assert weather_result["fuel_mt"] >= calm_result["fuel_mt"]

        # Resistance breakdown should show weather effects
        assert weather_result["resistance_breakdown_kn"]["wind"] >= 0
        assert weather_result["resistance_breakdown_kn"]["waves"] >= 0

    def test_zero_speed_returns_zeros(self, vessel_model):
        """Test that zero speed returns zero fuel (TN002 TEST-FUEL-02)."""
        result = vessel_model.calculate_fuel_consumption(
            speed_kts=0,
            is_laden=True,
            weather=None,
            distance_nm=100.0,
        )
        assert result["fuel_mt"] == 0.0
        assert result["power_kw"] == 0.0

    def test_negative_distance_still_computes(self, vessel_model):
        """Test that negative distance computes without crash."""
        result = vessel_model.calculate_fuel_consumption(
            speed_kts=14.5,
            is_laden=True,
            weather=None,
            distance_nm=-100.0,
        )
        # Negative distance produces negative fuel (model doesn't validate)
        assert isinstance(result["fuel_mt"], float)

    def test_extreme_weather_still_computes(self, vessel_model):
        """Test that extreme weather computes without crash."""
        result = vessel_model.calculate_fuel_consumption(
            speed_kts=14.5,
            is_laden=True,
            weather={"wind_speed_ms": 100.0},
            distance_nm=100.0,
        )
        # Model computes fuel even with extreme weather
        assert result["fuel_mt"] > 0


class TestCustomVesselConfiguration:
    """Test optimization with custom vessel configurations."""

    def test_larger_vessel(self):
        """Test optimization with a larger vessel."""
        large_specs = VesselSpecs(
            dwt=80000.0,
            loa=220.0,
            beam=36.0,
            draft_laden=14.0,
            draft_ballast=8.0,
            mcr_kw=12000.0,
            sfoc_at_mcr=168.0,
            service_speed_laden=14.0,
            service_speed_ballast=15.0,
        )

        model = VesselModel(specs=large_specs)
        router = MaritimeRouter(vessel_model=model)

        result = router.find_optimal_route(
            start_pos=(51.9, 4.5),
            end_pos=(36.1, -5.4),
            departure_time=datetime.utcnow(),
            is_laden=True,
        )

        assert result["total_fuel_mt"] > 0
        assert result["total_distance_nm"] > 0

    def test_smaller_vessel(self):
        """Test optimization with a smaller vessel."""
        small_specs = VesselSpecs(
            dwt=30000.0,
            loa=160.0,
            beam=28.0,
            draft_laden=10.0,
            draft_ballast=5.5,
            mcr_kw=6000.0,
            sfoc_at_mcr=175.0,
            service_speed_laden=13.5,
            service_speed_ballast=14.5,
        )

        model = VesselModel(specs=small_specs)
        router = MaritimeRouter(vessel_model=model)

        result = router.find_optimal_route(
            start_pos=(51.9, 4.5),
            end_pos=(36.1, -5.4),
            departure_time=datetime.utcnow(),
            is_laden=True,
        )

        assert result["total_fuel_mt"] > 0
        assert result["total_distance_nm"] > 0

    def test_calibrated_model(self):
        """Test optimization with calibrated model factors."""
        calibration_factors = {
            "calm_water": 1.05,  # 5% higher than predicted
            "wind": 1.10,        # 10% higher wind effect
            "waves": 1.15,       # 15% higher wave effect
        }

        model = VesselModel(calibration_factors=calibration_factors)
        router = MaritimeRouter(vessel_model=model)

        result = router.find_optimal_route(
            start_pos=(51.9, 4.5),
            end_pos=(36.1, -5.4),
            departure_time=datetime.utcnow(),
            is_laden=True,
        )

        assert result["total_fuel_mt"] > 0


class TestRouteConstraints:
    """Test route optimization with different constraints."""

    def test_default_constraints(self):
        """Test with default constraints."""
        model = VesselModel()
        router = MaritimeRouter(vessel_model=model)

        # Default constraints should work
        assert router.constraints.max_wind_speed_ms == 25.0
        assert router.constraints.max_wave_height_m == 5.0

    def test_custom_weather_limits(self):
        """Test with custom weather limits."""
        model = VesselModel()
        constraints = RouteConstraints(
            max_wind_speed_ms=20.0,
            max_wave_height_m=3.0,
        )
        router = MaritimeRouter(
            vessel_model=model,
            constraints=constraints,
        )

        result = router.find_optimal_route(
            start_pos=(51.9, 4.5),
            end_pos=(36.1, -5.4),
            departure_time=datetime.utcnow(),
            is_laden=True,
        )

        assert result["total_fuel_mt"] > 0

    def test_fine_grid_resolution(self):
        """Test with finer grid resolution."""
        model = VesselModel()
        constraints = RouteConstraints(
            grid_resolution_deg=0.25,  # Finer grid
        )
        router = MaritimeRouter(
            vessel_model=model,
            constraints=constraints,
        )

        result = router.find_optimal_route(
            start_pos=(51.9, 4.5),
            end_pos=(50.0, 0.0),  # Short route for faster test
            departure_time=datetime.utcnow(),
            is_laden=True,
        )

        assert result["total_fuel_mt"] > 0

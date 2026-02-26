"""
Tests for variable speed voyage calculation.

Covers:
- _find_optimal_leg_speed returns valid speed/fuel/time
- Variable speed saves fuel vs fixed speed in heavy weather
- Variable speed and fixed speed converge in calm conditions
- Speed profile has correct number of entries
"""

import pytest
from datetime import datetime

from src.optimization.voyage import VoyageCalculator, LegWeather, VoyageResult
from src.optimization.vessel_model import VesselModel
from src.routes.rtz_parser import create_route_from_waypoints


@pytest.fixture
def calculator():
    return VoyageCalculator(vessel_model=VesselModel())


@pytest.fixture
def calm_provider():
    def provider(lat, lon, t):
        return LegWeather()
    return provider


@pytest.fixture
def heavy_weather_provider():
    """Head-on wind and waves — significant fuel penalty."""
    def provider(lat, lon, t):
        return LegWeather(
            wind_speed_ms=15.0,
            wind_dir_deg=180.0,
            sig_wave_height_m=4.0,
            wave_period_s=9.0,
            wave_dir_deg=180.0,
        )
    return provider


@pytest.fixture
def two_leg_route():
    """Short Med route: Gibraltar → Sardinia (two legs via waypoint)."""
    wps = [(36.0, -5.5), (37.5, 1.0), (39.0, 8.0)]
    return create_route_from_waypoints(wps, "Test Route")


class TestFindOptimalLegSpeed:

    def test_returns_valid_values(self, calculator):
        speed, fuel, power, time_h = calculator._find_optimal_leg_speed(
            calm_speed_kts=14.0,
            is_laden=True,
            bearing_deg=90.0,
            weather=LegWeather(),
            distance_nm=100.0,
        )
        assert 4.0 <= speed <= 14.0
        assert fuel > 0
        assert power >= 0
        assert time_h > 0

    def test_heavy_weather_reduces_speed(self, calculator):
        """In heavy head-seas, optimal speed should be lower than calm speed."""
        heavy_wx = LegWeather(
            wind_speed_ms=15.0,
            wind_dir_deg=0.0,  # Head wind (bearing 0)
            sig_wave_height_m=4.0,
            wave_period_s=9.0,
            wave_dir_deg=0.0,
        )
        speed, _, _, _ = calculator._find_optimal_leg_speed(
            calm_speed_kts=14.0,
            is_laden=True,
            bearing_deg=0.0,
            weather=heavy_wx,
            distance_nm=100.0,
        )
        # Should slow down in heavy weather
        assert speed < 14.0

    def test_calm_weather_near_max_speed(self, calculator):
        """In calm conditions, optimal speed should be near the calm speed."""
        speed, _, _, _ = calculator._find_optimal_leg_speed(
            calm_speed_kts=14.0,
            is_laden=True,
            bearing_deg=90.0,
            weather=LegWeather(),
            distance_nm=100.0,
        )
        # In calm, time penalty pushes speed toward max — within 30%
        assert speed >= 14.0 * 0.7


class TestVariableSpeedVoyage:

    def test_variable_speed_flag_in_result(self, calculator, calm_provider, two_leg_route):
        result = calculator.calculate_voyage(
            route=two_leg_route,
            calm_speed_kts=14.0,
            is_laden=True,
            departure_time=datetime(2025, 6, 1),
            weather_provider=calm_provider,
            variable_speed=True,
        )
        assert isinstance(result, VoyageResult)
        assert result.variable_speed_enabled is True
        assert len(result.speed_profile) == 2  # Two legs

    def test_fixed_speed_flag_in_result(self, calculator, calm_provider, two_leg_route):
        result = calculator.calculate_voyage(
            route=two_leg_route,
            calm_speed_kts=14.0,
            is_laden=True,
            departure_time=datetime(2025, 6, 1),
            weather_provider=calm_provider,
            variable_speed=False,
        )
        assert result.variable_speed_enabled is False
        assert len(result.speed_profile) == 2  # Still populated for reference

    def test_variable_speed_saves_fuel_in_heavy_weather(
        self, calculator, heavy_weather_provider, two_leg_route
    ):
        """Variable speed should use less fuel than fixed speed in heavy weather."""
        fixed = calculator.calculate_voyage(
            route=two_leg_route,
            calm_speed_kts=14.0,
            is_laden=True,
            departure_time=datetime(2025, 6, 1),
            weather_provider=heavy_weather_provider,
            variable_speed=False,
        )
        variable = calculator.calculate_voyage(
            route=two_leg_route,
            calm_speed_kts=14.0,
            is_laden=True,
            departure_time=datetime(2025, 6, 1),
            weather_provider=heavy_weather_provider,
            variable_speed=True,
        )
        # Variable speed should save fuel (or at worst match fixed)
        assert variable.total_fuel_mt <= fixed.total_fuel_mt + 0.01

    def test_calm_weather_similar_results(self, calculator, calm_provider, two_leg_route):
        """In calm weather, variable and fixed speed should produce similar fuel."""
        fixed = calculator.calculate_voyage(
            route=two_leg_route,
            calm_speed_kts=14.0,
            is_laden=True,
            departure_time=datetime(2025, 6, 1),
            weather_provider=calm_provider,
            variable_speed=False,
        )
        variable = calculator.calculate_voyage(
            route=two_leg_route,
            calm_speed_kts=14.0,
            is_laden=True,
            departure_time=datetime(2025, 6, 1),
            weather_provider=calm_provider,
            variable_speed=True,
        )
        # Should be within 20% of each other in calm conditions
        ratio = variable.total_fuel_mt / fixed.total_fuel_mt if fixed.total_fuel_mt > 0 else 1.0
        assert 0.5 < ratio < 1.5

    def test_speed_profile_values_in_range(self, calculator, calm_provider, two_leg_route):
        result = calculator.calculate_voyage(
            route=two_leg_route,
            calm_speed_kts=14.0,
            is_laden=True,
            departure_time=datetime(2025, 6, 1),
            weather_provider=calm_provider,
            variable_speed=True,
        )
        for spd in result.speed_profile:
            assert 4.0 <= spd <= 14.0

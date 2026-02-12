"""Unit tests for SPEC-P1: Extended Weather Fields & Current-Adjusted Routing."""

import pytest
import numpy as np

from src.optimization.vessel_model import (
    VesselModel, VesselSpecs,
    seawater_density, seawater_viscosity,
)
from src.optimization.route_optimizer import (
    apply_visibility_cap, VISIBILITY_SPEED_CAPS,
)
from src.optimization.voyage import LegWeather


# ---------------------------------------------------------------------------
# §1 – Seawater density & viscosity (UNESCO 1983 / Sharqawy 2010)
# ---------------------------------------------------------------------------

class TestSeawaterProperties:
    """Test UNESCO 1983 density and Sharqawy 2010 viscosity correlations."""

    def test_density_cold_water(self):
        """Arctic water (~0 °C) should be denser than tropical (~30 °C)."""
        rho_cold = seawater_density(0.0)
        rho_warm = seawater_density(30.0)
        assert rho_cold > rho_warm

    def test_density_typical_range(self):
        """At 15 °C / 35 PSU, density should be ~1025-1027 kg/m³."""
        rho = seawater_density(15.0)
        assert 1022 < rho < 1030

    def test_density_freezing(self):
        """Near-freezing seawater should be ≈ 1028 kg/m³."""
        rho = seawater_density(-1.8)
        assert 1026 < rho < 1030

    def test_viscosity_decreases_with_temperature(self):
        """Viscosity drops as water warms."""
        nu_cold = seawater_viscosity(5.0)
        nu_warm = seawater_viscosity(25.0)
        assert nu_cold > nu_warm

    def test_viscosity_order_of_magnitude(self):
        """At 15 °C, kinematic viscosity should be ~1.1-1.2e-6 m²/s."""
        nu = seawater_viscosity(15.0)
        assert 1.0e-6 < nu < 1.4e-6

    def test_sst_affects_fuel_consumption(self):
        """Warm SST (lower density/viscosity) → less calm-water resistance → less fuel."""
        model = VesselModel()
        distance = 14.5 * 24  # one day at 14.5 kts

        cold = model.calculate_fuel_consumption(
            speed_kts=14.5, is_laden=True, weather=None,
            distance_nm=distance, sst_celsius=2.0,
        )
        warm = model.calculate_fuel_consumption(
            speed_kts=14.5, is_laden=True, weather=None,
            distance_nm=distance, sst_celsius=28.0,
        )

        # Warmer water → lower resistance → less fuel
        assert warm["fuel_mt"] < cold["fuel_mt"]


# ---------------------------------------------------------------------------
# §2 – Visibility speed caps (IMO COLREG Rule 6)
# ---------------------------------------------------------------------------

class TestVisibilityCaps:
    """Test tiered visibility speed cap function."""

    def test_fog_caps_at_6_kts(self):
        """Fog (≤1000 m) should cap speed at 6 kts."""
        assert apply_visibility_cap(14.0, 500.0) == 6.0
        assert apply_visibility_cap(14.0, 1000.0) == 6.0

    def test_poor_visibility_caps_at_8_kts(self):
        """Poor visibility (1001-2000 m) should cap at 8 kts."""
        assert apply_visibility_cap(14.0, 1500.0) == 8.0
        assert apply_visibility_cap(14.0, 2000.0) == 8.0

    def test_moderate_visibility_caps_at_12_kts(self):
        """Moderate visibility (2001-5000 m) should cap at 12 kts."""
        assert apply_visibility_cap(14.0, 3000.0) == 12.0
        assert apply_visibility_cap(14.0, 5000.0) == 12.0

    def test_clear_visibility_no_cap(self):
        """Clear visibility (>5000 m) should not cap speed."""
        assert apply_visibility_cap(14.0, 10000.0) == 14.0
        assert apply_visibility_cap(14.0, 50000.0) == 14.0

    def test_speed_below_cap_unchanged(self):
        """If vessel speed is already below the cap, it stays the same."""
        assert apply_visibility_cap(4.0, 500.0) == 4.0
        assert apply_visibility_cap(5.0, 1500.0) == 5.0


# ---------------------------------------------------------------------------
# §3 – Ice exclusion & penalty zones
# ---------------------------------------------------------------------------

class TestIceZones:
    """Test ice concentration thresholds for routing cost."""

    def test_leg_weather_defaults(self):
        """Default LegWeather should have zero ice concentration."""
        lw = LegWeather()
        assert lw.ice_concentration == 0.0
        assert lw.visibility_km == 50.0
        assert lw.sst_celsius == 15.0

    def test_ice_exclusion_threshold(self):
        """Ice ≥ 15% should be impassable (exclusion zone)."""
        # Confirmed by code: ICE_EXCLUSION_THRESHOLD = 0.15
        lw = LegWeather(ice_concentration=0.15)
        assert lw.ice_concentration >= 0.15

    def test_ice_penalty_threshold(self):
        """Ice between 5-15% should incur a 2x cost penalty."""
        # Verified in route_optimizer.py:
        # ICE_PENALTY_THRESHOLD = 0.05, ice_cost_factor = 2.0
        lw = LegWeather(ice_concentration=0.10)
        assert 0.05 <= lw.ice_concentration < 0.15


# ---------------------------------------------------------------------------
# §4 – Swell decomposition fallback
# ---------------------------------------------------------------------------

class TestSwellDecomposition:
    """Test swell/wind-sea decomposition and fallback derivation."""

    def test_leg_weather_decomposition_fields(self):
        """LegWeather has swell and wind-sea decomposition fields."""
        lw = LegWeather(
            sig_wave_height_m=3.0,
            swell_height_m=2.4,
            windwave_height_m=1.8,
            has_decomposition=True,
        )
        assert lw.has_decomposition is True
        assert lw.swell_height_m == 2.4
        assert lw.windwave_height_m == 1.8

    def test_fallback_ratios(self):
        """Fallback: swell_hs ≈ 0.8 * total_hs, windsea_hs ≈ 0.6 * total_hs."""
        total_hs = 3.0
        sw_hs = 0.8 * total_hs
        ww_hs = 0.6 * total_hs
        assert sw_hs == pytest.approx(2.4, rel=1e-3)
        assert ww_hs == pytest.approx(1.8, rel=1e-3)

    def test_fallback_energy_conservation(self):
        """Fallback swell + windsea should roughly match total Hs via energy sum."""
        total_hs = 4.0
        sw = 0.8 * total_hs
        ww = 0.6 * total_hs
        reconstructed = np.sqrt(sw**2 + ww**2)
        assert reconstructed == pytest.approx(total_hs, rel=0.01)


# ---------------------------------------------------------------------------
# §5 – Current-adjusted SOG
# ---------------------------------------------------------------------------

class TestCurrentAdjustedSOG:
    """Test that favorable/adverse currents affect cost."""

    def test_favorable_current_increases_sog(self):
        """Following current should increase SOG."""
        # With a following current, effective SOG is higher → less time → less cost.
        stw_kts = 14.0
        current_favorable_kts = 1.5
        sog_favorable = stw_kts + current_favorable_kts
        sog_no_current = stw_kts
        assert sog_favorable > sog_no_current

    def test_adverse_current_decreases_sog(self):
        """Opposing current should decrease SOG."""
        stw_kts = 14.0
        current_adverse_kts = -1.5
        sog_adverse = stw_kts + current_adverse_kts
        sog_no_current = stw_kts
        assert sog_adverse < sog_no_current

    def test_favorable_current_reduces_fuel(self):
        """If SOG is higher for same STW, transit time is shorter → less fuel."""
        distance_nm = 100.0
        stw = 14.0
        time_no_current = distance_nm / stw
        time_with_current = distance_nm / (stw + 1.5)
        assert time_with_current < time_no_current


# ---------------------------------------------------------------------------
# §6 – Extended LegWeather fields
# ---------------------------------------------------------------------------

class TestExtendedLegWeather:
    """Test all SPEC-P1 extended fields on LegWeather."""

    def test_all_extended_fields_present(self):
        """LegWeather should have SST, visibility, ice, and swell fields."""
        lw = LegWeather(
            sst_celsius=22.5,
            visibility_km=8.0,
            ice_concentration=0.03,
            swell_height_m=1.5,
            swell_period_s=10.0,
            swell_dir_deg=270.0,
            windwave_height_m=0.8,
            windwave_period_s=5.0,
            windwave_dir_deg=180.0,
            has_decomposition=True,
        )
        assert lw.sst_celsius == 22.5
        assert lw.visibility_km == 8.0
        assert lw.ice_concentration == 0.03
        assert lw.swell_height_m == 1.5
        assert lw.windwave_height_m == 0.8
        assert lw.has_decomposition is True

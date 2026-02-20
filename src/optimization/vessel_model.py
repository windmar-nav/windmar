"""
Vessel fuel consumption model for merchant ships.

Implements physics-based model using:
- Holtrop-Mennen resistance prediction
- SFOC curves for main engine
- Weather effects (wind, waves)
- Laden vs ballast conditions
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seawater property functions (SPEC-P1)
# ---------------------------------------------------------------------------

def seawater_density(sst_celsius: float) -> float:
    """UNESCO 1983 simplified equation of state (salinity=35 PSU)."""
    t = sst_celsius
    rho_fw = 999.842594 + 6.793952e-2 * t - 9.095290e-3 * t**2 + 1.001685e-4 * t**3
    return rho_fw + 0.824493 * 35 - 4.0899e-3 * 35 * t  # ~1022-1028 range


def seawater_viscosity(sst_celsius: float) -> float:
    """Kinematic viscosity of seawater (Sharqawy 2010 correlation)."""
    t = sst_celsius
    mu = 1.7910 - 6.144e-2 * t + 1.4510e-3 * t**2 - 1.6826e-5 * t**3  # mPa·s
    rho = seawater_density(t)
    return (mu * 1e-3) / rho  # m²/s


@dataclass
class VesselSpecs:
    """Vessel specifications. Defaults are for an MR Product Tanker (49k DWT)."""

    # Dimensions
    loa: float = 183.0  # Length overall (m)
    lpp: float = 176.0  # Length between perpendiculars (m)
    beam: float = 32.0  # Beam (m)
    draft_laden: float = 11.8  # Draft laden (m)
    draft_ballast: float = 6.5  # Draft ballast (m)
    dwt: float = 49000.0  # Deadweight tonnage (MT)
    displacement_laden: float = 65000.0  # Displacement laden (MT)
    displacement_ballast: float = 20000.0  # Displacement ballast (MT)

    # Block coefficient estimates
    cb_laden: float = 0.82  # Block coefficient laden
    cb_ballast: float = 0.75  # Block coefficient ballast

    # Wetted surface area (m²)
    wetted_surface_laden: float = 7500.0
    wetted_surface_ballast: float = 5200.0

    # Main engine
    mcr_kw: float = 8840.0  # Maximum continuous rating (kW)
    sfoc_at_mcr: float = 171.0  # Specific fuel oil consumption at MCR (g/kWh)

    # Service speeds
    service_speed_laden: float = 14.5  # Service speed laden (knots)
    service_speed_ballast: float = 15.0  # Service speed ballast (knots)

    # Frontal area for wind resistance
    frontal_area_laden: float = 450.0  # Above water frontal area laden (m²)
    frontal_area_ballast: float = 850.0  # Above water frontal area ballast (m²)

    # Lateral area for drift
    lateral_area_laden: float = 2100.0  # Lateral area laden (m²)
    lateral_area_ballast: float = 2800.0  # Lateral area ballast (m²)


class VesselModel:
    """
    Physics-based fuel consumption model for merchant ships.

    Calculates fuel consumption based on vessel specs, speed,
    loading condition, and weather conditions. Defaults to MR
    Product Tanker parameters; all specs are configurable.
    """

    # Seawater properties
    RHO_SW = 1025.0  # Seawater density (kg/m³)
    NU_SW = 1.19e-6  # Kinematic viscosity (m²/s at 15°C)

    # Air properties
    RHO_AIR = 1.225  # Air density (kg/m³)

    # Propulsion efficiency
    PROP_EFFICIENCY = 0.65  # Propeller efficiency
    HULL_EFFICIENCY = 1.05  # Hull efficiency factor
    RELATIVE_ROTATIVE_EFF = 1.00  # Relative rotative efficiency

    def __init__(
        self,
        specs: Optional[VesselSpecs] = None,
        calibration_factors: Optional[Dict[str, float]] = None,
        wave_method: str = "stawave1",
    ):
        """
        Initialize vessel model.

        Args:
            specs: Vessel specifications (defaults to MR tanker)
            calibration_factors: Optional calibration factors from noon reports
            wave_method: Wave added resistance method ('stawave1' or 'kwon')
        """
        self.specs = specs or VesselSpecs()
        self.calibration_factors = calibration_factors or {
            "calm_water": 1.0,
            "wind": 1.0,
            "waves": 1.0,
            "sfoc_factor": 1.0,
        }
        if wave_method not in ("stawave1", "kwon"):
            raise ValueError(f"Unknown wave_method: {wave_method!r}")
        self.wave_method = wave_method

    def calculate_fuel_consumption(
        self,
        speed_kts: float,
        is_laden: bool,
        weather: Optional[Dict[str, float]] = None,
        distance_nm: float = 1.0,
        sst_celsius: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Calculate fuel consumption for a voyage segment.

        Args:
            speed_kts: Vessel speed through water (knots)
            is_laden: True if laden, False if ballast
            weather: Weather conditions dict (wind_speed_ms, wind_dir_deg,
                     sig_wave_height_m, wave_dir_deg, heading_deg)
            distance_nm: Distance traveled (nautical miles)
            sst_celsius: Optional SST for dynamic seawater properties (SPEC-P1)

        Returns:
            Dictionary with:
                - fuel_mt: Total fuel consumed (metric tons)
                - power_kw: Engine power required (kW)
                - time_hours: Time taken (hours)
                - fuel_breakdown: Breakdown by component
        """
        # Guard against zero/negative speed (TN002 TEST-FUEL-02)
        if speed_kts <= 0:
            return {
                "fuel_mt": 0.0,
                "power_kw": 0.0,
                "required_power_kw": 0.0,
                "time_hours": 0.0,
                "fuel_breakdown": {"calm_water": 0.0, "wind": 0.0, "waves": 0.0},
                "resistance_breakdown_kn": {
                    "calm_water": 0.0, "wind": 0.0, "waves": 0.0, "total": 0.0,
                },
            }

        # Convert speed to m/s
        speed_ms = speed_kts * 0.51444

        # Get vessel parameters for loading condition
        draft = self.specs.draft_laden if is_laden else self.specs.draft_ballast
        displacement = (
            self.specs.displacement_laden if is_laden
            else self.specs.displacement_ballast
        )
        cb = self.specs.cb_laden if is_laden else self.specs.cb_ballast
        wetted_surface = (
            self.specs.wetted_surface_laden if is_laden
            else self.specs.wetted_surface_ballast
        )

        # Calculate calm water resistance (with SST-corrected properties when available)
        resistance_calm = self._holtrop_mennen_resistance(
            speed_ms, draft, displacement, cb, wetted_surface,
            sst_celsius=sst_celsius,
        )

        # Add wind resistance
        resistance_wind = 0.0
        if weather and "wind_speed_ms" in weather:
            resistance_wind = self._wind_resistance(
                weather["wind_speed_ms"],
                weather.get("wind_dir_deg", 0),
                weather.get("heading_deg", 0),
                is_laden,
            )

        # Add wave resistance
        resistance_waves = 0.0
        if weather and "sig_wave_height_m" in weather:
            resistance_waves = self._wave_resistance(
                weather["sig_wave_height_m"],
                weather.get("wave_dir_deg", 0),
                weather.get("heading_deg", 0),
                speed_ms,
                is_laden,
            )

        # Total resistance
        total_resistance = (
            resistance_calm * self.calibration_factors["calm_water"]
            + resistance_wind * self.calibration_factors["wind"]
            + resistance_waves * self.calibration_factors["waves"]
        )

        # Calculate required power
        tow_power_kw = (total_resistance * speed_ms) / 1000.0  # kW

        # Account for propulsion efficiencies
        brake_power_kw = tow_power_kw / (
            self.PROP_EFFICIENCY
            * self.HULL_EFFICIENCY
            * self.RELATIVE_ROTATIVE_EFF
        )

        # Store uncapped power (needed for speed reduction calculations)
        required_brake_power_kw = brake_power_kw

        # Ensure power is within engine limits for fuel calculation
        brake_power_kw = min(brake_power_kw, self.specs.mcr_kw)

        # Calculate SFOC at this load
        load_fraction = brake_power_kw / self.specs.mcr_kw
        sfoc = self._sfoc_curve(load_fraction)

        # Calculate time and fuel
        time_hours = distance_nm / speed_kts
        # SFOC is in g/kWh, so result is in grams
        fuel_grams = brake_power_kw * sfoc * time_hours
        fuel_mt = fuel_grams / 1_000_000.0  # grams to metric tons

        return {
            "fuel_mt": fuel_mt,
            "power_kw": brake_power_kw,
            "required_power_kw": required_brake_power_kw,
            "time_hours": time_hours,
            "fuel_breakdown": {
                "calm_water": (resistance_calm / total_resistance) * fuel_mt
                if total_resistance > 0
                else fuel_mt,
                "wind": (resistance_wind / total_resistance) * fuel_mt
                if total_resistance > 0
                else 0.0,
                "waves": (resistance_waves / total_resistance) * fuel_mt
                if total_resistance > 0
                else 0.0,
            },
            "resistance_breakdown_kn": {
                "calm_water": resistance_calm / 1000.0,
                "wind": resistance_wind / 1000.0,
                "waves": resistance_waves / 1000.0,
                "total": total_resistance / 1000.0,
            },
        }

    def _holtrop_mennen_resistance(
        self,
        speed_ms: float,
        draft: float,
        displacement: float,
        cb: float,
        wetted_surface: float,
        sst_celsius: Optional[float] = None,
    ) -> float:
        """
        Calculate calm water resistance using Holtrop-Mennen method.

        Simplified version for tankers. When sst_celsius is provided,
        uses SST-corrected seawater density and viscosity (SPEC-P1).

        Args:
            speed_ms: Speed (m/s)
            draft: Draft (m)
            displacement: Displacement (MT)
            cb: Block coefficient
            wetted_surface: Wetted surface area (m²)
            sst_celsius: Optional sea surface temperature for dynamic rho/nu

        Returns:
            Total resistance (N)
        """
        # Seawater properties — dynamic from SST when available (SPEC-P1)
        if sst_celsius is not None:
            rho_sw = seawater_density(sst_celsius)
            nu_sw = seawater_viscosity(sst_celsius)
        else:
            rho_sw = self.RHO_SW
            nu_sw = self.NU_SW

        # Calculate Froude number
        froude = speed_ms / np.sqrt(9.81 * self.specs.lpp)

        # Calculate Reynolds number
        reynolds = speed_ms * self.specs.lpp / nu_sw

        # Frictional resistance coefficient (ITTC 1957)
        cf = 0.075 / (np.log10(reynolds) - 2) ** 2

        # Hull roughness allowance (ITTC standard for in-service hull)
        delta_cf = 0.00025

        # Form factor (Holtrop-Mennen for tankers)
        k1 = (
            0.93
            + 0.4871 * (self.specs.beam / self.specs.lpp)
            - 0.2156 * (self.specs.beam / draft)
            + 0.1027 * cb
        )
        k1 = max(0.1, k1)  # Floor for extreme B/T ratios (e.g. ballast)

        # Frictional resistance (including hull roughness)
        rf = 0.5 * rho_sw * speed_ms**2 * wetted_surface * (cf + delta_cf) * (1 + k1)

        # Wave-making resistance (empirical for full-form ships)
        # For tankers (CB > 0.75) at low Froude numbers (Fn < 0.25),
        # Rw is a small fraction of total resistance, scaling with Fn².
        rw_ratio = 4.0 * froude**2
        rw = rw_ratio * rf

        # Appendage resistance (rudder, bilge keels ~5% of frictional)
        rapp = 0.05 * rf

        # Total resistance
        total_resistance = rf + rw + rapp

        return total_resistance

    def _wind_resistance(
        self,
        wind_speed_ms: float,
        wind_dir_deg: float,
        heading_deg: float,
        is_laden: bool,
    ) -> float:
        """
        Calculate wind resistance using Blendermann method.

        Relative angle convention: 0° = headwind, 180° = tailwind.

        Args:
            wind_speed_ms: True wind speed (m/s)
            wind_dir_deg: True wind direction (coming from, degrees)
            heading_deg: Vessel heading (degrees)
            is_laden: Loading condition

        Returns:
            Wind resistance (N), always >= 0
        """
        # Relative angle: 0° = headwind, 90° = beam, 180° = tailwind
        relative_angle = abs(((wind_dir_deg - heading_deg) + 180) % 360 - 180)
        relative_angle_rad = np.radians(relative_angle)

        frontal_area = (
            self.specs.frontal_area_laden if is_laden
            else self.specs.frontal_area_ballast
        )
        lateral_area = (
            self.specs.lateral_area_laden if is_laden
            else self.specs.lateral_area_ballast
        )

        # Longitudinal drag coefficient (Blendermann for merchant vessels)
        # Headwind (0°) → max drag ~0.8, tailwind (180°) → 0 drag
        cx_drag = 0.8 * np.cos(relative_angle_rad)
        direct_resistance = (
            max(0.0, cx_drag)
            * 0.5
            * self.RHO_AIR
            * wind_speed_ms**2
            * frontal_area
        )

        # Transverse wind creates drift, adding ~10% to effective resistance
        cy = 0.9 * abs(np.sin(relative_angle_rad))
        drift_resistance = (
            0.1
            * cy
            * 0.5
            * self.RHO_AIR
            * wind_speed_ms**2
            * lateral_area
        )

        return direct_resistance + drift_resistance

    def _wave_resistance(
        self,
        sig_wave_height_m: float,
        wave_dir_deg: float,
        heading_deg: float,
        speed_ms: float,
        is_laden: bool,
    ) -> float:
        """
        Calculate added resistance in waves.

        Dispatches to STAWAVE-1 (ISO 15016) or Kwon's method based on
        self.wave_method setting.

        Args:
            sig_wave_height_m: Significant wave height (m)
            wave_dir_deg: Wave direction (coming from, degrees)
            heading_deg: Vessel heading (degrees)
            speed_ms: Vessel speed (m/s)
            is_laden: Loading condition

        Returns:
            Added wave resistance (N)
        """
        if self.wave_method == "kwon":
            return self._kwon_wave_resistance(
                sig_wave_height_m, wave_dir_deg, heading_deg, speed_ms, is_laden,
            )
        return self._stawave1_wave_resistance(
            sig_wave_height_m, wave_dir_deg, heading_deg, speed_ms, is_laden,
        )

    def _stawave1_wave_resistance(
        self,
        sig_wave_height_m: float,
        wave_dir_deg: float,
        heading_deg: float,
        speed_ms: float,
        is_laden: bool,
    ) -> float:
        """
        STAWAVE-1 added resistance in waves (ISO 15016).

        Args:
            sig_wave_height_m: Significant wave height (m)
            wave_dir_deg: Wave direction (coming from, degrees)
            heading_deg: Vessel heading (degrees)
            speed_ms: Vessel speed (m/s)
            is_laden: Loading condition

        Returns:
            Added wave resistance (N)
        """
        # Relative angle: 0° = head seas, 180° = following seas
        relative_angle = abs(((wave_dir_deg - heading_deg) + 180) % 360 - 180)
        relative_angle_rad = np.radians(relative_angle)

        # Directional factor (head seas = 1, following seas = 0)
        directional_factor = (1 + np.cos(relative_angle_rad)) / 2

        # STAWAVE-1 added resistance in waves
        # R_AW = (1/16) * rho * g * Hs² * B * sqrt(B/Lpp) * alpha_BK
        alpha_bk = 1.0  # Block coefficient correction (~1.0 for CB > 0.75)
        raw = (
            (1.0 / 16.0)
            * self.RHO_SW
            * 9.81
            * sig_wave_height_m**2
            * self.specs.beam
            * np.sqrt(self.specs.beam / self.specs.lpp)
            * alpha_bk
            * directional_factor
        )

        return raw

    def _kwon_wave_resistance(
        self,
        sig_wave_height_m: float,
        wave_dir_deg: float,
        heading_deg: float,
        speed_ms: float,
        is_laden: bool,
    ) -> float:
        """
        Kwon's method for added resistance in waves (TN001).

        Estimates involuntary speed loss as a percentage, then converts
        to an equivalent added resistance using the cubic power-speed
        relationship.

        Args:
            sig_wave_height_m: Significant wave height (m)
            wave_dir_deg: Wave direction (coming from, degrees)
            heading_deg: Vessel heading (degrees)
            speed_ms: Vessel speed (m/s)
            is_laden: Loading condition

        Returns:
            Added wave resistance (N)
        """
        if speed_ms <= 0:
            return 0.0

        # Relative angle: 0° = head seas, 180° = following seas
        relative_angle = abs(((wave_dir_deg - heading_deg) + 180) % 360 - 180)

        cb = self.specs.cb_laden if is_laden else self.specs.cb_ballast
        lpp = self.specs.lpp

        # Kwon base speed loss coefficient (% per metre Hs)
        # For tankers/bulk carriers (CB > 0.75): ~3% per metre Hs head seas
        # Adjusted by Cb and Lpp
        cb_factor = 1.7 - 0.9 * cb  # Higher CB = less speed loss
        length_factor = max(0.5, min(1.5, 180.0 / lpp))  # Shorter ship = more loss

        base_loss_pct_per_m = 3.0 * cb_factor * length_factor

        # Directional reduction factor
        # Head seas (0°): 1.0, beam seas (90°): 0.7, following seas (180°): 0.2
        if relative_angle <= 30:
            dir_factor = 1.0
        elif relative_angle <= 60:
            dir_factor = 0.9
        elif relative_angle <= 90:
            dir_factor = 0.7
        elif relative_angle <= 150:
            dir_factor = 0.4
        else:
            dir_factor = 0.2

        # Speed loss percentage
        delta_v_pct = base_loss_pct_per_m * sig_wave_height_m * dir_factor
        delta_v_pct = min(delta_v_pct, 50.0)  # Cap at 50%

        # Convert speed loss to equivalent added resistance:
        # P ∝ V³ → ΔP/P ≈ 3 * ΔV/V for small losses
        # R = P/V → ΔR/R ≈ 2 * ΔV/V (since P = R*V)
        # For added resistance at constant speed:
        # R_added = R_calm * (2 * delta_v_pct / 100)
        # We approximate R_calm from tow power at current speed
        draft = self.specs.draft_laden if is_laden else self.specs.draft_ballast
        displacement = (
            self.specs.displacement_laden if is_laden
            else self.specs.displacement_ballast
        )
        wetted_surface = (
            self.specs.wetted_surface_laden if is_laden
            else self.specs.wetted_surface_ballast
        )

        r_calm = self._holtrop_mennen_resistance(
            speed_ms, draft, displacement, cb, wetted_surface,
        )

        r_wave = r_calm * 2.0 * (delta_v_pct / 100.0)
        return r_wave

    def _sfoc_curve(self, load_fraction: float) -> float:
        """
        Calculate specific fuel oil consumption at given load.

        Uses typical 2-stroke diesel SFOC curve, scaled by calibration sfoc_factor.

        Args:
            load_fraction: Engine load as fraction of MCR (0-1)

        Returns:
            SFOC in g/kWh
        """
        # Ensure load is within reasonable range
        load_fraction = max(0.15, min(1.0, load_fraction))

        # Typical SFOC curve for modern 2-stroke diesel
        # SFOC is optimal around 75-85% load
        if load_fraction < 0.75:
            # Below optimal load, SFOC increases
            sfoc = self.specs.sfoc_at_mcr * (1.0 + 0.15 * (0.75 - load_fraction))
        else:
            # At and above optimal load
            sfoc = self.specs.sfoc_at_mcr * (1.0 + 0.05 * (load_fraction - 0.75))

        # Apply calibration SFOC factor (engine degradation / actual measurement)
        sfoc *= self.calibration_factors.get('sfoc_factor', 1.0)

        return sfoc

    def get_optimal_speed(
        self,
        is_laden: bool,
        weather: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Calculate optimal speed for fuel efficiency.

        Args:
            is_laden: Loading condition
            weather: Weather conditions

        Returns:
            Optimal speed in knots
        """
        # Find speed at optimal engine efficiency (minimum SFOC).
        # The SFOC curve has its optimum at ~75-85% MCR load; the speed
        # that produces that load is the most fuel-efficient operating point.
        service_speed = (
            self.specs.service_speed_laden if is_laden
            else self.specs.service_speed_ballast
        )

        speeds = np.linspace(service_speed - 3, service_speed + 2, 20)
        sfoc_values = []

        for speed in speeds:
            result = self.calculate_fuel_consumption(
                speed, is_laden, weather, distance_nm=1.0
            )
            load = result["power_kw"] / self.specs.mcr_kw
            sfoc_values.append(self._sfoc_curve(load))

        # Find speed at minimum SFOC (best engine efficiency)
        optimal_idx = int(np.argmin(sfoc_values))
        return float(speeds[optimal_idx])

    def predict_performance(
        self,
        is_laden: bool,
        weather: Optional[Dict[str, float]] = None,
        engine_load_pct: float = 85.0,
        current_speed_ms: float = 0.0,
        current_dir_deg: float = 0.0,
        heading_deg: float = 0.0,
    ) -> Dict[str, float]:
        """
        Predict achievable speed and fuel consumption for given conditions.

        Solves the inverse problem: given a target engine power output and
        weather conditions, find the equilibrium speed where required brake
        power equals target power. Then applies ocean current for SOG.

        Args:
            is_laden: Loading condition
            weather: Weather conditions dict (wind_speed_ms, wind_dir_deg,
                     sig_wave_height_m, wave_dir_deg, heading_deg)
            engine_load_pct: Target engine load as % of MCR (0-100)
            current_speed_ms: Ocean current speed (m/s)
            current_dir_deg: Ocean current direction (flowing toward, degrees)
            heading_deg: Vessel heading (degrees, 0=North)

        Returns:
            Dictionary with:
                - stw_kts: Speed through water (knots)
                - sog_kts: Speed over ground (knots, after current)
                - fuel_per_day_mt: Daily fuel consumption (MT/day)
                - fuel_per_nm_mt: Fuel per nautical mile (MT/nm)
                - power_kw: Engine brake power (kW)
                - load_pct: Actual engine load (%)
                - sfoc_gkwh: SFOC at this load (g/kWh)
                - resistance_breakdown_kn: Resistance by component (kN)
                - speed_loss_from_service_pct: % speed loss vs calm water
                - current_effect_kts: Current contribution to SOG (kts)
        """
        engine_load_pct = max(15.0, min(100.0, engine_load_pct))
        target_power_kw = self.specs.mcr_kw * (engine_load_pct / 100.0)

        # Inject heading into weather dict for consistent angle calculations
        if weather:
            weather = dict(weather)
            weather['heading_deg'] = heading_deg

        # Bisection: find speed where required_power = target_power
        # Required power increases monotonically with speed (cubic-ish)
        v_lo, v_hi = 2.0, 25.0  # knots search range

        def _power_at_speed(speed_kts: float) -> float:
            """Required brake power at this speed."""
            r = self.calculate_fuel_consumption(
                speed_kts, is_laden, weather, distance_nm=1.0,
            )
            return r["required_power_kw"]

        # Ensure target is within achievable range
        p_lo = _power_at_speed(v_lo)
        p_hi = _power_at_speed(v_hi)

        if target_power_kw <= p_lo:
            # Even minimum speed exceeds target — use minimum
            stw_kts = v_lo
        elif target_power_kw >= p_hi:
            # Target exceeds maximum speed power — cap at v_hi
            stw_kts = v_hi
        else:
            # Bisection (30 iterations → precision ~0.001 kts)
            for _ in range(30):
                v_mid = (v_lo + v_hi) / 2.0
                p_mid = _power_at_speed(v_mid)
                if p_mid < target_power_kw:
                    v_lo = v_mid
                else:
                    v_hi = v_mid
            stw_kts = (v_lo + v_hi) / 2.0

        # Calculate full results at equilibrium speed
        result = self.calculate_fuel_consumption(
            stw_kts, is_laden, weather, distance_nm=stw_kts * 24,
        )

        # Current effect on SOG
        # Project current along vessel heading
        current_effect_kts = 0.0
        if current_speed_ms > 0:
            # Current direction is "flowing toward" — same convention as heading
            relative_current_angle = math.radians(current_dir_deg - heading_deg)
            current_along_kts = (current_speed_ms / 0.51444) * math.cos(relative_current_angle)
            current_effect_kts = current_along_kts

        sog_kts = max(0.0, stw_kts + current_effect_kts)

        # Fuel metrics
        fuel_per_day_mt = result["fuel_mt"]  # Already 24h distance
        fuel_per_nm_mt = fuel_per_day_mt / (sog_kts * 24) if sog_kts > 0 else 0.0

        # Speed loss from calm-water service speed
        service_speed = (
            self.specs.service_speed_laden if is_laden
            else self.specs.service_speed_ballast
        )
        # Calm-water speed at same power
        calm_result = self.calculate_fuel_consumption(
            stw_kts, is_laden, weather=None, distance_nm=1.0,
        )
        # Find what speed we'd get in calm water at same power
        calm_stw = stw_kts  # Start with same speed
        if weather:
            # Re-run bisection without weather
            v_lo_c, v_hi_c = 2.0, 25.0
            for _ in range(30):
                v_mid = (v_lo_c + v_hi_c) / 2.0
                r = self.calculate_fuel_consumption(v_mid, is_laden, None, distance_nm=1.0)
                if r["required_power_kw"] < target_power_kw:
                    v_lo_c = v_mid
                else:
                    v_hi_c = v_mid
            calm_stw = (v_lo_c + v_hi_c) / 2.0

        speed_loss_pct = ((calm_stw - stw_kts) / calm_stw * 100) if calm_stw > 0 else 0.0

        load_pct = result["power_kw"] / self.specs.mcr_kw * 100
        sfoc = self._sfoc_curve(result["power_kw"] / self.specs.mcr_kw)

        return {
            "stw_kts": round(stw_kts, 2),
            "sog_kts": round(sog_kts, 2),
            "fuel_per_day_mt": round(fuel_per_day_mt, 3),
            "fuel_per_nm_mt": round(fuel_per_nm_mt, 4),
            "power_kw": round(result["power_kw"], 0),
            "load_pct": round(load_pct, 1),
            "sfoc_gkwh": round(sfoc, 1),
            "resistance_breakdown_kn": result["resistance_breakdown_kn"],
            "speed_loss_from_weather_pct": round(max(0, speed_loss_pct), 1),
            "calm_water_speed_kts": round(calm_stw, 2),
            "current_effect_kts": round(current_effect_kts, 2),
            "service_speed_kts": service_speed,
        }

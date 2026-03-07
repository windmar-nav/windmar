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

from src.optimization import numba_kernels as nk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seawater property functions (SPEC-P1)
# ---------------------------------------------------------------------------


def seawater_density(sst_celsius: float) -> float:
    """UNESCO 1983 simplified equation of state (salinity=35 PSU)."""
    return nk.seawater_density(sst_celsius)


def seawater_viscosity(sst_celsius: float) -> float:
    """Kinematic viscosity of seawater (Sharqawy 2010 correlation)."""
    return nk.seawater_viscosity(sst_celsius)


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
                    "calm_water": 0.0,
                    "wind": 0.0,
                    "waves": 0.0,
                    "total": 0.0,
                },
            }

        # Convert speed to m/s
        speed_ms = speed_kts * 0.51444

        # Get vessel parameters for loading condition
        draft = self.specs.draft_laden if is_laden else self.specs.draft_ballast
        displacement = (
            self.specs.displacement_laden
            if is_laden
            else self.specs.displacement_ballast
        )
        cb = self.specs.cb_laden if is_laden else self.specs.cb_ballast
        wetted_surface = (
            self.specs.wetted_surface_laden
            if is_laden
            else self.specs.wetted_surface_ballast
        )

        # Calculate calm water resistance (with SST-corrected properties when available)
        resistance_calm = self._holtrop_mennen_resistance(
            speed_ms,
            draft,
            displacement,
            cb,
            wetted_surface,
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
            self.PROP_EFFICIENCY * self.HULL_EFFICIENCY * self.RELATIVE_ROTATIVE_EFF
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
                "calm_water": (
                    (resistance_calm / total_resistance) * fuel_mt
                    if total_resistance > 0
                    else fuel_mt
                ),
                "wind": (
                    (resistance_wind / total_resistance) * fuel_mt
                    if total_resistance > 0
                    else 0.0
                ),
                "waves": (
                    (resistance_waves / total_resistance) * fuel_mt
                    if total_resistance > 0
                    else 0.0
                ),
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
        if sst_celsius is not None:
            rho_sw = nk.seawater_density(sst_celsius)
            nu_sw = nk.seawater_viscosity(sst_celsius)
        else:
            rho_sw = self.RHO_SW
            nu_sw = self.NU_SW

        return nk.holtrop_mennen_resistance(
            speed_ms,
            draft,
            displacement,
            cb,
            wetted_surface,
            self.specs.lpp,
            self.specs.beam,
            rho_sw,
            nu_sw,
        )

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
        frontal_area = (
            self.specs.frontal_area_laden
            if is_laden
            else self.specs.frontal_area_ballast
        )
        lateral_area = (
            self.specs.lateral_area_laden
            if is_laden
            else self.specs.lateral_area_ballast
        )

        return nk.wind_resistance(
            wind_speed_ms,
            wind_dir_deg,
            heading_deg,
            frontal_area,
            lateral_area,
            self.RHO_AIR,
        )

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
                sig_wave_height_m,
                wave_dir_deg,
                heading_deg,
                speed_ms,
                is_laden,
            )
        return self._stawave1_wave_resistance(
            sig_wave_height_m,
            wave_dir_deg,
            heading_deg,
            speed_ms,
            is_laden,
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
        return nk.stawave1_wave_resistance(
            sig_wave_height_m,
            wave_dir_deg,
            heading_deg,
            speed_ms,
            self.specs.beam,
            self.specs.lpp,
            self.RHO_SW,
        )

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

        cb = self.specs.cb_laden if is_laden else self.specs.cb_ballast

        delta_v_pct = nk.kwon_speed_loss_pct(
            sig_wave_height_m,
            wave_dir_deg,
            heading_deg,
            cb,
            self.specs.lpp,
        )

        # Convert speed loss to equivalent added resistance
        draft = self.specs.draft_laden if is_laden else self.specs.draft_ballast
        displacement = (
            self.specs.displacement_laden
            if is_laden
            else self.specs.displacement_ballast
        )
        wetted_surface = (
            self.specs.wetted_surface_laden
            if is_laden
            else self.specs.wetted_surface_ballast
        )

        r_calm = self._holtrop_mennen_resistance(
            speed_ms,
            draft,
            displacement,
            cb,
            wetted_surface,
        )

        return r_calm * 2.0 * (delta_v_pct / 100.0)

    def _sfoc_curve(self, load_fraction: float) -> float:
        """
        Calculate specific fuel oil consumption at given load.

        Uses typical 2-stroke diesel SFOC curve, scaled by calibration sfoc_factor.

        Args:
            load_fraction: Engine load as fraction of MCR (0-1)

        Returns:
            SFOC in g/kWh
        """
        return nk.sfoc_curve(
            load_fraction,
            self.specs.sfoc_at_mcr,
            self.calibration_factors.get("sfoc_factor", 1.0),
        )

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
            self.specs.service_speed_laden
            if is_laden
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
            weather["heading_deg"] = heading_deg

        # Bisection: find speed where required_power = target_power
        # Required power increases monotonically with speed (cubic-ish)
        v_lo, v_hi = 2.0, 25.0  # knots search range

        def _power_at_speed(speed_kts: float) -> float:
            """Required brake power at this speed."""
            r = self.calculate_fuel_consumption(
                speed_kts,
                is_laden,
                weather,
                distance_nm=1.0,
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
            stw_kts,
            is_laden,
            weather,
            distance_nm=stw_kts * 24,
        )

        # Current effect on SOG
        # Project current along vessel heading
        current_effect_kts = 0.0
        if current_speed_ms > 0:
            # Current direction is "flowing toward" — same convention as heading
            relative_current_angle = math.radians(current_dir_deg - heading_deg)
            current_along_kts = (current_speed_ms / 0.51444) * math.cos(
                relative_current_angle
            )
            current_effect_kts = current_along_kts

        sog_kts = max(0.0, stw_kts + current_effect_kts)

        # Fuel metrics
        fuel_per_day_mt = result["fuel_mt"]  # Already 24h distance
        fuel_per_nm_mt = fuel_per_day_mt / (sog_kts * 24) if sog_kts > 0 else 0.0

        # Speed loss from calm-water service speed
        service_speed = (
            self.specs.service_speed_laden
            if is_laden
            else self.specs.service_speed_ballast
        )
        # Calm-water speed at same power
        calm_result = self.calculate_fuel_consumption(
            stw_kts,
            is_laden,
            weather=None,
            distance_nm=1.0,
        )
        # Find what speed we'd get in calm water at same power
        calm_stw = stw_kts  # Start with same speed
        if weather:
            # Re-run bisection without weather
            v_lo_c, v_hi_c = 2.0, 25.0
            for _ in range(30):
                v_mid = (v_lo_c + v_hi_c) / 2.0
                r = self.calculate_fuel_consumption(
                    v_mid, is_laden, None, distance_nm=1.0
                )
                if r["required_power_kw"] < target_power_kw:
                    v_lo_c = v_mid
                else:
                    v_hi_c = v_mid
            calm_stw = (v_lo_c + v_hi_c) / 2.0

        speed_loss_pct = (
            ((calm_stw - stw_kts) / calm_stw * 100) if calm_stw > 0 else 0.0
        )

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

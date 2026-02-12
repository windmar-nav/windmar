"""
Voyage calculation module.

Calculates per-leg SOG, ETA, and fuel consumption along a route
considering weather conditions at each waypoint.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.routes.rtz_parser import Route, Waypoint, haversine_distance, calculate_bearing
from src.optimization.vessel_model import VesselModel, VesselSpecs

logger = logging.getLogger(__name__)


@dataclass
class LegWeather:
    """Weather conditions for a leg."""
    wind_speed_ms: float = 0.0
    wind_dir_deg: float = 0.0
    sig_wave_height_m: float = 0.0
    wave_period_s: float = 0.0
    wave_dir_deg: float = 0.0
    current_speed_ms: float = 0.0
    current_dir_deg: float = 0.0

    # Wave decomposition (when available from CMEMS)
    windwave_height_m: float = 0.0
    windwave_period_s: float = 0.0
    windwave_dir_deg: float = 0.0
    swell_height_m: float = 0.0
    swell_period_s: float = 0.0
    swell_dir_deg: float = 0.0
    has_decomposition: bool = False

    # Extended fields (SPEC-P1)
    sst_celsius: float = 15.0  # Sea surface temperature
    visibility_km: float = 50.0  # Visibility (default: clear)
    ice_concentration: float = 0.0  # Sea ice fraction (0-1)


@dataclass
class LegResult:
    """Calculation result for a single leg."""
    leg_index: int
    from_wp: Waypoint
    to_wp: Waypoint

    # Leg geometry
    distance_nm: float
    bearing_deg: float

    # Weather at leg midpoint
    weather: LegWeather

    # Speed calculations
    calm_speed_kts: float  # Input calm water speed
    stw_kts: float  # Speed through water (after weather loss)
    sog_kts: float  # Speed over ground (after current)
    speed_loss_pct: float  # Percentage speed loss due to weather

    # Time and ETA
    time_hours: float
    departure_time: datetime
    arrival_time: datetime

    # Fuel
    fuel_mt: float
    power_kw: float

    # Resistance breakdown
    resistance_breakdown: Dict[str, float]


@dataclass
class VoyageResult:
    """Complete voyage calculation result."""
    route_name: str
    departure_time: datetime
    arrival_time: datetime

    # Totals
    total_distance_nm: float
    total_time_hours: float
    total_fuel_mt: float
    avg_sog_kts: float
    avg_stw_kts: float

    # Per-leg details
    legs: List[LegResult]

    # Vessel info
    vessel_specs: Dict
    calm_speed_kts: float
    is_laden: bool


class VoyageCalculator:
    """
    Calculate voyage performance along a route with weather.

    Takes a route, weather data provider, vessel model, and calculates
    per-leg SOG, ETA, and fuel consumption.
    """

    def __init__(
        self,
        vessel_model: Optional[VesselModel] = None,
        vessel_specs: Optional[VesselSpecs] = None,
    ):
        """
        Initialize voyage calculator.

        Args:
            vessel_model: VesselModel instance (creates default if None)
            vessel_specs: VesselSpecs (used if vessel_model is None)
        """
        if vessel_model:
            self.vessel_model = vessel_model
        else:
            specs = vessel_specs or VesselSpecs()
            self.vessel_model = VesselModel(specs=specs)

    def calculate_voyage(
        self,
        route: Route,
        calm_speed_kts: float,
        is_laden: bool,
        departure_time: datetime,
        weather_provider: Optional[callable] = None,
    ) -> VoyageResult:
        """
        Calculate complete voyage with per-leg details.

        Args:
            route: Route with waypoints
            calm_speed_kts: Calm water speed in knots
            is_laden: True if laden, False if ballast
            departure_time: Voyage start time
            weather_provider: Optional function(lat, lon, time) -> LegWeather

        Returns:
            VoyageResult with per-leg calculations
        """
        legs_result = []
        current_time = departure_time
        total_fuel = 0.0

        route_legs = route.legs

        for i, leg in enumerate(route_legs):
            # Get weather at leg midpoint
            mid_lat = (leg.from_wp.lat + leg.to_wp.lat) / 2
            mid_lon = (leg.from_wp.lon + leg.to_wp.lon) / 2
            leg_mid_time = current_time + timedelta(hours=leg.distance_nm / calm_speed_kts / 2)

            if weather_provider:
                weather = weather_provider(mid_lat, mid_lon, leg_mid_time)
            else:
                weather = LegWeather()  # Calm conditions

            # Calculate speed through water with weather effects
            stw_kts, speed_loss_pct, fuel_result = self._calculate_leg_performance(
                calm_speed_kts=calm_speed_kts,
                is_laden=is_laden,
                bearing_deg=leg.bearing_deg,
                weather=weather,
                distance_nm=leg.distance_nm,
            )

            # Calculate SOG considering current
            sog_kts = self._calculate_sog(
                stw_kts=stw_kts,
                heading_deg=leg.bearing_deg,
                current_speed_ms=weather.current_speed_ms,
                current_dir_deg=weather.current_dir_deg,
            )

            # Recompute speed loss from SOG to reflect total impact (weather + current)
            speed_loss_pct = max(0.0, ((calm_speed_kts - sog_kts) / calm_speed_kts) * 100)

            # Calculate time for this leg
            if sog_kts > 0:
                leg_time_hours = leg.distance_nm / sog_kts
            else:
                leg_time_hours = float('inf')

            arrival_time = current_time + timedelta(hours=leg_time_hours)

            leg_result = LegResult(
                leg_index=i,
                from_wp=leg.from_wp,
                to_wp=leg.to_wp,
                distance_nm=leg.distance_nm,
                bearing_deg=leg.bearing_deg,
                weather=weather,
                calm_speed_kts=calm_speed_kts,
                stw_kts=stw_kts,
                sog_kts=sog_kts,
                speed_loss_pct=speed_loss_pct,
                time_hours=leg_time_hours,
                departure_time=current_time,
                arrival_time=arrival_time,
                fuel_mt=fuel_result['fuel_mt'],
                power_kw=fuel_result['power_kw'],
                resistance_breakdown=fuel_result.get('resistance_breakdown_kn', {}),
            )

            legs_result.append(leg_result)
            total_fuel += fuel_result['fuel_mt']
            current_time = arrival_time

        # Calculate totals
        total_distance = sum(leg.distance_nm for leg in legs_result)
        total_time = sum(leg.time_hours for leg in legs_result)
        avg_sog = total_distance / total_time if total_time > 0 else 0
        avg_stw = sum(leg.stw_kts * leg.time_hours for leg in legs_result) / total_time if total_time > 0 else 0

        return VoyageResult(
            route_name=route.name,
            departure_time=departure_time,
            arrival_time=current_time,
            total_distance_nm=total_distance,
            total_time_hours=total_time,
            total_fuel_mt=total_fuel,
            avg_sog_kts=avg_sog,
            avg_stw_kts=avg_stw,
            legs=legs_result,
            vessel_specs={
                'dwt': self.vessel_model.specs.dwt,
                'loa': self.vessel_model.specs.loa,
                'service_speed_laden': self.vessel_model.specs.service_speed_laden,
                'service_speed_ballast': self.vessel_model.specs.service_speed_ballast,
            },
            calm_speed_kts=calm_speed_kts,
            is_laden=is_laden,
        )

    def _calculate_leg_performance(
        self,
        calm_speed_kts: float,
        is_laden: bool,
        bearing_deg: float,
        weather: LegWeather,
        distance_nm: float,
    ) -> Tuple[float, float, Dict]:
        """
        Calculate speed through water and fuel for a leg.

        Returns:
            Tuple of (stw_kts, speed_loss_pct, fuel_result_dict)
        """
        # Build weather dict for vessel model
        weather_dict = None
        if weather.wind_speed_ms > 0 or weather.sig_wave_height_m > 0:
            weather_dict = {
                'wind_speed_ms': weather.wind_speed_ms,
                'wind_dir_deg': weather.wind_dir_deg,
                'heading_deg': bearing_deg,
            }
            if weather.sig_wave_height_m > 0:
                weather_dict['sig_wave_height_m'] = weather.sig_wave_height_m
                weather_dict['wave_dir_deg'] = weather.wave_dir_deg

        # Calculate fuel at calm speed first to get resistance
        fuel_result = self.vessel_model.calculate_fuel_consumption(
            speed_kts=calm_speed_kts,
            is_laden=is_laden,
            weather=weather_dict,
            distance_nm=distance_nm,
        )

        # Calculate speed loss due to weather
        # Method: Compare power required with and without weather
        # If power exceeds available, reduce speed

        if weather_dict:
            # Use UNCAPPED required power to determine speed reduction.
            # The capped power_kw hides weather differences because it's
            # always MCR when near service speed.
            required_power = fuel_result['required_power_kw']
            mcr = self.vessel_model.specs.mcr_kw
            available_power = mcr * 0.9  # 90% MCR operating limit

            if required_power > available_power:
                # Engine cannot provide enough power at this speed.
                # Reduce speed: Power ~ speed^3, so speed ~ power^(1/3)
                speed_factor = (available_power / required_power) ** (1/3)
                stw_kts = calm_speed_kts * speed_factor
                stw_kts = max(stw_kts, calm_speed_kts * 0.5)  # Floor at 50%

                # Recalculate fuel at reduced speed
                fuel_result = self.vessel_model.calculate_fuel_consumption(
                    speed_kts=stw_kts,
                    is_laden=is_laden,
                    weather=weather_dict,
                    distance_nm=distance_nm,
                )
            else:
                # Engine can maintain commanded speed.
                # Fuel consumption already reflects added weather resistance.
                stw_kts = calm_speed_kts

            speed_loss_pct = ((calm_speed_kts - stw_kts) / calm_speed_kts) * 100
        else:
            stw_kts = calm_speed_kts
            speed_loss_pct = 0.0

        return stw_kts, speed_loss_pct, fuel_result

    def _calculate_sog(
        self,
        stw_kts: float,
        heading_deg: float,
        current_speed_ms: float,
        current_dir_deg: float,
    ) -> float:
        """
        Calculate speed over ground from STW and current.

        Current direction is the direction current is FLOWING TO.

        Args:
            stw_kts: Speed through water in knots
            heading_deg: Vessel heading in degrees
            current_speed_ms: Current speed in m/s
            current_dir_deg: Current direction (flowing to) in degrees

        Returns:
            Speed over ground in knots
        """
        if current_speed_ms <= 0:
            return stw_kts

        # Convert current to knots
        current_kts = current_speed_ms * 1.94384

        # Vector addition
        # Vessel velocity vector
        heading_rad = np.radians(heading_deg)
        vx = stw_kts * np.sin(heading_rad)
        vy = stw_kts * np.cos(heading_rad)

        # Current velocity vector (direction is where it flows TO)
        current_rad = np.radians(current_dir_deg)
        cx = current_kts * np.sin(current_rad)
        cy = current_kts * np.cos(current_rad)

        # Ground velocity
        gx = vx + cx
        gy = vy + cy

        sog = np.sqrt(gx**2 + gy**2)
        return float(sog)


def interpolate_weather_along_leg(
    from_lat: float,
    from_lon: float,
    to_lat: float,
    to_lon: float,
    departure_time: datetime,
    speed_kts: float,
    weather_func: callable,
    num_points: int = 5,
) -> LegWeather:
    """
    Get average weather conditions along a leg by sampling multiple points.

    Args:
        from_lat, from_lon: Starting position
        to_lat, to_lon: Ending position
        departure_time: Time leaving start point
        speed_kts: Estimated speed
        weather_func: Function(lat, lon, time) -> dict with weather
        num_points: Number of points to sample

    Returns:
        Average LegWeather
    """
    distance = haversine_distance(from_lat, from_lon, to_lat, to_lon)

    winds_ms = []
    winds_dir = []
    waves_m = []
    waves_dir = []

    for i in range(num_points):
        frac = i / (num_points - 1) if num_points > 1 else 0.5

        lat = from_lat + frac * (to_lat - from_lat)
        lon = from_lon + frac * (to_lon - from_lon)
        time = departure_time + timedelta(hours=frac * distance / speed_kts)

        wx = weather_func(lat, lon, time)

        winds_ms.append(wx.get('wind_speed_ms', 0))
        winds_dir.append(wx.get('wind_dir_deg', 0))
        waves_m.append(wx.get('sig_wave_height_m', 0))
        waves_dir.append(wx.get('wave_dir_deg', 0))

    return LegWeather(
        wind_speed_ms=np.mean(winds_ms),
        wind_dir_deg=np.mean(winds_dir),  # Simplified - should use circular mean
        sig_wave_height_m=np.mean(waves_m),
        wave_dir_deg=np.mean(waves_dir),
    )

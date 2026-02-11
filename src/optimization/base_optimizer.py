"""
Abstract base class for route optimization engines.

All optimization engines (A*, VISIR, etc.) must implement this interface
so they can be swapped via configuration or per-request selection.

``OptimizedRoute`` lives here (rather than in route_optimizer.py) so
that both the ABC and every concrete engine can import it without
circular dependencies.

Shared geometry helpers, path smoothing, and route statistics are also
implemented here so subclasses don't duplicate them.
"""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

from src.optimization.vessel_model import VesselModel
from src.optimization.voyage import LegWeather
from src.optimization.seakeeping import SafetyConstraints, SafetyStatus
from src.data.land_mask import is_path_clear

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Shared result dataclass
# -------------------------------------------------------------------

@dataclass
class OptimizedRoute:
    """Result of route optimization."""
    waypoints: List[Tuple[float, float]]  # (lat, lon) pairs
    total_fuel_mt: float
    total_time_hours: float
    total_distance_nm: float

    # Comparison with direct route
    direct_fuel_mt: float
    direct_time_hours: float
    fuel_savings_pct: float
    time_savings_pct: float

    # Per-leg details
    leg_details: List[Dict]

    # Speed profile (for variable speed optimization)
    speed_profile: List[float]  # Optimal speed per leg (kts)
    avg_speed_kts: float  # Average speed over voyage

    # Safety assessment
    safety_status: str  # "safe", "marginal", "dangerous"
    safety_warnings: List[str]
    max_roll_deg: float
    max_pitch_deg: float
    max_accel_ms2: float

    # Metadata
    grid_resolution_deg: float
    cells_explored: int
    optimization_time_ms: float
    variable_speed_enabled: bool

    # Speed strategy scenarios (populated when baseline provided)
    scenarios: List = field(default_factory=list)

    # Baseline reference (from voyage calculation)
    baseline_fuel_mt: float = 0.0
    baseline_time_hours: float = 0.0
    baseline_distance_nm: float = 0.0


# -------------------------------------------------------------------
# Abstract optimizer
# -------------------------------------------------------------------

class BaseOptimizer(ABC):
    """
    Abstract interface for route optimization engines.

    Subclasses must implement ``optimize_route`` and return an
    ``OptimizedRoute`` dataclass so the API layer can treat every
    engine identically.

    Provides shared geometry, smoothing, and route statistics methods.
    """

    def __init__(self, vessel_model: Optional[VesselModel] = None, **kwargs):
        self.vessel_model = vessel_model or VesselModel()

    @abstractmethod
    def optimize_route(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        departure_time: datetime,
        calm_speed_kts: float,
        is_laden: bool,
        weather_provider: Callable[[float, float, datetime], LegWeather],
        max_cells: int = 50_000,
        avoid_land: bool = True,
    ) -> OptimizedRoute:
        """
        Find an optimal route from *origin* to *destination*.

        Parameters
        ----------
        origin : (lat, lon)
        destination : (lat, lon)
        departure_time : UTC departure
        calm_speed_kts : calm-water service speed
        is_laden : vessel loading condition
        weather_provider : callable(lat, lon, time) -> LegWeather
        max_cells : exploration budget (interpretation is engine-specific)
        avoid_land : whether to enforce land avoidance

        Returns
        -------
        OptimizedRoute
        """
        ...

    # Convenience -------------------------------------------------------

    @property
    def engine_name(self) -> str:
        """Human-readable engine identifier."""
        return self.__class__.__name__

    # -------------------------------------------------------------------
    # Geometry helpers (shared by all engines)
    # -------------------------------------------------------------------

    @staticmethod
    def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great circle distance in nautical miles."""
        R = 3440.065  # Earth radius in nm
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
             * math.sin(dlon / 2) ** 2)
        return R * 2 * math.asin(math.sqrt(a))

    @staticmethod
    def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate initial bearing from point 1 to point 2 (degrees)."""
        dlon = math.radians(lon2 - lon1)
        x = math.sin(dlon) * math.cos(math.radians(lat2))
        y = (math.cos(math.radians(lat1)) * math.sin(math.radians(lat2))
             - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2))
             * math.cos(dlon))
        return (math.degrees(math.atan2(x, y)) + 360) % 360

    @staticmethod
    def current_effect(heading_deg: float, current_speed_ms: float, current_dir_deg: float) -> float:
        """
        Calculate effect of current on speed over ground.

        Returns speed adjustment in knots (positive = favorable).
        """
        if current_speed_ms <= 0:
            return 0.0
        current_kts = current_speed_ms * 1.94384
        relative_angle = abs(((current_dir_deg - heading_deg) + 180) % 360 - 180)
        return current_kts * math.cos(math.radians(relative_angle))

    @staticmethod
    def estimate_wave_period(weather: LegWeather) -> float:
        """Estimate wave period from weather data, with heuristic fallback."""
        if weather.wave_period_s > 0:
            return weather.wave_period_s
        return 5.0 + weather.sig_wave_height_m

    # -------------------------------------------------------------------
    # Path smoothing (Douglas-Peucker with land avoidance)
    # -------------------------------------------------------------------

    @staticmethod
    def smooth_path(
        waypoints: List[Tuple[float, float]],
        tolerance_nm: float = 5.0,
    ) -> List[Tuple[float, float]]:
        """
        Smooth path using Douglas-Peucker algorithm.

        Removes unnecessary waypoints while keeping path shape.
        Ensures simplified path doesn't cross land.
        """
        if len(waypoints) <= 2:
            return waypoints

        def perp_dist(pt, a, b):
            dx, dy = b[0] - a[0], b[1] - a[1]
            if dx == 0 and dy == 0:
                return math.sqrt((pt[0] - a[0]) ** 2 + (pt[1] - a[1]) ** 2) * 60
            t = max(0, min(1, ((pt[0] - a[0]) * dx + (pt[1] - a[1]) * dy) / (dx * dx + dy * dy)))
            px, py = a[0] + t * dx, a[1] + t * dy
            return math.sqrt((pt[0] - px) ** 2 + (pt[1] - py) ** 2) * 60

        def simplify(pts, eps):
            if len(pts) <= 2:
                return pts
            mx_d, mx_i = 0, 0
            for i in range(1, len(pts) - 1):
                d = perp_dist(pts[i], pts[0], pts[-1])
                if d > mx_d:
                    mx_d, mx_i = d, i
            if mx_d > eps:
                left = simplify(pts[: mx_i + 1], eps)
                right = simplify(pts[mx_i:], eps)
                return left[:-1] + right
            # Before collapsing, check land crossing
            if not is_path_clear(pts[0][0], pts[0][1], pts[-1][0], pts[-1][1]):
                mid = len(pts) // 2
                left = simplify(pts[: mid + 1], eps)
                right = simplify(pts[mid:], eps)
                return left[:-1] + right
            return [pts[0], pts[-1]]

        smoothed = simplify(waypoints, tolerance_nm)

        # Subdivide long segments to prevent Mercator rendering from
        # crossing land (straight lines in screen-space diverge from
        # the geographic path on segments > ~120 nm).
        max_seg_nm = 120.0
        result = [smoothed[0]]
        for i in range(1, len(smoothed)):
            prev = result[-1]
            cur = smoothed[i]
            # Approximate distance in nm (Pythagorean on lat/lon, 60nm per degree)
            dlat = (cur[0] - prev[0]) * 60
            dlon = (cur[1] - prev[1]) * 60 * math.cos(math.radians((prev[0] + cur[0]) / 2))
            seg_nm = math.sqrt(dlat * dlat + dlon * dlon)
            if seg_nm > max_seg_nm:
                n_sub = int(math.ceil(seg_nm / max_seg_nm))
                for j in range(1, n_sub):
                    t = j / n_sub
                    mid_lat = prev[0] + t * (cur[0] - prev[0])
                    mid_lon = prev[1] + t * (cur[1] - prev[1])
                    result.append((mid_lat, mid_lon))
            result.append(cur)
        return result

    # -------------------------------------------------------------------
    # Route statistics (shared by all engines)
    # -------------------------------------------------------------------

    def calculate_route_stats(
        self,
        waypoints: List[Tuple[float, float]],
        departure_time: datetime,
        calm_speed_kts: float,
        is_laden: bool,
        weather_provider: Callable[[float, float, datetime], LegWeather],
        safety_constraints: "SafetyConstraints",
        find_optimal_speed: Optional[Callable] = None,
    ) -> Tuple[float, float, float, List[Dict], Dict, List[float]]:
        """
        Calculate total fuel, time, and distance for a route.

        Parameters
        ----------
        waypoints : route waypoints
        departure_time : UTC departure
        calm_speed_kts : base speed
        is_laden : loading condition
        weather_provider : callable(lat, lon, time) -> LegWeather
        safety_constraints : seakeeping safety model
        find_optimal_speed : optional callable(dist, weather, bearing, is_laden) -> (speed, fuel, time)
            If provided, used for per-leg speed optimization.

        Returns
        -------
        (total_fuel, total_time, total_dist, leg_details, safety_summary, speed_profile)
        """
        total_fuel = 0.0
        total_time = 0.0
        total_dist = 0.0
        leg_details: List[Dict] = []
        speed_profile: List[float] = []

        max_roll = 0.0
        max_pitch = 0.0
        max_accel = 0.0
        all_warnings: List[str] = []
        worst = SafetyStatus.SAFE
        cur_time = departure_time

        for i in range(len(waypoints) - 1):
            f_wp = waypoints[i]
            t_wp = waypoints[i + 1]
            dist = self.haversine(f_wp[0], f_wp[1], t_wp[0], t_wp[1])
            brg = self.bearing(f_wp[0], f_wp[1], t_wp[0], t_wp[1])

            mid_lat = (f_wp[0] + t_wp[0]) / 2
            mid_lon = (f_wp[1] + t_wp[1]) / 2
            mid_time = cur_time + timedelta(hours=dist / calm_speed_kts / 2)

            try:
                weather = weather_provider(mid_lat, mid_lon, mid_time)
            except Exception:
                weather = LegWeather()

            # Per-leg speed optimization
            if find_optimal_speed is not None:
                spd, fuel_mt, time_h = find_optimal_speed(dist, weather, brg, is_laden)
            else:
                spd = calm_speed_kts
                weather_dict = {
                    'wind_speed_ms': weather.wind_speed_ms,
                    'wind_dir_deg': weather.wind_dir_deg,
                    'heading_deg': brg,
                    'sig_wave_height_m': weather.sig_wave_height_m,
                    'wave_dir_deg': weather.wave_dir_deg,
                }
                result = self.vessel_model.calculate_fuel_consumption(
                    speed_kts=calm_speed_kts, is_laden=is_laden,
                    weather=weather_dict, distance_nm=dist,
                )
                fuel_mt = result['fuel_mt']
                ce = self.current_effect(brg, weather.current_speed_ms, weather.current_dir_deg)
                sog = max(calm_speed_kts + ce, 0.1)
                time_h = dist / sog

            speed_profile.append(spd)

            ce = self.current_effect(brg, weather.current_speed_ms, weather.current_dir_deg)
            sog = max(spd + ce, 0.1)

            total_fuel += fuel_mt
            total_time += time_h
            total_dist += dist

            # Safety assessment
            leg_safety = None
            if weather.sig_wave_height_m > 0:
                wp = self.estimate_wave_period(weather)
                leg_safety = safety_constraints.assess_safety(
                    wave_height_m=weather.sig_wave_height_m,
                    wave_period_s=wp,
                    wave_dir_deg=weather.wave_dir_deg,
                    heading_deg=brg,
                    speed_kts=spd,
                    is_laden=is_laden,
                )
                max_roll = max(max_roll, leg_safety.motions.roll_amplitude_deg)
                max_pitch = max(max_pitch, leg_safety.motions.pitch_amplitude_deg)
                max_accel = max(max_accel, leg_safety.motions.bridge_accel_ms2)
                for w in leg_safety.warnings:
                    if w not in all_warnings:
                        all_warnings.append(w)
                if leg_safety.status == SafetyStatus.DANGEROUS:
                    worst = SafetyStatus.DANGEROUS
                elif leg_safety.status == SafetyStatus.MARGINAL and worst != SafetyStatus.DANGEROUS:
                    worst = SafetyStatus.MARGINAL

            leg_details.append({
                'from': f_wp,
                'to': t_wp,
                'distance_nm': dist,
                'bearing_deg': brg,
                'fuel_mt': fuel_mt,
                'time_hours': time_h,
                'sog_kts': sog,
                'stw_kts': spd,
                'wind_speed_ms': weather.wind_speed_ms,
                'wave_height_m': weather.sig_wave_height_m,
                'safety_status': leg_safety.status.value if leg_safety else 'safe',
                'roll_deg': leg_safety.motions.roll_amplitude_deg if leg_safety else 0.0,
                'pitch_deg': leg_safety.motions.pitch_amplitude_deg if leg_safety else 0.0,
            })
            cur_time += timedelta(hours=time_h)

        safety_summary = {
            'status': worst.value,
            'warnings': all_warnings,
            'max_roll_deg': max_roll,
            'max_pitch_deg': max_pitch,
            'max_accel_ms2': max_accel,
        }
        return total_fuel, total_time, total_dist, leg_details, safety_summary, speed_profile

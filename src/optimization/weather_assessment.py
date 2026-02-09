"""
Route weather assessment and provisioning.

Evaluates what weather data is needed for a voyage, checks DB availability,
bulk-loads all grids, and builds a TemporalGridWeatherProvider.

This replaces the single-snapshot get_wind_field/get_wave_field/get_current_field
calls with a DB-first, multi-timestep approach that eliminates live API calls
during route optimization.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from src.optimization.temporal_weather_provider import (
    GridDict,
    TemporalGridWeatherProvider,
    WeatherProvenance,
)

logger = logging.getLogger(__name__)


@dataclass
class WeatherAssessmentResult:
    """Result of assessing weather data needs for a voyage."""
    # Voyage estimate
    estimated_passage_hours: float
    weather_window_hours: float  # passage + safety margin

    # Required forecast hours (every 3h within weather window)
    required_forecast_hours: List[int]

    # Corridor bounding box (with margin)
    corridor_bbox: Tuple[float, float, float, float]  # (lat_min, lat_max, lon_min, lon_max)

    # Per-source availability
    availability: Dict[str, Dict]  # source -> {run_time, available_hours, coverage_pct}

    # Gap analysis
    beyond_forecast_horizon: bool  # True if voyage extends past available forecast
    gap_warnings: List[str] = field(default_factory=list)


class RouteWeatherAssessment:
    """Orchestrates weather assessment and provisioning for route optimization."""

    # All parameters we want to provision
    WIND_PARAMS = ["wind_u", "wind_v"]
    WAVE_PARAMS = ["wave_hs", "wave_tp", "wave_dir"]
    SWELL_PARAMS = ["swell_hs", "swell_tp", "swell_dir"]
    WINDWAVE_PARAMS = ["windwave_hs", "windwave_tp", "windwave_dir"]
    CURRENT_PARAMS = ["current_u", "current_v"]

    def __init__(self, db_weather):
        """
        Args:
            db_weather: DbWeatherProvider instance.
        """
        self.db_weather = db_weather

    def assess(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        departure_time: datetime,
        calm_speed_kts: float,
    ) -> WeatherAssessmentResult:
        """Assess weather data needs for a voyage.

        Determines the forecast hours needed, corridor bbox, and checks
        DB availability per source.
        """
        # Great-circle distance
        lat1, lon1 = math.radians(origin[0]), math.radians(origin[1])
        lat2, lon2 = math.radians(destination[0]), math.radians(destination[1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        distance_nm = c * 3440.065  # Earth radius in nm

        # Estimated passage hours + 25% safety margin
        estimated_hours = distance_nm / max(calm_speed_kts, 1.0)
        weather_window_hours = estimated_hours * 1.25

        # Required forecast hours (every 3h)
        max_fh = min(int(math.ceil(weather_window_hours / 3) * 3), 120)
        required_hours = list(range(0, max_fh + 1, 3))

        # Corridor bounding box with 5° margin
        margin = 5.0
        lat_min = min(origin[0], destination[0]) - margin
        lat_max = max(origin[0], destination[0]) + margin
        lon_min = min(origin[1], destination[1]) - margin
        lon_max = max(origin[1], destination[1]) + margin
        lat_min = max(lat_min, -85.0)
        lat_max = min(lat_max, 85.0)
        bbox = (lat_min, lat_max, lon_min, lon_max)

        # Check availability per source
        availability = {}
        for source in ["gfs", "cmems_wave", "cmems_current"]:
            run_time, avail_hours = self.db_weather.get_available_hours_by_source(source)
            if run_time is not None:
                avail_set = set(avail_hours)
                covered = len(avail_set.intersection(required_hours))
                coverage_pct = (covered / len(required_hours) * 100) if required_hours else 0
                availability[source] = {
                    "run_time": run_time,
                    "available_hours": avail_hours,
                    "coverage_pct": round(coverage_pct, 1),
                }
            else:
                availability[source] = {
                    "run_time": None,
                    "available_hours": [],
                    "coverage_pct": 0.0,
                }

        # Gap analysis
        beyond_horizon = weather_window_hours > 120
        warnings = []
        if beyond_horizon:
            warnings.append(
                f"Voyage ({estimated_hours:.0f}h) exceeds 120h forecast horizon. "
                f"Weather beyond 120h will clamp to last available forecast."
            )

        for source, info in availability.items():
            if info["coverage_pct"] < 50:
                warnings.append(
                    f"{source}: only {info['coverage_pct']:.0f}% coverage "
                    f"({len(info['available_hours'])} hours available)"
                )

        return WeatherAssessmentResult(
            estimated_passage_hours=round(estimated_hours, 1),
            weather_window_hours=round(weather_window_hours, 1),
            required_forecast_hours=required_hours,
            corridor_bbox=bbox,
            availability=availability,
            beyond_forecast_horizon=beyond_horizon,
            gap_warnings=warnings,
        )

    def provision(
        self,
        assessment: WeatherAssessmentResult,
    ) -> Optional[TemporalGridWeatherProvider]:
        """Bulk-load all weather grids from DB and build TemporalGridWeatherProvider.

        Returns None if DB has insufficient data (caller should fall back to
        the old GridWeatherProvider path).
        """
        bbox = assessment.corridor_bbox
        required_hours = assessment.required_forecast_hours

        grids: GridDict = {}

        # Determine the run_time to use (prefer wave, then wind, then current)
        run_time = None
        for source in ["cmems_wave", "gfs", "cmems_current"]:
            info = assessment.availability.get(source, {})
            if info.get("run_time"):
                rt = info["run_time"]
                if rt.tzinfo is None:
                    rt = rt.replace(tzinfo=timezone.utc)
                if run_time is None:
                    run_time = rt

        if run_time is None:
            logger.warning("No weather data available in DB — cannot provision temporal provider")
            return None

        # Load wind grids
        wind_grids = self.db_weather.get_grids_for_timeline(
            "gfs", self.WIND_PARAMS,
            *bbox, required_hours,
        )
        for param in self.WIND_PARAMS:
            if param in wind_grids and wind_grids[param]:
                grids[param] = wind_grids[param]

        # Load wave grids (combined + decomposition)
        all_wave_params = self.WAVE_PARAMS + self.SWELL_PARAMS + self.WINDWAVE_PARAMS
        wave_grids = self.db_weather.get_grids_for_timeline(
            "cmems_wave", all_wave_params,
            *bbox, required_hours,
        )
        for param in all_wave_params:
            if param in wave_grids and wave_grids[param]:
                grids[param] = wave_grids[param]

        # Load current grids
        current_grids = self.db_weather.get_grids_for_timeline(
            "cmems_current", self.CURRENT_PARAMS,
            *bbox, required_hours,
        )
        for param in self.CURRENT_PARAMS:
            if param in current_grids and current_grids[param]:
                grids[param] = current_grids[param]

        # Check if we have at least wind or wave data
        has_wind = any(p in grids for p in self.WIND_PARAMS)
        has_wave = "wave_hs" in grids
        if not has_wind and not has_wave:
            logger.warning("Insufficient temporal data in DB (no wind or wave grids)")
            return None

        # Build provenance
        provenance = {}
        for source, params in [
            ("GFS", self.WIND_PARAMS),
            ("CMEMS_wave", all_wave_params),
            ("CMEMS_current", self.CURRENT_PARAMS),
        ]:
            max_hours = 0
            for p in params:
                if p in grids:
                    max_hours = max(max_hours, max(grids[p].keys(), default=0))
            if max_hours > 0:
                provenance[source] = WeatherProvenance.from_lead_hours(max_hours, source)

        total_grids = sum(len(v) for v in grids.values())
        logger.info(
            f"Weather provisioned: {len(grids)} params, {total_grids} total grids, "
            f"run_time={run_time}"
        )

        if assessment.gap_warnings:
            for w in assessment.gap_warnings:
                logger.warning(f"Weather gap: {w}")

        return TemporalGridWeatherProvider(
            run_time=run_time,
            grids=grids,
            provenance=provenance,
        )

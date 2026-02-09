"""
Time-varying weather provider for route optimization.

Stores multi-timestep grids per weather parameter and performs trilinear
interpolation (lat, lon, time) to provide weather that changes along the
voyage.  Drop-in replacement for GridWeatherProvider's callable signature.

Key differences from GridWeatherProvider:
- Stores Dict[forecast_hour, 2D_grid] per parameter
- get_weather() uses the time argument (temporal interpolation)
- Tracks data provenance (forecast, hindcast, climatology)
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.optimization.grid_weather_provider import GridWeatherProvider
from src.optimization.voyage import LegWeather

logger = logging.getLogger(__name__)


@dataclass
class WeatherProvenance:
    """Metadata about the source and confidence of weather data."""
    source_type: str  # "forecast", "hindcast", "climatology", "blended"
    model_name: str  # "GFS", "CMEMS_wave", "CMEMS_current", "ERA5", etc.
    forecast_lead_hours: float  # hours ahead of model run time
    confidence: str  # "high" (<72h), "medium" (72-120h), "low" (>120h / climatology)

    @staticmethod
    def from_lead_hours(lead_hours: float, model_name: str = "multi") -> "WeatherProvenance":
        if lead_hours < 72:
            confidence = "high"
        elif lead_hours < 120:
            confidence = "medium"
        else:
            confidence = "low"
        return WeatherProvenance(
            source_type="forecast",
            model_name=model_name,
            forecast_lead_hours=lead_hours,
            confidence=confidence,
        )


# Type alias: parameter name -> {forecast_hour -> (lats_1d, lons_1d, data_2d)}
GridDict = Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]]


class TemporalGridWeatherProvider:
    """Time-varying weather grids with trilinear interpolation.

    Constructor args:
        run_time: Forecast model run time (UTC).
        grids: Nested dict mapping parameter name -> {forecast_hour -> (lats, lons, data)}.
            Parameters: wind_u, wind_v, wave_hs, wave_tp, wave_dir,
                        swell_hs, swell_tp, swell_dir,
                        current_u, current_v
        provenance: Dict mapping source category to WeatherProvenance.
    """

    # Parameters that use vector (U/V) decomposition
    VECTOR_PARAMS = {"wind": ("wind_u", "wind_v"), "current": ("current_u", "current_v")}

    # Scalar wave parameters
    WAVE_PARAMS = ("wave_hs", "wave_tp", "wave_dir")
    SWELL_PARAMS = ("swell_hs", "swell_tp", "swell_dir")
    WINDWAVE_PARAMS = ("windwave_hs", "windwave_tp", "windwave_dir")

    def __init__(
        self,
        run_time: datetime,
        grids: GridDict,
        provenance: Optional[Dict[str, WeatherProvenance]] = None,
    ):
        self.run_time = run_time
        if self.run_time.tzinfo is None:
            self.run_time = self.run_time.replace(tzinfo=timezone.utc)

        self.grids = grids
        self.provenance = provenance or {}

        # Build sorted list of available forecast hours per parameter
        self._sorted_hours: Dict[str, List[int]] = {}
        for param, hour_map in grids.items():
            self._sorted_hours[param] = sorted(hour_map.keys())

        # Log summary
        params_summary = {p: len(h) for p, h in self._sorted_hours.items()}
        logger.info(f"TemporalGridWeatherProvider initialized: run_time={run_time}, params={params_summary}")

    def get_weather(self, lat: float, lon: float, time: datetime) -> LegWeather:
        """Get weather at (lat, lon, time) via trilinear interpolation.

        Matches the weather_provider callable signature: (lat, lon, time) -> LegWeather.
        """
        # Compute forecast hour offset
        if time.tzinfo is None:
            time = time.replace(tzinfo=timezone.utc)
        fh = (time - self.run_time).total_seconds() / 3600.0

        # Wind
        wu = self._interp_temporal("wind_u", lat, lon, fh)
        wv = self._interp_temporal("wind_v", lat, lon, fh)
        wind_speed = math.sqrt(wu * wu + wv * wv)
        wind_dir = (270.0 - math.degrees(math.atan2(wv, wu))) % 360.0

        # Combined waves
        wave_hs = self._interp_temporal("wave_hs", lat, lon, fh)
        wave_tp = self._interp_temporal("wave_tp", lat, lon, fh)
        wave_dir = self._interp_temporal("wave_dir", lat, lon, fh)
        if wave_tp <= 0 and wave_hs > 0:
            wave_tp = 5.0 + wave_hs

        # Swell decomposition
        swell_hs = self._interp_temporal("swell_hs", lat, lon, fh)
        swell_tp = self._interp_temporal("swell_tp", lat, lon, fh)
        swell_dir = self._interp_temporal("swell_dir", lat, lon, fh)

        # Wind-wave decomposition
        ww_hs = self._interp_temporal("windwave_hs", lat, lon, fh)
        ww_tp = self._interp_temporal("windwave_tp", lat, lon, fh)
        ww_dir = self._interp_temporal("windwave_dir", lat, lon, fh)

        has_decomp = swell_hs > 0 or ww_hs > 0

        # Currents
        cu = self._interp_temporal("current_u", lat, lon, fh)
        cv = self._interp_temporal("current_v", lat, lon, fh)
        current_speed = math.sqrt(cu * cu + cv * cv)
        current_dir = (270.0 - math.degrees(math.atan2(cv, cu))) % 360.0

        return LegWeather(
            wind_speed_ms=wind_speed,
            wind_dir_deg=wind_dir,
            sig_wave_height_m=max(wave_hs, 0.0),
            wave_period_s=max(wave_tp, 0.0),
            wave_dir_deg=wave_dir,
            current_speed_ms=current_speed,
            current_dir_deg=current_dir,
            windwave_height_m=max(ww_hs, 0.0),
            windwave_period_s=max(ww_tp, 0.0),
            windwave_dir_deg=ww_dir,
            swell_height_m=max(swell_hs, 0.0),
            swell_period_s=max(swell_tp, 0.0),
            swell_dir_deg=swell_dir,
            has_decomposition=has_decomp,
        )

    def get_provenance(self, time: datetime) -> WeatherProvenance:
        """Return provenance metadata for a given query time."""
        if time.tzinfo is None:
            time = time.replace(tzinfo=timezone.utc)
        lead_hours = (time - self.run_time).total_seconds() / 3600.0
        return WeatherProvenance.from_lead_hours(lead_hours)

    # ------------------------------------------------------------------
    # Internal interpolation
    # ------------------------------------------------------------------

    def _interp_temporal(
        self,
        param: str,
        lat: float,
        lon: float,
        forecast_hour: float,
    ) -> float:
        """Trilinear interpolation: spatial bilinear at two bracketing hours, then linear time blend.

        Returns 0.0 for NaN/inf results (coastal grid cells with missing data).
        """
        if param not in self.grids or not self._sorted_hours.get(param):
            return 0.0

        hours = self._sorted_hours[param]
        hour_map = self.grids[param]

        # Clamp to available range
        if forecast_hour <= hours[0]:
            lats, lons, data = hour_map[hours[0]]
            v = GridWeatherProvider._interp(lat, lon, lats, lons, data)
            return v if math.isfinite(v) else 0.0

        if forecast_hour >= hours[-1]:
            lats, lons, data = hour_map[hours[-1]]
            v = GridWeatherProvider._interp(lat, lon, lats, lons, data)
            return v if math.isfinite(v) else 0.0

        # Find bracketing hours
        idx = 0
        for i, h in enumerate(hours):
            if h > forecast_hour:
                idx = i
                break

        h0 = hours[idx - 1]
        h1 = hours[idx]

        # Spatial interpolation at each bracketing hour
        lats0, lons0, data0 = hour_map[h0]
        lats1, lons1, data1 = hour_map[h1]

        v0 = GridWeatherProvider._interp(lat, lon, lats0, lons0, data0)
        v1 = GridWeatherProvider._interp(lat, lon, lats1, lons1, data1)

        # Handle NaN/inf from coastal grid cells
        if not math.isfinite(v0):
            v0 = 0.0
        if not math.isfinite(v1):
            v1 = 0.0

        # Temporal linear blend
        span = h1 - h0
        if span <= 0:
            return v0
        alpha = (forecast_hour - h0) / span
        return v0 * (1.0 - alpha) + v1 * alpha

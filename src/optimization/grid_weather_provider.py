"""
Pre-fetched grid weather provider for fast A* route optimization.

Instead of calling external APIs per grid cell (~30+ min for a route),
this class pre-fetches wind/wave/current grids once for the corridor
bounding box and serves weather via fast numpy bilinear interpolation.

Expected speedup: ~1000x (30+ min → 2-5 seconds).
"""

import math
import logging
from datetime import datetime
from typing import Optional

import numpy as np

from src.optimization.voyage import LegWeather

logger = logging.getLogger(__name__)


class GridWeatherProvider:
    """Pre-fetched grid weather for fast A* optimization."""

    def __init__(self, wind_data, wave_data, current_data,
                 sst_data=None, visibility_data=None, ice_data=None):
        """
        Initialize from WeatherData objects returned by get_wind_field(),
        get_wave_field(), get_current_field(), etc.

        Args:
            wind_data: WeatherData with u_component, v_component
            wave_data: WeatherData with values (sig wave height), wave_period, wave_direction
            current_data: WeatherData with u_component, v_component
            sst_data: WeatherData with sst (°C) — optional
            visibility_data: WeatherData with visibility (km) — optional
            ice_data: WeatherData with ice_concentration (0-1) — optional
        """
        # Wind grid
        self.wind_lats = np.asarray(wind_data.lats, dtype=np.float64)
        self.wind_lons = np.asarray(wind_data.lons, dtype=np.float64)
        self.wind_u = np.asarray(wind_data.u_component, dtype=np.float64)
        self.wind_v = np.asarray(wind_data.v_component, dtype=np.float64)

        # Wave grid
        self.wave_lats = np.asarray(wave_data.lats, dtype=np.float64)
        self.wave_lons = np.asarray(wave_data.lons, dtype=np.float64)
        self.wave_hs = np.asarray(wave_data.values, dtype=np.float64)
        self.wave_period = (
            np.asarray(wave_data.wave_period, dtype=np.float64)
            if wave_data.wave_period is not None
            else None
        )
        self.wave_direction = (
            np.asarray(wave_data.wave_direction, dtype=np.float64)
            if wave_data.wave_direction is not None
            else None
        )

        # Swell decomposition (SPEC-P1) — from CMEMS partitioned wave data
        self.swell_hs = (
            np.asarray(wave_data.swell_height, dtype=np.float64)
            if getattr(wave_data, 'swell_height', None) is not None
            else None
        )
        self.swell_period = (
            np.asarray(wave_data.swell_period, dtype=np.float64)
            if getattr(wave_data, 'swell_period', None) is not None
            else None
        )
        self.swell_direction = (
            np.asarray(wave_data.swell_direction, dtype=np.float64)
            if getattr(wave_data, 'swell_direction', None) is not None
            else None
        )
        self.windsea_hs = (
            np.asarray(wave_data.windwave_height, dtype=np.float64)
            if getattr(wave_data, 'windwave_height', None) is not None
            else None
        )
        self.windsea_period = (
            np.asarray(wave_data.windwave_period, dtype=np.float64)
            if getattr(wave_data, 'windwave_period', None) is not None
            else None
        )
        self.windsea_direction = (
            np.asarray(wave_data.windwave_direction, dtype=np.float64)
            if getattr(wave_data, 'windwave_direction', None) is not None
            else None
        )
        self._has_swell_decomposition = self.swell_hs is not None

        # Current grid
        self.current_lats = np.asarray(current_data.lats, dtype=np.float64)
        self.current_lons = np.asarray(current_data.lons, dtype=np.float64)
        self.current_u = np.asarray(current_data.u_component, dtype=np.float64)
        self.current_v = np.asarray(current_data.v_component, dtype=np.float64)

        # SST grid (optional, SPEC-P1)
        self.sst_lats = None
        self.sst_lons = None
        self.sst_data = None
        if sst_data is not None and sst_data.values is not None:
            self.sst_lats = np.asarray(sst_data.lats, dtype=np.float64)
            self.sst_lons = np.asarray(sst_data.lons, dtype=np.float64)
            self.sst_data = np.asarray(sst_data.values, dtype=np.float64)

        # Visibility grid (optional, SPEC-P1)
        self.vis_lats = None
        self.vis_lons = None
        self.vis_data = None
        if visibility_data is not None and visibility_data.values is not None:
            self.vis_lats = np.asarray(visibility_data.lats, dtype=np.float64)
            self.vis_lons = np.asarray(visibility_data.lons, dtype=np.float64)
            self.vis_data = np.asarray(visibility_data.values, dtype=np.float64)

        # Ice concentration grid (optional, SPEC-P1)
        self.ice_lats = None
        self.ice_lons = None
        self.ice_data = None
        if ice_data is not None and ice_data.values is not None:
            self.ice_lats = np.asarray(ice_data.lats, dtype=np.float64)
            self.ice_lons = np.asarray(ice_data.lons, dtype=np.float64)
            self.ice_data = np.asarray(ice_data.values, dtype=np.float64)

        extras = []
        if self.sst_data is not None:
            extras.append(f"sst {self.sst_data.shape}")
        if self.vis_data is not None:
            extras.append(f"vis {self.vis_data.shape}")
        if self.ice_data is not None:
            extras.append(f"ice {self.ice_data.shape}")

        logger.info(
            f"GridWeatherProvider initialized: "
            f"wind {self.wind_u.shape}, wave {self.wave_hs.shape}, "
            f"current {self.current_u.shape}"
            + (f", {', '.join(extras)}" if extras else "")
        )

    def get_weather(self, lat: float, lon: float, time: datetime) -> LegWeather:
        """
        Get weather at a point via bilinear interpolation from pre-fetched grids.

        Matches the weather_provider callable signature: (lat, lon, time) -> LegWeather.
        The time parameter is accepted but ignored (single-snapshot grid).
        """
        # Wind
        wu = self._interp(lat, lon, self.wind_lats, self.wind_lons, self.wind_u)
        wv = self._interp(lat, lon, self.wind_lats, self.wind_lons, self.wind_v)
        wind_speed = math.sqrt(wu * wu + wv * wv)
        wind_dir = (270.0 - math.degrees(math.atan2(wv, wu))) % 360.0

        # Waves
        wave_hs = self._interp(lat, lon, self.wave_lats, self.wave_lons, self.wave_hs)
        wave_period = 0.0
        wave_dir = 0.0
        if self.wave_period is not None:
            wave_period = self._interp(lat, lon, self.wave_lats, self.wave_lons, self.wave_period)
        if self.wave_direction is not None:
            wave_dir = self._interp(lat, lon, self.wave_lats, self.wave_lons, self.wave_direction)

        # Fallback wave period estimate if no data
        if wave_period <= 0 and wave_hs > 0:
            wave_period = 5.0 + wave_hs

        # Currents
        cu = self._interp(lat, lon, self.current_lats, self.current_lons, self.current_u)
        cv = self._interp(lat, lon, self.current_lats, self.current_lons, self.current_v)
        current_speed = math.sqrt(cu * cu + cv * cv)
        current_dir = (270.0 - math.degrees(math.atan2(cv, cu))) % 360.0

        # SST (SPEC-P1)
        sst = 15.0
        if self.sst_data is not None:
            sst = self._interp(lat, lon, self.sst_lats, self.sst_lons, self.sst_data)

        # Visibility (SPEC-P1)
        vis = 50.0
        if self.vis_data is not None:
            vis = self._interp(lat, lon, self.vis_lats, self.vis_lons, self.vis_data)

        # Ice concentration (SPEC-P1)
        ice = 0.0
        if self.ice_data is not None:
            ice = self._interp(lat, lon, self.ice_lats, self.ice_lons, self.ice_data)
            ice = max(0.0, min(1.0, ice))

        # Swell decomposition (SPEC-P1) — with fallback from total Hs
        has_decomp = False
        if self._has_swell_decomposition:
            sw_hs = self._interp(lat, lon, self.wave_lats, self.wave_lons, self.swell_hs)
            sw_tp = self._interp(lat, lon, self.wave_lats, self.wave_lons, self.swell_period) if self.swell_period is not None else 0.0
            sw_dir = self._interp(lat, lon, self.wave_lats, self.wave_lons, self.swell_direction) if self.swell_direction is not None else 0.0
            ww_hs = self._interp(lat, lon, self.wave_lats, self.wave_lons, self.windsea_hs) if self.windsea_hs is not None else 0.0
            ww_tp = self._interp(lat, lon, self.wave_lats, self.wave_lons, self.windsea_period) if self.windsea_period is not None else 0.0
            ww_dir = self._interp(lat, lon, self.wave_lats, self.wave_lons, self.windsea_direction) if self.windsea_direction is not None else 0.0
            has_decomp = True
        else:
            # Fallback: derive from total Hs (SPEC-P1 §2.1)
            total_hs = max(wave_hs, 0.0)
            sw_hs = 0.8 * total_hs
            ww_hs = 0.6 * total_hs
            sw_tp = max(wave_period, 0.0)
            sw_dir = wave_dir
            ww_tp = max(wave_period * 0.7, 0.0) if wave_period > 0 else 0.0
            ww_dir = wave_dir
            if total_hs > 0:
                has_decomp = True

        return LegWeather(
            wind_speed_ms=wind_speed,
            wind_dir_deg=wind_dir,
            sig_wave_height_m=max(wave_hs, 0.0),
            wave_period_s=max(wave_period, 0.0),
            wave_dir_deg=wave_dir,
            current_speed_ms=current_speed,
            current_dir_deg=current_dir,
            swell_height_m=max(sw_hs, 0.0),
            swell_period_s=max(sw_tp, 0.0),
            swell_dir_deg=sw_dir,
            windwave_height_m=max(ww_hs, 0.0),
            windwave_period_s=max(ww_tp, 0.0),
            windwave_dir_deg=ww_dir,
            has_decomposition=has_decomp,
            sst_celsius=sst,
            visibility_km=max(vis, 0.0),
            ice_concentration=ice,
        )

    @staticmethod
    def _interp(
        lat: float, lon: float,
        lats: np.ndarray, lons: np.ndarray,
        data: np.ndarray,
    ) -> float:
        """
        Bilinear interpolation on a regular lat/lon grid.

        ~10 numpy ops per call — fast enough for 50k+ A* cells.
        """
        ny, nx = data.shape

        # Find fractional indices
        lat_min, lat_max = float(lats[0]), float(lats[-1])
        lon_min, lon_max = float(lons[0]), float(lons[-1])

        # Handle grids that may be descending in latitude
        if lat_min > lat_max:
            lat_min, lat_max = lat_max, lat_min
            # Flip data so row 0 = south
            data = data[::-1]

        # Clamp to grid bounds
        lat_c = max(lat_min, min(lat, lat_max))
        lon_c = max(lon_min, min(lon, lon_max))

        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min

        if lat_range == 0 or lon_range == 0:
            return float(data[0, 0])

        # Fractional row/col
        fi = (lat_c - lat_min) / lat_range * (ny - 1)
        fj = (lon_c - lon_min) / lon_range * (nx - 1)

        i0 = int(fi)
        j0 = int(fj)
        i1 = min(i0 + 1, ny - 1)
        j1 = min(j0 + 1, nx - 1)

        di = fi - i0
        dj = fj - j0

        # Bilinear interpolation (handle NaN from coastal/land cells)
        corners = [data[i0, j0], data[i1, j0], data[i0, j1], data[i1, j1]]
        if any(np.isnan(c) for c in corners):
            # Average only the valid (non-NaN) corners
            valid = [float(c) for c in corners if not np.isnan(c)]
            return sum(valid) / len(valid) if valid else 0.0

        val = (
            corners[0] * (1 - di) * (1 - dj)
            + corners[1] * di * (1 - dj)
            + corners[2] * (1 - di) * dj
            + corners[3] * di * dj
        )

        return float(val)

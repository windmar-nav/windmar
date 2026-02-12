"""
Copernicus Data Provider for WINDMAR.

Fetches weather and ocean data from:
- Copernicus Marine Service (CMEMS) - waves, currents
- Climate Data Store (CDS) - wind forecasts

Requires:
- pip install copernicusmarine xarray netcdf4
- pip install cdsapi

Authentication:
- CMEMS: ~/.copernicusmarine/.copernicusmarine-credentials or environment variables
- CDS: ~/.cdsapirc file with API key
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WeatherData:
    """Container for weather grid data."""
    parameter: str
    time: datetime
    lats: np.ndarray
    lons: np.ndarray
    values: np.ndarray  # 2D array [lat, lon]
    unit: str

    # For vector data (wind, currents)
    u_component: Optional[np.ndarray] = None
    v_component: Optional[np.ndarray] = None

    # For wave data - combined fields
    wave_period: Optional[np.ndarray] = None  # Peak wave period (s)
    wave_direction: Optional[np.ndarray] = None  # Mean wave direction (deg)

    # Wave decomposition: wind-wave component
    windwave_height: Optional[np.ndarray] = None  # VHM0_WW (m)
    windwave_period: Optional[np.ndarray] = None  # VTM01_WW (s)
    windwave_direction: Optional[np.ndarray] = None  # VMDR_WW (deg)

    # Wave decomposition: primary swell component
    swell_height: Optional[np.ndarray] = None  # VHM0_SW1 (m)
    swell_period: Optional[np.ndarray] = None  # VTM01_SW1 (s)
    swell_direction: Optional[np.ndarray] = None  # VMDR_SW1 (deg)

    # Extended fields (SPEC-P1)
    sst: Optional[np.ndarray] = None  # Sea surface temperature (°C)
    visibility: Optional[np.ndarray] = None  # Visibility (km)
    ice_concentration: Optional[np.ndarray] = None  # Sea ice fraction (0-1)


@dataclass
class PointWeather:
    """Weather at a specific point."""
    lat: float
    lon: float
    time: datetime
    wind_speed_ms: float
    wind_dir_deg: float
    wave_height_m: float
    wave_period_s: float
    wave_dir_deg: float
    current_speed_ms: float = 0.0
    current_dir_deg: float = 0.0

    # Wave decomposition
    windwave_height_m: float = 0.0
    windwave_period_s: float = 0.0
    windwave_dir_deg: float = 0.0
    swell_height_m: float = 0.0
    swell_period_s: float = 0.0
    swell_dir_deg: float = 0.0

    # Extended fields (SPEC-P1)
    sst_celsius: float = 15.0  # Sea surface temperature
    visibility_km: float = 50.0  # Visibility (default: clear)
    ice_concentration: float = 0.0  # Sea ice fraction (0-1)


class CopernicusDataProvider:
    """
    Unified data provider for Copernicus services.

    Handles data fetching, caching, and interpolation for:
    - Wind (from CDS ERA5 or ECMWF)
    - Waves (from CMEMS global wave model)
    - Currents (from CMEMS global physics)
    """

    # CMEMS dataset IDs
    CMEMS_WAVE_DATASET = "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i"
    CMEMS_PHYSICS_DATASET = "cmems_mod_glo_phy_anfc_0.083deg_PT1H-m"

    # CDS dataset for wind
    CDS_WIND_DATASET = "reanalysis-era5-single-levels"

    def __init__(
        self,
        cache_dir: str = "data/copernicus_cache",
        cmems_username: Optional[str] = None,
        cmems_password: Optional[str] = None,
    ):
        """
        Initialize Copernicus data provider.

        Args:
            cache_dir: Directory to cache downloaded data
            cmems_username: CMEMS username (or set COPERNICUSMARINE_SERVICE_USERNAME env var)
            cmems_password: CMEMS password (or set COPERNICUSMARINE_SERVICE_PASSWORD env var)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # CMEMS credentials — resolve from param, then COPERNICUSMARINE_SERVICE_* env vars
        self.cmems_username = cmems_username or os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME")
        self.cmems_password = cmems_password or os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD")

        # Cached xarray datasets
        self._wind_data: Optional[any] = None
        self._wave_data: Optional[any] = None
        self._current_data: Optional[any] = None

        # Check for required packages
        self._check_dependencies()

    def _check_dependencies(self):
        """Check if required packages are installed."""
        self._has_copernicusmarine = False
        self._has_cdsapi = False
        self._has_xarray = False

        try:
            import xarray
            self._has_xarray = True
        except ImportError:
            logger.warning("xarray not installed. Run: pip install xarray netcdf4")

        try:
            import copernicusmarine
            self._has_copernicusmarine = True
        except ImportError:
            logger.warning("copernicusmarine not installed. Run: pip install copernicusmarine")

        try:
            import cdsapi
            self._has_cdsapi = True
        except ImportError:
            logger.warning("cdsapi not installed. Run: pip install cdsapi")

    def fetch_wind_data(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Optional[WeatherData]:
        """
        Fetch wind data from CDS ERA5.

        Args:
            lat_min, lat_max: Latitude bounds
            lon_min, lon_max: Longitude bounds
            start_time: Start of time range (default: now)
            end_time: End of time range (default: now + 5 days)

        Returns:
            WeatherData with u/v wind components
        """
        if not self._has_cdsapi or not self._has_xarray:
            logger.warning("CDS API not available, returning None")
            return None

        if not os.environ.get("CDSAPI_KEY"):
            logger.warning("CDS API key not configured (set CDSAPI_KEY), returning None")
            return None

        import cdsapi
        import xarray as xr

        if start_time is None:
            start_time = datetime.utcnow()
        if end_time is None:
            end_time = start_time + timedelta(days=5)

        # ERA5 is reanalysis data with ~5-day lag; clamp to latest available
        era5_lag = timedelta(days=5)
        latest_available = datetime.utcnow() - era5_lag
        if start_time > latest_available:
            logger.info(
                f"ERA5 data not yet available for {start_time.date()}, "
                f"using latest available: {latest_available.date()}"
            )
            start_time = latest_available

        # Generate cache filename
        cache_file = self._get_cache_path(
            "wind", lat_min, lat_max, lon_min, lon_max, start_time
        )

        # Check cache
        if cache_file.exists():
            logger.info(f"Loading wind data from cache: {cache_file}")
            ds = xr.open_dataset(cache_file)
        else:
            logger.info("Downloading wind data from CDS...")

            try:
                client = cdsapi.Client()

                # Request ERA5 10m wind components
                client.retrieve(
                    self.CDS_WIND_DATASET,
                    {
                        'product_type': 'reanalysis',
                        'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind'],
                        'year': start_time.strftime('%Y'),
                        'month': start_time.strftime('%m'),
                        'day': [start_time.strftime('%d')],
                        'time': ['00:00', '06:00', '12:00', '18:00'],
                        'area': [lat_max, lon_min, lat_min, lon_max],
                        'format': 'netcdf',
                    },
                    str(cache_file)
                )

                ds = xr.open_dataset(cache_file)

            except Exception as e:
                logger.error(f"Failed to download wind data: {e}")
                return None

        # Extract data
        try:
            u10 = ds['u10'].values
            v10 = ds['v10'].values
            lats = ds['latitude'].values
            lons = ds['longitude'].values
            time = ds['time'].values[0] if 'time' in ds.dims else start_time

            # Take first time step if multiple
            if len(u10.shape) == 3:
                u10 = u10[0]
                v10 = v10[0]

            return WeatherData(
                parameter="wind",
                time=time if isinstance(time, datetime) else start_time,
                lats=lats,
                lons=lons,
                values=np.sqrt(u10**2 + v10**2),  # Wind speed
                unit="m/s",
                u_component=u10,
                v_component=v10,
            )

        except Exception as e:
            logger.error(f"Failed to parse wind data: {e}")
            return None

    def fetch_wave_data(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        start_time: Optional[datetime] = None,
    ) -> Optional[WeatherData]:
        """
        Fetch wave data from CMEMS.

        Args:
            lat_min, lat_max: Latitude bounds
            lon_min, lon_max: Longitude bounds
            start_time: Reference time (default: now)

        Returns:
            WeatherData with significant wave height
        """
        if not self._has_copernicusmarine or not self._has_xarray:
            logger.warning("CMEMS API not available, returning None")
            return None

        if not self.cmems_username or not self.cmems_password:
            logger.warning("CMEMS credentials not configured, returning None")
            return None

        import copernicusmarine
        import xarray as xr

        if start_time is None:
            start_time = datetime.utcnow()

        # Generate cache filename
        cache_file = self._get_cache_path(
            "wave", lat_min, lat_max, lon_min, lon_max, start_time
        )

        # Check cache
        if cache_file.exists():
            logger.info(f"Loading wave data from cache: {cache_file}")
            try:
                ds = xr.open_dataset(cache_file)
            except Exception as e:
                logger.warning(f"Corrupted wave cache, deleting and re-downloading: {e}")
                cache_file.unlink(missing_ok=True)
                ds = None
        else:
            ds = None

        if ds is None:
            logger.info("Downloading wave data from CMEMS...")

            try:
                ds = copernicusmarine.open_dataset(
                    dataset_id=self.CMEMS_WAVE_DATASET,
                    variables=[
                        "VHM0", "VTPK", "VMDR",        # Combined: Hs, peak period, direction
                        "VHM0_WW", "VTM01_WW", "VMDR_WW",  # Wind-wave component
                        "VHM0_SW1", "VTM01_SW1", "VMDR_SW1",  # Primary swell component
                    ],
                    minimum_longitude=lon_min,
                    maximum_longitude=lon_max,
                    minimum_latitude=lat_min,
                    maximum_latitude=lat_max,
                    start_datetime=start_time.strftime("%Y-%m-%dT%H:%M:%S"),
                    end_datetime=(start_time + timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%S"),
                    username=self.cmems_username,
                    password=self.cmems_password,
                )

                if ds is None:
                    logger.error("CMEMS returned None for wave data")
                    return None

                # Save to cache
                ds.to_netcdf(cache_file)

            except Exception as e:
                logger.error(f"Failed to download wave data: {e}")
                return None

        # Extract data
        try:
            # VHM0 = Significant wave height
            hs = ds['VHM0'].values
            lats = ds['latitude'].values
            lons = ds['longitude'].values

            # Take first time step
            if len(hs.shape) == 3:
                hs = hs[0]

            # VTPK = Peak wave period (if available)
            tp = None
            if 'VTPK' in ds:
                tp = ds['VTPK'].values
                if len(tp.shape) == 3:
                    tp = tp[0]
                logger.info("Extracted wave period (VTPK) from CMEMS")

            # VMDR = Mean wave direction (if available)
            wave_dir = None
            if 'VMDR' in ds:
                wave_dir = ds['VMDR'].values
                if len(wave_dir.shape) == 3:
                    wave_dir = wave_dir[0]
                logger.info("Extracted wave direction (VMDR) from CMEMS")

            # Extract wind-wave decomposition (optional — graceful if missing)
            def _extract_var(name):
                if name in ds:
                    v = ds[name].values
                    if len(v.shape) == 3:
                        v = v[0]
                    return v
                return None

            ww_hs = _extract_var('VHM0_WW')
            ww_tp = _extract_var('VTM01_WW')
            ww_dir = _extract_var('VMDR_WW')
            sw_hs = _extract_var('VHM0_SW1')
            sw_tp = _extract_var('VTM01_SW1')
            sw_dir = _extract_var('VMDR_SW1')

            has_decomp = ww_hs is not None and sw_hs is not None
            if has_decomp:
                logger.info("Extracted wind-wave/swell decomposition from CMEMS")
            else:
                logger.info("Swell decomposition not available in this dataset")

            # Replace NaN (land pixels) with 0.0 for JSON serialization
            def _clean(arr):
                return np.nan_to_num(arr, nan=0.0) if arr is not None else None

            hs = _clean(hs)
            tp = _clean(tp)
            wave_dir = _clean(wave_dir)
            ww_hs = _clean(ww_hs)
            ww_tp = _clean(ww_tp)
            ww_dir = _clean(ww_dir)
            sw_hs = _clean(sw_hs)
            sw_tp = _clean(sw_tp)
            sw_dir = _clean(sw_dir)

            return WeatherData(
                parameter="wave_height",
                time=start_time,
                lats=lats,
                lons=lons,
                values=hs,
                unit="m",
                wave_period=tp,
                wave_direction=wave_dir,
                windwave_height=ww_hs,
                windwave_period=ww_tp,
                windwave_direction=ww_dir,
                swell_height=sw_hs,
                swell_period=sw_tp,
                swell_direction=sw_dir,
            )

        except Exception as e:
            logger.error(f"Failed to parse wave data: {e}")
            return None

    # ------------------------------------------------------------------
    # Wave forecast (multi-timestep)
    # ------------------------------------------------------------------
    WAVE_FORECAST_HOURS = list(range(0, 121, 3))  # 0-120h every 3h = 41 steps

    def fetch_wave_forecast(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> Optional[Dict[int, "WeatherData"]]:
        """
        Fetch 0-120h wave forecast from CMEMS in a single download.

        Returns:
            Dict mapping forecast_hour → WeatherData, or None on failure.
        """
        if not self._has_copernicusmarine or not self._has_xarray:
            logger.warning("CMEMS API not available for wave forecast")
            return None

        import copernicusmarine
        import xarray as xr

        logger.info(f"Wave forecast bbox: lat[{lat_min:.1f},{lat_max:.1f}] lon[{lon_min:.1f},{lon_max:.1f}]")

        now = datetime.utcnow()
        # CMEMS analysis+forecast dataset — request next 120 hours
        start_dt = now - timedelta(hours=1)  # slight overlap to ensure t=0
        end_dt = now + timedelta(hours=122)

        cache_key = now.strftime("%Y%m%d_%H")
        cache_file = self.cache_dir / f"wave_forecast_{cache_key}_lat{lat_min:.0f}_{lat_max:.0f}_lon{lon_min:.0f}_{lon_max:.0f}.nc"

        try:
            if cache_file.exists():
                logger.info(f"Loading wave forecast from cache: {cache_file}")
                try:
                    ds = xr.open_dataset(cache_file)
                except Exception as e:
                    logger.warning(f"Corrupted wave forecast cache, deleting and re-downloading: {e}")
                    cache_file.unlink(missing_ok=True)
                    ds = None
            else:
                ds = None

            if ds is None:
                logger.info(f"Downloading CMEMS wave forecast {start_dt} → {end_dt}")
                ds = copernicusmarine.open_dataset(
                    dataset_id=self.CMEMS_WAVE_DATASET,
                    variables=[
                        "VHM0", "VTPK", "VMDR",
                        "VHM0_WW", "VTM01_WW", "VMDR_WW",
                        "VHM0_SW1", "VTM01_SW1", "VMDR_SW1",
                    ],
                    minimum_longitude=lon_min,
                    maximum_longitude=lon_max,
                    minimum_latitude=lat_min,
                    maximum_latitude=lat_max,
                    start_datetime=start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                    end_datetime=end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                    username=self.cmems_username,
                    password=self.cmems_password,
                )
                if ds is None:
                    logger.error("CMEMS returned None for wave forecast")
                    return None
                # Load into memory first to avoid hung lazy reads during to_netcdf
                logger.info("Loading wave forecast data into memory...")
                ds = ds.load()
                ds.to_netcdf(cache_file)
                ds.close()
                logger.info(f"Wave forecast cached: {cache_file}")
                # Validate file isn't truncated (should be at least 1MB for any real forecast)
                fsize = cache_file.stat().st_size
                if fsize < 1_000_000:
                    logger.warning(f"Wave forecast cache suspiciously small ({fsize} bytes), deleting")
                    cache_file.unlink(missing_ok=True)
                    return None
                # Reopen from local file
                ds = xr.open_dataset(cache_file)

            # Extract coordinate arrays
            lats = ds["latitude"].values
            lons = ds["longitude"].values
            times = ds["time"].values  # numpy datetime64 array

            def _clean(arr):
                return np.nan_to_num(arr, nan=0.0) if arr is not None else None

            def _extract_var(name, t_idx):
                if name in ds:
                    v = ds[name].values
                    if len(v.shape) == 3 and t_idx < v.shape[0]:
                        return _clean(v[t_idx])
                return None

            # Map each available timestep to a forecast hour
            import pandas as pd

            base_time = pd.Timestamp(times[0]).to_pydatetime()
            frames: Dict[int, WeatherData] = {}

            for t_idx in range(len(times)):
                ts = pd.Timestamp(times[t_idx]).to_pydatetime()
                fh = round((ts - base_time).total_seconds() / 3600)
                # Only keep 3-hourly steps within 0-120h
                if fh < 0 or fh > 120 or fh % 3 != 0:
                    continue

                hs = _extract_var("VHM0", t_idx)
                if hs is None:
                    continue

                frames[fh] = WeatherData(
                    parameter="wave_height",
                    time=ts,
                    lats=lats,
                    lons=lons,
                    values=hs,
                    unit="m",
                    wave_period=_extract_var("VTPK", t_idx),
                    wave_direction=_extract_var("VMDR", t_idx),
                    windwave_height=_extract_var("VHM0_WW", t_idx),
                    windwave_period=_extract_var("VTM01_WW", t_idx),
                    windwave_direction=_extract_var("VMDR_WW", t_idx),
                    swell_height=_extract_var("VHM0_SW1", t_idx),
                    swell_period=_extract_var("VTM01_SW1", t_idx),
                    swell_direction=_extract_var("VMDR_SW1", t_idx),
                )

            logger.info(f"Wave forecast: {len(frames)} frames extracted (hours: {sorted(frames.keys())})")
            return frames if frames else None

        except Exception as e:
            logger.error(f"Failed to fetch wave forecast: {e}")
            return None

    def fetch_current_data(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        start_time: Optional[datetime] = None,
    ) -> Optional[WeatherData]:
        """
        Fetch ocean current data from CMEMS.

        Args:
            lat_min, lat_max: Latitude bounds
            lon_min, lon_max: Longitude bounds
            start_time: Reference time (default: now)

        Returns:
            WeatherData with u/v current components
        """
        if not self._has_copernicusmarine or not self._has_xarray:
            logger.warning("CMEMS API not available, returning None")
            return None

        if not self.cmems_username or not self.cmems_password:
            logger.warning("CMEMS credentials not configured, returning None")
            return None

        import copernicusmarine
        import xarray as xr

        if start_time is None:
            start_time = datetime.utcnow()

        cache_file = self._get_cache_path(
            "current", lat_min, lat_max, lon_min, lon_max, start_time
        )

        if cache_file.exists():
            logger.info(f"Loading current data from cache: {cache_file}")
            try:
                ds = xr.open_dataset(cache_file)
            except Exception as e:
                logger.warning(f"Corrupted current cache, deleting and re-downloading: {e}")
                cache_file.unlink(missing_ok=True)
                ds = None
        else:
            ds = None

        if ds is None:
            logger.info("Downloading current data from CMEMS...")

            try:
                ds = copernicusmarine.open_dataset(
                    dataset_id=self.CMEMS_PHYSICS_DATASET,
                    variables=["uo", "vo"],  # Eastward/Northward velocity
                    minimum_longitude=lon_min,
                    maximum_longitude=lon_max,
                    minimum_latitude=lat_min,
                    maximum_latitude=lat_max,
                    start_datetime=start_time.strftime("%Y-%m-%dT%H:%M:%S"),
                    end_datetime=(start_time + timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%S"),
                    minimum_depth=0,
                    maximum_depth=10,  # Surface currents
                    username=self.cmems_username,
                    password=self.cmems_password,
                )

                if ds is None:
                    logger.error("CMEMS returned None for current data")
                    return None

                ds.to_netcdf(cache_file)

            except Exception as e:
                logger.error(f"Failed to download current data: {e}")
                return None

        try:
            uo = ds['uo'].values
            vo = ds['vo'].values
            lats = ds['latitude'].values
            lons = ds['longitude'].values

            # Take first time/depth
            if len(uo.shape) == 4:
                uo = uo[0, 0]
                vo = vo[0, 0]
            elif len(uo.shape) == 3:
                uo = uo[0]
                vo = vo[0]

            # Replace NaN (land pixels) with 0.0 for JSON serialization
            uo = np.nan_to_num(uo, nan=0.0)
            vo = np.nan_to_num(vo, nan=0.0)

            return WeatherData(
                parameter="current",
                time=start_time,
                lats=lats,
                lons=lons,
                values=np.sqrt(uo**2 + vo**2),
                unit="m/s",
                u_component=uo,
                v_component=vo,
            )

        except Exception as e:
            logger.error(f"Failed to parse current data: {e}")
            return None

    # ------------------------------------------------------------------
    # Current forecast (multi-timestep)
    # ------------------------------------------------------------------
    CURRENT_FORECAST_HOURS = list(range(0, 121, 3))  # 0-120h every 3h

    @staticmethod
    def _cap_bbox(lat_min, lat_max, lon_min, lon_max, max_span=40.0):
        """Cap bounding box to max_span degrees, centered on the original bbox."""
        lat_mid = (lat_min + lat_max) / 2
        lon_mid = (lon_min + lon_max) / 2
        half = max_span / 2
        return (
            max(lat_mid - half, -85),
            min(lat_mid + half, 85),
            max(lon_mid - half, -180),
            min(lon_mid + half, 180),
        )

    def fetch_current_forecast(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> Optional[Dict[int, "WeatherData"]]:
        """
        Fetch 0-120h surface current forecast from CMEMS in a single download.

        Uses the CMEMS physics dataset (hourly means, 1/12 deg resolution).
        Bbox is capped to 40° span to keep data volume manageable (~500MB).

        Returns:
            Dict mapping forecast_hour -> WeatherData with u/v current components,
            or None on failure.
        """
        if not self._has_copernicusmarine or not self._has_xarray:
            logger.warning("CMEMS API not available for current forecast")
            return None

        import copernicusmarine
        import xarray as xr

        logger.info(f"Current forecast bbox: lat[{lat_min:.1f},{lat_max:.1f}] lon[{lon_min:.1f},{lon_max:.1f}]")

        now = datetime.utcnow()
        start_dt = now - timedelta(hours=1)
        end_dt = now + timedelta(hours=122)

        cache_key = now.strftime("%Y%m%d_%H")
        cache_file = (
            self.cache_dir
            / f"current_forecast_{cache_key}_lat{lat_min:.0f}_{lat_max:.0f}_lon{lon_min:.0f}_{lon_max:.0f}.nc"
        )

        try:
            if cache_file.exists():
                logger.info(f"Loading current forecast from cache: {cache_file}")
                try:
                    ds = xr.open_dataset(cache_file)
                except Exception as e:
                    logger.warning(f"Corrupted current forecast cache, deleting and re-downloading: {e}")
                    cache_file.unlink(missing_ok=True)
                    ds = None
            else:
                ds = None

            if ds is None:
                logger.info(f"Downloading CMEMS current forecast {start_dt} -> {end_dt}")
                ds = copernicusmarine.open_dataset(
                    dataset_id=self.CMEMS_PHYSICS_DATASET,
                    variables=["uo", "vo"],
                    minimum_longitude=lon_min,
                    maximum_longitude=lon_max,
                    minimum_latitude=lat_min,
                    maximum_latitude=lat_max,
                    start_datetime=start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                    end_datetime=end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                    minimum_depth=0,
                    maximum_depth=10,
                    username=self.cmems_username,
                    password=self.cmems_password,
                )
                if ds is None:
                    logger.error("CMEMS returned None for current forecast")
                    return None
                # Load into memory first to avoid hung lazy reads during to_netcdf
                logger.info("Loading current forecast data into memory...")
                ds = ds.load()
                ds.to_netcdf(cache_file)
                ds.close()
                logger.info(f"Current forecast cached: {cache_file}")
                fsize = cache_file.stat().st_size
                if fsize < 1_000_000:
                    logger.warning(f"Current forecast cache suspiciously small ({fsize} bytes), deleting")
                    cache_file.unlink(missing_ok=True)
                    return None
                ds = xr.open_dataset(cache_file)

            lats = ds["latitude"].values
            lons = ds["longitude"].values
            times = ds["time"].values

            import pandas as pd

            base_time = pd.Timestamp(times[0]).to_pydatetime()
            frames: Dict[int, WeatherData] = {}

            for t_idx in range(len(times)):
                ts = pd.Timestamp(times[t_idx]).to_pydatetime()
                fh = round((ts - base_time).total_seconds() / 3600)
                if fh < 0 or fh > 120 or fh % 3 != 0:
                    continue

                uo = ds["uo"].values
                vo = ds["vo"].values

                # Handle depth dimension
                if len(uo.shape) == 4:
                    uo_2d = uo[t_idx, 0]
                    vo_2d = vo[t_idx, 0]
                elif len(uo.shape) == 3:
                    uo_2d = uo[t_idx]
                    vo_2d = vo[t_idx]
                else:
                    continue

                uo_2d = np.nan_to_num(uo_2d, nan=0.0)
                vo_2d = np.nan_to_num(vo_2d, nan=0.0)

                frames[fh] = WeatherData(
                    parameter="current",
                    time=ts,
                    lats=lats,
                    lons=lons,
                    values=np.sqrt(uo_2d ** 2 + vo_2d ** 2),
                    unit="m/s",
                    u_component=uo_2d,
                    v_component=vo_2d,
                )

            logger.info(f"Current forecast: {len(frames)} frames extracted (hours: {sorted(frames.keys())})")
            return frames if frames else None

        except Exception as e:
            logger.error(f"Failed to fetch current forecast: {e}")
            return None

    # ------------------------------------------------------------------
    # SST (Sea Surface Temperature) from CMEMS physics
    # ------------------------------------------------------------------
    def fetch_sst_data(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        start_time: Optional[datetime] = None,
    ) -> Optional[WeatherData]:
        """
        Fetch sea surface temperature from CMEMS physics dataset.

        Returns:
            WeatherData with sst field (°C)
        """
        if not self._has_copernicusmarine or not self._has_xarray:
            logger.warning("CMEMS API not available for SST")
            return None

        if not self.cmems_username or not self.cmems_password:
            logger.warning("CMEMS credentials not configured for SST")
            return None

        import copernicusmarine
        import xarray as xr

        if start_time is None:
            start_time = datetime.utcnow()

        cache_file = self._get_cache_path(
            "sst", lat_min, lat_max, lon_min, lon_max, start_time
        )

        if cache_file.exists():
            logger.info(f"Loading SST data from cache: {cache_file}")
            try:
                ds = xr.open_dataset(cache_file)
            except Exception as e:
                logger.warning(f"Corrupted SST cache, re-downloading: {e}")
                cache_file.unlink(missing_ok=True)
                ds = None
        else:
            ds = None

        if ds is None:
            logger.info("Downloading SST data from CMEMS...")
            try:
                ds = copernicusmarine.open_dataset(
                    dataset_id=self.CMEMS_PHYSICS_DATASET,
                    variables=["thetao"],
                    minimum_longitude=lon_min,
                    maximum_longitude=lon_max,
                    minimum_latitude=lat_min,
                    maximum_latitude=lat_max,
                    start_datetime=start_time.strftime("%Y-%m-%dT%H:%M:%S"),
                    end_datetime=(start_time + timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%S"),
                    minimum_depth=0,
                    maximum_depth=2,
                    username=self.cmems_username,
                    password=self.cmems_password,
                )
                if ds is None:
                    logger.error("CMEMS returned None for SST data")
                    return None
                ds.to_netcdf(cache_file)
            except Exception as e:
                logger.error(f"Failed to download SST data: {e}")
                return None

        try:
            sst = ds['thetao'].values
            lats = ds['latitude'].values
            lons = ds['longitude'].values

            # Take first time/depth
            if len(sst.shape) == 4:
                sst = sst[0, 0]
            elif len(sst.shape) == 3:
                sst = sst[0]

            sst = np.nan_to_num(sst, nan=15.0)

            return WeatherData(
                parameter="sst",
                time=start_time,
                lats=lats,
                lons=lons,
                values=sst,
                unit="°C",
                sst=sst,
            )
        except Exception as e:
            logger.error(f"Failed to parse SST data: {e}")
            return None

    # ------------------------------------------------------------------
    # Sea Ice Concentration from CMEMS
    # ------------------------------------------------------------------
    CMEMS_ICE_DATASET = "cmems_mod_glo_phy_anfc_0.083deg_PT1H-m"

    def fetch_ice_data(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        start_time: Optional[datetime] = None,
    ) -> Optional[WeatherData]:
        """
        Fetch sea ice concentration from CMEMS.

        Returns:
            WeatherData with ice_concentration field (0-1 fraction)
        """
        if not self._has_copernicusmarine or not self._has_xarray:
            logger.warning("CMEMS API not available for ice data")
            return None

        if not self.cmems_username or not self.cmems_password:
            logger.warning("CMEMS credentials not configured for ice data")
            return None

        # Only fetch if region includes high latitudes (>55° or <-55°)
        if abs(lat_max) < 55 and abs(lat_min) < 55:
            logger.info("Region below 55° latitude — skipping ice data fetch")
            return None

        import copernicusmarine
        import xarray as xr

        if start_time is None:
            start_time = datetime.utcnow()

        cache_file = self._get_cache_path(
            "ice", lat_min, lat_max, lon_min, lon_max, start_time
        )

        if cache_file.exists():
            logger.info(f"Loading ice data from cache: {cache_file}")
            try:
                ds = xr.open_dataset(cache_file)
            except Exception as e:
                logger.warning(f"Corrupted ice cache, re-downloading: {e}")
                cache_file.unlink(missing_ok=True)
                ds = None
        else:
            ds = None

        if ds is None:
            logger.info("Downloading ice concentration from CMEMS...")
            try:
                ds = copernicusmarine.open_dataset(
                    dataset_id=self.CMEMS_ICE_DATASET,
                    variables=["siconc"],
                    minimum_longitude=lon_min,
                    maximum_longitude=lon_max,
                    minimum_latitude=lat_min,
                    maximum_latitude=lat_max,
                    start_datetime=start_time.strftime("%Y-%m-%dT%H:%M:%S"),
                    end_datetime=(start_time + timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%S"),
                    username=self.cmems_username,
                    password=self.cmems_password,
                )
                if ds is None:
                    logger.error("CMEMS returned None for ice data")
                    return None
                ds.to_netcdf(cache_file)
            except Exception as e:
                logger.error(f"Failed to download ice data: {e}")
                return None

        try:
            siconc = ds['siconc'].values
            lats = ds['latitude'].values
            lons = ds['longitude'].values

            if len(siconc.shape) == 3:
                siconc = siconc[0]

            # Clamp to 0-1 and replace NaN (open ocean = 0)
            siconc = np.nan_to_num(siconc, nan=0.0)
            siconc = np.clip(siconc, 0.0, 1.0)

            return WeatherData(
                parameter="ice_concentration",
                time=start_time,
                lats=lats,
                lons=lons,
                values=siconc,
                unit="fraction",
                ice_concentration=siconc,
            )
        except Exception as e:
            logger.error(f"Failed to parse ice data: {e}")
            return None

    def get_weather_at_point(
        self,
        lat: float,
        lon: float,
        time: datetime,
        wind_data: Optional[WeatherData] = None,
        wave_data: Optional[WeatherData] = None,
        current_data: Optional[WeatherData] = None,
    ) -> PointWeather:
        """
        Interpolate weather data at a specific point.

        Args:
            lat, lon: Position
            time: Time
            wind_data, wave_data, current_data: Pre-fetched data (optional)

        Returns:
            PointWeather with all parameters
        """
        result = PointWeather(
            lat=lat,
            lon=lon,
            time=time,
            wind_speed_ms=0.0,
            wind_dir_deg=0.0,
            wave_height_m=0.0,
            wave_period_s=0.0,
            wave_dir_deg=0.0,
            current_speed_ms=0.0,
            current_dir_deg=0.0,
        )

        # Interpolate wind
        if wind_data is not None and wind_data.u_component is not None:
            u, v = self._interpolate_vector(
                wind_data.lats, wind_data.lons,
                wind_data.u_component, wind_data.v_component,
                lat, lon
            )
            result.wind_speed_ms = float(np.sqrt(u**2 + v**2))
            result.wind_dir_deg = float((np.degrees(np.arctan2(-u, -v)) + 360) % 360)

        # Interpolate waves
        if wave_data is not None:
            result.wave_height_m = float(self._interpolate_scalar(
                wave_data.lats, wave_data.lons, wave_data.values, lat, lon
            ))

            # Interpolate wave period if available
            if wave_data.wave_period is not None:
                result.wave_period_s = float(self._interpolate_scalar(
                    wave_data.lats, wave_data.lons, wave_data.wave_period, lat, lon
                ))
            else:
                # Fallback: estimate from wave height
                result.wave_period_s = 5.0 + result.wave_height_m

            # Interpolate wave direction if available
            if wave_data.wave_direction is not None:
                result.wave_dir_deg = float(self._interpolate_scalar(
                    wave_data.lats, wave_data.lons, wave_data.wave_direction, lat, lon
                ))

        # Interpolate currents
        if current_data is not None and current_data.u_component is not None:
            u, v = self._interpolate_vector(
                current_data.lats, current_data.lons,
                current_data.u_component, current_data.v_component,
                lat, lon
            )
            result.current_speed_ms = float(np.sqrt(u**2 + v**2))
            result.current_dir_deg = float((np.degrees(np.arctan2(u, v)) + 360) % 360)

        return result

    def _interpolate_scalar(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        values: np.ndarray,
        lat: float,
        lon: float,
    ) -> float:
        """Bilinear interpolation for scalar field."""
        from scipy.interpolate import RegularGridInterpolator

        try:
            # Handle NaN values
            values = np.nan_to_num(values, nan=0.0)

            interp = RegularGridInterpolator(
                (lats, lons), values,
                method='linear',
                bounds_error=False,
                fill_value=0.0
            )
            return float(interp([lat, lon])[0])
        except Exception:
            return 0.0

    def _interpolate_vector(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        lat: float,
        lon: float,
    ) -> Tuple[float, float]:
        """Bilinear interpolation for vector field."""
        u_val = self._interpolate_scalar(lats, lons, u, lat, lon)
        v_val = self._interpolate_scalar(lats, lons, v, lat, lon)
        return u_val, v_val

    def _get_cache_path(
        self,
        data_type: str,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        time: datetime,
    ) -> Path:
        """Generate cache file path."""
        time_str = time.strftime("%Y%m%d_%H")
        filename = f"{data_type}_{time_str}_lat{lat_min:.1f}_{lat_max:.1f}_lon{lon_min:.1f}_{lon_max:.1f}.nc"
        return self.cache_dir / filename

    def clear_cache(self, older_than_days: int = 7) -> int:
        """Remove old cached files."""
        cutoff = datetime.now() - timedelta(days=older_than_days)
        count = 0

        for f in self.cache_dir.glob("*.nc"):
            if f.stat().st_mtime < cutoff.timestamp():
                f.unlink()
                count += 1

        logger.info(f"Cleared {count} old cache files")
        return count


# Fallback: synthetic data generator for when APIs are not available
class SyntheticDataProvider:
    """
    Generates synthetic weather data for development/demo.

    Use this when Copernicus APIs are not configured.
    """

    def generate_wind_field(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        resolution: float = 1.0,
        time: Optional[datetime] = None,
    ) -> WeatherData:
        """Generate synthetic wind field."""
        if time is None:
            time = datetime.utcnow()

        lats = np.arange(lat_min, lat_max + resolution, resolution)
        lons = np.arange(lon_min, lon_max + resolution, resolution)

        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Base westerlies
        base_u = 5.0 + 3.0 * np.sin(np.radians(lat_grid * 2))
        base_v = 2.0 * np.cos(np.radians(lon_grid * 3 + lat_grid * 2))

        # Add weather system
        hour_factor = np.sin(time.hour * np.pi / 12) if time else 0.5
        center_lat = 45.0 + 5.0 * hour_factor
        center_lon = 0.0 + 10.0 * hour_factor

        dist = np.sqrt((lat_grid - center_lat)**2 + (lon_grid - center_lon)**2)
        system_strength = 8.0 * np.exp(-dist / 10.0)

        angle_to_center = np.arctan2(lat_grid - center_lat, lon_grid - center_lon)
        u_cyclonic = -system_strength * np.sin(angle_to_center + np.pi/2)
        v_cyclonic = system_strength * np.cos(angle_to_center + np.pi/2)

        u_wind = base_u + u_cyclonic + np.random.randn(*lat_grid.shape) * 0.5
        v_wind = base_v + v_cyclonic + np.random.randn(*lat_grid.shape) * 0.5

        return WeatherData(
            parameter="wind",
            time=time,
            lats=lats,
            lons=lons,
            values=np.sqrt(u_wind**2 + v_wind**2),
            unit="m/s",
            u_component=u_wind,
            v_component=v_wind,
        )

    def generate_wave_field(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        resolution: float = 1.0,
        wind_data: Optional[WeatherData] = None,
    ) -> WeatherData:
        """Generate synthetic wave field with wind-wave/swell decomposition."""
        time = datetime.utcnow()

        lats = np.arange(lat_min, lat_max + resolution, resolution)
        lons = np.arange(lon_min, lon_max + resolution, resolution)

        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Wind-wave component: driven by local wind
        if wind_data is not None and wind_data.values is not None:
            wind_speed = wind_data.values
            ww_height = 0.12 * wind_speed + np.random.randn(*wind_speed.shape) * 0.2
            # Wind-wave direction follows wind direction
            if wind_data.u_component is not None and wind_data.v_component is not None:
                ww_dir = np.degrees(np.arctan2(-wind_data.u_component, -wind_data.v_component)) % 360
            else:
                ww_dir = np.full_like(ww_height, 270.0)
        else:
            ww_height = 0.8 + 0.5 * np.sin(np.radians(lat_grid * 3))
            ww_dir = np.full_like(ww_height, 270.0)

        ww_height = np.maximum(ww_height, 0.2)
        ww_period = 3.0 + 0.8 * ww_height  # Short-period wind sea

        # Swell component: long-period waves from distant storms
        # Swell typically comes from a consistent direction, independent of local wind
        swell_base = 1.0 + 0.8 * np.sin(np.radians(lat_grid * 2 + 30))
        sw_height = np.maximum(swell_base + np.random.randn(*lat_grid.shape) * 0.15, 0.3)
        sw_period = 10.0 + 2.0 * sw_height  # Long-period swell
        sw_dir = np.full_like(sw_height, 300.0) + np.random.randn(*lat_grid.shape) * 5  # NW swell

        # Combined sea state (RSS of components)
        wave_height = np.sqrt(ww_height**2 + sw_height**2)
        # Combined period: energy-weighted
        total_energy = ww_height**2 + sw_height**2
        wave_period = np.where(
            total_energy > 0,
            (ww_height**2 * ww_period + sw_height**2 * sw_period) / total_energy,
            8.0,
        )
        # Combined direction: dominant component
        wave_dir = np.where(sw_height > ww_height, sw_dir, ww_dir)

        return WeatherData(
            parameter="wave_height",
            time=time,
            lats=lats,
            lons=lons,
            values=wave_height,
            unit="m",
            wave_period=wave_period,
            wave_direction=wave_dir % 360,
            windwave_height=ww_height,
            windwave_period=ww_period,
            windwave_direction=ww_dir % 360,
            swell_height=sw_height,
            swell_period=sw_period,
            swell_direction=sw_dir % 360,
        )


    def generate_sst_field(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        resolution: float = 1.0,
        time: Optional[datetime] = None,
    ) -> WeatherData:
        """Generate synthetic SST field based on latitude."""
        if time is None:
            time = datetime.utcnow()

        lats = np.arange(lat_min, lat_max + resolution, resolution)
        lons = np.arange(lon_min, lon_max + resolution, resolution)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # SST decreases with latitude: ~28°C at equator, ~0°C at poles
        # Seasonal variation: ±3°C
        month = time.month
        seasonal = 3.0 * np.cos(np.radians((month - 7) * 30))  # Peak in July (NH)
        base_sst = 28.0 - 0.5 * np.abs(lat_grid)
        sst = base_sst + seasonal * np.sign(lat_grid) + np.random.randn(*lat_grid.shape) * 0.3
        sst = np.clip(sst, -2.0, 32.0)

        return WeatherData(
            parameter="sst",
            time=time,
            lats=lats,
            lons=lons,
            values=sst,
            unit="°C",
            sst=sst,
        )

    def generate_visibility_field(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        resolution: float = 1.0,
        time: Optional[datetime] = None,
    ) -> WeatherData:
        """Generate synthetic visibility field."""
        if time is None:
            time = datetime.utcnow()

        lats = np.arange(lat_min, lat_max + resolution, resolution)
        lons = np.arange(lon_min, lon_max + resolution, resolution)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Generally good visibility at sea (20-50 km), reduced near coasts and in high latitudes
        base_vis = 30.0 + 10.0 * np.random.rand(*lat_grid.shape)
        # Reduced visibility at high latitudes (fog/mist)
        high_lat_reduction = np.maximum(0, (np.abs(lat_grid) - 50) * 0.5)
        vis = np.maximum(base_vis - high_lat_reduction, 1.0)

        return WeatherData(
            parameter="visibility",
            time=time,
            lats=lats,
            lons=lons,
            values=vis,
            unit="km",
            visibility=vis,
        )

    def generate_ice_field(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        resolution: float = 1.0,
        time: Optional[datetime] = None,
    ) -> WeatherData:
        """Generate synthetic ice concentration field."""
        if time is None:
            time = datetime.utcnow()

        lats = np.arange(lat_min, lat_max + resolution, resolution)
        lons = np.arange(lon_min, lon_max + resolution, resolution)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Ice only at high latitudes (>65°)
        # Seasonal: more in winter, less in summer
        month = time.month
        # NH winter months
        nh_seasonal = 1.0 if month in [12, 1, 2, 3] else 0.5 if month in [4, 11] else 0.2
        sh_seasonal = 1.0 if month in [6, 7, 8, 9] else 0.5 if month in [5, 10] else 0.2

        ice = np.zeros_like(lat_grid)
        # Northern hemisphere ice
        nh_mask = lat_grid > 65
        ice[nh_mask] = np.clip((lat_grid[nh_mask] - 65) / 15 * nh_seasonal, 0, 1)
        # Southern hemisphere ice
        sh_mask = lat_grid < -60
        ice[sh_mask] = np.clip((-lat_grid[sh_mask] - 60) / 15 * sh_seasonal, 0, 1)

        return WeatherData(
            parameter="ice_concentration",
            time=time,
            lats=lats,
            lons=lons,
            values=ice,
            unit="fraction",
            ice_concentration=ice,
        )

    def generate_current_field(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        resolution: float = 1.0,
        time: Optional[datetime] = None,
    ) -> WeatherData:
        """Generate synthetic ocean current field.

        Models major surface current patterns:
        - Gulf Stream / North Atlantic Drift
        - Mediterranean circulation
        - General wind-driven surface currents
        """
        if time is None:
            time = datetime.utcnow()

        lats = np.arange(lat_min, lat_max + resolution, resolution)
        lons = np.arange(lon_min, lon_max + resolution, resolution)

        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Base eastward drift (wind-driven surface current, ~2% of wind)
        base_u = 0.15 + 0.1 * np.sin(np.radians(lat_grid * 2))
        base_v = 0.05 * np.cos(np.radians(lon_grid * 3))

        # Gulf Stream / North Atlantic Drift influence
        # Strong northeastward flow centered around 40-45N, -40 to 0W
        gs_lat_center = 42.0
        gs_strength = 0.8 * np.exp(-((lat_grid - gs_lat_center) ** 2) / 50)
        gs_u = gs_strength * 0.6
        gs_v = gs_strength * 0.3

        # Mediterranean counter-clockwise gyre
        med_lat_center = 37.0
        med_lon_center = 18.0
        med_dist = np.sqrt((lat_grid - med_lat_center) ** 2 + (lon_grid - med_lon_center) ** 2)
        med_strength = 0.3 * np.exp(-med_dist / 8)
        med_angle = np.arctan2(lat_grid - med_lat_center, lon_grid - med_lon_center)
        med_u = -med_strength * np.sin(med_angle)
        med_v = med_strength * np.cos(med_angle)

        u_current = base_u + gs_u + med_u + np.random.randn(*lat_grid.shape) * 0.02
        v_current = base_v + gs_v + med_v + np.random.randn(*lat_grid.shape) * 0.02

        return WeatherData(
            parameter="current",
            time=time,
            lats=lats,
            lons=lons,
            values=np.sqrt(u_current ** 2 + v_current ** 2),
            unit="m/s",
            u_component=u_current,
            v_component=v_current,
        )


class GFSDataProvider:
    """
    Near-real-time wind data from NOAA GFS (Global Forecast System).

    GFS provides 0.25° resolution wind data updated every 6 hours,
    available ~3.5 hours after each model run. This is much more
    current than ERA5 reanalysis which has a ~5-day lag.

    Data source: NOAA NOMADS GRIB filter (pre-filtered GRIB2 download).
    """

    # GFS run hours
    RUN_HOURS = [0, 6, 12, 18]

    # Approximate delay before GFS data becomes available
    AVAILABILITY_LAG_HOURS = 3.5

    NOMADS_BASE = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"

    def __init__(self, cache_dir: str = "data/gfs_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_latest_run(self) -> Tuple[str, str]:
        """
        Determine the latest available GFS run.

        Returns:
            Tuple of (date_str "YYYYMMDD", hour_str "HH")
        """
        now = datetime.utcnow()
        # Subtract availability lag to find what's actually ready
        available_time = now - timedelta(hours=self.AVAILABILITY_LAG_HOURS)

        # Round down to nearest GFS run hour
        run_hour = max(h for h in self.RUN_HOURS if h <= available_time.hour)
        run_date = available_time.strftime("%Y%m%d")

        return run_date, f"{run_hour:02d}"

    def _to_gfs_lon(self, lon: float) -> float:
        """Convert -180..180 longitude to GFS 0..360 convention."""
        return lon % 360

    def _download_grib(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        run_date: str,
        run_hour: str,
        forecast_hour: int = 0,
    ) -> Optional[Path]:
        """
        Download a subregion GRIB2 file from NOMADS.

        Args:
            forecast_hour: GFS forecast hour (0=analysis, 3-120 in 3h steps)

        Returns:
            Path to downloaded GRIB2 file, or None on failure.
        """
        import urllib.request
        import urllib.parse

        # Convert longitudes to GFS 0-360 convention
        gfs_lon_min = self._to_gfs_lon(lon_min)
        gfs_lon_max = self._to_gfs_lon(lon_max)

        # Handle wrap-around (e.g. lon_min=-15 → 345, lon_max=40 → 40)
        # If gfs_lon_min > gfs_lon_max, the region crosses the prime meridian
        # in GFS coordinates. NOMADS handles this correctly.
        if gfs_lon_min > gfs_lon_max:
            gfs_lon_min = lon_min
            gfs_lon_max = lon_max

        cache_file = (
            self.cache_dir
            / f"gfs_{run_date}_{run_hour}_f{forecast_hour:03d}_lat{lat_min:.0f}_{lat_max:.0f}_lon{lon_min:.0f}_{lon_max:.0f}.grib2"
        )

        # Use cache if file exists and is from current run
        if cache_file.exists():
            logger.info(f"GFS cache hit: {cache_file.name}")
            return cache_file

        # Build NOMADS GRIB filter URL
        params = {
            "file": f"gfs.t{run_hour}z.pgrb2.0p25.f{forecast_hour:03d}",
            "lev_10_m_above_ground": "on",
            "var_UGRD": "on",
            "var_VGRD": "on",
            "subregion": "",
            "leftlon": str(gfs_lon_min),
            "rightlon": str(gfs_lon_max),
            "toplat": str(lat_max),
            "bottomlat": str(lat_min),
            "dir": f"/gfs.{run_date}/{run_hour}/atmos",
        }

        url = f"{self.NOMADS_BASE}?{urllib.parse.urlencode(params)}"
        logger.info(f"Downloading GFS GRIB2: run={run_date}/{run_hour}z f{forecast_hour:03d}, region=[{lat_min},{lat_max}]x[{lon_min},{lon_max}]")

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Windmar/2.1"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()

            if len(data) < 100:
                logger.warning(f"GFS download too small ({len(data)} bytes), likely an error page")
                return None

            cache_file.write_bytes(data)
            logger.info(f"GFS GRIB2 saved: {cache_file.name} ({len(data)} bytes)")
            return cache_file

        except Exception as e:
            logger.warning(f"GFS download failed: {e}")
            return None

    def fetch_wind_data(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        time: Optional[datetime] = None,
        forecast_hour: int = 0,
    ) -> Optional[WeatherData]:
        """
        Fetch near-real-time wind data from GFS.

        Args:
            forecast_hour: GFS forecast hour (0=analysis, 3-120 in 3h steps)

        Returns:
            WeatherData with u/v wind components, or None on failure.
        """
        try:
            import pygrib
        except ImportError:
            logger.warning("pygrib not installed, GFS provider unavailable")
            return None

        if time is None:
            time = datetime.utcnow()

        run_date, run_hour = self._get_latest_run()

        grib_path = self._download_grib(lat_min, lat_max, lon_min, lon_max, run_date, run_hour, forecast_hour)
        if grib_path is None:
            return None

        try:
            grbs = pygrib.open(str(grib_path))

            u_msgs = grbs.select(shortName='10u')
            v_msgs = grbs.select(shortName='10v')

            if not u_msgs or not v_msgs:
                logger.warning("GFS GRIB2 missing U/V wind messages")
                grbs.close()
                return None

            u_msg = u_msgs[0]
            v_msg = v_msgs[0]

            u_data = u_msg.values  # 2D numpy array
            v_data = v_msg.values

            lats_2d, lons_2d = u_msg.latlons()

            # Extract 1D coordinate vectors
            lats = lats_2d[:, 0]
            lons = lons_2d[0, :]

            grbs.close()

            # Convert GFS 0-360 longitudes to -180..180
            lon_shift = lons > 180
            if np.any(lon_shift):
                lons[lon_shift] -= 360
                # Re-sort by longitude to keep ascending order
                sort_idx = np.argsort(lons)
                lons = lons[sort_idx]
                u_data = u_data[:, sort_idx]
                v_data = v_data[:, sort_idx]

            # Replace NaN with 0 (land pixels in some GRIB files)
            u_data = np.nan_to_num(u_data, nan=0.0)
            v_data = np.nan_to_num(v_data, nan=0.0)

            speed = np.sqrt(u_data**2 + v_data**2)

            ref_time = datetime.strptime(f"{run_date}{run_hour}", "%Y%m%d%H") + timedelta(hours=forecast_hour)

            logger.info(
                f"GFS wind data fetched: {len(lats)}x{len(lons)} grid, "
                f"run={run_date}/{run_hour}z f{forecast_hour:03d}, "
                f"valid={ref_time.strftime('%Y-%m-%d %H:%M')}Z, "
                f"wind range={speed.min():.1f}-{speed.max():.1f} m/s"
            )

            return WeatherData(
                parameter="wind",
                time=ref_time,
                lats=lats,
                lons=lons,
                values=speed,
                unit="m/s",
                u_component=u_data,
                v_component=v_data,
            )

        except Exception as e:
            logger.error(f"Failed to parse GFS GRIB2: {e}")
            return None

    def fetch_visibility_data(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        time: Optional[datetime] = None,
        forecast_hour: int = 0,
    ) -> Optional[WeatherData]:
        """
        Fetch visibility data from GFS.

        GFS provides surface visibility (VIS) in meters.

        Returns:
            WeatherData with visibility field (km)
        """
        try:
            import pygrib
        except ImportError:
            logger.warning("pygrib not installed, visibility data unavailable")
            return None

        if time is None:
            time = datetime.utcnow()

        run_date, run_hour = self._get_latest_run()

        # Build NOMADS URL with VIS variable
        import urllib.request
        import urllib.parse

        gfs_lon_min = self._to_gfs_lon(lon_min)
        gfs_lon_max = self._to_gfs_lon(lon_max)
        if gfs_lon_min > gfs_lon_max:
            gfs_lon_min = lon_min
            gfs_lon_max = lon_max

        cache_file = (
            self.cache_dir
            / f"gfs_vis_{run_date}_{run_hour}_f{forecast_hour:03d}_lat{lat_min:.0f}_{lat_max:.0f}_lon{lon_min:.0f}_{lon_max:.0f}.grib2"
        )

        if cache_file.exists():
            logger.info(f"GFS visibility cache hit: {cache_file.name}")
        else:
            params = {
                "file": f"gfs.t{run_hour}z.pgrb2.0p25.f{forecast_hour:03d}",
                "lev_surface": "on",
                "var_VIS": "on",
                "subregion": "",
                "leftlon": str(gfs_lon_min),
                "rightlon": str(gfs_lon_max),
                "toplat": str(lat_max),
                "bottomlat": str(lat_min),
                "dir": f"/gfs.{run_date}/{run_hour}/atmos",
            }
            url = f"{self.NOMADS_BASE}?{urllib.parse.urlencode(params)}"
            logger.info(f"Downloading GFS visibility: f{forecast_hour:03d}")

            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Windmar/2.1"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = resp.read()
                if len(data) < 100:
                    logger.warning(f"GFS visibility download too small ({len(data)} bytes)")
                    return None
                cache_file.write_bytes(data)
            except Exception as e:
                logger.warning(f"GFS visibility download failed: {e}")
                return None

        try:
            grbs = pygrib.open(str(cache_file))
            vis_msgs = grbs.select(shortName='vis')
            if not vis_msgs:
                grbs.close()
                logger.warning("GFS GRIB2 missing VIS message")
                return None

            vis_msg = vis_msgs[0]
            vis_data = vis_msg.values  # meters
            lats_2d, lons_2d = vis_msg.latlons()
            lats = lats_2d[:, 0]
            lons = lons_2d[0, :]
            grbs.close()

            # Convert 0-360 to -180..180
            lon_shift = lons > 180
            if np.any(lon_shift):
                lons[lon_shift] -= 360
                sort_idx = np.argsort(lons)
                lons = lons[sort_idx]
                vis_data = vis_data[:, sort_idx]

            # Convert meters to km, replace NaN
            vis_km = np.nan_to_num(vis_data, nan=50000.0) / 1000.0
            vis_km = np.clip(vis_km, 0.0, 100.0)

            ref_time = datetime.strptime(f"{run_date}{run_hour}", "%Y%m%d%H") + timedelta(hours=forecast_hour)

            logger.info(
                f"GFS visibility fetched: {len(lats)}x{len(lons)} grid, "
                f"range={vis_km.min():.1f}-{vis_km.max():.1f} km"
            )

            return WeatherData(
                parameter="visibility",
                time=ref_time,
                lats=lats,
                lons=lons,
                values=vis_km,
                unit="km",
                visibility=vis_km,
            )
        except Exception as e:
            logger.error(f"Failed to parse GFS visibility GRIB2: {e}")
            return None

    # All GFS forecast hours: f000 to f120 in 3h steps
    FORECAST_HOURS = list(range(0, 121, 3))  # 41 files

    def prefetch_forecast_hours(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> Dict[int, Path]:
        """
        Download all forecast hours (f000-f120, 3h steps) for the current GFS run.

        Skips already-cached files. Rate-limits at 2s between NOMADS requests.

        Returns:
            Dict mapping forecast_hour → Path for successfully downloaded files.
        """
        import time as _time

        run_date, run_hour = self._get_latest_run()
        results: Dict[int, Path] = {}

        for fh in self.FORECAST_HOURS:
            path = self._download_grib(lat_min, lat_max, lon_min, lon_max, run_date, run_hour, fh)
            if path is not None:
                results[fh] = path
            else:
                logger.warning(f"Failed to download GFS f{fh:03d}")
            # Rate limit: only sleep if we actually hit the network (no cache hit)
            # Check if file was just created (within last 5 seconds)
            if path is not None and (_time.time() - path.stat().st_mtime) < 5:
                _time.sleep(2)

        logger.info(f"GFS prefetch complete: {len(results)}/{len(self.FORECAST_HOURS)} forecast hours cached")
        return results

    def get_cached_forecast_hours(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> List[Dict]:
        """
        Check which forecast hours are cached for the current GFS run.

        Returns:
            List of dicts with forecast_hour, valid_time, and cached status.
        """
        run_date, run_hour = self._get_latest_run()
        run_time = datetime.strptime(f"{run_date}{run_hour}", "%Y%m%d%H")
        result = []

        for fh in self.FORECAST_HOURS:
            cache_file = (
                self.cache_dir
                / f"gfs_{run_date}_{run_hour}_f{fh:03d}_lat{lat_min:.0f}_{lat_max:.0f}_lon{lon_min:.0f}_{lon_max:.0f}.grib2"
            )
            valid_time = run_time + timedelta(hours=fh)
            result.append({
                "forecast_hour": fh,
                "valid_time": valid_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "cached": cache_file.exists(),
            })

        return result

    def clear_old_cache(self, keep_hours: int = 12) -> int:
        """Remove GRIB cache files older than keep_hours."""
        cutoff = datetime.utcnow() - timedelta(hours=keep_hours)
        count = 0
        for f in self.cache_dir.glob("*.grib2"):
            if f.stat().st_mtime < cutoff.timestamp():
                f.unlink()
                count += 1
        if count:
            logger.info(f"Cleared {count} old GFS cache files")
        return count


class ClimatologyProvider:
    """
    Provides climatological (historical average) weather data.

    Uses ERA5 monthly means for wind and waves.
    This is the fallback when forecast horizon is exceeded.

    Data source: Copernicus CDS ERA5 Monthly Averaged Data
    """

    # Forecast horizon in days (after this, blend to climatology)
    FORECAST_HORIZON_DAYS = 10
    BLEND_WINDOW_DAYS = 2  # Days over which to transition

    # ERA5 monthly means dataset
    CDS_MONTHLY_DATASET = "reanalysis-era5-single-levels-monthly-means"

    def __init__(self, cache_dir: str = "data/climatology_cache"):
        """
        Initialize climatology provider.

        Args:
            cache_dir: Directory to cache climatology data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Check dependencies
        self._has_cdsapi = False
        self._has_xarray = False
        try:
            import xarray
            self._has_xarray = True
        except ImportError:
            pass
        try:
            import cdsapi
            self._has_cdsapi = True
        except ImportError:
            pass

        # In-memory cache for monthly data
        self._monthly_cache: Dict[str, any] = {}

    def get_climatology_at_point(
        self,
        lat: float,
        lon: float,
        month: int,
    ) -> PointWeather:
        """
        Get climatological weather for a location and month.

        Args:
            lat, lon: Position
            month: Month (1-12)

        Returns:
            PointWeather with climatological values
        """
        # Try to get from ERA5 monthly means
        clim_data = self._get_monthly_data(month, lat, lon)

        if clim_data:
            return clim_data

        # Fallback: use built-in climatology tables
        return self._builtin_climatology(lat, lon, month)

    def _get_monthly_data(
        self,
        month: int,
        lat: float,
        lon: float,
    ) -> Optional[PointWeather]:
        """Fetch ERA5 monthly mean data."""
        if not self._has_cdsapi or not self._has_xarray:
            return None

        cache_key = f"month_{month:02d}"

        # Check in-memory cache
        if cache_key in self._monthly_cache:
            return self._interpolate_from_cache(
                self._monthly_cache[cache_key], lat, lon, month
            )

        # Check file cache
        cache_file = self.cache_dir / f"era5_monthly_{month:02d}.nc"

        if cache_file.exists():
            import xarray as xr
            try:
                ds = xr.open_dataset(cache_file)
                self._monthly_cache[cache_key] = ds
                return self._interpolate_from_cache(ds, lat, lon, month)
            except Exception as e:
                logger.warning(f"Failed to load cached climatology: {e}")

        # Download from CDS
        try:
            import cdsapi
            import xarray as xr

            logger.info(f"Downloading ERA5 monthly mean for month {month}...")

            client = cdsapi.Client()

            # Request monthly means for this month across multiple years
            # to get a robust average
            client.retrieve(
                self.CDS_MONTHLY_DATASET,
                {
                    'product_type': 'monthly_averaged_reanalysis',
                    'variable': [
                        '10m_u_component_of_wind',
                        '10m_v_component_of_wind',
                        'significant_height_of_combined_wind_waves_and_swell',
                        'mean_wave_direction',
                    ],
                    'year': ['2019', '2020', '2021', '2022', '2023'],
                    'month': [f'{month:02d}'],
                    'time': '00:00',
                    'format': 'netcdf',
                },
                str(cache_file)
            )

            ds = xr.open_dataset(cache_file)

            # Average across years
            ds = ds.mean(dim='time')
            self._monthly_cache[cache_key] = ds

            return self._interpolate_from_cache(ds, lat, lon, month)

        except Exception as e:
            logger.warning(f"Failed to download ERA5 monthly data: {e}")
            return None

    def _interpolate_from_cache(
        self,
        ds: any,
        lat: float,
        lon: float,
        month: int,
    ) -> PointWeather:
        """Interpolate climatology values from xarray dataset."""
        from scipy.interpolate import RegularGridInterpolator

        try:
            lats = ds['latitude'].values
            lons = ds['longitude'].values

            # Normalize longitude to dataset range
            if lon < 0 and lons.min() >= 0:
                lon = lon + 360

            # Get variables (names may vary)
            u10 = ds['u10'].values if 'u10' in ds else ds['10m_u_component_of_wind'].values
            v10 = ds['v10'].values if 'v10' in ds else ds['10m_v_component_of_wind'].values

            # Wave height (may not be in monthly means)
            if 'swh' in ds:
                wave_h = ds['swh'].values
            elif 'significant_height_of_combined_wind_waves_and_swell' in ds:
                wave_h = ds['significant_height_of_combined_wind_waves_and_swell'].values
            else:
                wave_h = None

            # Handle dimensions
            if len(u10.shape) > 2:
                u10 = u10.mean(axis=0)
                v10 = v10.mean(axis=0)
                if wave_h is not None:
                    wave_h = wave_h.mean(axis=0)

            # Interpolate
            def interp_scalar(values):
                values = np.nan_to_num(values, nan=0.0)
                interp = RegularGridInterpolator(
                    (lats, lons), values,
                    method='linear', bounds_error=False, fill_value=0.0
                )
                return float(interp([lat, lon])[0])

            u_val = interp_scalar(u10)
            v_val = interp_scalar(v10)

            wind_speed = np.sqrt(u_val**2 + v_val**2)
            wind_dir = (np.degrees(np.arctan2(-u_val, -v_val)) + 360) % 360

            wave_height = interp_scalar(wave_h) if wave_h is not None else self._estimate_wave_height(wind_speed)

            return PointWeather(
                lat=lat,
                lon=lon,
                time=datetime(2000, month, 15),  # Placeholder time
                wind_speed_ms=wind_speed,
                wind_dir_deg=wind_dir,
                wave_height_m=wave_height,
                wave_period_s=5.0 + wave_height,  # Estimate
                wave_dir_deg=wind_dir,  # Assume waves follow wind
                current_speed_ms=0.0,
                current_dir_deg=0.0,
            )

        except Exception as e:
            logger.warning(f"Failed to interpolate climatology: {e}")
            return self._builtin_climatology(lat, lon, month)

    def _estimate_wave_height(self, wind_speed_ms: float) -> float:
        """Estimate wave height from wind speed (simplified)."""
        # Simplified Pierson-Moskowitz relationship
        # Hs ≈ 0.21 * U^2 / g for fully developed seas
        # Use a more conservative estimate
        return min(0.15 * wind_speed_ms, 8.0)

    def _builtin_climatology(
        self,
        lat: float,
        lon: float,
        month: int,
    ) -> PointWeather:
        """
        Built-in climatology based on general oceanic patterns.

        This is the fallback when ERA5 data is unavailable.
        Based on typical patterns from Pilot Charts.
        """
        # Determine ocean basin
        is_north = lat > 0
        is_atlantic = -80 < lon < 0
        is_pacific = lon < -80 or lon > 100

        # Seasonal factor (Northern Hemisphere winter = more wind)
        winter_months = [12, 1, 2, 3] if is_north else [6, 7, 8, 9]
        is_winter = month in winter_months
        seasonal_factor = 1.3 if is_winter else 0.9

        # Latitude-based wind patterns
        abs_lat = abs(lat)

        if abs_lat < 10:
            # ITCZ / Doldrums
            base_wind = 3.0
            wind_dir = 90 if is_north else 270  # Light easterlies
        elif abs_lat < 30:
            # Trade wind belt
            base_wind = 7.0
            wind_dir = 45 if is_north else 315  # NE trades / SE trades
        elif abs_lat < 50:
            # Westerlies
            base_wind = 9.0
            wind_dir = 250 if is_north else 290
        else:
            # Roaring 40s/50s
            base_wind = 12.0
            wind_dir = 270

        # Apply seasonal adjustment
        wind_speed = base_wind * seasonal_factor

        # Wave height from wind (simplified)
        wave_height = self._estimate_wave_height(wind_speed)

        # North Atlantic / North Pacific winter storms
        if is_north and (is_atlantic or is_pacific) and abs_lat > 40 and is_winter:
            wind_speed *= 1.2
            wave_height *= 1.3

        return PointWeather(
            lat=lat,
            lon=lon,
            time=datetime(2000, month, 15),
            wind_speed_ms=wind_speed,
            wind_dir_deg=wind_dir,
            wave_height_m=wave_height,
            wave_period_s=5.0 + wave_height,
            wave_dir_deg=wind_dir,
            current_speed_ms=0.0,
            current_dir_deg=0.0,
        )


@dataclass
class WeatherDataSource:
    """Indicates the source of weather data."""
    source: str  # "forecast", "climatology", or "blended"
    forecast_weight: float  # 1.0 = pure forecast, 0.0 = pure climatology
    forecast_age_hours: float  # How old is the forecast
    message: Optional[str] = None


class UnifiedWeatherProvider:
    """
    Unified weather provider that seamlessly blends forecast and climatology.

    - Uses Copernicus forecast data when available
    - Transitions to climatology beyond forecast horizon
    - Provides data source metadata for UI
    """

    def __init__(
        self,
        copernicus: Optional[CopernicusDataProvider] = None,
        climatology: Optional[ClimatologyProvider] = None,
        cache_dir: str = "data/weather_cache",
    ):
        """
        Initialize unified provider.

        Args:
            copernicus: Copernicus provider (created if None)
            climatology: Climatology provider (created if None)
            cache_dir: Cache directory
        """
        self.copernicus = copernicus or CopernicusDataProvider(cache_dir=cache_dir)
        self.climatology = climatology or ClimatologyProvider(cache_dir=f"{cache_dir}/climatology")

        # Forecast horizon settings
        self.forecast_horizon_days = ClimatologyProvider.FORECAST_HORIZON_DAYS
        self.blend_window_days = ClimatologyProvider.BLEND_WINDOW_DAYS

        # Cache for fetched forecast data
        self._forecast_cache: Dict[str, Tuple[WeatherData, datetime]] = {}
        self._forecast_valid_time: Optional[datetime] = None

    def get_weather_at_point(
        self,
        lat: float,
        lon: float,
        time: datetime,
    ) -> Tuple[PointWeather, WeatherDataSource]:
        """
        Get weather at a point, blending forecast and climatology as needed.

        Args:
            lat, lon: Position
            time: Requested time

        Returns:
            Tuple of (PointWeather, WeatherDataSource)
        """
        now = datetime.utcnow()
        hours_ahead = (time - now).total_seconds() / 3600
        days_ahead = hours_ahead / 24

        # Determine blend weight
        if days_ahead <= self.forecast_horizon_days:
            # Within forecast horizon - use forecast
            forecast_weight = 1.0
            source_type = "forecast"
        elif days_ahead <= self.forecast_horizon_days + self.blend_window_days:
            # In blend window - transition
            blend_progress = (days_ahead - self.forecast_horizon_days) / self.blend_window_days
            forecast_weight = 1.0 - blend_progress
            source_type = "blended"
        else:
            # Beyond blend window - pure climatology
            forecast_weight = 0.0
            source_type = "climatology"

        # Get data from appropriate sources
        if forecast_weight > 0:
            forecast_wx = self._get_forecast_weather(lat, lon, time)
        else:
            forecast_wx = None

        if forecast_weight < 1.0:
            clim_wx = self.climatology.get_climatology_at_point(lat, lon, time.month)
        else:
            clim_wx = None

        # Blend if needed
        if forecast_weight == 1.0 and forecast_wx:
            result_wx = forecast_wx
        elif forecast_weight == 0.0 and clim_wx:
            result_wx = clim_wx
        elif forecast_wx and clim_wx:
            result_wx = self._blend_weather(forecast_wx, clim_wx, forecast_weight, lat, lon, time)
        elif forecast_wx:
            result_wx = forecast_wx
        elif clim_wx:
            result_wx = clim_wx
        else:
            # Fallback to built-in climatology
            result_wx = self.climatology._builtin_climatology(lat, lon, time.month)

        # Create source metadata
        source = WeatherDataSource(
            source=source_type,
            forecast_weight=forecast_weight,
            forecast_age_hours=hours_ahead,
            message=self._get_source_message(source_type, days_ahead),
        )

        return result_wx, source

    def _get_forecast_weather(
        self,
        lat: float,
        lon: float,
        time: datetime,
    ) -> Optional[PointWeather]:
        """Get forecast weather from Copernicus."""
        # Use a bounding box around the point
        margin = 2.0  # degrees
        lat_min, lat_max = lat - margin, lat + margin
        lon_min, lon_max = lon - margin, lon + margin

        try:
            # Fetch wind data
            wind_data = self.copernicus.fetch_wind_data(
                lat_min, lat_max, lon_min, lon_max,
                start_time=time,
                end_time=time + timedelta(hours=6),
            )

            # Fetch wave data
            wave_data = self.copernicus.fetch_wave_data(
                lat_min, lat_max, lon_min, lon_max,
                start_time=time,
            )

            # Fetch current data
            current_data = self.copernicus.fetch_current_data(
                lat_min, lat_max, lon_min, lon_max,
                start_time=time,
            )

            return self.copernicus.get_weather_at_point(
                lat, lon, time,
                wind_data=wind_data,
                wave_data=wave_data,
                current_data=current_data,
            )

        except Exception as e:
            logger.warning(f"Failed to get forecast weather: {e}")
            return None

    def _blend_weather(
        self,
        forecast: PointWeather,
        climatology: PointWeather,
        forecast_weight: float,
        lat: float,
        lon: float,
        time: datetime,
    ) -> PointWeather:
        """Blend forecast and climatology weather."""
        cw = 1.0 - forecast_weight  # climatology weight

        # Blend scalar values
        wind_speed = forecast.wind_speed_ms * forecast_weight + climatology.wind_speed_ms * cw
        wave_height = forecast.wave_height_m * forecast_weight + climatology.wave_height_m * cw
        wave_period = forecast.wave_period_s * forecast_weight + climatology.wave_period_s * cw

        # Blend directions using circular mean
        def blend_direction(d1: float, d2: float, w1: float) -> float:
            r1, r2 = np.radians(d1), np.radians(d2)
            x = w1 * np.cos(r1) + (1 - w1) * np.cos(r2)
            y = w1 * np.sin(r1) + (1 - w1) * np.sin(r2)
            return (np.degrees(np.arctan2(y, x)) + 360) % 360

        wind_dir = blend_direction(forecast.wind_dir_deg, climatology.wind_dir_deg, forecast_weight)
        wave_dir = blend_direction(forecast.wave_dir_deg, climatology.wave_dir_deg, forecast_weight)

        # Currents (forecast only, climatology typically doesn't have)
        current_speed = forecast.current_speed_ms * forecast_weight
        current_dir = forecast.current_dir_deg

        return PointWeather(
            lat=lat,
            lon=lon,
            time=time,
            wind_speed_ms=wind_speed,
            wind_dir_deg=wind_dir,
            wave_height_m=wave_height,
            wave_period_s=wave_period,
            wave_dir_deg=wave_dir,
            current_speed_ms=current_speed,
            current_dir_deg=current_dir,
        )

    def _get_source_message(self, source_type: str, days_ahead: float) -> str:
        """Get human-readable message about data source."""
        if source_type == "forecast":
            return f"Forecast data (T+{days_ahead:.1f} days)"
        elif source_type == "blended":
            return f"Blended forecast/climatology (T+{days_ahead:.1f} days, beyond {self.forecast_horizon_days}-day forecast)"
        else:
            return f"Climatological average (T+{days_ahead:.1f} days, beyond forecast horizon)"

    def get_forecast_horizon(self) -> timedelta:
        """Get the current forecast horizon."""
        return timedelta(days=self.forecast_horizon_days)

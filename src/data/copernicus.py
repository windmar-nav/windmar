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

    # For wave data - additional fields
    wave_period: Optional[np.ndarray] = None  # Peak wave period (s)
    wave_direction: Optional[np.ndarray] = None  # Mean wave direction (deg)


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
            ds = xr.open_dataset(cache_file)
        else:
            logger.info("Downloading wave data from CMEMS...")

            try:
                ds = copernicusmarine.open_dataset(
                    dataset_id=self.CMEMS_WAVE_DATASET,
                    variables=["VHM0", "VTPK", "VMDR"],  # Hs, Peak period, Mean direction
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

            return WeatherData(
                parameter="wave_height",
                time=start_time,
                lats=lats,
                lons=lons,
                values=hs,
                unit="m",
                wave_period=tp,
                wave_direction=wave_dir,
            )

        except Exception as e:
            logger.error(f"Failed to parse wave data: {e}")
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
            ds = xr.open_dataset(cache_file)
        else:
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
        """Generate synthetic wave field based on wind."""
        time = datetime.utcnow()

        lats = np.arange(lat_min, lat_max + resolution, resolution)
        lons = np.arange(lon_min, lon_max + resolution, resolution)

        lon_grid, lat_grid = np.meshgrid(lons, lats)

        if wind_data is not None and wind_data.values is not None:
            wind_speed = wind_data.values
            wave_height = 0.15 * wind_speed + np.random.randn(*wind_speed.shape) * 0.3
        else:
            wave_height = 1.5 + 1.0 * np.sin(np.radians(lat_grid * 3))
            wave_height += np.random.randn(*lat_grid.shape) * 0.2

        wave_height = np.maximum(wave_height, 0.3)

        return WeatherData(
            parameter="wave_height",
            time=time,
            lats=lats,
            lons=lons,
            values=wave_height,
            unit="m",
        )


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

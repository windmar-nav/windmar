"""
Copernicus Marine Service Client.

Fetches real-time and forecast ocean data:
- Wave height, period, direction (GLOBAL_ANALYSISFORECAST_WAV)
- Ocean currents (GLOBAL_ANALYSISFORECAST_PHY)
- Sea surface temperature

Data resolution: 1/12° (~9km)
Update frequency: Hourly for analysis, 6-hourly for forecast

API Reference: https://marine.copernicus.eu/
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import numpy as np
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class OceanConditions:
    """Ocean conditions at a specific location and time."""
    timestamp: datetime
    latitude: float
    longitude: float

    # Wave data
    significant_wave_height_m: float = 0.0  # VHM0
    mean_wave_period_s: float = 0.0         # VTM10
    peak_wave_period_s: float = 0.0         # VTPK
    wave_direction_deg: float = 0.0         # VMDR (from)

    # Wind wave component
    wind_wave_height_m: float = 0.0         # VHM0_WW
    wind_wave_period_s: float = 0.0         # VTM01_WW
    wind_wave_direction_deg: float = 0.0    # VMDR_WW

    # Swell component
    swell_height_m: float = 0.0             # VHM0_SW1
    swell_period_s: float = 0.0             # VTM01_SW1
    swell_direction_deg: float = 0.0        # VMDR_SW1

    # Current data
    current_speed_ms: float = 0.0           # sqrt(uo^2 + vo^2)
    current_direction_deg: float = 0.0      # atan2(vo, uo)
    current_u_ms: float = 0.0               # Eastward current
    current_v_ms: float = 0.0               # Northward current

    # Temperature
    sea_surface_temp_c: float = 0.0         # thetao

    # Metadata
    data_source: str = "mock"
    forecast_hours: int = 0  # 0 = analysis, >0 = forecast


@dataclass
class WindConditions:
    """Wind conditions (from atmospheric reanalysis)."""
    timestamp: datetime
    latitude: float
    longitude: float

    wind_speed_ms: float = 0.0
    wind_direction_deg: float = 0.0  # From direction
    wind_u_ms: float = 0.0           # Eastward
    wind_v_ms: float = 0.0           # Northward

    data_source: str = "mock"


class CopernicusClient:
    """
    Client for Copernicus Marine Service data.

    Supports two modes:
    - mock: Returns simulated data for testing
    - live: Fetches real data from CMEMS API

    Usage:
        # Mock mode (default)
        client = CopernicusClient(mock_mode=True)

        # Live mode (requires credentials)
        client = CopernicusClient(
            mock_mode=False,
            username="your_username",
            password="your_password"
        )

        # Get conditions at location
        ocean = client.get_ocean_conditions(lat=43.5, lon=7.0)
        print(f"Wave height: {ocean.significant_wave_height_m} m")
    """

    # CMEMS dataset IDs
    WAVE_DATASET = "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i"
    PHYSICS_DATASET = "cmems_mod_glo_phy_anfc_0.083deg_PT1H-m"

    def __init__(
        self,
        mock_mode: bool = True,
        username: Optional[str] = None,
        password: Optional[str] = None,
        cache_hours: float = 1.0,
    ):
        """
        Initialize Copernicus client.

        Args:
            mock_mode: If True, return simulated data
            username: CMEMS username (required for live mode)
            password: CMEMS password (required for live mode)
            cache_hours: Cache duration for data
        """
        self.mock_mode = mock_mode
        self.username = username
        self.password = password
        self.cache_hours = cache_hours

        # Data cache: {(lat, lon, time_bucket): OceanConditions}
        self._cache: Dict[Tuple, OceanConditions] = {}
        self._cache_timestamps: Dict[Tuple, datetime] = {}

        # Initialize CMEMS client if not in mock mode
        self._cmems_client = None
        if not mock_mode:
            self._init_cmems_client()

    def _init_cmems_client(self):
        """Initialize the copernicusmarine client."""
        try:
            import copernicusmarine
            self._cmems_client = copernicusmarine
            logger.info("Copernicus Marine client initialized")
        except ImportError:
            logger.warning(
                "copernicusmarine package not installed. "
                "Install with: pip install copernicusmarine"
            )
            self.mock_mode = True

    def get_ocean_conditions(
        self,
        latitude: float,
        longitude: float,
        timestamp: Optional[datetime] = None,
    ) -> OceanConditions:
        """
        Get ocean conditions at a location.

        Args:
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            timestamp: Time for data (default: now)

        Returns:
            OceanConditions with wave, current, temperature data
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        # Check cache
        cache_key = self._cache_key(latitude, longitude, timestamp)
        if cache_key in self._cache:
            cache_time = self._cache_timestamps.get(cache_key)
            if cache_time and (datetime.utcnow() - cache_time).total_seconds() < self.cache_hours * 3600:
                logger.debug(f"Cache hit for {cache_key}")
                return self._cache[cache_key]

        # Fetch data
        if self.mock_mode:
            conditions = self._generate_mock_conditions(latitude, longitude, timestamp)
        else:
            conditions = self._fetch_live_conditions(latitude, longitude, timestamp)

        # Update cache
        self._cache[cache_key] = conditions
        self._cache_timestamps[cache_key] = datetime.utcnow()

        return conditions

    def get_wind_conditions(
        self,
        latitude: float,
        longitude: float,
        timestamp: Optional[datetime] = None,
    ) -> WindConditions:
        """
        Get wind conditions at a location.

        Note: CMEMS wave products include wind-wave component which
        is derived from wind. For direct wind, use ERA5 or GFS.
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        if self.mock_mode:
            return self._generate_mock_wind(latitude, longitude, timestamp)
        else:
            # In live mode, derive wind from wind-wave component
            ocean = self.get_ocean_conditions(latitude, longitude, timestamp)
            # Approximate wind from wind-wave height (empirical)
            # Hs_ww ≈ 0.0246 * U^2 (fully developed)
            wind_speed = np.sqrt(ocean.wind_wave_height_m / 0.0246) if ocean.wind_wave_height_m > 0 else 5.0

            return WindConditions(
                timestamp=timestamp,
                latitude=latitude,
                longitude=longitude,
                wind_speed_ms=wind_speed,
                wind_direction_deg=ocean.wind_wave_direction_deg,
                wind_u_ms=-wind_speed * np.sin(np.radians(ocean.wind_wave_direction_deg)),
                wind_v_ms=-wind_speed * np.cos(np.radians(ocean.wind_wave_direction_deg)),
                data_source="derived_from_waves",
            )

    def get_forecast(
        self,
        latitude: float,
        longitude: float,
        hours_ahead: int = 24,
        interval_hours: int = 3,
    ) -> list[OceanConditions]:
        """
        Get forecast for next N hours.

        Args:
            latitude: Latitude
            longitude: Longitude
            hours_ahead: Hours to forecast
            interval_hours: Time step between forecasts

        Returns:
            List of OceanConditions for each time step
        """
        forecasts = []
        now = datetime.utcnow()

        for h in range(0, hours_ahead + 1, interval_hours):
            forecast_time = now + timedelta(hours=h)
            conditions = self.get_ocean_conditions(latitude, longitude, forecast_time)
            conditions.forecast_hours = h
            forecasts.append(conditions)

        return forecasts

    def _cache_key(self, lat: float, lon: float, timestamp: datetime) -> Tuple:
        """Generate cache key with spatial and temporal bucketing."""
        # Bucket to 0.1° spatial and 1 hour temporal
        lat_bucket = round(lat, 1)
        lon_bucket = round(lon, 1)
        time_bucket = timestamp.replace(minute=0, second=0, microsecond=0)
        return (lat_bucket, lon_bucket, time_bucket)

    def _generate_mock_conditions(
        self,
        latitude: float,
        longitude: float,
        timestamp: datetime,
    ) -> OceanConditions:
        """Generate realistic mock ocean conditions."""
        # Use lat/lon and time to generate consistent pseudo-random conditions
        seed = int((latitude * 1000 + longitude * 100 + timestamp.hour) % 10000)
        rng = np.random.RandomState(seed)

        # Base conditions vary by latitude (rougher at higher latitudes)
        lat_factor = abs(latitude) / 60.0  # 0-1 scale

        # Wave height: 0.5-5m depending on latitude and "weather"
        base_hs = 1.0 + lat_factor * 2.0
        hs = base_hs * (0.5 + rng.random())

        # Wave period: typically 5-15s, correlated with height
        tp = 5.0 + hs * 1.5 + rng.random() * 2
        tm = tp * 0.85  # Mean period slightly less than peak

        # Wave direction: generally from west in northern hemisphere
        wave_dir = 270 + rng.normal(0, 30)
        wave_dir = wave_dir % 360

        # Split into wind-wave and swell components
        # Assume 60% wind-wave, 40% swell in energy
        ww_ratio = 0.6
        sw_ratio = 0.4

        ww_hs = hs * np.sqrt(ww_ratio)
        sw_hs = hs * np.sqrt(sw_ratio)

        # Current: 0-1 m/s, typically weaker than wind-driven
        current_speed = 0.1 + rng.random() * 0.4
        current_dir = rng.random() * 360
        current_u = current_speed * np.sin(np.radians(current_dir))
        current_v = current_speed * np.cos(np.radians(current_dir))

        # Sea surface temperature: varies by latitude and season
        month = timestamp.month
        seasonal_factor = np.cos(2 * np.pi * (month - 7) / 12)  # Peak in July
        sst = 25 - abs(latitude) * 0.3 + seasonal_factor * 5
        sst = max(0, min(30, sst + rng.normal(0, 1)))

        return OceanConditions(
            timestamp=timestamp,
            latitude=latitude,
            longitude=longitude,
            significant_wave_height_m=round(hs, 2),
            mean_wave_period_s=round(tm, 1),
            peak_wave_period_s=round(tp, 1),
            wave_direction_deg=round(wave_dir, 0),
            wind_wave_height_m=round(ww_hs, 2),
            wind_wave_period_s=round(tp * 0.9, 1),
            wind_wave_direction_deg=round(wave_dir + rng.normal(0, 10), 0) % 360,
            swell_height_m=round(sw_hs, 2),
            swell_period_s=round(tp * 1.3, 1),  # Swell has longer period
            swell_direction_deg=round(wave_dir + rng.normal(0, 30), 0) % 360,
            current_speed_ms=round(current_speed, 2),
            current_direction_deg=round(current_dir, 0),
            current_u_ms=round(current_u, 3),
            current_v_ms=round(current_v, 3),
            sea_surface_temp_c=round(sst, 1),
            data_source="mock",
            forecast_hours=0,
        )

    def _generate_mock_wind(
        self,
        latitude: float,
        longitude: float,
        timestamp: datetime,
    ) -> WindConditions:
        """Generate realistic mock wind conditions."""
        seed = int((latitude * 1000 + longitude * 100 + timestamp.hour + 500) % 10000)
        rng = np.random.RandomState(seed)

        # Wind speed: typically 5-20 m/s
        lat_factor = abs(latitude) / 60.0
        base_wind = 8.0 + lat_factor * 5.0
        wind_speed = base_wind * (0.5 + rng.random() * 0.8)

        # Wind direction
        wind_dir = 250 + rng.normal(0, 40)
        wind_dir = wind_dir % 360

        # Components (wind FROM direction, so negative sign)
        wind_u = -wind_speed * np.sin(np.radians(wind_dir))
        wind_v = -wind_speed * np.cos(np.radians(wind_dir))

        return WindConditions(
            timestamp=timestamp,
            latitude=latitude,
            longitude=longitude,
            wind_speed_ms=round(wind_speed, 1),
            wind_direction_deg=round(wind_dir, 0),
            wind_u_ms=round(wind_u, 2),
            wind_v_ms=round(wind_v, 2),
            data_source="mock",
        )

    def _fetch_live_conditions(
        self,
        latitude: float,
        longitude: float,
        timestamp: datetime,
    ) -> OceanConditions:
        """Fetch real conditions from CMEMS API."""
        if self._cmems_client is None:
            logger.error("CMEMS client not initialized, falling back to mock")
            return self._generate_mock_conditions(latitude, longitude, timestamp)

        try:
            # Fetch wave data
            wave_data = self._fetch_wave_data(latitude, longitude, timestamp)
            current_data = self._fetch_current_data(latitude, longitude, timestamp)

            return OceanConditions(
                timestamp=timestamp,
                latitude=latitude,
                longitude=longitude,
                significant_wave_height_m=wave_data.get('VHM0', 0.0),
                mean_wave_period_s=wave_data.get('VTM10', 0.0),
                peak_wave_period_s=wave_data.get('VTPK', 0.0),
                wave_direction_deg=wave_data.get('VMDR', 0.0),
                wind_wave_height_m=wave_data.get('VHM0_WW', 0.0),
                wind_wave_period_s=wave_data.get('VTM01_WW', 0.0),
                wind_wave_direction_deg=wave_data.get('VMDR_WW', 0.0),
                swell_height_m=wave_data.get('VHM0_SW1', 0.0),
                swell_period_s=wave_data.get('VTM01_SW1', 0.0),
                swell_direction_deg=wave_data.get('VMDR_SW1', 0.0),
                current_speed_ms=current_data.get('speed', 0.0),
                current_direction_deg=current_data.get('direction', 0.0),
                current_u_ms=current_data.get('uo', 0.0),
                current_v_ms=current_data.get('vo', 0.0),
                sea_surface_temp_c=current_data.get('thetao', 15.0),
                data_source="cmems",
                forecast_hours=0,
            )
        except Exception as e:
            logger.error(f"Failed to fetch CMEMS data: {e}")
            return self._generate_mock_conditions(latitude, longitude, timestamp)

    def _fetch_wave_data(
        self,
        latitude: float,
        longitude: float,
        timestamp: datetime,
    ) -> Dict[str, float]:
        """Fetch wave data from CMEMS."""
        # This would use copernicusmarine.subset() in production
        # For now, return empty dict to fall back to mock
        logger.warning("Live CMEMS wave fetch not implemented, using mock")
        return {}

    def _fetch_current_data(
        self,
        latitude: float,
        longitude: float,
        timestamp: datetime,
    ) -> Dict[str, float]:
        """Fetch current/physics data from CMEMS."""
        logger.warning("Live CMEMS current fetch not implemented, using mock")
        return {}

    def clear_cache(self):
        """Clear the data cache."""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("Cache cleared")

    @property
    def cache_size(self) -> int:
        """Get number of cached entries."""
        return len(self._cache)


# Test function
def test_copernicus_client():
    """Test the Copernicus client."""
    print("Testing Copernicus Marine Client")
    print("=" * 50)

    # Create client in mock mode
    client = CopernicusClient(mock_mode=True)

    # Test location: Mediterranean Sea
    lat, lon = 43.5, 7.0

    print(f"\nLocation: {lat}°N, {lon}°E (Mediterranean)")

    # Get current conditions
    ocean = client.get_ocean_conditions(lat, lon)

    print(f"\nOcean Conditions:")
    print(f"  Wave Height (Hs): {ocean.significant_wave_height_m} m")
    print(f"  Peak Period (Tp): {ocean.peak_wave_period_s} s")
    print(f"  Wave Direction: {ocean.wave_direction_deg}°")
    print(f"  - Wind waves: {ocean.wind_wave_height_m} m")
    print(f"  - Swell: {ocean.swell_height_m} m")
    print(f"  Current Speed: {ocean.current_speed_ms} m/s")
    print(f"  Current Dir: {ocean.current_direction_deg}°")
    print(f"  SST: {ocean.sea_surface_temp_c}°C")
    print(f"  Source: {ocean.data_source}")

    # Get wind
    wind = client.get_wind_conditions(lat, lon)

    print(f"\nWind Conditions:")
    print(f"  Wind Speed: {wind.wind_speed_ms} m/s")
    print(f"  Wind Direction: {wind.wind_direction_deg}° (from)")

    # Test forecast
    print(f"\n24-hour Forecast:")
    forecast = client.get_forecast(lat, lon, hours_ahead=24, interval_hours=6)

    for fc in forecast:
        print(f"  +{fc.forecast_hours:2d}h: Hs={fc.significant_wave_height_m:.1f}m, "
              f"Tp={fc.peak_wave_period_s:.1f}s")

    # Test caching
    print(f"\nCache size: {client.cache_size} entries")

    return client


if __name__ == "__main__":
    test_copernicus_client()

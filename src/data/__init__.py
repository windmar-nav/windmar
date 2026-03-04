"""Data providers for weather and ocean data."""

from .copernicus import (
    CopernicusDataProvider,
    SyntheticDataProvider,
    WeatherData,
    PointWeather,
)

# Real-time Copernicus client for sensor fusion
from .copernicus_client import (
    CopernicusClient,
    OceanConditions,
    WindConditions,
)

# Emission Control Area zones (IMO MARPOL Annex VI)
from .eca_zones import (
    ECAZone,
    ECAManager,
    ECA_ZONES,
    BALTIC_SEA_ECA,
    NORTH_SEA_ECA,
    NORTH_AMERICAN_ECA,
    NORTH_AMERICAN_PACIFIC_ECA,
    US_CARIBBEAN_ECA,
    eca_manager,
)

__all__ = [
    # Weather data providers
    "CopernicusDataProvider",
    "SyntheticDataProvider",
    "WeatherData",
    "PointWeather",
    # Real-time Copernicus client
    "CopernicusClient",
    "OceanConditions",
    "WindConditions",
    # ECA zones
    "ECAZone",
    "ECAManager",
    "ECA_ZONES",
    "BALTIC_SEA_ECA",
    "NORTH_SEA_ECA",
    "NORTH_AMERICAN_ECA",
    "NORTH_AMERICAN_PACIFIC_ECA",
    "US_CARIBBEAN_ECA",
    "eca_manager",
]

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

__all__ = [
    'CopernicusDataProvider',
    'SyntheticDataProvider',
    'WeatherData',
    'PointWeather',
    'CopernicusClient',
    'OceanConditions',
    'WindConditions',
]

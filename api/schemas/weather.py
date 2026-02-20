"""Weather-related API schemas."""

from datetime import datetime
from typing import Dict, List

from pydantic import BaseModel


class WindDataPoint(BaseModel):
    """Wind data at a point."""
    lat: float
    lon: float
    u: float  # U component (m/s)
    v: float  # V component (m/s)
    speed_kts: float
    dir_deg: float


class WeatherGridResponse(BaseModel):
    """Weather grid data for visualization."""
    parameter: str
    time: datetime
    bbox: Dict[str, float]
    resolution: float
    nx: int
    ny: int
    lats: List[float]
    lons: List[float]
    data: List[List[float]]  # 2D grid


class VelocityDataResponse(BaseModel):
    """Wind velocity data in leaflet-velocity format."""
    header: Dict
    data_u: List[float]
    data_v: List[float]

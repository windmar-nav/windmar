"""Voyage calculation API schemas."""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from .common import Position, WaypointModel


class VoyageRequest(BaseModel):
    """Request for voyage calculation."""
    waypoints: List[Position]
    calm_speed_kts: float = Field(..., gt=0, lt=30, description="Calm water speed in knots")
    is_laden: bool = True
    departure_time: Optional[datetime] = None
    use_weather: bool = True


class LegResultModel(BaseModel):
    """Result for a single leg."""
    leg_index: int
    from_wp: WaypointModel
    to_wp: WaypointModel
    distance_nm: float
    bearing_deg: float

    # Weather
    wind_speed_kts: float
    wind_dir_deg: float
    wave_height_m: float
    wave_dir_deg: float
    current_speed_ms: float = 0.0
    current_dir_deg: float = 0.0

    # Speeds
    calm_speed_kts: float
    stw_kts: float
    sog_kts: float
    speed_loss_pct: float

    # Time
    time_hours: float
    departure_time: datetime
    arrival_time: datetime

    # Fuel
    fuel_mt: float
    power_kw: float

    # Data source info (forecast, climatology, blended)
    data_source: Optional[str] = None
    forecast_weight: Optional[float] = None


class DataSourceSummary(BaseModel):
    """Summary of data sources used in voyage calculation."""
    forecast_legs: int
    blended_legs: int
    climatology_legs: int
    forecast_horizon_days: float
    warning: Optional[str] = None


class VoyageResponse(BaseModel):
    """Complete voyage calculation response."""
    route_name: str
    departure_time: datetime
    arrival_time: datetime

    total_distance_nm: float
    total_time_hours: float
    total_fuel_mt: float
    avg_sog_kts: float
    avg_stw_kts: float

    legs: List[LegResultModel]

    calm_speed_kts: float
    is_laden: bool

    # Data source summary
    data_sources: Optional[DataSourceSummary] = None


class MonteCarloRequest(BaseModel):
    """Request for Monte Carlo voyage simulation."""
    waypoints: List[Position]
    calm_speed_kts: float = Field(..., gt=0, lt=30)
    is_laden: bool = True
    departure_time: Optional[datetime] = None
    n_simulations: int = Field(100, ge=10, le=500)


class PercentileFloat(BaseModel):
    p10: float
    p50: float
    p90: float


class PercentileString(BaseModel):
    p10: str
    p50: str
    p90: str


class MonteCarloResponse(BaseModel):
    """Monte Carlo simulation result."""
    n_simulations: int
    eta: PercentileString
    fuel_mt: PercentileFloat
    total_time_hours: PercentileFloat
    computation_time_ms: float

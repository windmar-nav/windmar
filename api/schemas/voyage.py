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


# =============================================================================
# Voyage History (persistence) schemas
# =============================================================================


class SaveVoyageLeg(BaseModel):
    """Leg data for saving a voyage."""
    leg_index: int
    from_name: Optional[str] = None
    from_lat: float
    from_lon: float
    to_name: Optional[str] = None
    to_lat: float
    to_lon: float
    distance_nm: float
    bearing_deg: Optional[float] = None
    wind_speed_kts: Optional[float] = None
    wind_dir_deg: Optional[float] = None
    wave_height_m: Optional[float] = None
    wave_dir_deg: Optional[float] = None
    current_speed_ms: Optional[float] = None
    current_dir_deg: Optional[float] = None
    calm_speed_kts: Optional[float] = None
    stw_kts: Optional[float] = None
    sog_kts: Optional[float] = None
    speed_loss_pct: Optional[float] = None
    time_hours: float
    departure_time: Optional[datetime] = None
    arrival_time: Optional[datetime] = None
    fuel_mt: float
    power_kw: Optional[float] = None
    data_source: Optional[str] = None


class SaveVoyageRequest(BaseModel):
    """Request to persist a calculated voyage."""
    name: Optional[str] = None
    departure_port: Optional[str] = None
    arrival_port: Optional[str] = None
    departure_time: datetime
    arrival_time: datetime
    total_distance_nm: float = Field(..., gt=0)
    total_time_hours: float = Field(..., gt=0)
    total_fuel_mt: float = Field(..., ge=0)
    avg_sog_kts: Optional[float] = None
    avg_stw_kts: Optional[float] = None
    calm_speed_kts: float = Field(..., gt=0, lt=30)
    is_laden: bool = True
    vessel_specs_snapshot: Optional[Dict] = None
    cii_estimate: Optional[Dict] = None
    notes: Optional[str] = None
    legs: List[SaveVoyageLeg]


class VoyageLegResponse(BaseModel):
    """Leg data in a voyage detail response."""
    id: str
    leg_index: int
    from_name: Optional[str] = None
    from_lat: float
    from_lon: float
    to_name: Optional[str] = None
    to_lat: float
    to_lon: float
    distance_nm: float
    bearing_deg: Optional[float] = None
    wind_speed_kts: Optional[float] = None
    wind_dir_deg: Optional[float] = None
    wave_height_m: Optional[float] = None
    wave_dir_deg: Optional[float] = None
    current_speed_ms: Optional[float] = None
    current_dir_deg: Optional[float] = None
    calm_speed_kts: Optional[float] = None
    stw_kts: Optional[float] = None
    sog_kts: Optional[float] = None
    speed_loss_pct: Optional[float] = None
    time_hours: float
    departure_time: Optional[datetime] = None
    arrival_time: Optional[datetime] = None
    fuel_mt: float
    power_kw: Optional[float] = None
    data_source: Optional[str] = None


class VoyageSummaryResponse(BaseModel):
    """Summary of a saved voyage (for list view)."""
    id: str
    name: Optional[str] = None
    departure_port: Optional[str] = None
    arrival_port: Optional[str] = None
    departure_time: datetime
    arrival_time: datetime
    total_distance_nm: float
    total_time_hours: float
    total_fuel_mt: float
    avg_sog_kts: Optional[float] = None
    calm_speed_kts: float
    is_laden: bool
    cii_estimate: Optional[Dict] = None
    created_at: datetime


class VoyageDetailResponse(VoyageSummaryResponse):
    """Full voyage detail with legs."""
    avg_stw_kts: Optional[float] = None
    vessel_specs_snapshot: Optional[Dict] = None
    notes: Optional[str] = None
    updated_at: datetime
    legs: List[VoyageLegResponse]


class VoyageListResponse(BaseModel):
    """Paginated list of saved voyages."""
    voyages: List[VoyageSummaryResponse]
    total: int
    limit: int
    offset: int


class NoonReportEntry(BaseModel):
    """Synthetic noon report entry generated from voyage legs."""
    report_number: int
    timestamp: datetime
    lat: float
    lon: float
    sog_kts: Optional[float] = None
    stw_kts: Optional[float] = None
    course_deg: Optional[float] = None
    distance_since_last_nm: float
    fuel_since_last_mt: float
    cumulative_distance_nm: float
    cumulative_fuel_mt: float
    wind_speed_kts: Optional[float] = None
    wind_dir_deg: Optional[float] = None
    wave_height_m: Optional[float] = None
    wave_dir_deg: Optional[float] = None
    current_speed_ms: Optional[float] = None
    current_dir_deg: Optional[float] = None


class NoonReportsResponse(BaseModel):
    """List of noon reports for a voyage."""
    voyage_id: str
    voyage_name: Optional[str] = None
    departure_time: datetime
    arrival_time: datetime
    reports: List[NoonReportEntry]


class DepartureReportData(BaseModel):
    """Departure report fields."""
    vessel_name: Optional[str] = None
    dwt: Optional[float] = None
    departure_port: Optional[str] = None
    departure_time: datetime
    loading_condition: str
    destination: Optional[str] = None
    eta: datetime
    planned_distance_nm: float
    planned_speed_kts: float
    estimated_fuel_mt: float
    weather_at_departure: Optional[Dict] = None


class ArrivalReportData(BaseModel):
    """Arrival report fields."""
    vessel_name: Optional[str] = None
    arrival_port: Optional[str] = None
    arrival_time: datetime
    actual_voyage_time_hours: float
    total_fuel_consumed_mt: float
    average_speed_kts: float
    total_distance_nm: float
    weather_summary: Optional[Dict] = None
    cii_estimate: Optional[Dict] = None


class VoyageReportsResponse(BaseModel):
    """Departure + arrival reports for a voyage."""
    voyage_id: str
    departure_report: DepartureReportData
    arrival_report: ArrivalReportData
    noon_reports: List[NoonReportEntry]


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

"""Route optimization API schemas."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from .common import Position


class WeatherProvenanceModel(BaseModel):
    """Weather data source provenance metadata."""
    source_type: str  # "forecast", "hindcast", "climatology", "blended"
    model_name: str  # "GFS", "CMEMS_wave", etc.
    forecast_lead_hours: float
    confidence: str  # "high", "medium", "low"


class OptimizationRequest(BaseModel):
    """Request for route optimization."""
    origin: Position
    destination: Position
    calm_speed_kts: float = Field(..., gt=0, lt=30, description="Calm water speed in knots")
    is_laden: bool = True
    departure_time: Optional["datetime"] = None
    optimization_target: str = Field("fuel", description="Minimize 'fuel' or 'time'")
    grid_resolution_deg: float = Field(0.2, ge=0.05, le=2.0, description="Grid resolution in degrees")
    max_time_factor: float = Field(1.15, ge=1.0, le=2.0,
        description="Max voyage time as multiple of direct time (1.15 = 15% longer allowed)")
    engine: str = Field("astar", description="Optimization engine: 'astar' (A* pathfinding) or 'visir' (VISIR graph-based Dijkstra)")
    # All user waypoints for multi-segment optimization (respects intermediate via-points)
    route_waypoints: Optional[List[Position]] = None
    # Baseline from voyage calculation (enables dual-strategy comparison)
    baseline_fuel_mt: Optional[float] = None
    baseline_time_hours: Optional[float] = None
    baseline_distance_nm: Optional[float] = None
    # Safety weight: 0.0 = pure fuel optimization, 1.0 = full safety penalties
    safety_weight: float = Field(0.0, ge=0.0, le=1.0, description="Safety penalty weight: 0=fuel optimal, 1=safety priority")


class OptimizationLegModel(BaseModel):
    """Optimized route leg details."""
    from_lat: float
    from_lon: float
    to_lat: float
    to_lon: float
    distance_nm: float
    bearing_deg: float
    fuel_mt: float
    time_hours: float
    sog_kts: float
    stw_kts: float  # Speed through water (optimized per leg)
    wind_speed_ms: float
    wave_height_m: float
    # Safety metrics per leg
    safety_status: Optional[str] = None
    roll_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    # Weather provenance per leg
    data_source: Optional[str] = None  # "forecast (high confidence)" etc.
    # Extended weather fields (SPEC-P1)
    swell_hs_m: Optional[float] = None
    windsea_hs_m: Optional[float] = None
    current_effect_kts: Optional[float] = None
    visibility_m: Optional[float] = None
    sst_celsius: Optional[float] = None
    ice_concentration: Optional[float] = None


class SafetySummary(BaseModel):
    """Safety assessment summary for optimized route."""
    status: str  # "safe", "marginal", "dangerous"
    warnings: List[str]
    max_roll_deg: float
    max_pitch_deg: float
    max_accel_ms2: float


class SpeedScenarioModel(BaseModel):
    """One speed strategy applied to the optimized path."""
    strategy: str
    label: str
    total_fuel_mt: float
    total_time_hours: float
    total_distance_nm: float
    avg_speed_kts: float
    speed_profile: List[float]
    legs: List[OptimizationLegModel]
    fuel_savings_pct: float
    time_savings_pct: float


class OptimizationResponse(BaseModel):
    """Route optimization result."""
    waypoints: List[Position]
    total_fuel_mt: float
    total_time_hours: float
    total_distance_nm: float

    # Comparison with direct route
    direct_fuel_mt: float
    direct_time_hours: float
    fuel_savings_pct: float
    time_savings_pct: float

    # Per-leg details
    legs: List[OptimizationLegModel]

    # Speed profile (variable speed optimization)
    speed_profile: List[float]  # Optimal speed per leg (kts)
    avg_speed_kts: float
    variable_speed_enabled: bool

    # Engine used
    engine: str = "astar"  # "astar" or "visir"

    # Safety assessment
    safety: Optional[SafetySummary] = None

    # Speed strategy scenarios
    scenarios: List[SpeedScenarioModel] = []
    baseline_fuel_mt: Optional[float] = None
    baseline_time_hours: Optional[float] = None
    baseline_distance_nm: Optional[float] = None

    # Weather provenance
    weather_provenance: Optional[List[WeatherProvenanceModel]] = None
    temporal_weather: bool = False  # True if time-varying weather was used

    # Metadata
    optimization_target: str
    grid_resolution_deg: float
    cells_explored: int
    optimization_time_ms: float


# Fix forward reference for OptimizationRequest.departure_time
from datetime import datetime  # noqa: E402
OptimizationRequest.model_rebuild()

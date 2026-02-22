"""Route optimization API schemas."""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .common import Position

_VALID_ZONE_TYPES = {"tss", "eca", "piracy", "ice", "war_risk"}


class WeatherProvenanceModel(BaseModel):
    """Weather data source provenance metadata."""
    source_type: str = Field(..., max_length=50)
    model_name: str = Field(..., max_length=100)
    forecast_lead_hours: float
    confidence: Literal["high", "medium", "low"]


class OptimizationRequest(BaseModel):
    """Request for route optimization."""
    origin: Position
    destination: Position
    calm_speed_kts: float = Field(..., gt=0, lt=30, description="Calm water speed in knots")
    is_laden: bool = True
    departure_time: Optional["datetime"] = None
    optimization_target: Literal["fuel", "time"] = Field("fuel", description="Minimize 'fuel' or 'time'")
    grid_resolution_deg: float = Field(0.2, ge=0.05, le=2.0, description="Grid resolution in degrees")
    max_time_factor: float = Field(1.15, ge=1.0, le=2.0,
        description="Max voyage time as multiple of direct time (1.15 = 15% longer allowed)")
    engine: Literal["astar", "visir"] = Field("astar", description="Optimization engine: 'astar' or 'visir'")
    # All user waypoints for multi-segment optimization (respects intermediate via-points)
    route_waypoints: Optional[List[Position]] = Field(None, max_length=50)
    # Baseline from voyage calculation (enables dual-strategy comparison)
    baseline_fuel_mt: Optional[float] = None
    baseline_time_hours: Optional[float] = None
    baseline_distance_nm: Optional[float] = None
    # Safety weight: 0.0 = pure fuel optimization, 1.0 = full safety penalties
    safety_weight: float = Field(0.0, ge=0.0, le=1.0, description="Safety penalty weight: 0=fuel optimal, 1=safety priority")
    # Pareto front: when True, run A* with multiple lambda values and return Pareto-optimal set
    pareto: bool = Field(False, description="Return Pareto front of fuel/time trade-offs")
    # Variable resolution: when True, use two-tier grid (0.5° ocean + 0.1° nearshore)
    variable_resolution: bool = Field(False, description="Enable variable resolution grid (fine nearshore, coarse ocean)")
    # Zone types to enforce during routing (empty list = no enforcement)
    enforced_zone_types: Optional[List[str]] = Field(None, max_length=10, description="Zone types to enforce (e.g. ['tss','eca']). None = enforce all.")

    @field_validator("enforced_zone_types")
    @classmethod
    def validate_zone_types(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            for zt in v:
                if zt not in _VALID_ZONE_TYPES:
                    raise ValueError(f"Invalid zone type '{zt}'. Must be one of: {sorted(_VALID_ZONE_TYPES)}")
        return v


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
    status: Literal["safe", "marginal", "dangerous"]
    warnings: List[str] = Field(default_factory=list, max_length=50)
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


class ParetoSolutionModel(BaseModel):
    """One point on the Pareto front."""
    lambda_value: float
    fuel_mt: float
    time_hours: float
    distance_nm: float
    speed_profile: List[float]
    is_selected: bool = False


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
    engine: Literal["astar", "visir"] = "astar"
    variable_resolution_enabled: bool = False

    # Safety assessment
    safety: Optional[SafetySummary] = None

    # Speed strategy scenarios
    scenarios: List[SpeedScenarioModel] = []
    baseline_fuel_mt: Optional[float] = None
    baseline_time_hours: Optional[float] = None
    baseline_distance_nm: Optional[float] = None

    # Pareto front (populated when pareto=True)
    pareto_front: Optional[List[ParetoSolutionModel]] = None

    # Safety fallback: True when hard limits were relaxed to find a route
    safety_degraded: bool = False

    # Weather provenance
    weather_provenance: Optional[List[WeatherProvenanceModel]] = None
    temporal_weather: bool = False  # True if time-varying weather was used

    # Metadata
    optimization_target: Literal["fuel", "time"]
    grid_resolution_deg: float
    cells_explored: int
    optimization_time_ms: float


# Fix forward reference for OptimizationRequest.departure_time
from datetime import datetime  # noqa: E402
OptimizationRequest.model_rebuild()

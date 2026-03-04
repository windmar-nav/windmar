"""Charter Party Weather Clause API schemas."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Shared input model
# ---------------------------------------------------------------------------


class LegWeatherInput(BaseModel):
    """Single leg weather/performance input for manual analysis."""

    wind_speed_kts: float = Field(0, ge=0, description="Wind speed in knots")
    wave_height_m: float = Field(
        0, ge=0, description="Significant wave height in metres"
    )
    current_speed_ms: float = Field(0, ge=0, description="Current speed in m/s")
    time_hours: float = Field(..., gt=0, description="Leg duration in hours")
    distance_nm: float = Field(0, ge=0, description="Leg distance in nautical miles")
    sog_kts: float = Field(0, ge=0, description="Speed over ground in knots")
    fuel_mt: float = Field(0, ge=0, description="Fuel consumed in metric tons")


# ---------------------------------------------------------------------------
# Good Weather Days
# ---------------------------------------------------------------------------


class GoodWeatherRequest(BaseModel):
    """Good weather day analysis from a saved voyage."""

    voyage_id: str = Field(..., min_length=1, description="Voyage UUID")
    bf_threshold: int = Field(
        4, ge=0, le=12, description="Max Beaufort force for good weather"
    )
    wave_threshold_m: Optional[float] = Field(
        None, ge=0, description="Max wave height (m)"
    )
    current_threshold_kts: Optional[float] = Field(
        None, ge=0, description="Max current speed (kts)"
    )


class GoodWeatherFromLegsRequest(BaseModel):
    """Good weather day analysis from manual leg data."""

    legs: List[LegWeatherInput] = Field(..., min_length=1, max_length=200)
    bf_threshold: int = Field(4, ge=0, le=12)
    wave_threshold_m: Optional[float] = Field(None, ge=0)
    current_threshold_kts: Optional[float] = Field(None, ge=0)


# ---------------------------------------------------------------------------
# Warranty Verification
# ---------------------------------------------------------------------------


class WarrantyVerificationRequest(BaseModel):
    """Warranty verification from a saved voyage."""

    voyage_id: str = Field(..., min_length=1, description="Voyage UUID")
    warranted_speed_kts: float = Field(
        ..., gt=0, le=30, description="Chartered speed (kts)"
    )
    warranted_consumption_mt_day: float = Field(
        ..., gt=0, description="Chartered daily consumption (MT/day)"
    )
    bf_threshold: int = Field(4, ge=0, le=12)
    speed_tolerance_pct: float = Field(
        0, ge=0, le=20, description="Allowable speed deficit %"
    )
    consumption_tolerance_pct: float = Field(
        0, ge=0, le=20, description="Allowable consumption excess %"
    )


class WarrantyFromLegsRequest(BaseModel):
    """Warranty verification from manual leg data."""

    legs: List[LegWeatherInput] = Field(..., min_length=1, max_length=200)
    warranted_speed_kts: float = Field(..., gt=0, le=30)
    warranted_consumption_mt_day: float = Field(..., gt=0)
    bf_threshold: int = Field(4, ge=0, le=12)
    speed_tolerance_pct: float = Field(0, ge=0, le=20)
    consumption_tolerance_pct: float = Field(0, ge=0, le=20)


# ---------------------------------------------------------------------------
# Off-Hire
# ---------------------------------------------------------------------------


class OffHireRequest(BaseModel):
    """Off-hire detection from engine log entries."""

    date_from: Optional[datetime] = Field(None, description="Start of analysis window")
    date_to: Optional[datetime] = Field(None, description="End of analysis window")
    rpm_threshold: float = Field(
        10, ge=0, description="RPM below which engine is stopped"
    )
    speed_threshold: float = Field(
        1, ge=0, description="Speed (kts) below which vessel is drifting"
    )
    gap_hours: float = Field(6, ge=1, description="Timestamp gap threshold (hours)")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class BeaufortEntry(BaseModel):
    """Single Beaufort scale entry."""

    force: int
    wind_min_kts: int
    wind_max_kts: int
    wave_height_m: float
    description: str


class BeaufortScaleResponse(BaseModel):
    """Beaufort scale reference data."""

    scale: List[BeaufortEntry]


class GoodWeatherLegResponse(BaseModel):
    """Per-leg good weather classification."""

    leg_index: int
    wind_speed_kts: float
    wave_height_m: float
    current_speed_ms: float
    bf_force: int
    is_good_weather: bool
    time_hours: float


class GoodWeatherResponse(BaseModel):
    """Good weather day analysis result."""

    total_days: float
    good_weather_days: float
    bad_weather_days: float
    good_weather_pct: float
    bf_threshold: int
    wave_threshold_m: Optional[float]
    current_threshold_kts: Optional[float]
    legs: List[GoodWeatherLegResponse]


class WarrantyLegDetailResponse(BaseModel):
    """Per-leg warranty verification detail."""

    leg_index: int
    sog_kts: float
    fuel_mt: float
    time_hours: float
    distance_nm: float
    bf_force: int
    is_good_weather: bool


class WarrantyVerificationResponse(BaseModel):
    """Warranty verification result."""

    warranted_speed_kts: float
    achieved_speed_kts: float
    speed_margin_kts: float
    speed_compliant: bool
    warranted_consumption_mt_day: float
    achieved_consumption_mt_day: float
    consumption_margin_mt: float
    consumption_compliant: bool
    good_weather_hours: float
    total_hours: float
    legs_assessed: int
    legs_good_weather: int
    legs: List[WarrantyLegDetailResponse]


class OffHireEventResponse(BaseModel):
    """Single off-hire event."""

    start_time: datetime
    end_time: datetime
    duration_hours: float
    reason: str
    avg_speed_kts: Optional[float]


class OffHireResponse(BaseModel):
    """Off-hire analysis result."""

    total_hours: float
    on_hire_hours: float
    off_hire_hours: float
    off_hire_pct: float
    events: List[OffHireEventResponse]

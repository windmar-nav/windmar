"""CII compliance API schemas."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class CIIFuelConsumption(BaseModel):
    """Fuel consumption by type in metric tons."""

    hfo: float = Field(0, ge=0, description="Heavy Fuel Oil (MT)")
    lfo: float = Field(0, ge=0, description="Light Fuel Oil (MT)")
    vlsfo: float = Field(0, ge=0, description="Very Low Sulphur Fuel Oil (MT)")
    mdo: float = Field(0, ge=0, description="Marine Diesel Oil (MT)")
    mgo: float = Field(0, ge=0, description="Marine Gas Oil (MT)")
    lng: float = Field(0, ge=0, description="LNG (MT)")
    lpg_propane: float = Field(0, ge=0, description="LPG Propane (MT)")
    lpg_butane: float = Field(0, ge=0, description="LPG Butane (MT)")
    methanol: float = Field(0, ge=0, description="Methanol (MT)")
    ethanol: float = Field(0, ge=0, description="Ethanol (MT)")

    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.model_dump().items() if v > 0}


class CIICalculateRequest(BaseModel):
    """Request for CII calculation."""

    fuel_consumption_mt: CIIFuelConsumption
    total_distance_nm: float = Field(..., gt=0)
    dwt: float = Field(..., gt=0)
    vessel_type: str = Field("tanker", description="IMO vessel type category")
    year: int = Field(2024, ge=2019, le=2040)
    gt: Optional[float] = Field(
        None, gt=0, description="Gross tonnage (for cruise/ro-ro passenger)"
    )


class CIIProjectRequest(BaseModel):
    """Request for multi-year CII projection."""

    annual_fuel_mt: CIIFuelConsumption
    annual_distance_nm: float = Field(..., gt=0)
    dwt: float = Field(..., gt=0)
    vessel_type: str = Field("tanker")
    start_year: int = Field(2024, ge=2019, le=2040)
    end_year: int = Field(2030, ge=2019, le=2040)
    fuel_efficiency_improvement_pct: float = Field(
        0, ge=0, le=20, description="Annual efficiency improvement %"
    )
    gt: Optional[float] = Field(None, gt=0)


class CIIReductionRequest(BaseModel):
    """Request for CII reduction calculation."""

    current_fuel_mt: CIIFuelConsumption
    current_distance_nm: float = Field(..., gt=0)
    dwt: float = Field(..., gt=0)
    vessel_type: str = Field("tanker")
    target_rating: str = Field("C", description="Target rating: A, B, C, or D")
    target_year: int = Field(2026, ge=2019, le=2040)
    gt: Optional[float] = Field(None, gt=0)


# ---------------------------------------------------------------------------
# Speed Sweep (Simulator tab)
# ---------------------------------------------------------------------------


class CIISpeedSweepRequest(BaseModel):
    """Request for CII speed sweep simulation."""

    dwt: float = Field(..., gt=0)
    vessel_type: str = Field("tanker")
    distance_nm: float = Field(..., gt=0, description="Single-voyage distance (nm)")
    voyages_per_year: int = Field(12, ge=1, le=100)
    fuel_type: str = Field("vlsfo")
    year: int = Field(2026, ge=2019, le=2040)
    speed_min_kts: float = Field(8.0, ge=3.0, le=25.0)
    speed_max_kts: float = Field(16.0, ge=3.0, le=25.0)
    speed_step_kts: float = Field(0.5, ge=0.1, le=2.0)
    is_laden: bool = True


class CIISpeedSweepPoint(BaseModel):
    """Single data point in the speed sweep curve."""

    speed_kts: float
    fuel_per_voyage_mt: float
    annual_fuel_mt: float
    annual_co2_mt: float
    attained_cii: float
    required_cii: float
    rating: str


class CIISpeedSweepResponse(BaseModel):
    """Response for CII speed sweep."""

    points: List[CIISpeedSweepPoint]
    optimal_speed_kts: float
    rating_boundaries: Dict[str, float]


# ---------------------------------------------------------------------------
# Thresholds (Projection chart enhancement)
# ---------------------------------------------------------------------------


class CIIThresholdYear(BaseModel):
    """Rating boundaries for a single year."""

    year: int
    required_cii: float
    boundaries: Dict[str, float]
    reduction_factor: float


class CIIThresholdsResponse(BaseModel):
    """Multi-year rating boundary thresholds."""

    years: List[CIIThresholdYear]
    vessel_type: str
    capacity: float


# ---------------------------------------------------------------------------
# Fleet comparison
# ---------------------------------------------------------------------------


class CIIFleetVessel(BaseModel):
    """Single vessel in a fleet CII batch request."""

    name: str = Field(..., min_length=1, max_length=100)
    dwt: float = Field(..., gt=0)
    vessel_type: str = Field("tanker")
    fuel_consumption_mt: Dict[str, float] = Field(
        ..., description="Fuel type -> MT consumed annually"
    )
    total_distance_nm: float = Field(..., gt=0)
    year: int = Field(2026, ge=2019, le=2040)
    gt: Optional[float] = Field(None, gt=0)


class CIIFleetRequest(BaseModel):
    """Batch CII calculation for multiple vessels."""

    vessels: List[CIIFleetVessel] = Field(..., min_length=1, max_length=20)


class CIIFleetResult(BaseModel):
    """CII result for one vessel in a fleet batch."""

    name: str
    rating: str
    attained_cii: float
    required_cii: float
    compliance_status: str
    total_co2_mt: float
    margin_to_downgrade: float
    margin_to_upgrade: float


class CIIFleetResponse(BaseModel):
    """Batch CII results for the fleet."""

    results: List[CIIFleetResult]
    summary: Dict[str, int]

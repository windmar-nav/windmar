"""CII compliance API schemas."""

from typing import Dict, Optional

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
    gt: Optional[float] = Field(None, gt=0, description="Gross tonnage (for cruise/ro-ro passenger)")


class CIIProjectRequest(BaseModel):
    """Request for multi-year CII projection."""
    annual_fuel_mt: CIIFuelConsumption
    annual_distance_nm: float = Field(..., gt=0)
    dwt: float = Field(..., gt=0)
    vessel_type: str = Field("tanker")
    start_year: int = Field(2024, ge=2019, le=2040)
    end_year: int = Field(2030, ge=2019, le=2040)
    fuel_efficiency_improvement_pct: float = Field(0, ge=0, le=20, description="Annual efficiency improvement %")
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

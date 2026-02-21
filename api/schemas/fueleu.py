"""FuelEU Maritime compliance API schemas."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class FuelEUFuelConsumption(BaseModel):
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


class FuelEUCalculateRequest(BaseModel):
    """Request for GHG intensity calculation."""
    fuel_consumption_mt: FuelEUFuelConsumption
    year: int = Field(2025, ge=2025, le=2050)


class FuelEUComplianceRequest(BaseModel):
    """Request for compliance balance calculation."""
    fuel_consumption_mt: FuelEUFuelConsumption
    year: int = Field(2025, ge=2025, le=2050)


class FuelEUPenaltyRequest(BaseModel):
    """Request for penalty exposure calculation."""
    fuel_consumption_mt: FuelEUFuelConsumption
    year: int = Field(2025, ge=2025, le=2050)
    consecutive_deficit_years: int = Field(0, ge=0, le=20)


class FuelEUPoolingVessel(BaseModel):
    """Single vessel in a pooling request."""
    name: str = Field(..., min_length=1, max_length=100)
    fuel_mt: Dict[str, float] = Field(..., description="Fuel type -> MT consumed")


class FuelEUPoolingRequest(BaseModel):
    """Request for fleet pooling simulation."""
    vessels: List[FuelEUPoolingVessel] = Field(..., min_length=1, max_length=20)
    year: int = Field(2025, ge=2025, le=2050)


class FuelEUProjectRequest(BaseModel):
    """Request for multi-year compliance projection."""
    fuel_consumption_mt: FuelEUFuelConsumption
    start_year: int = Field(2025, ge=2025, le=2050)
    end_year: int = Field(2050, ge=2025, le=2050)
    annual_efficiency_improvement_pct: float = Field(0, ge=0, le=20)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class FuelEUFuelBreakdown(BaseModel):
    """Per-fuel breakdown of energy and emissions."""
    fuel_type: str
    mass_mt: float
    energy_mj: float
    wtt_gco2eq: float
    ttw_gco2eq: float
    wtw_gco2eq: float
    wtw_intensity: float


class FuelEUCalculateResponse(BaseModel):
    """Response for GHG intensity calculation."""
    ghg_intensity: float
    total_energy_mj: float
    total_co2eq_g: float
    fuel_breakdown: List[FuelEUFuelBreakdown]


class FuelEUComplianceResponse(BaseModel):
    """Response for compliance balance calculation."""
    year: int
    ghg_intensity: float
    ghg_limit: float
    reduction_target_pct: float
    compliance_balance_gco2eq: float
    total_energy_mj: float
    status: str


class FuelEUPenaltyResponse(BaseModel):
    """Response for penalty calculation."""
    compliance_balance_gco2eq: float
    non_compliant_energy_mj: float
    vlsfo_equivalent_mt: float
    penalty_eur: float
    penalty_per_mt_fuel: float


class FuelEUPoolingVesselResult(BaseModel):
    """Individual vessel result within a pooling response."""
    name: str
    ghg_intensity: float
    total_energy_mj: float
    total_co2eq_g: float
    individual_balance_gco2eq: float
    status: str


class FuelEUPoolingResponse(BaseModel):
    """Response for fleet pooling simulation."""
    fleet_ghg_intensity: float
    fleet_total_energy_mj: float
    fleet_total_co2eq_g: float
    fleet_balance_gco2eq: float
    per_vessel: List[FuelEUPoolingVesselResult]
    status: str


class FuelEUProjectionYear(BaseModel):
    """Single year in a multi-year projection response."""
    year: int
    ghg_intensity: float
    ghg_limit: float
    reduction_target_pct: float
    compliance_balance_gco2eq: float
    total_energy_mj: float
    status: str
    penalty_eur: float


class FuelEUProjectResponse(BaseModel):
    """Response for multi-year compliance projection."""
    projections: List[FuelEUProjectionYear]


class FuelEULimitYear(BaseModel):
    """GHG limit for a target year."""
    year: int
    reduction_pct: float
    ghg_limit: float


class FuelEULimitsResponse(BaseModel):
    """Response for GHG limits listing."""
    limits: List[FuelEULimitYear]
    reference_ghg: float


class FuelEUFuelInfo(BaseModel):
    """Fuel type emission factor info."""
    id: str
    name: str
    lcv_mj_per_g: float
    wtt_gco2eq_per_mj: float
    ttw_gco2eq_per_mj: float
    wtw_gco2eq_per_mj: float


class FuelEUFuelTypesResponse(BaseModel):
    """Response for fuel types listing."""
    fuel_types: List[FuelEUFuelInfo]

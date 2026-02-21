"""
FuelEU Maritime (EU 2023/1805) compliance API router.

Handles GHG intensity calculations, compliance balance,
penalty exposure, fleet pooling, and multi-year projections.
"""

from dataclasses import asdict

from fastapi import APIRouter

from api.schemas.fueleu import (
    FuelEUCalculateRequest, FuelEUCalculateResponse,
    FuelEUComplianceRequest, FuelEUComplianceResponse,
    FuelEUPenaltyRequest, FuelEUPenaltyResponse,
    FuelEUPoolingRequest, FuelEUPoolingResponse,
    FuelEUProjectRequest, FuelEUProjectResponse,
    FuelEULimitsResponse, FuelEULimitYear,
    FuelEUFuelTypesResponse, FuelEUFuelInfo,
)
from src.compliance.fueleu import FuelEUCalculator, REFERENCE_GHG

router = APIRouter(prefix="/api/fueleu", tags=["FuelEU Maritime"])

_calc = FuelEUCalculator()


# ---- reference data endpoints -----------------------------------------------

@router.get("/fuel-types", response_model=FuelEUFuelTypesResponse)
async def get_fueleu_fuel_types():
    """List supported fuel types with WtW emission factors."""
    fuels = _calc.get_fuel_info()
    return FuelEUFuelTypesResponse(
        fuel_types=[FuelEUFuelInfo(**f) for f in fuels],
    )


@router.get("/limits", response_model=FuelEULimitsResponse)
async def get_fueleu_limits():
    """Return GHG intensity limits for all regulation target years."""
    limits = _calc.get_limits_by_year()
    return FuelEULimitsResponse(
        limits=[FuelEULimitYear(**lim) for lim in limits],
        reference_ghg=REFERENCE_GHG,
    )


# ---- calculation endpoints --------------------------------------------------

@router.post("/calculate", response_model=FuelEUCalculateResponse)
async def calculate_ghg_intensity(request: FuelEUCalculateRequest):
    """Calculate Well-to-Wake GHG intensity for given fuel consumption."""
    result = _calc.calculate_ghg_intensity(request.fuel_consumption_mt.to_dict())
    return FuelEUCalculateResponse(
        ghg_intensity=result.ghg_intensity,
        total_energy_mj=result.total_energy_mj,
        total_co2eq_g=result.total_co2eq_g,
        fuel_breakdown=[asdict(fb) for fb in result.fuel_breakdown],
    )


@router.post("/compliance", response_model=FuelEUComplianceResponse)
async def calculate_compliance(request: FuelEUComplianceRequest):
    """Calculate compliance balance (surplus/deficit) against annual limit."""
    result = _calc.calculate_compliance_balance(
        request.fuel_consumption_mt.to_dict(), request.year,
    )
    return FuelEUComplianceResponse(**asdict(result))


@router.post("/penalty", response_model=FuelEUPenaltyResponse)
async def calculate_penalty(request: FuelEUPenaltyRequest):
    """Calculate penalty exposure for a deficit position."""
    result = _calc.calculate_penalty(
        request.fuel_consumption_mt.to_dict(),
        request.year,
        consecutive_deficit_years=request.consecutive_deficit_years,
    )
    return FuelEUPenaltyResponse(**asdict(result))


@router.post("/pooling", response_model=FuelEUPoolingResponse)
async def simulate_pooling(request: FuelEUPoolingRequest):
    """Simulate fleet pooling — aggregate vessels into a shared compliance pool."""
    vessels = [{"name": v.name, "fuel_mt": v.fuel_mt} for v in request.vessels]
    result = _calc.simulate_pooling(vessels, request.year)
    return FuelEUPoolingResponse(
        fleet_ghg_intensity=result.fleet_ghg_intensity,
        fleet_total_energy_mj=result.fleet_total_energy_mj,
        fleet_total_co2eq_g=result.fleet_total_co2eq_g,
        fleet_balance_gco2eq=result.fleet_balance_gco2eq,
        per_vessel=[asdict(v) for v in result.per_vessel],
        status=result.status,
    )


@router.post("/project", response_model=FuelEUProjectResponse)
async def project_compliance(request: FuelEUProjectRequest):
    """Project compliance across years as GHG limits tighten."""
    projections = _calc.project_compliance(
        request.fuel_consumption_mt.to_dict(),
        start_year=request.start_year,
        end_year=request.end_year,
        annual_efficiency_improvement_pct=request.annual_efficiency_improvement_pct,
    )
    return FuelEUProjectResponse(
        projections=[asdict(p) for p in projections],
    )

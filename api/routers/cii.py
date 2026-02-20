"""
CII (Carbon Intensity Indicator) compliance API router.

Handles CII calculations, projections, and reduction targets
per IMO MEPC.354(78) and MEPC.355(78).
"""

from fastapi import APIRouter, HTTPException

from api.schemas.cii import CIICalculateRequest, CIIProjectRequest, CIIReductionRequest
from src.compliance.cii import (
    CIICalculator, VesselType as CIIVesselType, CIIRating,
)

router = APIRouter(prefix="/api/cii", tags=["CII Compliance"])


# ---- helpers (CII-only) ---------------------------------------------------

def _resolve_vessel_type(name: str) -> CIIVesselType:
    """Resolve vessel type string to enum."""
    mapping = {vt.value: vt for vt in CIIVesselType}
    if name in mapping:
        return mapping[name]
    raise HTTPException(status_code=400, detail=f"Unknown vessel type: {name}. Valid: {list(mapping.keys())}")


def _resolve_target_rating(name: str) -> CIIRating:
    """Resolve rating string to enum."""
    mapping = {r.value: r for r in CIIRating}
    if name.upper() in mapping:
        return mapping[name.upper()]
    raise HTTPException(status_code=400, detail=f"Unknown rating: {name}. Valid: A, B, C, D, E")


def _compliance_status(rating: CIIRating) -> str:
    if rating in (CIIRating.A, CIIRating.B):
        return "Compliant"
    elif rating == CIIRating.C:
        return "At Risk"
    else:
        return "Non-Compliant"


# ---- endpoints -------------------------------------------------------------

@router.get("/vessel-types")
async def get_cii_vessel_types():
    """List available IMO vessel type categories for CII calculations."""
    vessel_types = [
        {"id": vt.value, "name": vt.value.replace("_", " ").title()}
        for vt in CIIVesselType
    ]
    return {"vessel_types": vessel_types}


@router.get("/fuel-types")
async def get_cii_fuel_types():
    """List available fuel types and their CO2 emission factors."""
    fuel_types = [
        {"id": fuel, "name": fuel.upper().replace("_", " "), "co2_factor": factor}
        for fuel, factor in CIICalculator.CO2_FACTORS.items()
    ]
    return {"fuel_types": fuel_types}


@router.post("/calculate")
async def calculate_cii(request: CIICalculateRequest):
    """Calculate CII rating for given fuel consumption and distance."""
    vtype = _resolve_vessel_type(request.vessel_type)
    gt = request.gt if vtype in (CIIVesselType.CRUISE_PASSENGER, CIIVesselType.RO_RO_PASSENGER) else None

    try:
        calc = CIICalculator(vessel_type=vtype, dwt=request.dwt, year=request.year, gt=gt)
        result = calc.calculate(request.fuel_consumption_mt.to_dict(), request.total_distance_nm)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "year": result.year,
        "rating": result.rating.value,
        "compliance_status": _compliance_status(result.rating),
        "attained_cii": result.attained_cii,
        "required_cii": result.required_cii,
        "rating_boundaries": result.rating_boundaries,
        "reduction_factor": result.reduction_factor,
        "total_co2_mt": result.total_co2_mt,
        "total_distance_nm": result.total_distance_nm,
        "capacity": result.capacity,
        "vessel_type": result.vessel_type.value,
        "margin_to_downgrade": result.margin_to_downgrade,
        "margin_to_upgrade": result.margin_to_upgrade,
    }


@router.post("/project")
async def project_cii(request: CIIProjectRequest):
    """Project CII rating across multiple years with optional efficiency improvements."""
    vtype = _resolve_vessel_type(request.vessel_type)
    gt = request.gt if vtype in (CIIVesselType.CRUISE_PASSENGER, CIIVesselType.RO_RO_PASSENGER) else None

    try:
        calc = CIICalculator(vessel_type=vtype, dwt=request.dwt, year=request.start_year, gt=gt)
        years = list(range(request.start_year, request.end_year + 1))
        projections = calc.project_rating(
            request.annual_fuel_mt.to_dict(),
            request.annual_distance_nm,
            years=years,
            fuel_reduction_rate=request.fuel_efficiency_improvement_pct,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    proj_list = [
        {
            "year": p.year,
            "rating": p.rating.value,
            "attained_cii": p.attained_cii,
            "required_cii": p.required_cii,
            "reduction_factor": p.reduction_factor,
            "status": p.status,
        }
        for p in projections
    ]

    # Build summary
    current_rating = projections[0].rating.value if projections else "?"
    final_rating = projections[-1].rating.value if projections else "?"
    years_until_d = next(
        (p.year - projections[0].year for p in projections if p.rating in (CIIRating.D, CIIRating.E)),
        "N/A"
    )
    years_until_e = next(
        (p.year - projections[0].year for p in projections if p.rating == CIIRating.E),
        "N/A"
    )

    if final_rating in ("D", "E"):
        recommendation = f"Action required: rating degrades to {final_rating} by {projections[-1].year}."
    elif final_rating == "C":
        recommendation = "Borderline: rating reaches C. Consider efficiency improvements."
    else:
        recommendation = f"On track: rating remains {final_rating} through {projections[-1].year}."

    return {
        "projections": proj_list,
        "summary": {
            "current_rating": current_rating,
            "final_rating": final_rating,
            "years_until_d_rating": years_until_d,
            "years_until_e_rating": years_until_e,
            "recommendation": recommendation,
        },
    }


@router.post("/reduction")
async def calculate_cii_reduction(request: CIIReductionRequest):
    """Calculate fuel reduction needed to achieve a target CII rating."""
    vtype = _resolve_vessel_type(request.vessel_type)
    target = _resolve_target_rating(request.target_rating)
    gt = request.gt if vtype in (CIIVesselType.CRUISE_PASSENGER, CIIVesselType.RO_RO_PASSENGER) else None

    try:
        calc = CIICalculator(vessel_type=vtype, dwt=request.dwt, year=request.target_year, gt=gt)
        result = calc.calculate_required_reduction(
            request.current_fuel_mt.to_dict(),
            request.current_distance_nm,
            target_rating=target,
            target_year=request.target_year,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return result

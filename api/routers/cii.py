"""
CII (Carbon Intensity Indicator) compliance API router.

Handles CII calculations, projections, and reduction targets
per IMO MEPC.354(78) and MEPC.355(78).
"""

from fastapi import APIRouter, HTTPException, Query

from api.schemas.cii import (
    CIICalculateRequest,
    CIIProjectRequest,
    CIIReductionRequest,
    CIISpeedSweepRequest,
    CIISpeedSweepResponse,
    CIISpeedSweepPoint,
    CIIThresholdsResponse,
    CIIThresholdYear,
    CIIFleetRequest,
    CIIFleetResponse,
    CIIFleetResult,
)
from api.state import get_app_state
from src.compliance.cii import (
    CIICalculator,
    VesselType as CIIVesselType,
    CIIRating,
)

router = APIRouter(prefix="/api/cii", tags=["CII Compliance"])


# ---- helpers (CII-only) ---------------------------------------------------


def _resolve_vessel_type(name: str) -> CIIVesselType:
    """Resolve vessel type string to enum."""
    mapping = {vt.value: vt for vt in CIIVesselType}
    if name in mapping:
        return mapping[name]
    raise HTTPException(
        status_code=400,
        detail=f"Unknown vessel type: {name}. Valid: {list(mapping.keys())}",
    )


def _resolve_target_rating(name: str) -> CIIRating:
    """Resolve rating string to enum."""
    mapping = {r.value: r for r in CIIRating}
    if name.upper() in mapping:
        return mapping[name.upper()]
    raise HTTPException(
        status_code=400, detail=f"Unknown rating: {name}. Valid: A, B, C, D, E"
    )


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
    gt = (
        request.gt
        if vtype in (CIIVesselType.CRUISE_PASSENGER, CIIVesselType.RO_RO_PASSENGER)
        else None
    )

    try:
        calc = CIICalculator(
            vessel_type=vtype, dwt=request.dwt, year=request.year, gt=gt
        )
        result = calc.calculate(
            request.fuel_consumption_mt.to_dict(), request.total_distance_nm
        )
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
    gt = (
        request.gt
        if vtype in (CIIVesselType.CRUISE_PASSENGER, CIIVesselType.RO_RO_PASSENGER)
        else None
    )

    try:
        calc = CIICalculator(
            vessel_type=vtype, dwt=request.dwt, year=request.start_year, gt=gt
        )
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
        (
            p.year - projections[0].year
            for p in projections
            if p.rating in (CIIRating.D, CIIRating.E)
        ),
        "N/A",
    )
    years_until_e = next(
        (p.year - projections[0].year for p in projections if p.rating == CIIRating.E),
        "N/A",
    )

    if final_rating in ("D", "E"):
        recommendation = f"Action required: rating degrades to {final_rating} by {projections[-1].year}."
    elif final_rating == "C":
        recommendation = (
            "Borderline: rating reaches C. Consider efficiency improvements."
        )
    else:
        recommendation = (
            f"On track: rating remains {final_rating} through {projections[-1].year}."
        )

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
    gt = (
        request.gt
        if vtype in (CIIVesselType.CRUISE_PASSENGER, CIIVesselType.RO_RO_PASSENGER)
        else None
    )

    try:
        calc = CIICalculator(
            vessel_type=vtype, dwt=request.dwt, year=request.target_year, gt=gt
        )
        result = calc.calculate_required_reduction(
            request.current_fuel_mt.to_dict(),
            request.current_distance_nm,
            target_rating=target,
            target_year=request.target_year,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return result


# ---- Phase 2b: Simulator endpoints ----------------------------------------


@router.post("/speed-sweep", response_model=CIISpeedSweepResponse)
async def speed_sweep(request: CIISpeedSweepRequest):
    """Compute CII across a range of speeds using the vessel physics model."""
    if request.speed_min_kts >= request.speed_max_kts:
        raise HTTPException(
            status_code=400, detail="speed_min_kts must be less than speed_max_kts"
        )

    vtype = _resolve_vessel_type(request.vessel_type)
    vessel_model = get_app_state().vessel.model

    # Validate fuel type
    if request.fuel_type.lower() not in CIICalculator.CO2_FACTORS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown fuel type: {request.fuel_type}. Valid: {list(CIICalculator.CO2_FACTORS.keys())}",
        )

    calc = CIICalculator(vessel_type=vtype, dwt=request.dwt, year=request.year)
    info = calc.get_rating_boundaries_for_year(request.year)

    points: list[CIISpeedSweepPoint] = []
    best_rating_order = 5  # E=5, A=1
    optimal_speed = request.speed_min_kts
    rating_order = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}

    speed = request.speed_min_kts
    while speed <= request.speed_max_kts + 1e-9:
        # Get fuel consumption from the physics model for one voyage
        fuel_result = vessel_model.calculate_fuel_consumption(
            speed_kts=speed,
            is_laden=request.is_laden,
            weather=None,
            distance_nm=request.distance_nm,
        )
        fuel_per_voyage = fuel_result["fuel_mt"]
        annual_fuel = fuel_per_voyage * request.voyages_per_year
        annual_distance = request.distance_nm * request.voyages_per_year

        # CII calculation
        cii_result = calc.calculate(
            {request.fuel_type: annual_fuel},
            annual_distance,
            year=request.year,
        )

        point = CIISpeedSweepPoint(
            speed_kts=round(speed, 1),
            fuel_per_voyage_mt=round(fuel_per_voyage, 2),
            annual_fuel_mt=round(annual_fuel, 2),
            annual_co2_mt=cii_result.total_co2_mt,
            attained_cii=cii_result.attained_cii,
            required_cii=cii_result.required_cii,
            rating=cii_result.rating.value,
        )
        points.append(point)

        # Track optimal: best rating, and within same rating, lowest speed
        r = rating_order.get(cii_result.rating.value, 5)
        if r < best_rating_order:
            best_rating_order = r
            optimal_speed = speed

        speed += request.speed_step_kts

    return CIISpeedSweepResponse(
        points=points,
        optimal_speed_kts=round(optimal_speed, 1),
        rating_boundaries=info["boundaries"],
    )


@router.get("/thresholds", response_model=CIIThresholdsResponse)
async def get_thresholds(
    dwt: float = Query(..., gt=0),
    vessel_type: str = Query("tanker"),
    gt: float | None = Query(None, gt=0),
):
    """Return A-E rating boundary values for each year from 2019 to 2035."""
    vtype = _resolve_vessel_type(vessel_type)
    gt_val = (
        gt
        if vtype in (CIIVesselType.CRUISE_PASSENGER, CIIVesselType.RO_RO_PASSENGER)
        else None
    )

    try:
        calc = CIICalculator(vessel_type=vtype, dwt=dwt, year=2024, gt=gt_val)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    years_out: list[CIIThresholdYear] = []
    for yr in range(2019, 2036):
        info = calc.get_rating_boundaries_for_year(yr)
        years_out.append(
            CIIThresholdYear(
                year=yr,
                required_cii=info["required_cii"],
                boundaries=info["boundaries"],
                reduction_factor=info["reduction_factor"],
            )
        )

    return CIIThresholdsResponse(
        years=years_out,
        vessel_type=vessel_type,
        capacity=calc.capacity,
    )


@router.post("/fleet", response_model=CIIFleetResponse)
async def calculate_fleet_cii(request: CIIFleetRequest):
    """Batch CII calculation for multiple vessels."""
    results: list[CIIFleetResult] = []
    summary: dict[str, int] = {
        "A": 0,
        "B": 0,
        "C": 0,
        "D": 0,
        "E": 0,
        "total": len(request.vessels),
    }

    for vessel in request.vessels:
        vtype = _resolve_vessel_type(vessel.vessel_type)
        gt_val = (
            vessel.gt
            if vtype in (CIIVesselType.CRUISE_PASSENGER, CIIVesselType.RO_RO_PASSENGER)
            else None
        )

        try:
            calc = CIICalculator(
                vessel_type=vtype, dwt=vessel.dwt, year=vessel.year, gt=gt_val
            )
            cii_result = calc.calculate(
                vessel.fuel_consumption_mt, vessel.total_distance_nm, year=vessel.year
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Vessel '{vessel.name}': {e}")

        rating = cii_result.rating.value
        summary[rating] = summary.get(rating, 0) + 1

        results.append(
            CIIFleetResult(
                name=vessel.name,
                rating=rating,
                attained_cii=cii_result.attained_cii,
                required_cii=cii_result.required_cii,
                compliance_status=_compliance_status(cii_result.rating),
                total_co2_mt=cii_result.total_co2_mt,
                margin_to_downgrade=cii_result.margin_to_downgrade,
                margin_to_upgrade=cii_result.margin_to_upgrade,
            )
        )

    return CIIFleetResponse(results=results, summary=summary)

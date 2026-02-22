"""
Vessel configuration, calibration, and performance API router.

Handles vessel specifications, noon report ingestion, model calibration,
performance curves, fuel scenarios, and performance prediction.
"""

import logging
import math
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile

from api.auth import get_api_key
from api.database import get_db_context
from api.demo import require_not_demo
from api.models import VesselSpec
from api.rate_limit import limiter, get_rate_limit_string
from api.schemas import (
    CalibrationFactorsModel,
    CalibrationResponse,
    NoonReportModel,
    PerformancePredictionRequest,
    VesselConfig,
)
from api.state import get_vessel_state
from src.optimization.vessel_calibration import (
    CalibrationFactors, NoonReport, VesselCalibrator,
)
from src.optimization.vessel_model import VesselSpecs

logger = logging.getLogger(__name__)

# File upload size limits
MAX_CSV_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB for CSV/Excel files

router = APIRouter(prefix="/api/vessel", tags=["Vessel"])


# ============================================================================
# DB Persistence Helpers (also used by startup in main.py)
# ============================================================================

def save_vessel_specs_to_db(specs_dict: dict) -> None:
    """Persist vessel specs to the vessel_specs table (upsert by name='default')."""
    _d = VesselSpecs  # canonical defaults
    with get_db_context() as db:
        row = db.query(VesselSpec).filter(VesselSpec.name == "default").first()
        dwt = specs_dict.get("dwt", _d.dwt)
        speed = specs_dict.get("service_speed_laden", _d.service_speed_laden)
        vals = {
            "name": "default",
            "length": specs_dict.get("loa", _d.loa),
            "beam": specs_dict.get("beam", _d.beam),
            "draft": specs_dict.get("draft_laden", _d.draft_laden),
            "deadweight": dwt,
            "displacement": dwt * 1.33,
            "block_coefficient": _d.cb_laden,
            "engine_power": specs_dict.get("mcr_kw", _d.mcr_kw),
            "service_speed": speed,
            "max_speed": speed + 2.0,
        }
        extra = {
            "draft_ballast": specs_dict.get("draft_ballast"),
            "sfoc_at_mcr": specs_dict.get("sfoc_at_mcr"),
            "service_speed_ballast": specs_dict.get("service_speed_ballast"),
        }
        if row is None:
            row = VesselSpec(**vals, extra_metadata=extra)
            db.add(row)
        else:
            for k, v in vals.items():
                if k != "name":
                    setattr(row, k, v)
            row.extra_metadata = extra
            row.updated_at = datetime.utcnow()
        logger.info("Vessel specs persisted to DB (name='default')")


def load_vessel_specs_from_db() -> Optional[dict]:
    """Load vessel specs from DB. Returns dict for VesselSpecs() or None."""
    with get_db_context() as db:
        row = db.query(VesselSpec).filter(VesselSpec.name == "default").first()
        if row is None:
            return None
        extra = row.extra_metadata or {}
        return {
            "loa": row.length,
            "beam": row.beam,
            "draft_laden": row.draft,
            "dwt": row.deadweight,
            "mcr_kw": row.engine_power,
            "service_speed_laden": row.service_speed,
            "draft_ballast": extra.get("draft_ballast", 6.5),
            "sfoc_at_mcr": extra.get("sfoc_at_mcr", 171.0),
            "service_speed_ballast": extra.get("service_speed_ballast", 13.0),
        }


# ============================================================================
# Vessel Specifications
# ============================================================================

@router.get("/specs")
async def get_vessel_specs():
    """Get current vessel specifications."""
    specs = get_vessel_state().specs
    return {
        "dwt": specs.dwt,
        "loa": specs.loa,
        "beam": specs.beam,
        "draft_laden": specs.draft_laden,
        "draft_ballast": specs.draft_ballast,
        "mcr_kw": specs.mcr_kw,
        "sfoc_at_mcr": specs.sfoc_at_mcr,
        "service_speed_laden": specs.service_speed_laden,
        "service_speed_ballast": specs.service_speed_ballast,
    }


@router.post("/specs", dependencies=[Depends(require_not_demo("Vessel configuration"))])
@limiter.limit(get_rate_limit_string())
async def update_vessel_specs(
    request: Request,
    config: VesselConfig,
    api_key=Depends(get_api_key),
):
    """
    Update vessel specifications.

    Requires authentication via API key.
    """
    _vs = get_vessel_state()
    try:
        _vs.update_specs({
            'dwt': config.dwt,
            'loa': config.loa,
            'beam': config.beam,
            'draft_laden': config.draft_laden,
            'draft_ballast': config.draft_ballast,
            'mcr_kw': config.mcr_kw,
            'sfoc_at_mcr': config.sfoc_at_mcr,
            'service_speed_laden': config.service_speed_laden,
            'service_speed_ballast': config.service_speed_ballast,
        })

        # Persist to DB so specs survive container restarts
        try:
            save_vessel_specs_to_db({
                'dwt': config.dwt, 'loa': config.loa, 'beam': config.beam,
                'draft_laden': config.draft_laden, 'draft_ballast': config.draft_ballast,
                'mcr_kw': config.mcr_kw, 'sfoc_at_mcr': config.sfoc_at_mcr,
                'service_speed_laden': config.service_speed_laden,
                'service_speed_ballast': config.service_speed_ballast,
            })
        except Exception as persist_err:
            logger.warning("Failed to persist vessel specs to DB: %s", persist_err)

        return {"status": "success", "message": "Vessel specs updated and persisted"}

    except Exception as e:
        logger.error(f"Failed to update vessel specs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Calibration
# ============================================================================

@router.get("/calibration")
async def get_calibration():
    """Get current vessel calibration factors."""
    cal = get_vessel_state().calibration

    if cal is None:
        return {
            "calibrated": False,
            "factors": {
                "calm_water": 1.0,
                "wind": 1.0,
                "waves": 1.0,
                "sfoc_factor": 1.0,
            },
            "message": "No calibration data. Using default theoretical model."
        }

    return {
        "calibrated": True,
        "factors": {
            "calm_water": cal.calm_water,
            "wind": cal.wind,
            "waves": cal.waves,
            "sfoc_factor": cal.sfoc_factor,
        },
        "calibrated_at": cal.calibrated_at.isoformat() if cal.calibrated_at else None,
        "num_reports_used": cal.num_reports_used,
        "calibration_error_mt": cal.calibration_error,
        "days_since_drydock": cal.days_since_drydock,
    }


@router.post("/calibration/set", dependencies=[Depends(require_not_demo("Vessel calibration"))])
@limiter.limit(get_rate_limit_string())
async def set_calibration_factors(
    request: Request,
    factors: CalibrationFactorsModel,
    api_key=Depends(get_api_key),
):
    """
    Manually set calibration factors.

    Requires authentication via API key.
    """
    get_vessel_state().update_calibration(CalibrationFactors(
        calm_water=factors.calm_water,
        wind=factors.wind,
        waves=factors.waves,
        sfoc_factor=factors.sfoc_factor,
        calibrated_at=datetime.utcnow(),
        num_reports_used=0,
        days_since_drydock=factors.days_since_drydock,
    ))

    return {"status": "success", "message": "Calibration factors updated"}


# ============================================================================
# Noon Reports
# ============================================================================

@router.get("/noon-reports")
async def get_noon_reports():
    """Get list of uploaded noon reports."""
    _vs = get_vessel_state()
    return {
        "count": len(_vs.calibrator.noon_reports),
        "reports": [
            {
                "timestamp": r.timestamp.isoformat(),
                "latitude": r.latitude,
                "longitude": r.longitude,
                "speed_kts": r.speed_over_ground_kts,
                "fuel_mt": r.fuel_consumption_mt,
                "period_hours": r.period_hours,
                "is_laden": r.is_laden,
            }
            for r in _vs.calibrator.noon_reports
        ]
    }


@router.post("/noon-reports", dependencies=[Depends(require_not_demo("Noon report upload"))])
@limiter.limit(get_rate_limit_string())
async def add_noon_report(
    request: Request,
    report: NoonReportModel,
    api_key=Depends(get_api_key),
):
    """
    Add a single noon report for calibration.

    Requires authentication via API key.
    """
    _vs = get_vessel_state()
    nr = NoonReport(
        timestamp=report.timestamp,
        latitude=report.latitude,
        longitude=report.longitude,
        speed_over_ground_kts=report.speed_over_ground_kts,
        speed_through_water_kts=report.speed_through_water_kts,
        fuel_consumption_mt=report.fuel_consumption_mt,
        period_hours=report.period_hours,
        is_laden=report.is_laden,
        heading_deg=report.heading_deg,
        wind_speed_kts=report.wind_speed_kts,
        wind_direction_deg=report.wind_direction_deg,
        wave_height_m=report.wave_height_m,
        wave_direction_deg=report.wave_direction_deg,
        engine_power_kw=report.engine_power_kw,
    )

    _vs.calibrator.add_noon_report(nr)

    return {
        "status": "success",
        "total_reports": len(_vs.calibrator.noon_reports),
    }


@router.post("/noon-reports/upload-csv", dependencies=[Depends(require_not_demo("Noon report upload"))])
@limiter.limit("10/minute")
async def upload_noon_reports_csv(
    request: Request,
    file: UploadFile = File(...),
    api_key=Depends(get_api_key),
):
    """
    Upload noon reports from CSV file.

    Requires authentication via API key.
    Maximum file size: 50 MB.

    Expected columns:
    - timestamp (ISO format or common date format)
    - latitude, longitude
    - speed_over_ground_kts
    - fuel_consumption_mt
    - period_hours (optional, default 24)
    - is_laden (optional, default true)
    - wind_speed_kts, wind_direction_deg (optional)
    - wave_height_m, wave_direction_deg (optional)
    - heading_deg (optional)
    """
    _vs = get_vessel_state()
    try:
        # Validate file extension
        if file.filename and not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only .csv files accepted")

        # Read and validate file size
        content = await file.read()
        if len(content) > MAX_CSV_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {MAX_CSV_SIZE_BYTES // (1024*1024)} MB"
            )

        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            count = _vs.calibrator.add_noon_reports_from_csv(tmp_path)
        finally:
            tmp_path.unlink()

        return {
            "status": "success",
            "imported": count,
            "total_reports": len(_vs.calibrator.noon_reports),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to import CSV: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")


@router.post("/noon-reports/upload-excel", dependencies=[Depends(require_not_demo("Noon report upload"))])
@limiter.limit("10/minute")
async def upload_noon_reports_excel(
    request: Request,
    file: UploadFile = File(...),
    api_key=Depends(get_api_key),
):
    """
    Upload noon reports from an Excel file (.xlsx/.xls).

    Uses ExcelParser to auto-detect column mappings.
    """
    _vs = get_vessel_state()
    try:
        # Validate file extension
        _EXCEL_EXTS = {".xlsx", ".xls"}
        suffix = ".xlsx"
        if file.filename:
            suffix = Path(file.filename).suffix.lower() or ".xlsx"
            if suffix not in _EXCEL_EXTS:
                raise HTTPException(status_code=400, detail="Only .xlsx/.xls files accepted")

        content = await file.read()
        if len(content) > MAX_CSV_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {MAX_CSV_SIZE_BYTES // (1024*1024)} MB"
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            count = _vs.calibrator.add_noon_reports_from_excel(tmp_path)
        finally:
            tmp_path.unlink()

        return {
            "status": "success",
            "imported": count,
            "total_reports": len(_vs.calibrator.noon_reports),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to import Excel: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to parse Excel: {str(e)}")


@router.delete("/noon-reports", dependencies=[Depends(require_not_demo("Noon report deletion"))])
@limiter.limit(get_rate_limit_string())
async def clear_noon_reports(
    request: Request,
    api_key=Depends(get_api_key),
):
    """
    Clear all uploaded noon reports.

    Requires authentication via API key.
    """
    get_vessel_state().calibrator.noon_reports = []
    return {"status": "success", "message": "All noon reports cleared"}


@router.post("/calibrate", response_model=CalibrationResponse, dependencies=[Depends(require_not_demo("Vessel calibration"))])
@limiter.limit("5/minute")
async def calibrate_vessel(
    request: Request,
    days_since_drydock: int = Query(0, ge=0, description="Days since last dry dock"),
    api_key=Depends(get_api_key),
):
    """
    Run calibration using uploaded noon reports.

    Requires authentication via API key.

    Finds optimal calibration factors that minimize prediction error
    compared to actual fuel consumption.
    """
    _vs = get_vessel_state()
    if len(_vs.calibrator.noon_reports) < VesselCalibrator.MIN_REPORTS:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {VesselCalibrator.MIN_REPORTS} noon reports for calibration. "
                   f"Currently have {len(_vs.calibrator.noon_reports)}."
        )

    try:
        result = _vs.calibrator.calibrate(days_since_drydock=days_since_drydock)

        # Apply calibration atomically (rebuilds model, calculators, optimizers)
        _vs.update_calibration(result.factors)

        # Save calibration to file
        _vs.calibrator.save_calibration("default", _vs.calibration)

        return CalibrationResponse(
            factors=CalibrationFactorsModel(
                calm_water=result.factors.calm_water,
                wind=result.factors.wind,
                waves=result.factors.waves,
                sfoc_factor=result.factors.sfoc_factor,
                calibrated_at=result.factors.calibrated_at,
                num_reports_used=result.factors.num_reports_used,
                calibration_error=result.factors.calibration_error,
                days_since_drydock=result.factors.days_since_drydock,
            ),
            reports_used=result.reports_used,
            reports_skipped=result.reports_skipped,
            mean_error_before_mt=result.mean_error_before,
            mean_error_after_mt=result.mean_error_after,
            improvement_pct=result.improvement_pct,
            residuals=result.residuals,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Calibration failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Calibration failed: {str(e)}")


@router.post("/calibration/estimate-fouling", dependencies=[Depends(require_not_demo("Vessel calibration"))])
@limiter.limit(get_rate_limit_string())
async def estimate_hull_fouling(
    request: Request,
    days_since_drydock: int = Query(..., ge=0),
    operating_regions: List[str] = Query(default=[], description="Operating regions: tropical, warm_temperate, cold, polar"),
):
    """
    Estimate hull fouling factor without calibration data.

    Useful when no noon reports are available but you know
    the vessel's operating history.
    """
    fouling = get_vessel_state().calibrator.estimate_hull_fouling(
        days_since_drydock=days_since_drydock,
        operating_regions=operating_regions,
    )

    return {
        "days_since_drydock": days_since_drydock,
        "operating_regions": operating_regions,
        "estimated_fouling_factor": round(fouling, 3),
        "resistance_increase_pct": round((fouling - 1) * 100, 1),
        "note": "This is an estimate. Calibration with actual noon reports is more accurate.",
    }


# ============================================================================
# Model Introspection
# ============================================================================

@router.get("/model-status")
async def get_vessel_model_status():
    """
    Full vessel model status: all specs, calibration state, computed values.

    Returns every VesselSpecs field grouped by category, current calibration
    factors with timestamp, and derived performance values (optimal speeds,
    daily fuel at service speed).
    """
    _vs = get_vessel_state()
    specs = _vs.specs
    cal = _vs.calibration
    model = _vs.model

    # Compute optimal speeds and daily fuel at service speed
    optimal_laden = model.get_optimal_speed(is_laden=True)
    optimal_ballast = model.get_optimal_speed(is_laden=False)

    fuel_at_service_laden = model.calculate_fuel_consumption(
        speed_kts=specs.service_speed_laden, is_laden=True, distance_nm=specs.service_speed_laden * 24,
    )
    fuel_at_service_ballast = model.calculate_fuel_consumption(
        speed_kts=specs.service_speed_ballast, is_laden=False, distance_nm=specs.service_speed_ballast * 24,
    )

    return {
        "specifications": {
            "dimensions": {
                "loa": specs.loa,
                "lpp": specs.lpp,
                "beam": specs.beam,
                "draft_laden": specs.draft_laden,
                "draft_ballast": specs.draft_ballast,
                "dwt": specs.dwt,
                "displacement_laden": specs.displacement_laden,
                "displacement_ballast": specs.displacement_ballast,
            },
            "hull_form": {
                "cb_laden": specs.cb_laden,
                "cb_ballast": specs.cb_ballast,
                "wetted_surface_laden": specs.wetted_surface_laden,
                "wetted_surface_ballast": specs.wetted_surface_ballast,
            },
            "engine": {
                "mcr_kw": specs.mcr_kw,
                "sfoc_at_mcr": specs.sfoc_at_mcr,
                "service_speed_laden": specs.service_speed_laden,
                "service_speed_ballast": specs.service_speed_ballast,
            },
            "areas": {
                "frontal_area_laden": specs.frontal_area_laden,
                "frontal_area_ballast": specs.frontal_area_ballast,
                "lateral_area_laden": specs.lateral_area_laden,
                "lateral_area_ballast": specs.lateral_area_ballast,
            },
        },
        "calibration": {
            "calibrated": cal is not None,
            "factors": {
                "calm_water": cal.calm_water if cal else 1.0,
                "wind": cal.wind if cal else 1.0,
                "waves": cal.waves if cal else 1.0,
                "sfoc_factor": cal.sfoc_factor if cal else 1.0,
            },
            "calibrated_at": cal.calibrated_at.isoformat() if cal and cal.calibrated_at else None,
            "num_reports_used": cal.num_reports_used if cal else 0,
            "calibration_error_mt": cal.calibration_error if cal else 0.0,
            "days_since_drydock": cal.days_since_drydock if cal else 0,
        },
        "wave_method": model.wave_method,
        "computed": {
            "optimal_speed_laden_kts": round(optimal_laden, 1),
            "optimal_speed_ballast_kts": round(optimal_ballast, 1),
            "daily_fuel_service_laden_mt": round(fuel_at_service_laden["fuel_mt"], 2),
            "daily_fuel_service_ballast_mt": round(fuel_at_service_ballast["fuel_mt"], 2),
        },
    }


@router.get("/model-curves")
async def get_vessel_model_curves():
    """
    Pre-computed model curves for frontend charting.

    Returns speed-indexed arrays for resistance, power, SFOC, and fuel
    consumption — both theoretical (calibration=1.0) and with current
    calibration factors applied.
    """
    _vs = get_vessel_state()
    specs = _vs.specs
    model = _vs.model
    cal = _vs.calibration

    # Speed range: 5-14.5 kts in 0.5 kts steps
    speeds = list(np.arange(5.0, 15.0, 0.5))

    resistance_theoretical = []
    resistance_calibrated = []
    power_kw_list = []
    sfoc_gkwh_list = []
    fuel_mt_per_day_list = []

    for spd in speeds:
        # Theoretical (no calibration)
        speed_ms = spd * 0.51444
        draft = specs.draft_laden
        displacement = specs.displacement_laden
        cb = specs.cb_laden
        ws = specs.wetted_surface_laden
        r_theo = model._holtrop_mennen_resistance(speed_ms, draft, displacement, cb, ws)
        resistance_theoretical.append(round(r_theo / 1000.0, 2))  # kN

        # Calibrated
        r_cal = r_theo * model.calibration_factors.get("calm_water", 1.0)
        resistance_calibrated.append(round(r_cal / 1000.0, 2))  # kN

        # Power and fuel from the full model (uses calibration)
        result = model.calculate_fuel_consumption(
            speed_kts=spd, is_laden=True, distance_nm=spd * 24,
        )
        power_kw = result["power_kw"]
        power_kw_list.append(round(power_kw, 0))

        load = min(power_kw / specs.mcr_kw, 1.0)
        sfoc = model._sfoc_curve(load)
        sfoc_gkwh_list.append(round(sfoc, 1))

        fuel_mt_per_day_list.append(round(result["fuel_mt"], 2))


    # SFOC vs engine load (15-100%)
    sfoc_loads = list(range(15, 105, 5))
    sfoc_at_loads = []
    sfoc_at_loads_theoretical = []
    for load_pct in sfoc_loads:
        lf = load_pct / 100.0
        # Theoretical (sfoc_factor=1.0)
        if lf < 0.75:
            theo = specs.sfoc_at_mcr * (1.0 + 0.15 * (0.75 - lf))
        else:
            theo = specs.sfoc_at_mcr * (1.0 + 0.05 * (lf - 0.75))
        sfoc_at_loads_theoretical.append(round(theo, 1))
        sfoc_at_loads.append(round(theo * model.calibration_factors.get("sfoc_factor", 1.0), 1))

    return {
        "speed_range_kts": [round(s, 1) for s in speeds],
        "resistance_theoretical_kn": resistance_theoretical,
        "resistance_calibrated_kn": resistance_calibrated,
        "power_kw": power_kw_list,
        "sfoc_gkwh": sfoc_gkwh_list,
        "fuel_mt_per_day": fuel_mt_per_day_list,
        "sfoc_curve": {
            "load_pct": sfoc_loads,
            "sfoc_theoretical_gkwh": sfoc_at_loads_theoretical,
            "sfoc_calibrated_gkwh": sfoc_at_loads,
        },
        "calibration": {
            "calibrated": cal is not None,
            "factors": {
                "calm_water": cal.calm_water if cal else 1.0,
                "wind": cal.wind if cal else 1.0,
                "waves": cal.waves if cal else 1.0,
                "sfoc_factor": cal.sfoc_factor if cal else 1.0,
            },
            "calibrated_at": cal.calibrated_at.isoformat() if cal and cal.calibrated_at else None,
            "num_reports_used": cal.num_reports_used if cal else 0,
            "calibration_error_mt": cal.calibration_error if cal else 0.0,
        },
    }


@router.get("/fuel-scenarios")
async def get_fuel_scenarios():
    """
    Compute fuel scenarios using the real physics model with current calibration.

    Returns 4 daily fuel scenarios: calm laden, head wind laden,
    rough seas laden, and calm ballast.
    """
    _vs = get_vessel_state()
    specs = _vs.specs
    model = _vs.model

    # Scenario 1: Calm water laden (24h at service speed)
    distance_calm_laden = specs.service_speed_laden * 24
    calm_laden = model.calculate_fuel_consumption(
        speed_kts=specs.service_speed_laden, is_laden=True, distance_nm=distance_calm_laden,
    )

    # Scenario 2: Head wind laden (20 kt = 10.3 m/s head wind)
    headwind_wx = {
        "wind_speed_ms": 10.3, "wind_dir_deg": 0, "heading_deg": 0,
        "sig_wave_height_m": 0.5, "wave_dir_deg": 0,
    }
    headwind_laden = model.calculate_fuel_consumption(
        speed_kts=specs.service_speed_laden, is_laden=True,
        weather=headwind_wx, distance_nm=distance_calm_laden,
    )

    # Scenario 3: Rough seas laden (3m waves head seas)
    roughsea_wx = {
        "wind_speed_ms": 8.0, "wind_dir_deg": 0, "heading_deg": 0,
        "sig_wave_height_m": 3.0, "wave_dir_deg": 0,
    }
    roughsea_laden = model.calculate_fuel_consumption(
        speed_kts=specs.service_speed_laden, is_laden=True,
        weather=roughsea_wx, distance_nm=distance_calm_laden,
    )

    # Scenario 4: Calm water ballast
    distance_calm_ballast = specs.service_speed_ballast * 24
    calm_ballast = model.calculate_fuel_consumption(
        speed_kts=specs.service_speed_ballast, is_laden=False, distance_nm=distance_calm_ballast,
    )

    scenarios = [
        {
            "name": "Calm Water (Laden)",
            "conditions": f"{specs.service_speed_laden} kts, no wind/waves",
            "fuel_mt": round(calm_laden["fuel_mt"], 2),
            "power_kw": round(calm_laden["power_kw"], 0),
        },
        {
            "name": "Head Wind (Laden)",
            "conditions": f"{specs.service_speed_laden} kts, 20 kt head wind",
            "fuel_mt": round(headwind_laden["fuel_mt"], 2),
            "power_kw": round(headwind_laden["power_kw"], 0),
        },
        {
            "name": "Rough Seas (Laden)",
            "conditions": f"{specs.service_speed_laden} kts, 3m waves",
            "fuel_mt": round(roughsea_laden["fuel_mt"], 2),
            "power_kw": round(roughsea_laden["power_kw"], 0),
        },
        {
            "name": "Calm Water (Ballast)",
            "conditions": f"{specs.service_speed_ballast} kts, no wind/waves",
            "fuel_mt": round(calm_ballast["fuel_mt"], 2),
            "power_kw": round(calm_ballast["power_kw"], 0),
        },
    ]

    return {"scenarios": scenarios}


# ============================================================================
# Performance Prediction
# ============================================================================

@router.post("/predict")
@limiter.limit("30/minute")
async def predict_vessel_performance(request: Request, req: PerformancePredictionRequest):
    """
    Predict vessel speed and fuel consumption under given conditions.

    Two modes:
    - **engine_load_pct**: Find achievable speed at given power + weather
    - **calm_speed_kts**: Find what happens to a target speed in weather
      (power required, actual STW if MCR exceeded, fuel burn)

    All directions are relative to bow (0=ahead, 90=beam, 180=astern).
    """
    model = get_vessel_state().model

    # Convert relative directions to absolute with heading=0
    heading = 0.0
    wind_abs = req.wind_relative_deg
    wave_abs = req.wave_relative_deg

    weather = None
    if req.wind_speed_kts > 0 or req.wave_height_m > 0:
        weather = {
            "wind_speed_ms": req.wind_speed_kts * 0.51444,
            "wind_dir_deg": wind_abs,
            "sig_wave_height_m": req.wave_height_m,
            "wave_dir_deg": wave_abs,
        }

    current_ms = req.current_speed_kts * 0.51444
    # Current: relative 0° = head current (opposing) → flowing toward 180°
    current_abs = (180.0 + req.current_relative_deg) % 360

    # Mode 2: calm_speed_kts — calculate fuel at this speed in given weather
    if req.calm_speed_kts is not None:
        stw = req.calm_speed_kts
        distance_24h = stw * 24
        r = model.calculate_fuel_consumption(stw, req.is_laden, weather, distance_nm=distance_24h)

        # Check if MCR is exceeded
        mcr_exceeded = bool(r["required_power_kw"] > model.specs.mcr_kw)
        required_power_raw = float(r["required_power_kw"])
        actual_load_pct = float(min(r["required_power_kw"], model.specs.mcr_kw) / model.specs.mcr_kw * 100)
        sfoc = float(model._sfoc_curve(actual_load_pct / 100))

        # If MCR exceeded, find achievable speed at 100% MCR
        if mcr_exceeded:
            capped = model.predict_performance(
                is_laden=req.is_laden, weather=weather, engine_load_pct=100.0,
                current_speed_ms=current_ms, current_dir_deg=current_abs, heading_deg=heading,
            )
            stw = capped["stw_kts"]
            # Recalculate at capped speed
            r = model.calculate_fuel_consumption(stw, req.is_laden, weather, distance_nm=stw * 24)

        # Current effect
        current_effect_kts = 0.0
        if current_ms > 0:
            rel_angle = math.radians(current_abs - heading)
            current_effect_kts = float((current_ms / 0.51444) * math.cos(rel_angle))
        sog = max(0.0, stw + current_effect_kts)

        # Speed loss from weather
        speed_loss_pct = float((req.calm_speed_kts - stw) / req.calm_speed_kts * 100) if req.calm_speed_kts > 0 else 0.0

        # Sanitise resistance_breakdown_kn (numpy → native float)
        rb = r["resistance_breakdown_kn"]
        rb_clean = {k: round(float(v), 4) for k, v in rb.items()}

        result = {
            "stw_kts": round(float(stw), 2),
            "sog_kts": round(float(sog), 2),
            "fuel_per_day_mt": round(float(r["fuel_mt"]), 3),
            "fuel_per_nm_mt": round(float(r["fuel_mt"]) / (sog * 24), 4) if sog > 0 else 0.0,
            "power_kw": round(float(min(r["required_power_kw"], model.specs.mcr_kw)), 0),
            "required_power_kw": round(float(required_power_raw), 0),
            "load_pct": round(actual_load_pct, 1),
            "sfoc_gkwh": round(sfoc, 1),
            "mcr_exceeded": mcr_exceeded,
            "resistance_breakdown_kn": rb_clean,
            "speed_loss_from_weather_pct": round(max(0.0, speed_loss_pct), 1),
            "calm_water_speed_kts": req.calm_speed_kts,
            "current_effect_kts": round(current_effect_kts, 2),
            "service_speed_kts": float(model.specs.service_speed_laden if req.is_laden else model.specs.service_speed_ballast),
            "mode": "calm_speed",
            "inputs": {
                "calm_speed_kts": req.calm_speed_kts,
                "wind_relative_deg": req.wind_relative_deg,
                "wave_relative_deg": req.wave_relative_deg,
                "current_relative_deg": req.current_relative_deg,
            },
        }
        return result

    # Mode 1: engine_load_pct (default 85%)
    load = req.engine_load_pct if req.engine_load_pct is not None else 85.0

    result = model.predict_performance(
        is_laden=req.is_laden,
        weather=weather,
        engine_load_pct=load,
        current_speed_ms=current_ms,
        current_dir_deg=current_abs,
        heading_deg=heading,
    )

    result["mode"] = "engine_load"
    result["inputs"] = {
        "engine_load_pct": load,
        "wind_relative_deg": req.wind_relative_deg,
        "wave_relative_deg": req.wave_relative_deg,
        "current_relative_deg": req.current_relative_deg,
    }

    return result

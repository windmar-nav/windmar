"""
Engine log ingestion and calibration API router.

Handles engine log file upload (Excel/CSV), CRUD operations on parsed entries,
summary statistics, and calibration from NOON report entries.
"""

import logging
import tempfile
import uuid as uuid_mod
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from sqlalchemy import func

from api.auth import get_api_key
from api.database import get_db
from api.demo import require_not_demo
from api.models import EngineLogEntry
from api.rate_limit import limiter, get_rate_limit_string
from api.schemas import (
    CalibrationFactorsModel,
    EngineLogCalibrateResponse,
    EngineLogEntryResponse,
    EngineLogSummaryResponse,
    EngineLogUploadResponse,
)
from api.state import get_vessel_state
from src.database.engine_log_parser import EngineLogParser
from src.optimization.vessel_calibration import NoonReport, VesselCalibrator

logger = logging.getLogger(__name__)

# 50 MB limit for Excel/CSV engine log uploads
MAX_EXCEL_UPLOAD_BYTES = 50 * 1024 * 1024

router = APIRouter(tags=["Engine Log"])


@router.post("/api/engine-log/upload", response_model=EngineLogUploadResponse, dependencies=[Depends(require_not_demo("Engine log upload"))])
@limiter.limit("10/minute")
async def upload_engine_log(
    request: Request,
    file: UploadFile = File(...),
    vessel_id: Optional[str] = Query(None, description="Vessel UUID to link entries"),
    sheet_name: Optional[str] = Query(None, description="Sheet name (default: E log)"),
    api_key=Depends(get_api_key),
    db=Depends(get_db),
):
    """Upload and parse an engine log Excel workbook."""
    content = await file.read()
    if len(content) > MAX_EXCEL_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum: {MAX_EXCEL_UPLOAD_BYTES // (1024 * 1024)} MB")
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    suffix = ".xlsx"
    if file.filename:
        suffix = Path(file.filename).suffix or ".xlsx"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        parser = EngineLogParser(tmp_path)
        entries = parser.parse(sheet_name=sheet_name)
    except (ValueError, FileNotFoundError) as e:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        logger.error(f"Engine log parse error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Parse error: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    if not entries:
        raise HTTPException(status_code=400, detail="No valid entries found in file")

    batch_id = uuid_mod.uuid4()
    vessel_uuid = uuid_mod.UUID(vessel_id) if vessel_id else None

    db_entries = []
    for entry in entries:
        db_entry = EngineLogEntry(
            vessel_id=vessel_uuid,
            timestamp=entry["timestamp"],
            lapse_hours=entry.get("lapse_hours"),
            place=entry.get("place"),
            event=entry.get("event"),
            rpm=entry.get("rpm"),
            engine_distance=entry.get("engine_distance"),
            speed_stw=entry.get("speed_stw"),
            me_power_kw=entry.get("me_power_kw"),
            me_load_pct=entry.get("me_load_pct"),
            me_fuel_index_pct=entry.get("me_fuel_index_pct"),
            shaft_power=entry.get("shaft_power"),
            shaft_torque_knm=entry.get("shaft_torque_knm"),
            slip_pct=entry.get("slip_pct"),
            hfo_me_mt=entry.get("hfo_me_mt"),
            hfo_ae_mt=entry.get("hfo_ae_mt"),
            hfo_boiler_mt=entry.get("hfo_boiler_mt"),
            hfo_total_mt=entry.get("hfo_total_mt"),
            mgo_me_mt=entry.get("mgo_me_mt"),
            mgo_ae_mt=entry.get("mgo_ae_mt"),
            mgo_total_mt=entry.get("mgo_total_mt"),
            methanol_me_mt=entry.get("methanol_me_mt"),
            rob_vlsfo_mt=entry.get("rob_vlsfo_mt"),
            rob_mgo_mt=entry.get("rob_mgo_mt"),
            rob_methanol_mt=entry.get("rob_methanol_mt"),
            rh_me=entry.get("rh_me"),
            rh_ae_total=entry.get("rh_ae_total"),
            tc_rpm=entry.get("tc_rpm"),
            scav_air_press_bar=entry.get("scav_air_press_bar"),
            fuel_temp_c=entry.get("fuel_temp_c"),
            sw_temp_c=entry.get("sw_temp_c"),
            upload_batch_id=batch_id,
            source_sheet=entry.get("source_sheet"),
            source_file=file.filename or entry.get("source_file"),
            extended_data=entry.get("extended_data"),
        )
        db_entries.append(db_entry)

    db.add_all(db_entries)
    db.commit()

    stats = parser.get_statistics()

    return EngineLogUploadResponse(
        status="success",
        batch_id=str(batch_id),
        imported=len(db_entries),
        skipped=0,
        date_range=stats.get("date_range"),
        events_summary=stats.get("events_breakdown"),
    )


@router.get("/api/engine-log/entries", response_model=List[EngineLogEntryResponse])
async def get_engine_log_entries(
    event: Optional[str] = Query(None, description="Filter by event type"),
    date_from: Optional[datetime] = Query(None, description="Start date"),
    date_to: Optional[datetime] = Query(None, description="End date"),
    min_rpm: Optional[float] = Query(None, ge=0, description="Minimum RPM"),
    batch_id: Optional[str] = Query(None, description="Filter by batch UUID"),
    limit: int = Query(100, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    db=Depends(get_db),
):
    """Query engine log entries with optional filters and pagination."""
    query = db.query(EngineLogEntry)
    if event:
        query = query.filter(EngineLogEntry.event == event.upper())
    if date_from:
        query = query.filter(EngineLogEntry.timestamp >= date_from)
    if date_to:
        query = query.filter(EngineLogEntry.timestamp <= date_to)
    if min_rpm is not None:
        query = query.filter(EngineLogEntry.rpm >= min_rpm)
    if batch_id:
        query = query.filter(EngineLogEntry.upload_batch_id == uuid_mod.UUID(batch_id))

    entries = query.order_by(EngineLogEntry.timestamp).offset(offset).limit(limit).all()

    return [
        EngineLogEntryResponse(
            id=str(e.id), timestamp=e.timestamp, lapse_hours=e.lapse_hours,
            place=e.place, event=e.event, rpm=e.rpm, engine_distance=e.engine_distance,
            speed_stw=e.speed_stw, me_power_kw=e.me_power_kw, me_load_pct=e.me_load_pct,
            me_fuel_index_pct=e.me_fuel_index_pct, shaft_power=e.shaft_power,
            shaft_torque_knm=e.shaft_torque_knm, slip_pct=e.slip_pct,
            hfo_me_mt=e.hfo_me_mt, hfo_ae_mt=e.hfo_ae_mt, hfo_boiler_mt=e.hfo_boiler_mt,
            hfo_total_mt=e.hfo_total_mt, mgo_me_mt=e.mgo_me_mt, mgo_ae_mt=e.mgo_ae_mt,
            mgo_total_mt=e.mgo_total_mt, methanol_me_mt=e.methanol_me_mt,
            rob_vlsfo_mt=e.rob_vlsfo_mt, rob_mgo_mt=e.rob_mgo_mt,
            rob_methanol_mt=e.rob_methanol_mt, rh_me=e.rh_me, rh_ae_total=e.rh_ae_total,
            tc_rpm=e.tc_rpm, scav_air_press_bar=e.scav_air_press_bar,
            fuel_temp_c=e.fuel_temp_c, sw_temp_c=e.sw_temp_c,
            upload_batch_id=str(e.upload_batch_id), source_sheet=e.source_sheet,
            source_file=e.source_file, extended_data=e.extended_data,
        )
        for e in entries
    ]


@router.get("/api/engine-log/summary", response_model=EngineLogSummaryResponse)
async def get_engine_log_summary(
    batch_id: Optional[str] = Query(None, description="Filter by batch UUID"),
    db=Depends(get_db),
):
    """Get aggregated summary statistics from engine log entries."""
    query = db.query(EngineLogEntry)
    if batch_id:
        query = query.filter(EngineLogEntry.upload_batch_id == uuid_mod.UUID(batch_id))

    total = query.count()
    if total == 0:
        return EngineLogSummaryResponse(total_entries=0)

    date_q = db.query(func.min(EngineLogEntry.timestamp), func.max(EngineLogEntry.timestamp))
    if batch_id:
        date_q = date_q.filter(EngineLogEntry.upload_batch_id == uuid_mod.UUID(batch_id))
    min_ts, max_ts = date_q.one()

    event_q = db.query(EngineLogEntry.event, func.count(EngineLogEntry.id)).group_by(EngineLogEntry.event)
    if batch_id:
        event_q = event_q.filter(EngineLogEntry.upload_batch_id == uuid_mod.UUID(batch_id))
    events_breakdown = {ev or "UNKNOWN": cnt for ev, cnt in event_q.all()}

    fuel_q = db.query(func.sum(EngineLogEntry.hfo_total_mt), func.sum(EngineLogEntry.mgo_total_mt), func.sum(EngineLogEntry.methanol_me_mt))
    if batch_id:
        fuel_q = fuel_q.filter(EngineLogEntry.upload_batch_id == uuid_mod.UUID(batch_id))
    hfo_sum, mgo_sum, meth_sum = fuel_q.one()

    rpm_q = db.query(func.avg(EngineLogEntry.rpm)).filter(EngineLogEntry.event == "NOON", EngineLogEntry.rpm > 0)
    if batch_id:
        rpm_q = rpm_q.filter(EngineLogEntry.upload_batch_id == uuid_mod.UUID(batch_id))
    avg_rpm = rpm_q.scalar()

    spd_q = db.query(func.avg(EngineLogEntry.speed_stw)).filter(EngineLogEntry.event == "NOON", EngineLogEntry.speed_stw > 0)
    if batch_id:
        spd_q = spd_q.filter(EngineLogEntry.upload_batch_id == uuid_mod.UUID(batch_id))
    avg_speed = spd_q.scalar()

    batch_q = db.query(
        EngineLogEntry.upload_batch_id, func.count(EngineLogEntry.id),
        func.min(EngineLogEntry.timestamp), func.max(EngineLogEntry.timestamp),
        func.min(EngineLogEntry.source_file),
    ).group_by(EngineLogEntry.upload_batch_id)
    batches = [
        {"batch_id": str(bid), "count": cnt,
         "date_start": ds.isoformat() if ds else None,
         "date_end": de.isoformat() if de else None, "source_file": sf}
        for bid, cnt, ds, de, sf in batch_q.all()
    ]

    return EngineLogSummaryResponse(
        total_entries=total,
        date_range={"start": min_ts.isoformat() if min_ts else None, "end": max_ts.isoformat() if max_ts else None},
        events_breakdown=events_breakdown,
        fuel_summary={"hfo_mt": round(float(hfo_sum or 0), 3), "mgo_mt": round(float(mgo_sum or 0), 3), "methanol_mt": round(float(meth_sum or 0), 3)},
        avg_rpm_at_sea=round(float(avg_rpm), 1) if avg_rpm else None,
        avg_speed_stw=round(float(avg_speed), 2) if avg_speed else None,
        batches=batches,
    )


@router.delete("/api/engine-log/batch/{batch_id}", dependencies=[Depends(require_not_demo("Engine log deletion"))])
@limiter.limit(get_rate_limit_string())
async def delete_engine_log_batch(
    request: Request,
    batch_id: str,
    api_key=Depends(get_api_key),
    db=Depends(get_db),
):
    """Delete all engine log entries for a given upload batch."""
    try:
        bid = uuid_mod.UUID(batch_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid batch_id UUID format")

    count = db.query(EngineLogEntry).filter(EngineLogEntry.upload_batch_id == bid).delete()
    db.commit()

    if count == 0:
        raise HTTPException(status_code=404, detail=f"No entries found for batch {batch_id}")

    return {"status": "deleted", "batch_id": batch_id, "deleted_count": count}


# ============================================================================
# Engine Log → Calibration Bridge
# ============================================================================

@router.post("/api/engine-log/calibrate", response_model=EngineLogCalibrateResponse, dependencies=[Depends(require_not_demo("Engine log calibration"))])
@limiter.limit("5/minute")
async def calibrate_from_engine_log(
    request: Request,
    batch_id: Optional[str] = Query(None, description="Filter to specific upload batch"),
    days_since_drydock: int = Query(0, ge=0, description="Days since last dry dock"),
    api_key=Depends(get_api_key),
    db=Depends(get_db),
):
    """
    Calibrate vessel model from engine log NOON entries.

    Converts NOON entries (speed, fuel, power, RPM) into NoonReport objects,
    feeds them to VesselCalibrator.calibrate(), and applies the resulting
    factors to the vessel model.
    """
    _vs = get_vessel_state()

    # Query NOON entries
    query = db.query(EngineLogEntry).filter(EngineLogEntry.event == "NOON")
    if batch_id:
        try:
            bid = uuid_mod.UUID(batch_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid batch_id UUID format")
        query = query.filter(EngineLogEntry.upload_batch_id == bid)

    noon_rows = query.order_by(EngineLogEntry.timestamp).all()

    # Convert to NoonReport objects, skipping invalid entries
    noon_reports: List[NoonReport] = []
    skipped = 0
    for row in noon_rows:
        speed = row.speed_stw
        hfo = row.hfo_total_mt or 0.0
        mgo = row.mgo_total_mt or 0.0
        fuel = hfo + mgo

        # Skip entries with no speed or no fuel
        if not speed or speed <= 0 or fuel <= 0:
            skipped += 1
            continue

        noon_reports.append(NoonReport(
            timestamp=row.timestamp,
            latitude=0.0,
            longitude=0.0,
            speed_over_ground_kts=speed,
            speed_through_water_kts=speed,
            fuel_consumption_mt=fuel,
            period_hours=row.lapse_hours if row.lapse_hours and row.lapse_hours > 0 else 24.0,
            is_laden=True,
            engine_power_kw=row.me_power_kw,
            engine_rpm=row.rpm,
        ))

    if len(noon_reports) < VesselCalibrator.MIN_REPORTS:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {VesselCalibrator.MIN_REPORTS} valid NOON entries for calibration. "
                   f"Found {len(noon_reports)} valid, {skipped} skipped."
        )

    try:
        # Feed reports to calibrator and run
        _vs.calibrator.noon_reports = noon_reports
        result = _vs.calibrator.calibrate(days_since_drydock=days_since_drydock)

        # Apply calibration atomically (rebuilds model, calculators, optimizers)
        _vs.update_calibration(result.factors)

        _vs.calibrator.save_calibration("default", _vs.calibration)

        return EngineLogCalibrateResponse(
            status="calibrated",
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
            entries_used=result.reports_used,
            entries_skipped=skipped,
            mean_error_before_mt=result.mean_error_before,
            mean_error_after_mt=result.mean_error_after,
            improvement_pct=result.improvement_pct,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Engine log calibration failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Calibration failed: {str(e)}")

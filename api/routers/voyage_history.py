"""
Voyage history CRUD, reporting, and PDF export router.

Separate from voyage.py (transient calculations) to keep concerns clean.
Write endpoints (POST, DELETE) are demo-guarded.
"""

import logging
import uuid as uuid_mod
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import func
import io

from api.auth import get_api_key
from api.database import get_db
from api.demo import require_not_demo
from api.models import Voyage, VoyageLeg
from api.rate_limit import limiter, get_rate_limit_string
from api.reports.noon_reports import generate_noon_reports
from api.reports.templates import build_departure_report, build_arrival_report
from api.reports.pdf_generator import generate_voyage_pdf
from api.schemas import (
    SaveVoyageRequest,
    VoyageDetailResponse,
    VoyageLegResponse,
    VoyageListResponse,
    VoyageSummaryResponse,
    NoonReportEntry,
    NoonReportsResponse,
    DepartureReportData,
    ArrivalReportData,
    VoyageReportsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Voyage History"])


# =============================================================================
# CRUD
# =============================================================================


@router.post(
    "/api/voyages",
    response_model=VoyageSummaryResponse,
    dependencies=[Depends(require_not_demo("Voyage management"))],
)
@limiter.limit(get_rate_limit_string())
async def save_voyage(
    request: Request,
    body: SaveVoyageRequest,
    api_key=Depends(get_api_key),
    db=Depends(get_db),
):
    """Persist a calculated voyage with all legs."""
    voyage = Voyage(
        name=body.name,
        departure_port=body.departure_port,
        arrival_port=body.arrival_port,
        departure_time=body.departure_time,
        arrival_time=body.arrival_time,
        total_distance_nm=body.total_distance_nm,
        total_time_hours=body.total_time_hours,
        total_fuel_mt=body.total_fuel_mt,
        avg_sog_kts=body.avg_sog_kts,
        avg_stw_kts=body.avg_stw_kts,
        calm_speed_kts=body.calm_speed_kts,
        is_laden=body.is_laden,
        vessel_specs_snapshot=body.vessel_specs_snapshot,
        cii_estimate=body.cii_estimate,
        notes=body.notes,
    )

    for leg_data in body.legs:
        leg = VoyageLeg(
            leg_index=leg_data.leg_index,
            from_name=leg_data.from_name,
            from_lat=leg_data.from_lat,
            from_lon=leg_data.from_lon,
            to_name=leg_data.to_name,
            to_lat=leg_data.to_lat,
            to_lon=leg_data.to_lon,
            distance_nm=leg_data.distance_nm,
            bearing_deg=leg_data.bearing_deg,
            wind_speed_kts=leg_data.wind_speed_kts,
            wind_dir_deg=leg_data.wind_dir_deg,
            wave_height_m=leg_data.wave_height_m,
            wave_dir_deg=leg_data.wave_dir_deg,
            current_speed_ms=leg_data.current_speed_ms,
            current_dir_deg=leg_data.current_dir_deg,
            calm_speed_kts=leg_data.calm_speed_kts,
            stw_kts=leg_data.stw_kts,
            sog_kts=leg_data.sog_kts,
            speed_loss_pct=leg_data.speed_loss_pct,
            time_hours=leg_data.time_hours,
            departure_time=leg_data.departure_time,
            arrival_time=leg_data.arrival_time,
            fuel_mt=leg_data.fuel_mt,
            power_kw=leg_data.power_kw,
            data_source=leg_data.data_source,
        )
        voyage.legs.append(leg)

    db.add(voyage)
    db.commit()
    db.refresh(voyage)

    return _voyage_to_summary(voyage)


@router.get("/api/voyages", response_model=VoyageListResponse)
async def list_voyages(
    name: Optional[str] = Query(None, description="Search by name (case-insensitive contains)"),
    date_from: Optional[datetime] = Query(None, description="Filter departure >= date"),
    date_to: Optional[datetime] = Query(None, description="Filter departure <= date"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db=Depends(get_db),
):
    """List saved voyages with optional filters and pagination."""
    query = db.query(Voyage)

    if name:
        query = query.filter(Voyage.name.ilike(f"%{name}%"))
    if date_from:
        query = query.filter(Voyage.departure_time >= date_from)
    if date_to:
        query = query.filter(Voyage.departure_time <= date_to)

    total = query.count()
    voyages = query.order_by(Voyage.departure_time.desc()).offset(offset).limit(limit).all()

    return VoyageListResponse(
        voyages=[_voyage_to_summary(v) for v in voyages],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/api/voyages/{voyage_id}", response_model=VoyageDetailResponse)
async def get_voyage(voyage_id: str, db=Depends(get_db)):
    """Get full voyage detail with all legs."""
    voyage = _get_voyage_or_404(voyage_id, db)
    return _voyage_to_detail(voyage)


@router.delete(
    "/api/voyages/{voyage_id}",
    dependencies=[Depends(require_not_demo("Voyage management"))],
)
@limiter.limit(get_rate_limit_string())
async def delete_voyage(
    request: Request,
    voyage_id: str,
    api_key=Depends(get_api_key),
    db=Depends(get_db),
):
    """Delete a saved voyage and all its legs."""
    voyage = _get_voyage_or_404(voyage_id, db)
    db.delete(voyage)
    db.commit()
    return {"status": "deleted", "voyage_id": voyage_id}


# =============================================================================
# Reports
# =============================================================================


@router.get("/api/voyages/{voyage_id}/noon-reports", response_model=NoonReportsResponse)
async def get_voyage_noon_reports(voyage_id: str, db=Depends(get_db)):
    """Generate noon reports (24h intervals) for a saved voyage."""
    voyage = _get_voyage_or_404(voyage_id, db)
    reports = generate_noon_reports(voyage)

    return NoonReportsResponse(
        voyage_id=str(voyage.id),
        voyage_name=voyage.name,
        departure_time=voyage.departure_time,
        arrival_time=voyage.arrival_time,
        reports=[NoonReportEntry(**r) for r in reports],
    )


@router.get("/api/voyages/{voyage_id}/reports", response_model=VoyageReportsResponse)
async def get_voyage_reports(voyage_id: str, db=Depends(get_db)):
    """Get departure, arrival, and noon reports for a voyage."""
    voyage = _get_voyage_or_404(voyage_id, db)

    dep = build_departure_report(voyage)
    arr = build_arrival_report(voyage)
    noon = generate_noon_reports(voyage)

    return VoyageReportsResponse(
        voyage_id=str(voyage.id),
        departure_report=DepartureReportData(**dep),
        arrival_report=ArrivalReportData(**arr),
        noon_reports=[NoonReportEntry(**r) for r in noon],
    )


@router.get("/api/voyages/{voyage_id}/pdf")
async def download_voyage_pdf(voyage_id: str, db=Depends(get_db)):
    """Download PDF report for a saved voyage."""
    voyage = _get_voyage_or_404(voyage_id, db)

    dep = build_departure_report(voyage)
    arr = build_arrival_report(voyage)
    noon = generate_noon_reports(voyage)

    pdf_bytes = generate_voyage_pdf(voyage, noon, dep, arr)

    filename = f"voyage-report-{voyage.name or voyage_id[:8]}.pdf"
    filename = filename.replace(" ", "_")

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# =============================================================================
# Helpers
# =============================================================================


def _get_voyage_or_404(voyage_id: str, db) -> Voyage:
    try:
        vid = uuid_mod.UUID(voyage_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid voyage_id UUID format")

    voyage = db.query(Voyage).filter(Voyage.id == vid).first()
    if not voyage:
        raise HTTPException(status_code=404, detail=f"Voyage {voyage_id} not found")
    return voyage


def _voyage_to_summary(voyage: Voyage) -> VoyageSummaryResponse:
    return VoyageSummaryResponse(
        id=str(voyage.id),
        name=voyage.name,
        departure_port=voyage.departure_port,
        arrival_port=voyage.arrival_port,
        departure_time=voyage.departure_time,
        arrival_time=voyage.arrival_time,
        total_distance_nm=voyage.total_distance_nm,
        total_time_hours=voyage.total_time_hours,
        total_fuel_mt=voyage.total_fuel_mt,
        avg_sog_kts=voyage.avg_sog_kts,
        calm_speed_kts=voyage.calm_speed_kts,
        is_laden=voyage.is_laden,
        cii_estimate=voyage.cii_estimate,
        created_at=voyage.created_at,
    )


def _voyage_to_detail(voyage: Voyage) -> VoyageDetailResponse:
    legs = sorted(voyage.legs, key=lambda l: l.leg_index)
    return VoyageDetailResponse(
        id=str(voyage.id),
        name=voyage.name,
        departure_port=voyage.departure_port,
        arrival_port=voyage.arrival_port,
        departure_time=voyage.departure_time,
        arrival_time=voyage.arrival_time,
        total_distance_nm=voyage.total_distance_nm,
        total_time_hours=voyage.total_time_hours,
        total_fuel_mt=voyage.total_fuel_mt,
        avg_sog_kts=voyage.avg_sog_kts,
        avg_stw_kts=voyage.avg_stw_kts,
        calm_speed_kts=voyage.calm_speed_kts,
        is_laden=voyage.is_laden,
        vessel_specs_snapshot=voyage.vessel_specs_snapshot,
        cii_estimate=voyage.cii_estimate,
        notes=voyage.notes,
        created_at=voyage.created_at,
        updated_at=voyage.updated_at,
        legs=[
            VoyageLegResponse(
                id=str(l.id),
                leg_index=l.leg_index,
                from_name=l.from_name,
                from_lat=l.from_lat,
                from_lon=l.from_lon,
                to_name=l.to_name,
                to_lat=l.to_lat,
                to_lon=l.to_lon,
                distance_nm=l.distance_nm,
                bearing_deg=l.bearing_deg,
                wind_speed_kts=l.wind_speed_kts,
                wind_dir_deg=l.wind_dir_deg,
                wave_height_m=l.wave_height_m,
                wave_dir_deg=l.wave_dir_deg,
                current_speed_ms=l.current_speed_ms,
                current_dir_deg=l.current_dir_deg,
                calm_speed_kts=l.calm_speed_kts,
                stw_kts=l.stw_kts,
                sog_kts=l.sog_kts,
                speed_loss_pct=l.speed_loss_pct,
                time_hours=l.time_hours,
                departure_time=l.departure_time,
                arrival_time=l.arrival_time,
                fuel_mt=l.fuel_mt,
                power_kw=l.power_kw,
                data_source=l.data_source,
            )
            for l in legs
        ],
    )

"""
Charter Party Weather Clause Tools API router.

Provides good weather day counting, warranty verification,
and off-hire detection for charter party analysis.
"""

import logging
import uuid as uuid_mod
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import asc

from api.auth import get_api_key
from api.database import get_db
from api.models import EngineLogEntry, Voyage, VoyageLeg
from api.rate_limit import limiter, get_rate_limit_string
from api.schemas.charter_party import (
    BeaufortEntry,
    BeaufortScaleResponse,
    GoodWeatherFromLegsRequest,
    GoodWeatherLegResponse,
    GoodWeatherRequest,
    GoodWeatherResponse,
    OffHireEventResponse,
    OffHireRequest,
    OffHireResponse,
    WarrantyFromLegsRequest,
    WarrantyLegDetailResponse,
    WarrantyVerificationRequest,
    WarrantyVerificationResponse,
)
from src.compliance.charter_party import BEAUFORT_SCALE, CharterPartyCalculator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/charter-party", tags=["Charter Party Tools"])

_calc = CharterPartyCalculator()


# ---- helpers ----------------------------------------------------------------


def _get_voyage_or_404(voyage_id: str, db) -> Voyage:
    try:
        vid = uuid_mod.UUID(voyage_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid voyage_id UUID format")
    voyage = db.query(Voyage).filter(Voyage.id == vid).first()
    if not voyage:
        raise HTTPException(status_code=404, detail=f"Voyage {voyage_id} not found")
    return voyage


def _voyage_legs_to_dicts(legs) -> list:
    """Convert VoyageLeg ORM objects to calculator input dicts."""
    return [
        {
            "wind_speed_kts": leg.wind_speed_kts or 0.0,
            "wave_height_m": leg.wave_height_m or 0.0,
            "current_speed_ms": leg.current_speed_ms or 0.0,
            "time_hours": leg.time_hours or 0.0,
            "distance_nm": leg.distance_nm or 0.0,
            "sog_kts": leg.sog_kts or 0.0,
            "fuel_mt": leg.fuel_mt or 0.0,
        }
        for leg in sorted(legs, key=lambda l: l.leg_index)
    ]


def _legs_request_to_dicts(legs) -> list:
    """Convert Pydantic LegWeatherInput objects to calculator input dicts."""
    return [
        {
            "wind_speed_kts": leg.wind_speed_kts,
            "wave_height_m": leg.wave_height_m,
            "current_speed_ms": leg.current_speed_ms,
            "time_hours": leg.time_hours,
            "distance_nm": leg.distance_nm,
            "sog_kts": leg.sog_kts,
            "fuel_mt": leg.fuel_mt,
        }
        for leg in legs
    ]


def _good_weather_to_response(result) -> GoodWeatherResponse:
    return GoodWeatherResponse(
        total_days=result.total_days,
        good_weather_days=result.good_weather_days,
        bad_weather_days=result.bad_weather_days,
        good_weather_pct=result.good_weather_pct,
        bf_threshold=result.bf_threshold,
        wave_threshold_m=result.wave_threshold_m,
        current_threshold_kts=result.current_threshold_kts,
        legs=[GoodWeatherLegResponse(**asdict(lg)) for lg in result.legs],
    )


def _warranty_to_response(result) -> WarrantyVerificationResponse:
    return WarrantyVerificationResponse(
        warranted_speed_kts=result.warranted_speed_kts,
        achieved_speed_kts=result.achieved_speed_kts,
        speed_margin_kts=result.speed_margin_kts,
        speed_compliant=result.speed_compliant,
        warranted_consumption_mt_day=result.warranted_consumption_mt_day,
        achieved_consumption_mt_day=result.achieved_consumption_mt_day,
        consumption_margin_mt=result.consumption_margin_mt,
        consumption_compliant=result.consumption_compliant,
        good_weather_hours=result.good_weather_hours,
        total_hours=result.total_hours,
        legs_assessed=result.legs_assessed,
        legs_good_weather=result.legs_good_weather,
        legs=[WarrantyLegDetailResponse(**asdict(lg)) for lg in result.legs],
    )


# ---- reference data ---------------------------------------------------------


@router.get("/beaufort-scale", response_model=BeaufortScaleResponse)
async def get_beaufort_scale():
    """Return the Beaufort wind force scale reference table."""
    return BeaufortScaleResponse(
        scale=[BeaufortEntry(**entry) for entry in BEAUFORT_SCALE],
    )


# ---- good weather days ------------------------------------------------------


@router.post("/good-weather", response_model=GoodWeatherResponse)
@limiter.limit(get_rate_limit_string())
async def analyze_good_weather(
    request: Request,
    body: GoodWeatherRequest,
    api_key=Depends(get_api_key),
    db=Depends(get_db),
):
    """Count good weather days for a saved voyage."""
    voyage = _get_voyage_or_404(body.voyage_id, db)
    legs = _voyage_legs_to_dicts(voyage.legs)
    result = _calc.count_good_weather_days(
        legs,
        bf_threshold=body.bf_threshold,
        wave_threshold_m=body.wave_threshold_m,
        current_threshold_kts=body.current_threshold_kts,
    )
    return _good_weather_to_response(result)


@router.post("/good-weather/from-legs", response_model=GoodWeatherResponse)
async def analyze_good_weather_from_legs(body: GoodWeatherFromLegsRequest):
    """Count good weather days from manual leg data (no DB required)."""
    legs = _legs_request_to_dicts(body.legs)
    result = _calc.count_good_weather_days(
        legs,
        bf_threshold=body.bf_threshold,
        wave_threshold_m=body.wave_threshold_m,
        current_threshold_kts=body.current_threshold_kts,
    )
    return _good_weather_to_response(result)


# ---- warranty verification --------------------------------------------------


@router.post("/verify-warranty", response_model=WarrantyVerificationResponse)
@limiter.limit(get_rate_limit_string())
async def verify_warranty(
    request: Request,
    body: WarrantyVerificationRequest,
    api_key=Depends(get_api_key),
    db=Depends(get_db),
):
    """Verify warranted speed and consumption for a saved voyage."""
    voyage = _get_voyage_or_404(body.voyage_id, db)
    legs = _voyage_legs_to_dicts(voyage.legs)
    result = _calc.verify_warranty(
        legs,
        warranted_speed_kts=body.warranted_speed_kts,
        warranted_consumption_mt_day=body.warranted_consumption_mt_day,
        bf_threshold=body.bf_threshold,
        speed_tolerance_pct=body.speed_tolerance_pct,
        consumption_tolerance_pct=body.consumption_tolerance_pct,
    )
    return _warranty_to_response(result)


@router.post("/verify-warranty/from-legs", response_model=WarrantyVerificationResponse)
async def verify_warranty_from_legs(body: WarrantyFromLegsRequest):
    """Verify warranted speed and consumption from manual leg data (no DB required)."""
    legs = _legs_request_to_dicts(body.legs)
    result = _calc.verify_warranty(
        legs,
        warranted_speed_kts=body.warranted_speed_kts,
        warranted_consumption_mt_day=body.warranted_consumption_mt_day,
        bf_threshold=body.bf_threshold,
        speed_tolerance_pct=body.speed_tolerance_pct,
        consumption_tolerance_pct=body.consumption_tolerance_pct,
    )
    return _warranty_to_response(result)


# ---- off-hire detection -----------------------------------------------------


@router.post("/off-hire", response_model=OffHireResponse)
@limiter.limit(get_rate_limit_string())
async def detect_off_hire(
    request: Request,
    body: OffHireRequest,
    api_key=Depends(get_api_key),
    db=Depends(get_db),
):
    """Detect off-hire events from engine log entries."""
    query = db.query(EngineLogEntry)

    if body.date_from:
        query = query.filter(EngineLogEntry.timestamp >= body.date_from)
    if body.date_to:
        query = query.filter(EngineLogEntry.timestamp <= body.date_to)

    entries_orm = query.order_by(asc(EngineLogEntry.timestamp)).all()

    if not entries_orm:
        return OffHireResponse(
            total_hours=0.0,
            on_hire_hours=0.0,
            off_hire_hours=0.0,
            off_hire_pct=0.0,
            events=[],
        )

    entries = [
        {
            "timestamp": e.timestamp,
            "rpm": e.rpm or 0.0,
            "speed_stw": e.speed_stw or 0.0,
            "event": e.event or "",
            "place": e.place or "",
        }
        for e in entries_orm
    ]

    result = _calc.detect_off_hire(
        entries,
        rpm_threshold=body.rpm_threshold,
        speed_threshold=body.speed_threshold,
        gap_hours=body.gap_hours,
    )

    return OffHireResponse(
        total_hours=result.total_hours,
        on_hire_hours=result.on_hire_hours,
        off_hire_hours=result.off_hire_hours,
        off_hire_pct=result.off_hire_pct,
        events=[OffHireEventResponse(**asdict(ev)) for ev in result.events],
    )

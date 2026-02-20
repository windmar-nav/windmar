"""Vessel configuration and calibration API schemas."""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from src.optimization.vessel_model import VesselSpecs


class VesselConfig(BaseModel):
    """Vessel configuration.

    Defaults sourced from VesselSpecs (MR tanker defaults, fully configurable).
    At runtime, values are overridden by DB-persisted specs on startup.
    """
    dwt: float = VesselSpecs.dwt
    loa: float = VesselSpecs.loa
    beam: float = VesselSpecs.beam
    draft_laden: float = VesselSpecs.draft_laden
    draft_ballast: float = VesselSpecs.draft_ballast
    mcr_kw: float = VesselSpecs.mcr_kw
    sfoc_at_mcr: float = VesselSpecs.sfoc_at_mcr
    service_speed_laden: float = VesselSpecs.service_speed_laden
    service_speed_ballast: float = VesselSpecs.service_speed_ballast


class NoonReportModel(BaseModel):
    """Noon report data for calibration."""
    timestamp: datetime
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    speed_over_ground_kts: float = Field(..., gt=0)
    speed_through_water_kts: Optional[float] = None
    fuel_consumption_mt: float = Field(..., gt=0)
    period_hours: float = Field(24.0, gt=0)
    is_laden: bool = True
    heading_deg: float = Field(0.0, ge=0, le=360)
    wind_speed_kts: Optional[float] = None
    wind_direction_deg: Optional[float] = None
    wave_height_m: Optional[float] = None
    wave_direction_deg: Optional[float] = None
    engine_power_kw: Optional[float] = None


class CalibrationFactorsModel(BaseModel):
    """Calibration factors for vessel model."""
    calm_water: float = Field(1.0, description="Hull fouling factor")
    wind: float = Field(1.0, description="Wind coefficient adjustment")
    waves: float = Field(1.0, description="Wave response adjustment")
    sfoc_factor: float = Field(1.0, description="SFOC multiplier")
    calibrated_at: Optional[datetime] = None
    num_reports_used: int = 0
    calibration_error: float = 0.0
    days_since_drydock: int = 0


class CalibrationResponse(BaseModel):
    """Calibration result response."""
    factors: CalibrationFactorsModel
    reports_used: int
    reports_skipped: int
    mean_error_before_mt: float
    mean_error_after_mt: float
    improvement_pct: float
    residuals: List[Dict]


class PerformancePredictionRequest(BaseModel):
    """Request for vessel performance prediction under given conditions.

    Two modes:
    - engine_load_pct set: find achievable speed at this power
    - calm_speed_kts set: find required power to maintain this calm-water
      speed through the given weather (speed may drop if MCR exceeded)

    All directions are RELATIVE to the vessel bow:
    0 = dead ahead (head wind / head seas / head current)
    90 = beam (port or starboard)
    180 = dead astern (following wind / following seas / following current)
    """
    is_laden: bool = True
    engine_load_pct: Optional[float] = Field(None, ge=15, le=100, description="Engine load as % of MCR (mode 1)")
    calm_speed_kts: Optional[float] = Field(None, gt=0, lt=25, description="Target calm-water speed in knots (mode 2)")
    wind_speed_kts: float = Field(0.0, ge=0, le=100, description="True wind speed (knots)")
    wind_relative_deg: float = Field(0.0, ge=0, le=180, description="Wind relative to bow: 0=ahead, 90=beam, 180=astern")
    wave_height_m: float = Field(0.0, ge=0, le=15, description="Significant wave height (m)")
    wave_relative_deg: float = Field(0.0, ge=0, le=180, description="Waves relative to bow: 0=head seas, 90=beam, 180=following")
    current_speed_kts: float = Field(0.0, ge=0, le=10, description="Current speed (knots)")
    current_relative_deg: float = Field(0.0, ge=0, le=180, description="Current relative to bow: 0=head current, 180=following")

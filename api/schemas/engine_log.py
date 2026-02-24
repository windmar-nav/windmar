"""Engine log API schemas."""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from .vessel import CalibrationFactorsModel


class EngineLogUploadResponse(BaseModel):
    """Response from engine log upload."""
    status: str
    batch_id: str
    imported: int
    skipped: int
    date_range: Optional[Dict] = None
    events_summary: Optional[Dict[str, int]] = None


class EngineLogEntryResponse(BaseModel):
    """Serialized engine log entry."""
    id: str
    timestamp: datetime
    lapse_hours: Optional[float] = None
    place: Optional[str] = None
    event: Optional[str] = None
    rpm: Optional[float] = None
    engine_distance: Optional[float] = None
    speed_stw: Optional[float] = None
    me_power_kw: Optional[float] = None
    me_load_pct: Optional[float] = None
    me_fuel_index_pct: Optional[float] = None
    shaft_power: Optional[float] = None
    shaft_torque_knm: Optional[float] = None
    slip_pct: Optional[float] = None
    hfo_me_mt: Optional[float] = None
    hfo_ae_mt: Optional[float] = None
    hfo_boiler_mt: Optional[float] = None
    hfo_total_mt: Optional[float] = None
    mgo_me_mt: Optional[float] = None
    mgo_ae_mt: Optional[float] = None
    mgo_total_mt: Optional[float] = None
    methanol_me_mt: Optional[float] = None
    rob_vlsfo_mt: Optional[float] = None
    rob_mgo_mt: Optional[float] = None
    rob_methanol_mt: Optional[float] = None
    rh_me: Optional[float] = None
    rh_ae_total: Optional[float] = None
    tc_rpm: Optional[float] = None
    scav_air_press_bar: Optional[float] = None
    fuel_temp_c: Optional[float] = None
    sw_temp_c: Optional[float] = None
    upload_batch_id: str
    source_sheet: Optional[str] = None
    source_file: Optional[str] = None
    extended_data: Optional[Dict] = None

    model_config = ConfigDict(from_attributes=True)


class EngineLogSummaryResponse(BaseModel):
    """Aggregated engine log summary."""
    total_entries: int
    date_range: Optional[Dict] = None
    events_breakdown: Optional[Dict[str, int]] = None
    fuel_summary: Optional[Dict] = None
    avg_rpm_at_sea: Optional[float] = None
    avg_speed_stw: Optional[float] = None
    batches: Optional[List[Dict]] = None


class EngineLogCalibrateResponse(BaseModel):
    """Response from engine-log-based calibration."""
    status: str
    factors: CalibrationFactorsModel
    entries_used: int
    entries_skipped: int
    mean_error_before_mt: float
    mean_error_after_mt: float
    improvement_pct: float

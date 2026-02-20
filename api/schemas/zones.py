"""Regulatory zones API schemas."""

from typing import List, Optional

from pydantic import BaseModel, Field


class ZoneCoordinate(BaseModel):
    """A coordinate in a zone polygon."""
    lat: float
    lon: float


class CreateZoneRequest(BaseModel):
    """Request to create a custom zone."""
    name: str
    zone_type: str = Field(..., description="eca, hra, tss, exclusion, custom, etc.")
    interaction: str = Field(..., description="mandatory, exclusion, penalty, advisory")
    coordinates: List[ZoneCoordinate]
    penalty_factor: float = Field(1.0, ge=1.0, le=10.0)
    notes: Optional[str] = None


class ZoneResponse(BaseModel):
    """Zone information response."""
    id: str
    name: str
    zone_type: str
    interaction: str
    penalty_factor: float
    is_builtin: bool
    coordinates: List[ZoneCoordinate]
    notes: Optional[str] = None

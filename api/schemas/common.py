"""Common shared schemas used across multiple domains."""

from typing import List

from pydantic import BaseModel, Field


class Position(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)


class WaypointModel(BaseModel):
    id: int
    name: str
    lat: float
    lon: float


class RouteModel(BaseModel):
    name: str
    waypoints: List[WaypointModel]

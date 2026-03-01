"""
Maritime routes API router.

Handles RTZ file parsing and route creation from waypoints.
"""

import logging
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from api.demo import require_not_demo
from api.rate_limit import limiter, get_rate_limit_string
from api.schemas.common import Position
from src.routes.rtz_parser import parse_rtz_string, create_route_from_waypoints

logger = logging.getLogger(__name__)

# 5 MB limit for RTZ files (matches main.py constant)
MAX_RTZ_SIZE_BYTES = 5 * 1024 * 1024

router = APIRouter(prefix="/api/routes", tags=["routes"])


@router.post("/parse-rtz", dependencies=[Depends(require_not_demo("Route file upload"))])
@limiter.limit(get_rate_limit_string())
async def parse_rtz(
    request: Request,
    file: UploadFile = File(...),
):
    """
    Parse an uploaded RTZ route file.

    Maximum file size: 5 MB.
    Returns waypoints in standard format.
    """
    try:
        # Validate file extension
        if file.filename and not file.filename.lower().endswith(".rtz"):
            raise HTTPException(status_code=400, detail="Only .rtz files accepted")

        content = await file.read()

        # Validate file size
        if len(content) > MAX_RTZ_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {MAX_RTZ_SIZE_BYTES // (1024*1024)} MB"
            )

        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        rtz_string = content.decode('utf-8')

        route = parse_rtz_string(rtz_string)

        return {
            "name": route.name,
            "waypoints": [
                {
                    "id": wp.id,
                    "name": wp.name,
                    "lat": wp.lat,
                    "lon": wp.lon,
                }
                for wp in route.waypoints
            ],
            "total_distance_nm": route.total_distance_nm,
            "legs": [
                {
                    "from": leg.from_wp.name,
                    "to": leg.to_wp.name,
                    "distance_nm": leg.distance_nm,
                    "bearing_deg": leg.bearing_deg,
                }
                for leg in route.legs
            ]
        }
    except Exception as e:
        logger.error(f"Failed to parse RTZ: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid RTZ file: {str(e)}")


@router.post("/from-waypoints")
async def create_route_from_wps(
    waypoints: List[Position],
    name: str = "Custom Route",
):
    """
    Create a route from a list of waypoints.

    Returns route with calculated distances and bearings.
    """
    if len(waypoints) < 2:
        raise HTTPException(status_code=400, detail="At least 2 waypoints required")

    wps = [(wp.lat, wp.lon) for wp in waypoints]
    route = create_route_from_waypoints(wps, name)

    return {
        "name": route.name,
        "waypoints": [
            {
                "id": wp.id,
                "name": wp.name,
                "lat": wp.lat,
                "lon": wp.lon,
            }
            for wp in route.waypoints
        ],
        "total_distance_nm": route.total_distance_nm,
        "legs": [
            {
                "from": leg.from_wp.name,
                "to": leg.to_wp.name,
                "distance_nm": leg.distance_nm,
                "bearing_deg": leg.bearing_deg,
            }
            for leg in route.legs
        ]
    }

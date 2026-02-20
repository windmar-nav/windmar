"""
Regulatory zones API router.

Handles CRUD and spatial queries for regulatory/custom zones
(ECA, HRA, TSS, exclusion, custom).
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from api.auth import get_api_key
from api.demo import require_not_demo
from api.rate_limit import limiter, get_rate_limit_string
from api.schemas.zones import CreateZoneRequest, ZoneCoordinate, ZoneResponse
from src.data.regulatory_zones import (
    get_zone_checker, Zone, ZoneProperties, ZoneType, ZoneInteraction,
)

router = APIRouter(prefix="/api/zones", tags=["zones"])


@router.get("")
async def get_all_zones():
    """
    Get all regulatory zones (built-in and custom).

    Returns GeoJSON FeatureCollection for map display.
    """
    zone_checker = get_zone_checker()
    return zone_checker.export_geojson()


@router.get("/list")
async def list_zones():
    """Get zones as a simple list."""
    zone_checker = get_zone_checker()
    zones = []
    for zone in zone_checker.get_all_zones():
        zones.append({
            "id": zone.id,
            "name": zone.properties.name,
            "zone_type": zone.properties.zone_type.value,
            "interaction": zone.properties.interaction.value,
            "penalty_factor": zone.properties.penalty_factor,
            "is_builtin": zone.is_builtin,
        })
    return {"zones": zones, "count": len(zones)}


@router.get("/at-point")
async def get_zones_at_point(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
):
    """Get all zones that contain a specific point."""
    zone_checker = get_zone_checker()
    zones = zone_checker.get_zones_at_point(lat, lon)

    return {
        "position": {"lat": lat, "lon": lon},
        "zones": [
            {
                "id": z.id,
                "name": z.properties.name,
                "zone_type": z.properties.zone_type.value,
                "interaction": z.properties.interaction.value,
                "penalty_factor": z.properties.penalty_factor,
            }
            for z in zones
        ]
    }


@router.get("/check-path")
async def check_path_zones(
    lat1: float = Query(..., ge=-90, le=90),
    lon1: float = Query(..., ge=-180, le=180),
    lat2: float = Query(..., ge=-90, le=90),
    lon2: float = Query(..., ge=-180, le=180),
):
    """Check which zones a path segment crosses."""
    zone_checker = get_zone_checker()
    zones_by_type = zone_checker.check_path_zones(lat1, lon1, lat2, lon2)
    penalty, warnings = zone_checker.get_path_penalty(lat1, lon1, lat2, lon2)

    return {
        "path": {
            "from": {"lat": lat1, "lon": lon1},
            "to": {"lat": lat2, "lon": lon2},
        },
        "zones": {
            interaction: [
                {"id": z.id, "name": z.properties.name}
                for z in zones
            ]
            for interaction, zones in zones_by_type.items()
        },
        "penalty_factor": penalty if penalty != float('inf') else None,
        "is_forbidden": penalty == float('inf'),
        "warnings": warnings,
    }


@router.get("/{zone_id}")
async def get_zone(zone_id: str):
    """Get a specific zone by ID."""
    zone_checker = get_zone_checker()
    zone = zone_checker.get_zone(zone_id)

    if zone is None:
        raise HTTPException(status_code=404, detail=f"Zone not found: {zone_id}")

    return zone.to_geojson()


@router.post("", response_model=ZoneResponse, dependencies=[Depends(require_not_demo("Zone creation"))])
@limiter.limit(get_rate_limit_string())
async def create_zone(
    http_request: Request,
    request: CreateZoneRequest,
    api_key=Depends(get_api_key),
):
    """
    Create a custom zone.

    Requires authentication via API key.

    Coordinates should be provided as a list of {lat, lon} objects
    forming a closed polygon (first and last point should match).
    """
    zone_checker = get_zone_checker()

    # Validate zone type
    try:
        zone_type = ZoneType(request.zone_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid zone_type. Valid values: {[t.value for t in ZoneType]}"
        )

    # Validate interaction
    try:
        interaction = ZoneInteraction(request.interaction)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid interaction. Valid values: {[i.value for i in ZoneInteraction]}"
        )

    # Convert coordinates
    coords = [(c.lat, c.lon) for c in request.coordinates]

    # Ensure polygon is closed
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    # Create zone
    zone_id = f"custom_{uuid.uuid4().hex[:8]}"
    zone = Zone(
        id=zone_id,
        properties=ZoneProperties(
            name=request.name,
            zone_type=zone_type,
            interaction=interaction,
            penalty_factor=request.penalty_factor,
            notes=request.notes,
        ),
        coordinates=coords,
        is_builtin=False,
    )

    zone_checker.add_zone(zone)

    return ZoneResponse(
        id=zone.id,
        name=zone.properties.name,
        zone_type=zone.properties.zone_type.value,
        interaction=zone.properties.interaction.value,
        penalty_factor=zone.properties.penalty_factor,
        is_builtin=zone.is_builtin,
        coordinates=[ZoneCoordinate(lat=c[0], lon=c[1]) for c in zone.coordinates],
        notes=zone.properties.notes,
    )


@router.delete("/{zone_id}", dependencies=[Depends(require_not_demo("Zone deletion"))])
@limiter.limit(get_rate_limit_string())
async def delete_zone(
    request: Request,
    zone_id: str,
    api_key=Depends(get_api_key),
):
    """
    Delete a custom zone.

    Requires authentication via API key.
    Built-in zones cannot be deleted.
    """
    zone_checker = get_zone_checker()

    zone = zone_checker.get_zone(zone_id)
    if zone is None:
        raise HTTPException(status_code=404, detail=f"Zone not found: {zone_id}")

    if zone.is_builtin:
        raise HTTPException(status_code=400, detail="Cannot delete built-in zones")

    zone_checker.remove_zone(zone_id)
    return {"status": "deleted", "zone_id": zone_id}

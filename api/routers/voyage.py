"""
Voyage calculation API router.

Handles voyage ETA/fuel computation, Monte Carlo simulation,
and weather-along-route queries.
"""

import asyncio
import logging
import time as _time
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api.schemas import (
    DataSourceSummary,
    LegResultModel,
    MonteCarloRequest,
    MonteCarloResponse,
    PercentileFloat,
    PercentileString,
    VoyageRequest,
    VoyageResponse,
    WaypointModel,
)
from api.state import get_app_state, get_vessel_state
from api.weather_service import (
    get_current_field,
    get_ice_field,
    get_sst_field,
    get_visibility_field,
    get_weather_at_point,
    get_wind_field,
    get_wave_field,
    get_voyage_data_sources,
    reset_voyage_data_sources,
    supplement_temporal_wind,
    weather_provider,
)
from src.data.copernicus import ClimatologyProvider
from src.optimization.grid_weather_provider import GridWeatherProvider
from src.optimization.weather_assessment import RouteWeatherAssessment
from src.routes.rtz_parser import create_route_from_waypoints, haversine_distance

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Voyage"])


@router.post("/api/voyage/calculate", response_model=VoyageResponse)
async def calculate_voyage(request: VoyageRequest):
    """
    Calculate voyage with per-leg SOG, ETA, and fuel.

    Takes waypoints, calm speed, and vessel condition.
    Returns detailed per-leg results including weather impact.

    Weather data is sourced from:
    - Forecast: Copernicus data for first 10 days
    - Blended: Transition from forecast to climatology (days 10-12)
    - Climatology: ERA5 monthly averages beyond forecast horizon
    """
    if len(request.waypoints) < 2:
        raise HTTPException(status_code=400, detail="At least 2 waypoints required")

    _vs = get_vessel_state()
    db_weather = get_app_state().weather_providers.get('db_weather')

    departure = request.departure_time or datetime.now(timezone.utc)
    t_start = _time.monotonic()
    logger.info(f"Voyage calculation started: {len(request.waypoints)} waypoints, speed={request.calm_speed_kts}kts, weather={request.use_weather}")

    # Create route from waypoints
    wps = [(wp.lat, wp.lon) for wp in request.waypoints]
    route = create_route_from_waypoints(wps, "Voyage Route")

    # Reset data source tracking
    reset_voyage_data_sources()

    # Pre-fetch weather grids for entire route bounding box
    # Try temporal (time-varying) weather first, fall back to single-snapshot
    wp_func = None
    data_source_type = None
    used_temporal = False
    if request.use_weather:
        margin = 3.0
        lats = [wp.lat for wp in request.waypoints]
        lons = [wp.lon for wp in request.waypoints]
        lat_min = max(min(lats) - margin, -85)
        lat_max = min(max(lats) + margin, 85)
        lon_min = min(lons) - margin
        lon_max = max(lons) + margin

        origin_pt = (request.waypoints[0].lat, request.waypoints[0].lon)
        dest_pt = (request.waypoints[-1].lat, request.waypoints[-1].lon)

        # ── Temporal weather provisioning (DB-first) ──────────────────
        temporal_wx = None
        if db_weather is not None:
            try:
                assessor = RouteWeatherAssessment(db_weather=db_weather)
                wx_needs = assessor.assess(
                    origin=origin_pt,
                    destination=dest_pt,
                    departure_time=departure,
                    calm_speed_kts=request.calm_speed_kts,
                )
                avail_parts = [f"{s}: {v.get('coverage_pct',0):.0f}%" for s, v in wx_needs.availability.items()]
                logger.info(
                    f"Voyage weather assessment: {wx_needs.estimated_passage_hours:.0f}h passage, "
                    f"need hours {wx_needs.required_forecast_hours[:5]}..., "
                    f"availability: {', '.join(avail_parts)}"
                )
                temporal_wx = assessor.provision(wx_needs)
                if temporal_wx is not None:
                    used_temporal = True
                    params_loaded = list(temporal_wx.grids.keys())
                    has_temporal_wind = any(p in temporal_wx.grids for p in ["wind_u", "wind_v"])
                    if not has_temporal_wind:
                        if supplement_temporal_wind(temporal_wx, lat_min, lat_max, lon_min, lon_max, departure):
                            has_temporal_wind = True
                            params_loaded = list(temporal_wx.grids.keys())
                    logger.info(
                        f"Voyage using temporal weather: {len(params_loaded)} params ({params_loaded}), "
                        f"wind={'yes' if has_temporal_wind else 'NO (calm assumed)'}"
                    )
                else:
                    logger.warning("Temporal provisioning returned None — falling back to single-snapshot")
            except Exception as e:
                logger.warning(f"Temporal weather provisioning failed for voyage, falling back: {e}", exc_info=True)

        # ── Fallback: single-snapshot GridWeatherProvider ─────────────
        grid_wx = None
        if temporal_wx is None:
            t0 = _time.monotonic()
            logger.info(f"  Pre-fetching single-snapshot weather for bbox [{lat_min:.1f},{lat_max:.1f},{lon_min:.1f},{lon_max:.1f}]")
            wind = get_wind_field(lat_min, lat_max, lon_min, lon_max, 0.5, departure)
            logger.info(f"  Wind loaded in {_time.monotonic()-t0:.1f}s")
            t1 = _time.monotonic()
            waves = get_wave_field(lat_min, lat_max, lon_min, lon_max, 0.5, wind)
            logger.info(f"  Waves loaded in {_time.monotonic()-t1:.1f}s")
            t2 = _time.monotonic()
            currents = get_current_field(lat_min, lat_max, lon_min, lon_max, 0.5)
            logger.info(f"  Currents loaded in {_time.monotonic()-t2:.1f}s")
            # Extended fields (SPEC-P1)
            sst = get_sst_field(lat_min, lat_max, lon_min, lon_max, 0.5, departure)
            vis = get_visibility_field(lat_min, lat_max, lon_min, lon_max, 0.5, departure)
            ice = get_ice_field(lat_min, lat_max, lon_min, lon_max, 0.5, departure)
            logger.info(f"  Total prefetch: {_time.monotonic()-t0:.1f}s (incl. SST/vis/ice)")
            grid_wx = GridWeatherProvider(wind, waves, currents, sst, vis, ice)

        data_source_type = "temporal" if used_temporal else "forecast"
        wx_callable = temporal_wx.get_weather if temporal_wx else grid_wx.get_weather

        # Wrapper that tracks data sources per leg
        def tracked_weather_provider(lat: float, lon: float, time: datetime):
            leg_wx = wx_callable(lat, lon, time)
            get_voyage_data_sources().append({
                'lat': lat, 'lon': lon, 'time': time.isoformat(),
                'source': data_source_type, 'forecast_weight': 1.0,
                'message': f'{"Temporal" if used_temporal else "Single-snapshot"} grid',
            })
            return leg_wx

        wp_func = tracked_weather_provider

    result = _vs.voyage_calculator.calculate_voyage(
        route=route,
        calm_speed_kts=request.calm_speed_kts,
        is_laden=request.is_laden,
        departure_time=departure,
        weather_provider=wp_func,
    )
    logger.info(f"Voyage calculation completed in {_time.monotonic()-t_start:.1f}s: {len(result.legs)} legs, {result.total_distance_nm:.0f}nm, {result.total_fuel_mt:.1f}mt fuel")

    # Build data source summary
    forecast_legs = sum(1 for ds in get_voyage_data_sources() if ds['source'] == 'forecast')
    blended_legs = sum(1 for ds in get_voyage_data_sources() if ds['source'] == 'blended')
    climatology_legs = sum(1 for ds in get_voyage_data_sources() if ds['source'] == 'climatology')

    data_source_warning = None
    if climatology_legs > 0:
        data_source_warning = (
            f"Voyage extends beyond {ClimatologyProvider.FORECAST_HORIZON_DAYS}-day forecast horizon. "
            f"{climatology_legs} leg(s) use climatological averages with higher uncertainty."
        )
    elif blended_legs > 0:
        data_source_warning = (
            f"Voyage approaches forecast horizon. "
            f"{blended_legs} leg(s) use blended forecast/climatology data."
        )

    data_sources_summary = DataSourceSummary(
        forecast_legs=forecast_legs,
        blended_legs=blended_legs,
        climatology_legs=climatology_legs,
        forecast_horizon_days=ClimatologyProvider.FORECAST_HORIZON_DAYS,
        warning=data_source_warning,
    ) if request.use_weather else None

    # Format response with data source info per leg
    legs_response = []
    for i, leg in enumerate(result.legs):
        # Get data source info for this leg
        leg_source = get_voyage_data_sources()[i] if i < len(get_voyage_data_sources()) else None

        legs_response.append(LegResultModel(
            leg_index=leg.leg_index,
            from_wp=WaypointModel(
                id=leg.from_wp.id,
                name=leg.from_wp.name,
                lat=leg.from_wp.lat,
                lon=leg.from_wp.lon,
            ),
            to_wp=WaypointModel(
                id=leg.to_wp.id,
                name=leg.to_wp.name,
                lat=leg.to_wp.lat,
                lon=leg.to_wp.lon,
            ),
            distance_nm=round(leg.distance_nm, 2),
            bearing_deg=round(leg.bearing_deg, 1),
            wind_speed_kts=round(leg.weather.wind_speed_ms * 1.94384, 1),
            wind_dir_deg=round(leg.weather.wind_dir_deg, 0),
            wave_height_m=round(leg.weather.sig_wave_height_m, 1),
            wave_dir_deg=round(leg.weather.wave_dir_deg, 0),
            current_speed_ms=round(leg.weather.current_speed_ms, 2),
            current_dir_deg=round(leg.weather.current_dir_deg, 0),
            calm_speed_kts=round(leg.calm_speed_kts, 1),
            stw_kts=round(leg.stw_kts, 1),
            sog_kts=round(leg.sog_kts, 1),
            speed_loss_pct=round(leg.speed_loss_pct, 1),
            time_hours=round(leg.time_hours, 2),
            departure_time=leg.departure_time,
            arrival_time=leg.arrival_time,
            fuel_mt=round(leg.fuel_mt, 2),
            power_kw=round(leg.power_kw, 0),
            data_source=leg_source['source'] if leg_source else None,
            forecast_weight=leg_source['forecast_weight'] if leg_source else None,
        ))

    return VoyageResponse(
        route_name=result.route_name,
        departure_time=result.departure_time,
        arrival_time=result.arrival_time,
        total_distance_nm=round(result.total_distance_nm, 2),
        total_time_hours=round(result.total_time_hours, 2),
        total_fuel_mt=round(result.total_fuel_mt, 2),
        avg_sog_kts=round(result.avg_sog_kts, 1),
        avg_stw_kts=round(result.avg_stw_kts, 1),
        legs=legs_response,
        calm_speed_kts=request.calm_speed_kts,
        is_laden=request.is_laden,
        data_sources=data_sources_summary,
    )


@router.post("/api/voyage/monte-carlo", response_model=MonteCarloResponse)
async def monte_carlo_simulation(request: MonteCarloRequest):
    """
    Run Monte Carlo simulation on a voyage.

    Perturbs weather conditions across N simulations and returns
    P10/P50/P90 confidence intervals for ETA, fuel, and time.
    """
    if len(request.waypoints) < 2:
        raise HTTPException(status_code=400, detail="At least 2 waypoints required")

    _vs = get_vessel_state()
    db_weather = get_app_state().weather_providers.get('db_weather')

    departure = request.departure_time or datetime.now(timezone.utc)

    wps = [(wp.lat, wp.lon) for wp in request.waypoints]
    route = create_route_from_waypoints(wps, "MC Simulation Route")

    # Pre-fetch wind grid for the route bbox so MC wind lookups are instant
    mc_weather_fn = weather_provider  # default fallback
    try:
        lats = [wp.lat for wp in request.waypoints]
        lons = [wp.lon for wp in request.waypoints]
        margin = 2.0
        bbox = (min(lats) - margin, max(lats) + margin,
                min(lons) - margin, max(lons) + margin)
        wind_data = get_wind_field(*bbox, 0.5, departure)
        wave_data = get_wave_field(*bbox, 0.5, wind_data)
        current_data = get_current_field(bbox[0], bbox[1], bbox[2], bbox[3])
        if wind_data and wave_data and current_data:
            grid_wx = GridWeatherProvider(wind_data, wave_data, current_data)
            mc_weather_fn = grid_wx.get_weather
            logger.info("MC: Pre-fetched route weather grid for fast wind lookups")
    except Exception as e:
        logger.warning(f"MC: Failed to pre-fetch route grid, using default provider: {e}")

    def _run():
        return _vs.monte_carlo_sim.run(
            route=route,
            calm_speed_kts=request.calm_speed_kts,
            is_laden=request.is_laden,
            departure_time=departure,
            weather_provider=mc_weather_fn,
            n_simulations=request.n_simulations,
            db_weather=db_weather,
        )

    try:
        mc_result = await asyncio.to_thread(_run)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Monte Carlo simulation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

    return MonteCarloResponse(
        n_simulations=mc_result.n_simulations,
        eta=PercentileString(
            p10=mc_result.eta_p10,
            p50=mc_result.eta_p50,
            p90=mc_result.eta_p90,
        ),
        fuel_mt=PercentileFloat(
            p10=mc_result.fuel_p10,
            p50=mc_result.fuel_p50,
            p90=mc_result.fuel_p90,
        ),
        total_time_hours=PercentileFloat(
            p10=mc_result.time_p10,
            p50=mc_result.time_p50,
            p90=mc_result.time_p90,
        ),
        computation_time_ms=mc_result.computation_time_ms,
    )


@router.get("/api/voyage/weather-along-route")
async def get_weather_along_route(
    waypoints: str = Query(..., description="Comma-separated lat,lon pairs: lat1,lon1;lat2,lon2;..."),
    time: Optional[datetime] = None,
    interpolation_points: int = Query(5, ge=1, le=20, description="Points to interpolate per leg"),
):
    """
    Get weather conditions along a route with distance-indexed interpolation.

    Returns weather at waypoints plus interpolated points along each leg,
    with cumulative distance for chart display.
    """
    if time is None:
        time = datetime.now(timezone.utc)

    # Parse waypoints
    try:
        wps = []
        for wp_str in waypoints.split(';'):
            lat, lon = wp_str.strip().split(',')
            wps.append((float(lat), float(lon)))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid waypoints format: {e}")

    if len(wps) < 2:
        raise HTTPException(status_code=400, detail="At least 2 waypoints required")

    # Build interpolated points along great circle per leg
    points = []
    cumulative_nm = 0.0

    for i in range(len(wps)):
        lat, lon = wps[i]

        if i > 0:
            prev_lat, prev_lon = wps[i - 1]

            # Interpolate along the leg (skip first point — already added as prev waypoint)
            for j in range(1, interpolation_points):
                frac = j / interpolation_points
                # Linear interpolation (good enough for nearby points)
                ilat = prev_lat + (lat - prev_lat) * frac
                ilon = prev_lon + (lon - prev_lon) * frac

                seg_dist = haversine_distance(
                    prev_lat if j == 1 else points[-1]["lat"],
                    prev_lon if j == 1 else points[-1]["lon"],
                    ilat, ilon,
                )
                cumulative_nm += seg_dist

                wx, _ = get_weather_at_point(ilat, ilon, time)
                points.append({
                    "distance_nm": round(cumulative_nm, 1),
                    "lat": round(ilat, 4),
                    "lon": round(ilon, 4),
                    "wind_speed_kts": round(wx['wind_speed_ms'] * 1.94384, 1),
                    "wind_dir_deg": round(wx['wind_dir_deg'], 0),
                    "wave_height_m": round(wx['sig_wave_height_m'], 1),
                    "wave_dir_deg": round(wx['wave_dir_deg'], 0),
                    "current_speed_ms": round(wx['current_speed_ms'], 2),
                    "current_dir_deg": round(wx['current_dir_deg'], 0),
                    "is_waypoint": False,
                    "waypoint_index": None,
                })

            # Distance from last interpolated point to this waypoint
            if points:
                seg_dist = haversine_distance(points[-1]["lat"], points[-1]["lon"], lat, lon)
                cumulative_nm += seg_dist
            # else first waypoint at distance 0

        # Add waypoint itself
        wx, _ = get_weather_at_point(lat, lon, time)
        points.append({
            "distance_nm": round(cumulative_nm, 1),
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "wind_speed_kts": round(wx['wind_speed_ms'] * 1.94384, 1),
            "wind_dir_deg": round(wx['wind_dir_deg'], 0),
            "wave_height_m": round(wx['sig_wave_height_m'], 1),
            "wave_dir_deg": round(wx['wave_dir_deg'], 0),
            "current_speed_ms": round(wx['current_speed_ms'], 2),
            "current_dir_deg": round(wx['current_dir_deg'], 0),
            "is_waypoint": True,
            "waypoint_index": i,
        })

    return {
        "time": time.isoformat(),
        "total_distance_nm": round(cumulative_nm, 1),
        "points": points,
    }

"""
Route optimization API router.

Handles A* and VISIR weather routing optimization,
and optimizer status queries.
"""

import asyncio
import logging
import math
import time as _time
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from api.rate_limit import limiter

from api.schemas import (
    OptimizationLegModel,
    OptimizationRequest,
    OptimizationResponse,
    ParetoSolutionModel,
    Position,
    SafetySummary,
    SpeedScenarioModel,
    WeatherProvenanceModel,
)
from api.state import get_app_state, get_vessel_state
from api.weather_service import (
    get_current_field,
    get_ice_field,
    get_sst_field,
    get_visibility_field,
    get_wind_field,
    get_wave_field,
    supplement_temporal_wind,
)
from src.optimization.grid_weather_provider import GridWeatherProvider
from src.optimization.route_optimizer import RouteOptimizer
from src.optimization.weather_assessment import RouteWeatherAssessment

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Optimization"])


def _safe_round(value: float, ndigits: int = 2, fallback: float = 0.0) -> float:
    """Round value, replacing NaN/Inf with fallback to prevent JSON serialization errors."""
    if math.isnan(value) or math.isinf(value):
        return fallback
    return round(value, ndigits)


@router.post("/api/optimize/route", response_model=OptimizationResponse)
@limiter.limit("10/minute")
async def optimize_route(request: Request, request_body: OptimizationRequest = None):
    """
    Find optimal route through weather.

    Supports two optimization engines selected via the ``engine`` field:
    - **astar** (default): A* grid search with weather-aware cost function
    - **visir**: VISIR-style Dijkstra with time-expanded graph and voluntary speed reduction

    Minimizes fuel consumption (or time) by routing around adverse weather.

    Grid resolution affects accuracy vs computation time:
    - 0.2° = ~12nm cells, good land avoidance (default for A*)
    - 0.25° = ~15nm cells, good balance (default for VISIR)
    - 0.5° = ~30nm cells, faster, less precise
    """
    # Run the entire optimization in a thread so the event loop stays
    # responsive (weather provisioning + VISIR can take 30s+).
    return await asyncio.to_thread(_optimize_route_sync, request_body)


def _run_engine_with_fallback(
    optimizer,
    engine_name: str,
    request: "OptimizationRequest",
    departure: "datetime",
    wx_provider,
    route_wps,
) -> "OptimizedRoute":
    """
    Run the optimization engine, retrying with relaxed safety limits on failure.

    First attempt uses normal hard limits.  If a ValueError containing
    "no route found" is raised (weather blocks departure), retry with
    ``skip_hard_limits=True``.  A successful retry sets
    ``result.safety_degraded = True`` so the frontend can warn the user.
    """
    from src.optimization.base_optimizer import OptimizedRoute  # noqa: F811

    def _call_engine():
        if engine_name == "visir":
            return optimizer.optimize_route(
                origin=(request.origin.lat, request.origin.lon),
                destination=(request.destination.lat, request.destination.lon),
                departure_time=departure,
                calm_speed_kts=request.calm_speed_kts,
                is_laden=request.is_laden,
                weather_provider=wx_provider,
                max_time_factor=request.max_time_factor,
            )
        elif request.pareto:
            return optimizer.optimize_route_pareto(
                origin=(request.origin.lat, request.origin.lon),
                destination=(request.destination.lat, request.destination.lon),
                departure_time=departure,
                calm_speed_kts=request.calm_speed_kts,
                is_laden=request.is_laden,
                weather_provider=wx_provider,
                max_time_factor=request.max_time_factor,
                baseline_time_hours=request.baseline_time_hours,
                baseline_fuel_mt=request.baseline_fuel_mt,
                baseline_distance_nm=request.baseline_distance_nm,
                route_waypoints=route_wps,
            )
        else:
            return optimizer.optimize_route(
                origin=(request.origin.lat, request.origin.lon),
                destination=(request.destination.lat, request.destination.lon),
                departure_time=departure,
                calm_speed_kts=request.calm_speed_kts,
                is_laden=request.is_laden,
                weather_provider=wx_provider,
                max_time_factor=request.max_time_factor,
                baseline_time_hours=request.baseline_time_hours,
                baseline_fuel_mt=request.baseline_fuel_mt,
                baseline_distance_nm=request.baseline_distance_nm,
                route_waypoints=route_wps,
            )

    # First attempt: normal safety limits
    optimizer.skip_hard_limits = False
    try:
        return _call_engine()
    except ValueError as e:
        msg = str(e).lower()
        if "no route found" not in msg:
            raise
        # Only retry with relaxed limits when the failure looks weather-related
        # (few nodes explored).  Topology failures (timeout, pq exhausted after
        # many nodes) won't benefit from relaxed safety limits — re-raise fast.
        explored = 0
        import re
        m = re.search(r"exploring (\d+) nodes", str(e))
        if m:
            explored = int(m.group(1))
        # Heuristic: >10 000 nodes explored means the grid is reachable but the
        # destination isn't — retrying with relaxed limits won't help.
        if explored > 10_000:
            logger.warning(
                f"{engine_name}: routing failed after {explored} nodes "
                "(topology / timeout), skipping safety retry"
            )
            raise
        logger.warning(
            f"{engine_name}: normal routing failed ({e}), retrying with relaxed safety limits"
        )

    # Retry: relax wave/wind hard limits
    optimizer.skip_hard_limits = True
    try:
        result = _call_engine()
        result.safety_degraded = True
        logger.info(f"{engine_name}: fallback route found with relaxed safety limits")
        return result
    finally:
        optimizer.skip_hard_limits = False


def _optimize_route_sync(request: "OptimizationRequest") -> "OptimizationResponse":
    """Synchronous route optimization logic (runs in a thread pool)."""
    _vs = get_vessel_state()
    db_weather = get_app_state().weather_providers.get('db_weather')

    departure = request.departure_time or datetime.utcnow()

    # Select optimization engine
    engine_name = request.engine.lower()
    if engine_name == "visir":
        active_optimizer = _vs.visir_optimizer
    else:
        active_optimizer = _vs.route_optimizer
    # VISIR uses coarser resolution than A* (0.25° vs 0.1°) for performance
    active_optimizer.resolution_deg = (
        max(request.grid_resolution_deg, 0.25) if engine_name == "visir"
        else request.grid_resolution_deg
    )
    active_optimizer.optimization_target = request.optimization_target
    active_optimizer.safety_weight = request.safety_weight
    # Variable resolution: two-tier grid (A* only, ignored by VISIR)
    if engine_name != "visir":
        active_optimizer.variable_resolution = request.variable_resolution
    # Zone enforcement: filter by visible zone types (None = all, [] = none)
    if request.enforced_zone_types is not None:
        active_optimizer.enforce_zones = len(request.enforced_zone_types) > 0
        active_optimizer.zone_checker.set_enforced_types(request.enforced_zone_types)
    else:
        active_optimizer.enforce_zones = True
        active_optimizer.zone_checker.set_enforced_types(None)

    try:
        # ── Temporal weather provisioning (DB-first) ──────────────────
        temporal_wx = None
        provenance_models = None
        used_temporal = False

        if db_weather is not None:
            try:
                assessor = RouteWeatherAssessment(db_weather=db_weather)
                wx_needs = assessor.assess(
                    origin=(request.origin.lat, request.origin.lon),
                    destination=(request.destination.lat, request.destination.lon),
                    departure_time=departure,
                    calm_speed_kts=request.calm_speed_kts,
                )
                avail_parts = [f"{s}: {v.get('coverage_pct',0):.0f}%" for s,v in wx_needs.availability.items()]
                logger.info(
                    f"Weather assessment: {wx_needs.estimated_passage_hours:.0f}h passage, "
                    f"need hours {wx_needs.required_forecast_hours[:5]}..., "
                    f"availability: {', '.join(avail_parts)}, "
                    f"warnings: {wx_needs.gap_warnings}"
                )
                temporal_wx = assessor.provision(wx_needs)
                if temporal_wx is not None:
                    used_temporal = True
                    params_loaded = list(temporal_wx.grids.keys())
                    has_temporal_wind = any(p in temporal_wx.grids for p in ["wind_u", "wind_v"])
                    if not has_temporal_wind:
                        bbox = wx_needs.corridor_bbox
                        if supplement_temporal_wind(temporal_wx, bbox[0], bbox[1], bbox[2], bbox[3], departure):
                            has_temporal_wind = True
                            params_loaded = list(temporal_wx.grids.keys())
                    hours_per_param = {p: sorted(temporal_wx.grids[p].keys()) for p in params_loaded[:3]}
                    logger.info(
                        f"Temporal provider: {len(params_loaded)} params ({params_loaded}), "
                        f"wind={'yes' if has_temporal_wind else 'NO (calm assumed)'}, "
                        f"hours sample: {hours_per_param}"
                    )
                    provenance_models = [
                        WeatherProvenanceModel(
                            source_type=p.source_type,
                            model_name=p.model_name,
                            forecast_lead_hours=_safe_round(p.forecast_lead_hours, 1),
                            confidence=p.confidence,
                        )
                        for p in temporal_wx.provenance.values()
                    ]
                    logger.info("Using temporal weather provider for route optimization")
                else:
                    logger.warning("Temporal provisioning returned None — falling back to single-snapshot")
            except Exception as e:
                logger.warning(f"Temporal weather provisioning failed, falling back: {e}", exc_info=True)

        # ── Fallback: single-snapshot GridWeatherProvider ─────────────
        if temporal_wx is None:
            margin = 5.0
            lat_min = min(request.origin.lat, request.destination.lat) - margin
            lat_max = max(request.origin.lat, request.destination.lat) + margin
            lon_min = min(request.origin.lon, request.destination.lon) - margin
            lon_max = max(request.origin.lon, request.destination.lon) + margin
            lat_min, lat_max = max(lat_min, -85), min(lat_max, 85)

            logger.info(f"Fallback: loading single-snapshot weather for bbox [{lat_min:.1f},{lat_max:.1f},{lon_min:.1f},{lon_max:.1f}]")
            t0 = _time.monotonic()
            wind = get_wind_field(lat_min, lat_max, lon_min, lon_max, request.grid_resolution_deg, departure)
            logger.info(f"  Wind loaded in {_time.monotonic()-t0:.1f}s: source={getattr(wind, 'source', '?')}")
            t1 = _time.monotonic()
            waves = get_wave_field(lat_min, lat_max, lon_min, lon_max, request.grid_resolution_deg, wind)
            logger.info(f"  Waves loaded in {_time.monotonic()-t1:.1f}s")
            t2 = _time.monotonic()
            currents = get_current_field(lat_min, lat_max, lon_min, lon_max, request.grid_resolution_deg)
            logger.info(f"  Currents loaded in {_time.monotonic()-t2:.1f}s")
            # Extended fields (SPEC-P1)
            sst = get_sst_field(lat_min, lat_max, lon_min, lon_max, request.grid_resolution_deg, departure)
            vis = get_visibility_field(lat_min, lat_max, lon_min, lon_max, request.grid_resolution_deg, departure)
            ice = get_ice_field(lat_min, lat_max, lon_min, lon_max, request.grid_resolution_deg, departure)
            logger.info(f"  Total fallback: {_time.monotonic()-t0:.1f}s (incl. SST/vis/ice)")
            grid_wx = GridWeatherProvider(wind, waves, currents, sst, vis, ice)

        # Select weather provider callable
        wx_provider = temporal_wx.get_weather if temporal_wx else grid_wx.get_weather

        # Convert route_waypoints for multi-segment optimization
        route_wps = None
        if request.route_waypoints and len(request.route_waypoints) > 2:
            route_wps = [(wp.lat, wp.lon) for wp in request.route_waypoints]

        # ── Run engine with safety-fallback retry ──────────────────
        result = _run_engine_with_fallback(
            active_optimizer, engine_name, request, departure,
            wx_provider, route_wps,
        )

        # Format response
        waypoints = [Position(lat=wp[0], lon=wp[1]) for wp in result.waypoints]

        # Compute cumulative time for provenance per leg
        cum_time_h = 0.0
        legs = []
        for leg in result.leg_details:
            # Per-leg provenance label
            data_source_label = None
            if used_temporal and temporal_wx is not None:
                leg_time = departure + timedelta(hours=cum_time_h + leg['time_hours'] / 2)
                prov = temporal_wx.get_provenance(leg_time)
                data_source_label = f"{prov.source_type} ({prov.confidence} confidence)"
            cum_time_h += leg['time_hours']

            legs.append(OptimizationLegModel(
                from_lat=leg['from'][0],
                from_lon=leg['from'][1],
                to_lat=leg['to'][0],
                to_lon=leg['to'][1],
                distance_nm=_safe_round(leg['distance_nm'], 2),
                bearing_deg=_safe_round(leg['bearing_deg'], 1),
                fuel_mt=_safe_round(leg['fuel_mt'], 3),
                time_hours=_safe_round(leg['time_hours'], 2),
                sog_kts=_safe_round(leg['sog_kts'], 1),
                stw_kts=_safe_round(leg.get('stw_kts', leg['sog_kts']), 1),
                wind_speed_ms=_safe_round(leg['wind_speed_ms'], 1),
                wave_height_m=_safe_round(leg['wave_height_m'], 1),
                safety_status=leg.get('safety_status'),
                roll_deg=_safe_round(leg['roll_deg'], 1) if leg.get('roll_deg') else None,
                pitch_deg=_safe_round(leg['pitch_deg'], 1) if leg.get('pitch_deg') else None,
                data_source=data_source_label,
                swell_hs_m=_safe_round(leg['swell_hs_m'], 2) if leg.get('swell_hs_m') is not None else None,
                windsea_hs_m=_safe_round(leg['windsea_hs_m'], 2) if leg.get('windsea_hs_m') is not None else None,
                current_effect_kts=_safe_round(leg['current_effect_kts'], 2) if leg.get('current_effect_kts') is not None else None,
                visibility_m=_safe_round(leg['visibility_m'], 0) if leg.get('visibility_m') is not None else None,
                sst_celsius=_safe_round(leg['sst_celsius'], 1) if leg.get('sst_celsius') is not None else None,
                ice_concentration=_safe_round(leg['ice_concentration'], 3) if leg.get('ice_concentration') is not None else None,
            ))

        # Build safety summary
        safety_summary = SafetySummary(
            status=result.safety_status,
            warnings=result.safety_warnings,
            max_roll_deg=_safe_round(result.max_roll_deg, 1),
            max_pitch_deg=_safe_round(result.max_pitch_deg, 1),
            max_accel_ms2=_safe_round(result.max_accel_ms2, 2),
        )

        # Build speed strategy scenarios
        scenario_models = []
        for sc in result.scenarios:
            sc_legs = []
            for leg in sc.leg_details:
                sc_legs.append(OptimizationLegModel(
                    from_lat=leg['from'][0],
                    from_lon=leg['from'][1],
                    to_lat=leg['to'][0],
                    to_lon=leg['to'][1],
                    distance_nm=_safe_round(leg['distance_nm'], 2),
                    bearing_deg=_safe_round(leg['bearing_deg'], 1),
                    fuel_mt=_safe_round(leg['fuel_mt'], 3),
                    time_hours=_safe_round(leg['time_hours'], 2),
                    sog_kts=_safe_round(leg['sog_kts'], 1),
                    stw_kts=_safe_round(leg.get('stw_kts', leg['sog_kts']), 1),
                    wind_speed_ms=_safe_round(leg['wind_speed_ms'], 1),
                    wave_height_m=_safe_round(leg['wave_height_m'], 1),
                    safety_status=leg.get('safety_status'),
                    roll_deg=_safe_round(leg['roll_deg'], 1) if leg.get('roll_deg') else None,
                    pitch_deg=_safe_round(leg['pitch_deg'], 1) if leg.get('pitch_deg') else None,
                    swell_hs_m=_safe_round(leg['swell_hs_m'], 2) if leg.get('swell_hs_m') is not None else None,
                    windsea_hs_m=_safe_round(leg['windsea_hs_m'], 2) if leg.get('windsea_hs_m') is not None else None,
                    current_effect_kts=_safe_round(leg['current_effect_kts'], 2) if leg.get('current_effect_kts') is not None else None,
                    visibility_m=_safe_round(leg['visibility_m'], 0) if leg.get('visibility_m') is not None else None,
                    sst_celsius=_safe_round(leg['sst_celsius'], 1) if leg.get('sst_celsius') is not None else None,
                    ice_concentration=_safe_round(leg['ice_concentration'], 3) if leg.get('ice_concentration') is not None else None,
                ))
            scenario_models.append(SpeedScenarioModel(
                strategy=sc.strategy,
                label=sc.label,
                total_fuel_mt=_safe_round(sc.total_fuel_mt, 2),
                total_time_hours=_safe_round(sc.total_time_hours, 2),
                total_distance_nm=_safe_round(sc.total_distance_nm, 1),
                avg_speed_kts=_safe_round(sc.avg_speed_kts, 1),
                speed_profile=[_safe_round(s, 1) for s in sc.speed_profile],
                legs=sc_legs,
                fuel_savings_pct=_safe_round(sc.fuel_savings_pct, 1),
                time_savings_pct=_safe_round(sc.time_savings_pct, 1),
            ))

        # Build Pareto front models (if available)
        pareto_models = None
        if result.pareto_front:
            pareto_models = [
                ParetoSolutionModel(
                    lambda_value=_safe_round(p.lambda_value, 3),
                    fuel_mt=_safe_round(p.fuel_mt, 2),
                    time_hours=_safe_round(p.time_hours, 2),
                    distance_nm=_safe_round(p.distance_nm, 1),
                    speed_profile=[_safe_round(s, 1) for s in p.speed_profile],
                    is_selected=p.is_selected,
                )
                for p in result.pareto_front
            ]

        return OptimizationResponse(
            waypoints=waypoints,
            total_fuel_mt=_safe_round(result.total_fuel_mt, 2),
            total_time_hours=_safe_round(result.total_time_hours, 2),
            total_distance_nm=_safe_round(result.total_distance_nm, 1),
            direct_fuel_mt=_safe_round(result.direct_fuel_mt, 2),
            direct_time_hours=_safe_round(result.direct_time_hours, 2),
            fuel_savings_pct=_safe_round(result.fuel_savings_pct, 1),
            time_savings_pct=_safe_round(result.time_savings_pct, 1),
            legs=legs,
            speed_profile=[_safe_round(s, 1) for s in result.speed_profile],
            avg_speed_kts=_safe_round(result.avg_speed_kts, 1),
            variable_speed_enabled=result.variable_speed_enabled,
            engine=engine_name,
            variable_resolution_enabled=request.variable_resolution and engine_name != "visir",
            safety=safety_summary,
            scenarios=scenario_models,
            pareto_front=pareto_models,
            baseline_fuel_mt=_safe_round(result.baseline_fuel_mt, 2) if result.baseline_fuel_mt else None,
            baseline_time_hours=_safe_round(result.baseline_time_hours, 2) if result.baseline_time_hours else None,
            baseline_distance_nm=_safe_round(result.baseline_distance_nm, 1) if result.baseline_distance_nm else None,
            safety_degraded=result.safety_degraded,
            weather_provenance=provenance_models,
            temporal_weather=used_temporal,
            optimization_target=request.optimization_target,
            grid_resolution_deg=active_optimizer.resolution_deg,
            cells_explored=result.cells_explored,
            optimization_time_ms=_safe_round(result.optimization_time_ms, 1),
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail={
            "error": "routing_failed",
            "message": str(e),
            "diagnostics": {
                "engine": engine_name,
                "reason": "weather_blocked" if "no route found" in str(e).lower() else "grid_error",
            },
        })
    except Exception as e:
        logger.error(f"Route optimization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@router.get("/api/optimize/status")
async def get_optimization_status():
    """Get current optimizer configuration."""
    _vs = get_vessel_state()
    return {
        "status": "ready",
        "default_resolution_deg": RouteOptimizer.DEFAULT_RESOLUTION_DEG,
        "default_max_cells": RouteOptimizer.DEFAULT_MAX_CELLS,
        "optimization_targets": ["fuel", "time"],
        "vessel_model": {
            "dwt": _vs.specs.dwt,
            "service_speed_laden": _vs.specs.service_speed_laden,
        }
    }

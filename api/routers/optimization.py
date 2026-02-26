"""
Route optimization API router.

Handles A* and VISIR weather routing optimization,
and optimizer status queries.
"""

import asyncio
import logging
import time as _time
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException

from api.schemas import (
    BenchmarkEngineResult,
    BenchmarkRequest,
    BenchmarkResponse,
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
from src.optimization.visir_optimizer import VisirOptimizer
from src.optimization.weather_assessment import RouteWeatherAssessment

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Optimization"])


@router.post("/api/optimize/route", response_model=OptimizationResponse)
async def optimize_route(request: OptimizationRequest):
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
    return await asyncio.to_thread(_optimize_route_sync, request)


def _optimize_route_sync(request: "OptimizationRequest") -> "OptimizationResponse":
    """Synchronous route optimization logic (runs in a thread pool)."""
    _vs = get_vessel_state()
    db_weather = get_app_state().weather_providers.get('db_weather')

    departure = request.departure_time or datetime.now(timezone.utc)

    # Create fresh optimizer per request (avoids race conditions with concurrent requests)
    engine_name = request.engine.lower()
    vessel_model = _vs.model
    resolution = (
        max(request.grid_resolution_deg, 0.25) if engine_name == "visir"
        else request.grid_resolution_deg
    )

    if engine_name == "visir":
        active_optimizer = VisirOptimizer(
            vessel_model=vessel_model,
            resolution_deg=resolution,
            optimization_target=request.optimization_target,
        )
        active_optimizer.safety_weight = request.safety_weight
    else:
        active_optimizer = RouteOptimizer(
            vessel_model=vessel_model,
            resolution_deg=resolution,
            optimization_target=request.optimization_target,
            variable_resolution=request.variable_resolution,
        )
        active_optimizer.safety_weight = request.safety_weight

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
                            forecast_lead_hours=round(p.forecast_lead_hours, 1),
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

        # A* engine accepts extra dev params; VISIR uses base interface
        if engine_name == "visir":
            # VISIR needs a wider time budget than A* — its 3D graph
            # (lat, lon, time) requires slack to explore alternate speeds.
            visir_time_factor = max(request.max_time_factor, 1.30)
            result = active_optimizer.optimize_route(
                origin=(request.origin.lat, request.origin.lon),
                destination=(request.destination.lat, request.destination.lon),
                departure_time=departure,
                calm_speed_kts=request.calm_speed_kts,
                is_laden=request.is_laden,
                weather_provider=wx_provider,
                max_time_factor=visir_time_factor,
            )
        elif request.pareto:
            result = active_optimizer.optimize_route_pareto(
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
            result = active_optimizer.optimize_route(
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
                distance_nm=round(leg['distance_nm'], 2),
                bearing_deg=round(leg['bearing_deg'], 1),
                fuel_mt=round(leg['fuel_mt'], 3),
                time_hours=round(leg['time_hours'], 2),
                sog_kts=round(leg['sog_kts'], 1),
                stw_kts=round(leg.get('stw_kts', leg['sog_kts']), 1),
                wind_speed_ms=round(leg['wind_speed_ms'], 1),
                wave_height_m=round(leg['wave_height_m'], 1),
                safety_status=leg.get('safety_status'),
                roll_deg=round(leg['roll_deg'], 1) if leg.get('roll_deg') else None,
                pitch_deg=round(leg['pitch_deg'], 1) if leg.get('pitch_deg') else None,
                data_source=data_source_label,
                swell_hs_m=round(leg['swell_hs_m'], 2) if leg.get('swell_hs_m') is not None else None,
                windsea_hs_m=round(leg['windsea_hs_m'], 2) if leg.get('windsea_hs_m') is not None else None,
                current_effect_kts=round(leg['current_effect_kts'], 2) if leg.get('current_effect_kts') is not None else None,
                visibility_m=round(leg['visibility_m'], 0) if leg.get('visibility_m') is not None else None,
                sst_celsius=round(leg['sst_celsius'], 1) if leg.get('sst_celsius') is not None else None,
                ice_concentration=round(leg['ice_concentration'], 3) if leg.get('ice_concentration') is not None else None,
            ))

        # Build safety summary
        safety_summary = SafetySummary(
            status=result.safety_status,
            warnings=result.safety_warnings,
            max_roll_deg=round(result.max_roll_deg, 1),
            max_pitch_deg=round(result.max_pitch_deg, 1),
            max_accel_ms2=round(result.max_accel_ms2, 2),
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
                    distance_nm=round(leg['distance_nm'], 2),
                    bearing_deg=round(leg['bearing_deg'], 1),
                    fuel_mt=round(leg['fuel_mt'], 3),
                    time_hours=round(leg['time_hours'], 2),
                    sog_kts=round(leg['sog_kts'], 1),
                    stw_kts=round(leg.get('stw_kts', leg['sog_kts']), 1),
                    wind_speed_ms=round(leg['wind_speed_ms'], 1),
                    wave_height_m=round(leg['wave_height_m'], 1),
                    safety_status=leg.get('safety_status'),
                    roll_deg=round(leg['roll_deg'], 1) if leg.get('roll_deg') else None,
                    pitch_deg=round(leg['pitch_deg'], 1) if leg.get('pitch_deg') else None,
                    swell_hs_m=round(leg['swell_hs_m'], 2) if leg.get('swell_hs_m') is not None else None,
                    windsea_hs_m=round(leg['windsea_hs_m'], 2) if leg.get('windsea_hs_m') is not None else None,
                    current_effect_kts=round(leg['current_effect_kts'], 2) if leg.get('current_effect_kts') is not None else None,
                    visibility_m=round(leg['visibility_m'], 0) if leg.get('visibility_m') is not None else None,
                    sst_celsius=round(leg['sst_celsius'], 1) if leg.get('sst_celsius') is not None else None,
                    ice_concentration=round(leg['ice_concentration'], 3) if leg.get('ice_concentration') is not None else None,
                ))
            scenario_models.append(SpeedScenarioModel(
                strategy=sc.strategy,
                label=sc.label,
                total_fuel_mt=round(sc.total_fuel_mt, 2),
                total_time_hours=round(sc.total_time_hours, 2),
                total_distance_nm=round(sc.total_distance_nm, 1),
                avg_speed_kts=round(sc.avg_speed_kts, 1),
                speed_profile=[round(s, 1) for s in sc.speed_profile],
                legs=sc_legs,
                fuel_savings_pct=round(sc.fuel_savings_pct, 1),
                time_savings_pct=round(sc.time_savings_pct, 1),
            ))

        # Build Pareto front models (if available)
        pareto_models = None
        if result.pareto_front:
            pareto_models = [
                ParetoSolutionModel(
                    lambda_value=round(p.lambda_value, 3),
                    fuel_mt=round(p.fuel_mt, 2),
                    time_hours=round(p.time_hours, 2),
                    distance_nm=round(p.distance_nm, 1),
                    speed_profile=[round(s, 1) for s in p.speed_profile],
                    is_selected=p.is_selected,
                )
                for p in result.pareto_front
            ]

        return OptimizationResponse(
            waypoints=waypoints,
            total_fuel_mt=round(result.total_fuel_mt, 2),
            total_time_hours=round(result.total_time_hours, 2),
            total_distance_nm=round(result.total_distance_nm, 1),
            direct_fuel_mt=round(result.direct_fuel_mt, 2),
            direct_time_hours=round(result.direct_time_hours, 2),
            fuel_savings_pct=round(result.fuel_savings_pct, 1),
            time_savings_pct=round(result.time_savings_pct, 1),
            legs=legs,
            speed_profile=[round(s, 1) for s in result.speed_profile],
            avg_speed_kts=round(result.avg_speed_kts, 1),
            variable_speed_enabled=result.variable_speed_enabled,
            engine=engine_name,
            variable_resolution_enabled=request.variable_resolution and engine_name != "visir",
            safety=safety_summary,
            scenarios=scenario_models,
            pareto_front=pareto_models,
            baseline_fuel_mt=round(result.baseline_fuel_mt, 2) if result.baseline_fuel_mt else None,
            baseline_time_hours=round(result.baseline_time_hours, 2) if result.baseline_time_hours else None,
            baseline_distance_nm=round(result.baseline_distance_nm, 1) if result.baseline_distance_nm else None,
            weather_provenance=provenance_models,
            temporal_weather=used_temporal,
            optimization_target=request.optimization_target,
            grid_resolution_deg=active_optimizer.resolution_deg,
            cells_explored=result.cells_explored,
            optimization_time_ms=round(result.optimization_time_ms, 1),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
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


@router.post("/api/optimize/benchmark", response_model=BenchmarkResponse)
async def benchmark_engines(request: BenchmarkRequest):
    """
    Run the same optimization on multiple engines and compare results.

    Returns wall-clock time, fuel, distance, and cells explored per engine.
    Useful for comparing A* vs VISIR performance on the same route.
    """
    return await asyncio.to_thread(_benchmark_sync, request)


def _benchmark_sync(request: "BenchmarkRequest") -> "BenchmarkResponse":
    """Synchronous benchmark logic (runs in a thread pool)."""
    _vs = get_vessel_state()
    db_weather = get_app_state().weather_providers.get('db_weather')
    departure = request.departure_time or datetime.now(timezone.utc)
    vessel_model = _vs.model

    allowed_engines = {"astar", "visir"}
    engines = [e.lower() for e in request.engines if e.lower() in allowed_engines]
    if not engines:
        engines = ["astar", "visir"]

    # Build shared weather provider once (snapshot fallback only for benchmark)
    margin = 5.0
    lat_min = min(request.origin.lat, request.destination.lat) - margin
    lat_max = max(request.origin.lat, request.destination.lat) + margin
    lon_min = min(request.origin.lon, request.destination.lon) - margin
    lon_max = max(request.origin.lon, request.destination.lon) + margin
    lat_min, lat_max = max(lat_min, -85), min(lat_max, 85)

    wind = get_wind_field(lat_min, lat_max, lon_min, lon_max, request.grid_resolution_deg, departure)
    waves = get_wave_field(lat_min, lat_max, lon_min, lon_max, request.grid_resolution_deg, wind)
    currents = get_current_field(lat_min, lat_max, lon_min, lon_max, request.grid_resolution_deg)
    sst = get_sst_field(lat_min, lat_max, lon_min, lon_max, request.grid_resolution_deg, departure)
    vis = get_visibility_field(lat_min, lat_max, lon_min, lon_max, request.grid_resolution_deg, departure)
    ice = get_ice_field(lat_min, lat_max, lon_min, lon_max, request.grid_resolution_deg, departure)
    grid_wx = GridWeatherProvider(wind, waves, currents, sst, vis, ice)
    wx_provider = grid_wx.get_weather

    results = []
    for engine_name in engines:
        try:
            resolution = (
                max(request.grid_resolution_deg, 0.25) if engine_name == "visir"
                else request.grid_resolution_deg
            )

            if engine_name == "visir":
                optimizer = VisirOptimizer(
                    vessel_model=vessel_model,
                    resolution_deg=resolution,
                    optimization_target=request.optimization_target,
                )
            else:
                optimizer = RouteOptimizer(
                    vessel_model=vessel_model,
                    resolution_deg=resolution,
                    optimization_target=request.optimization_target,
                    variable_resolution=request.variable_resolution,
                )
            optimizer.safety_weight = request.safety_weight

            result = optimizer.optimize_route(
                origin=(request.origin.lat, request.origin.lon),
                destination=(request.destination.lat, request.destination.lon),
                departure_time=departure,
                calm_speed_kts=request.calm_speed_kts,
                is_laden=request.is_laden,
                weather_provider=wx_provider,
                max_time_factor=request.max_time_factor,
            )

            results.append(BenchmarkEngineResult(
                engine=engine_name,
                total_fuel_mt=round(result.total_fuel_mt, 2),
                total_time_hours=round(result.total_time_hours, 2),
                total_distance_nm=round(result.total_distance_nm, 1),
                cells_explored=result.cells_explored,
                optimization_time_ms=round(result.optimization_time_ms, 1),
                waypoint_count=len(result.waypoints),
            ))
        except Exception as e:
            logger.error(f"Benchmark engine {engine_name} failed: {e}", exc_info=True)
            results.append(BenchmarkEngineResult(
                engine=engine_name,
                total_fuel_mt=0,
                total_time_hours=0,
                total_distance_nm=0,
                cells_explored=0,
                optimization_time_ms=0,
                waypoint_count=0,
                error=str(e),
            ))

    return BenchmarkResponse(
        results=results,
        grid_resolution_deg=request.grid_resolution_deg,
        optimization_target=request.optimization_target,
    )

"""
A* Route Optimizer for WINDMAR.

Finds optimal routes through weather using grid-based A* search.
Minimizes fuel consumption (or time) considering:
- Wind resistance
- Wave resistance
- Ocean currents
- Vessel hydrodynamics
- Land avoidance

Grid-based approach:
1. Discretize ocean into lat/lon cells
2. Each cell has weather-dependent transit cost
3. A* finds minimum-cost path from origin to destination
4. Smooth resulting path to create navigable waypoints
"""

import heapq
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from src.optimization.vessel_model import VesselModel, VesselSpecs
from src.optimization.voyage import LegWeather
from src.optimization.seakeeping import SafetyConstraints, SafetyStatus, create_default_safety_constraints
from src.data.land_mask import is_ocean, is_path_clear, get_land_mask_status
from src.data.regulatory_zones import get_zone_checker, ZoneChecker
from src.data.strait_waypoints import STRAITS, StraitDefinition
from src.optimization.base_optimizer import BaseOptimizer, OptimizedRoute, ParetoSolution

logger = logging.getLogger(__name__)

# SPEC-P1: Visibility speed caps (IMO COLREG Rule 6)
VISIBILITY_SPEED_CAPS = {
    1000: 6.0,   # Fog — bare minimum steerage
    2000: 8.0,   # Poor visibility
    5000: 12.0,  # Moderate visibility
}  # Above 5000m: no cap


def apply_visibility_cap(speed_kts: float, visibility_m: float) -> float:
    """Apply tiered COLREG Rule 6 speed cap based on visibility."""
    for vis_threshold, max_speed in sorted(VISIBILITY_SPEED_CAPS.items()):
        if visibility_m <= vis_threshold:
            return min(speed_kts, max_speed)
    return speed_kts


@dataclass
class GridCell:
    """A cell in the routing grid."""
    lat: float
    lon: float
    row: int
    col: int

    def __hash__(self):
        return hash((self.row, self.col))

    def __eq__(self, other):
        if isinstance(other, GridCell):
            return self.row == other.row and self.col == other.col
        return NotImplemented


@dataclass(order=True)
class SearchNode:
    """Node in A* search priority queue."""
    f_score: float  # g + h (total estimated cost)
    cell: GridCell = field(compare=False)
    g_score: float = field(compare=False)  # Cost from start
    arrival_time: datetime = field(compare=False)
    parent: Optional['SearchNode'] = field(compare=False, default=None)


@dataclass
class SpeedScenario:
    """One speed strategy applied to the optimized path."""
    strategy: str            # "constant_speed" or "match_eta"
    label: str               # "Same Speed" or "Match ETA"
    total_fuel_mt: float
    total_time_hours: float
    total_distance_nm: float  # same for both (same path)
    avg_speed_kts: float
    speed_profile: List[float]
    leg_details: List[Dict]
    fuel_savings_pct: float   # vs baseline
    time_savings_pct: float   # vs baseline


class RouteOptimizer(BaseOptimizer):
    """
    A* based route optimizer.

    Finds minimum-fuel (or minimum-time) routes through weather.
    """

    # Default grid settings
    DEFAULT_RESOLUTION_DEG = 0.2  # Grid cell size in degrees (~12nm at equator)
    DEFAULT_MAX_CELLS = 200_000  # Maximum cells to explore before giving up

    # Time penalty weight: fraction of full service-speed fuel rate applied per
    # hour of additional voyage time.  Lower values allow the optimizer to
    # explore weather-avoidance detours that add time but save fuel.
    # 0 = no time penalty (pure fuel), 1.0 = full penalty (original behaviour).
    TIME_PENALTY_WEIGHT = 0.3

    # 16-connected grid: 4 cardinal + 4 diagonal + 8 knight-move directions.
    # Knight moves enable ~26° and ~63° headings for smoother paths.
    DIRECTIONS = [
        # Cardinal (4)
        (-1, 0), (0, 1), (1, 0), (0, -1),
        # Diagonal (4)
        (-1, 1), (1, 1), (1, -1), (-1, -1),
        # Knight moves (8)
        (-2, 1), (-1, 2), (1, 2), (2, 1),
        (2, -1), (1, -2), (-1, -2), (-2, -1),
    ]

    # Speed optimization settings
    SPEED_RANGE_KTS = (10.0, 16.0)  # Min/max speeds to consider (slow steaming to design speed)
    SPEED_STEPS = 13  # Number of speeds to test per leg (0.5 kt increments: 10.0, 10.5, ..., 16.0)

    def __init__(
        self,
        vessel_model: Optional[VesselModel] = None,
        resolution_deg: float = DEFAULT_RESOLUTION_DEG,
        optimization_target: str = "fuel",  # "fuel" or "time"
        safety_constraints: Optional[SafetyConstraints] = None,
        enforce_safety: bool = True,
        zone_checker: Optional[ZoneChecker] = None,
        enforce_zones: bool = True,
        variable_speed: bool = True,  # Enable variable speed optimization
        variable_resolution: bool = False,  # Enable two-tier grid
    ):
        """
        Initialize route optimizer.

        Args:
            vessel_model: Vessel performance model
            resolution_deg: Grid resolution in degrees
            optimization_target: What to minimize ("fuel" or "time")
            safety_constraints: Safety constraint checker (seakeeping model)
            enforce_safety: Whether to penalize/forbid unsafe routes
            zone_checker: Regulatory zone checker
            enforce_zones: Whether to apply zone penalties/exclusions
            variable_speed: Enable per-leg speed optimization
            variable_resolution: Enable two-tier grid (0.5° ocean + 0.1° nearshore)
        """
        super().__init__(vessel_model=vessel_model)
        self.resolution_deg = resolution_deg
        self.optimization_target = optimization_target
        self.enforce_safety = enforce_safety
        self.enforce_zones = enforce_zones
        self.variable_speed = variable_speed
        self.variable_resolution = variable_resolution
        self.safety_weight: float = 0.0  # 0=pure fuel, 1=full safety penalties
        self.skip_hard_limits: bool = False  # When True, relax wave/wind hard limits

        # Safety constraints (seakeeping model)
        self.safety_constraints = safety_constraints or create_default_safety_constraints(
            lpp=self.vessel_model.specs.lpp,
            beam=self.vessel_model.specs.beam,
        )

        # Regulatory zone checker
        self.zone_checker = zone_checker or get_zone_checker()

        # Weather provider function (set before optimization)
        self.weather_provider: Optional[Callable] = None

        # Time-value penalty (computed per voyage in optimize_route)
        self._lambda_time: float = 0.0

    def optimize_route(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        departure_time: datetime,
        calm_speed_kts: float,
        is_laden: bool,
        weather_provider: Callable[[float, float, datetime], LegWeather],
        max_cells: int = DEFAULT_MAX_CELLS,
        avoid_land: bool = True,
        max_time_factor: float = 1.15,
        baseline_time_hours: Optional[float] = None,
        baseline_fuel_mt: Optional[float] = None,
        baseline_distance_nm: Optional[float] = None,
        route_waypoints: Optional[List[Tuple[float, float]]] = None,
    ) -> OptimizedRoute:
        """
        Find optimal route from origin to destination.

        If route_waypoints is provided (>2 points), optimizes each consecutive
        segment independently, respecting intermediate waypoints as via-points.

        Args:
            origin: (lat, lon) starting point
            destination: (lat, lon) ending point
            departure_time: When voyage starts
            calm_speed_kts: Calm water speed
            is_laden: Vessel loading condition
            weather_provider: Function(lat, lon, time) -> LegWeather
            max_cells: Maximum cells to explore
            avoid_land: Whether to avoid land masses
            route_waypoints: All user waypoints for multi-segment optimization

        Returns:
            OptimizedRoute with waypoints and statistics
        """
        import time
        start_time = time.time()

        self.weather_provider = weather_provider

        # Compute time-value penalty (lambda_time) for cost function.
        # Scaled by TIME_PENALTY_WEIGHT so weather-avoidance detours that add
        # modest extra time (5-15% longer) can still prove fuel-optimal.
        service_speed = (
            self.vessel_model.specs.service_speed_laden if is_laden
            else self.vessel_model.specs.service_speed_ballast
        )
        service_fuel_result = self.vessel_model.calculate_fuel_consumption(
            speed_kts=service_speed,
            is_laden=is_laden,
            weather=None,
            distance_nm=service_speed,  # 1 hour at service speed
        )
        self._lambda_time = service_fuel_result['fuel_mt'] * self.TIME_PENALTY_WEIGHT

        # Build grid around origin-destination corridor and run A*
        if self.variable_resolution:
            path, cells_explored = self._run_variable_resolution_astar(
                origin=origin,
                destination=destination,
                departure_time=departure_time,
                calm_speed_kts=calm_speed_kts,
                is_laden=is_laden,
                max_cells=max_cells,
            )
        else:
            grid = self._build_grid([origin, destination])

            # Inject strait waypoints as direct edges into the grid
            self._inject_strait_edges(grid)

            start_cell = self._get_cell(origin[0], origin[1], grid)
            end_cell = self._get_cell(destination[0], destination[1], grid)

            if start_cell is None or end_cell is None:
                raise ValueError("Origin or destination outside grid bounds")

            path, cells_explored = self._astar_search(
                start_cell=start_cell,
                end_cell=end_cell,
                grid=grid,
                departure_time=departure_time,
                calm_speed_kts=calm_speed_kts,
                is_laden=is_laden,
                max_cells=max_cells,
            )

        if path is None:
            raise ValueError(f"No route found after exploring {cells_explored} cells")

        waypoints = [(cell.lat, cell.lon) for cell in path]
        waypoints = self._smooth_path(waypoints)

        # Pin endpoints to actual origin/destination (grid cells may be offset)
        waypoints[0] = origin
        waypoints[-1] = destination

        # "Direct" route = user's original waypoints if provided, else straight line
        direct_wps = list(route_waypoints) if route_waypoints and len(route_waypoints) > 2 else [origin, destination]

        # Calculate direct route for comparison (constant speed to match voyage calculator)
        direct_fuel, direct_time, direct_distance, _, _, _ = self._calculate_route_stats(
            direct_wps, departure_time, calm_speed_kts, is_laden, use_variable_speed=False
        )

        # ── Scenario 1: Constant Speed (same calm_speed_kts on optimized path) ──
        cs_fuel, cs_time, cs_dist, cs_legs, cs_safety, cs_speeds = self._calculate_route_stats(
            waypoints, departure_time, calm_speed_kts, is_laden, use_variable_speed=False
        )

        # ── Scenario 2: Match ETA (slow-steam to match baseline or direct time) ──
        # Use baseline time if provided (from voyage calculation), else direct_time * max_time_factor
        eta_target_time = baseline_time_hours if baseline_time_hours is not None else direct_time * max_time_factor
        eta_fuel, eta_time, eta_dist, eta_legs, eta_safety, eta_speeds = (
            self._calculate_route_stats_time_constrained(
                waypoints, departure_time, calm_speed_kts, is_laden, eta_target_time
            )
        )

        optimization_time_ms = (time.time() - start_time) * 1000

        # Use baseline values for savings calculation if provided, else use direct route
        ref_fuel = baseline_fuel_mt if baseline_fuel_mt is not None else direct_fuel
        ref_time = baseline_time_hours if baseline_time_hours is not None else direct_time
        ref_dist = baseline_distance_nm if baseline_distance_nm is not None else direct_distance

        # Build scenarios
        scenarios = []

        # Scenario 1: Constant Speed
        cs_fuel_savings = ((ref_fuel - cs_fuel) / ref_fuel * 100) if ref_fuel > 0 else 0
        cs_time_savings = ((ref_time - cs_time) / ref_time * 100) if ref_time > 0 else 0
        scenarios.append(SpeedScenario(
            strategy="constant_speed",
            label="Same Speed",
            total_fuel_mt=cs_fuel,
            total_time_hours=cs_time,
            total_distance_nm=cs_dist,
            avg_speed_kts=cs_dist / cs_time if cs_time > 0 else calm_speed_kts,
            speed_profile=cs_speeds,
            leg_details=cs_legs,
            fuel_savings_pct=cs_fuel_savings,
            time_savings_pct=cs_time_savings,
        ))

        # Scenario 2: Match ETA
        eta_fuel_savings = ((ref_fuel - eta_fuel) / ref_fuel * 100) if ref_fuel > 0 else 0
        eta_time_savings = ((ref_time - eta_time) / ref_time * 100) if ref_time > 0 else 0
        scenarios.append(SpeedScenario(
            strategy="match_eta",
            label="Match ETA",
            total_fuel_mt=eta_fuel,
            total_time_hours=eta_time,
            total_distance_nm=eta_dist,
            avg_speed_kts=eta_dist / eta_time if eta_time > 0 else calm_speed_kts,
            speed_profile=eta_speeds,
            leg_details=eta_legs,
            fuel_savings_pct=eta_fuel_savings,
            time_savings_pct=eta_time_savings,
        ))

        # Default top-level fields use Constant Speed scenario for backward compat
        opt_fuel = cs_fuel
        opt_time = cs_time
        opt_distance = cs_dist
        leg_details = cs_legs
        speed_profile = cs_speeds
        safety_summary = cs_safety

        # Calculate savings vs direct route (top-level fields always vs direct)
        fuel_savings = ((direct_fuel - opt_fuel) / direct_fuel * 100) if direct_fuel > 0 else 0
        time_savings = ((direct_time - opt_time) / direct_time * 100) if direct_time > 0 else 0
        avg_speed = opt_distance / opt_time if opt_time > 0 else calm_speed_kts

        return OptimizedRoute(
            waypoints=waypoints,
            total_fuel_mt=opt_fuel,
            total_time_hours=opt_time,
            total_distance_nm=opt_distance,
            direct_fuel_mt=direct_fuel,
            direct_time_hours=direct_time,
            fuel_savings_pct=fuel_savings,
            time_savings_pct=time_savings,
            leg_details=leg_details,
            speed_profile=speed_profile,
            avg_speed_kts=avg_speed,
            safety_status=safety_summary['status'],
            safety_warnings=safety_summary['warnings'],
            max_roll_deg=safety_summary['max_roll_deg'],
            max_pitch_deg=safety_summary['max_pitch_deg'],
            max_accel_ms2=safety_summary['max_accel_ms2'],
            grid_resolution_deg=self.resolution_deg,
            cells_explored=cells_explored,
            optimization_time_ms=optimization_time_ms,
            variable_speed_enabled=self.variable_speed,
            scenarios=scenarios,
            baseline_fuel_mt=ref_fuel,
            baseline_time_hours=ref_time,
            baseline_distance_nm=ref_dist,
        )

    def optimize_route_pareto(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        departure_time: datetime,
        calm_speed_kts: float,
        is_laden: bool,
        weather_provider: Callable[[float, float, datetime], LegWeather],
        max_cells: int = DEFAULT_MAX_CELLS,
        avoid_land: bool = True,
        max_time_factor: float = 1.15,
        baseline_time_hours: Optional[float] = None,
        baseline_fuel_mt: Optional[float] = None,
        baseline_distance_nm: Optional[float] = None,
        route_waypoints: Optional[List[Tuple[float, float]]] = None,
        lambda_values: Optional[List[float]] = None,
    ) -> OptimizedRoute:
        """
        Run A* with multiple lambda (time penalty) values, return Pareto-optimal set.

        Builds the grid ONCE, then runs A* with each lambda value.
        Filters dominated solutions to return only the Pareto front.
        Default selection: lambda closest to 0.3 (current default).

        Args:
            lambda_values: Time penalty weights to sweep. Default: [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
            Other args: same as optimize_route()

        Returns:
            OptimizedRoute with pareto_front populated
        """
        import time
        start_time = time.time()

        if lambda_values is None:
            # Fuel scales ~speed³, so low lambdas always favour slow speeds.
            # Dense mid-range (1–5) fills the gap between slow-steaming and
            # service speed; high values (10–20) ensure at least one solution
            # near the user's calm_speed_kts.
            lambda_values = [0.0, 0.3, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0]

        self.weather_provider = weather_provider

        # Compute service-speed fuel rate (used to scale lambda)
        service_speed = (
            self.vessel_model.specs.service_speed_laden if is_laden
            else self.vessel_model.specs.service_speed_ballast
        )
        service_fuel_result = self.vessel_model.calculate_fuel_consumption(
            speed_kts=service_speed,
            is_laden=is_laden,
            weather=None,
            distance_nm=service_speed,
        )
        base_fuel_rate = service_fuel_result['fuel_mt']

        # Build grid ONCE
        grid = self._build_grid([origin, destination])
        self._inject_strait_edges(grid)

        start_cell = self._get_cell(origin[0], origin[1], grid)
        end_cell = self._get_cell(destination[0], destination[1], grid)

        if start_cell is None or end_cell is None:
            raise ValueError("Origin or destination outside grid bounds")

        # Run A* with each lambda value
        raw_solutions = []
        for lam in lambda_values:
            self._lambda_time = base_fuel_rate * lam

            path, cells_explored = self._astar_search(
                start_cell=start_cell,
                end_cell=end_cell,
                grid=grid,
                departure_time=departure_time,
                calm_speed_kts=calm_speed_kts,
                is_laden=is_laden,
                max_cells=max_cells,
            )

            if path is None:
                continue

            waypoints = [(cell.lat, cell.lon) for cell in path]
            waypoints = self._smooth_path(waypoints)
            waypoints[0] = origin
            waypoints[-1] = destination

            # Calculate stats with variable speed
            fuel, time_h, distance, _, _, speeds = self._calculate_route_stats(
                waypoints, departure_time, calm_speed_kts, is_laden,
                use_variable_speed=True,
            )

            raw_solutions.append(ParetoSolution(
                lambda_value=lam,
                fuel_mt=fuel,
                time_hours=time_h,
                distance_nm=distance,
                waypoints=waypoints,
                speed_profile=speeds,
            ))

        # Filter to Pareto front
        pareto = self._pareto_filter(raw_solutions)

        # Mark default selection: closest to baseline voyage in normalised fuel-time space.
        # Falls back to highest-lambda solution (nearest service speed) if no baseline.
        if pareto:
            ref_fuel = baseline_fuel_mt
            ref_time = baseline_time_hours
            if ref_fuel and ref_time:
                fuel_vals = [p.fuel_mt for p in pareto]
                time_vals = [p.time_hours for p in pareto]
                fuel_range = max(fuel_vals) - min(fuel_vals) or 1.0
                time_range = max(time_vals) - min(time_vals) or 1.0
                default_idx = min(
                    range(len(pareto)),
                    key=lambda i: (
                        ((pareto[i].fuel_mt - ref_fuel) / fuel_range) ** 2
                        + ((pareto[i].time_hours - ref_time) / time_range) ** 2
                    ),
                )
            else:
                # No baseline — pick highest lambda (closest to service speed)
                default_idx = max(range(len(pareto)), key=lambda i: pareto[i].lambda_value)
            pareto[default_idx].is_selected = True

        # Use selected solution for top-level fields
        selected = next((p for p in pareto if p.is_selected), pareto[0] if pareto else None)

        if selected is None:
            raise ValueError("No valid Pareto solutions found")

        # Use selected point's lambda so reported fuel matches Pareto chart
        self._lambda_time = base_fuel_rate * selected.lambda_value

        # Calculate direct route for comparison
        direct_wps = list(route_waypoints) if route_waypoints and len(route_waypoints) > 2 else [origin, destination]
        direct_fuel, direct_time, direct_distance, _, _, _ = self._calculate_route_stats(
            direct_wps, departure_time, calm_speed_kts, is_laden, use_variable_speed=False
        )

        # Full stats for selected solution
        opt_fuel, opt_time, opt_distance, leg_details, safety_summary, speed_profile = (
            self._calculate_route_stats(
                selected.waypoints, departure_time, calm_speed_kts, is_laden,
                use_variable_speed=True,
            )
        )

        optimization_time_ms = (time.time() - start_time) * 1000

        fuel_savings = ((direct_fuel - opt_fuel) / direct_fuel * 100) if direct_fuel > 0 else 0
        time_savings = ((direct_time - opt_time) / direct_time * 100) if direct_time > 0 else 0
        avg_speed = opt_distance / opt_time if opt_time > 0 else calm_speed_kts

        return OptimizedRoute(
            waypoints=selected.waypoints,
            total_fuel_mt=opt_fuel,
            total_time_hours=opt_time,
            total_distance_nm=opt_distance,
            direct_fuel_mt=direct_fuel,
            direct_time_hours=direct_time,
            fuel_savings_pct=fuel_savings,
            time_savings_pct=time_savings,
            leg_details=leg_details,
            speed_profile=speed_profile,
            avg_speed_kts=avg_speed,
            safety_status=safety_summary['status'],
            safety_warnings=safety_summary['warnings'],
            max_roll_deg=safety_summary['max_roll_deg'],
            max_pitch_deg=safety_summary['max_pitch_deg'],
            max_accel_ms2=safety_summary['max_accel_ms2'],
            grid_resolution_deg=self.resolution_deg,
            cells_explored=cells_explored,
            optimization_time_ms=optimization_time_ms,
            variable_speed_enabled=self.variable_speed,
            pareto_front=pareto,
            baseline_fuel_mt=baseline_fuel_mt or direct_fuel,
            baseline_time_hours=baseline_time_hours or direct_time,
            baseline_distance_nm=baseline_distance_nm or direct_distance,
        )

    @staticmethod
    def _pareto_filter(solutions: List[ParetoSolution]) -> List[ParetoSolution]:
        """
        Remove dominated solutions from a list.

        A dominates B iff A.fuel <= B.fuel AND A.time <= B.time, with strict < on at least one.
        Returns the non-dominated subset, sorted by fuel ascending.
        """
        if len(solutions) <= 1:
            return list(solutions)

        non_dominated = []
        for candidate in solutions:
            dominated = False
            for other in solutions:
                if other is candidate:
                    continue
                if (other.fuel_mt <= candidate.fuel_mt and
                    other.time_hours <= candidate.time_hours and
                    (other.fuel_mt < candidate.fuel_mt or other.time_hours < candidate.time_hours)):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(candidate)

        return sorted(non_dominated, key=lambda s: s.fuel_mt)

    def _run_variable_resolution_astar(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        departure_time: datetime,
        calm_speed_kts: float,
        is_laden: bool,
        max_cells: int,
    ) -> Tuple[Optional[List[GridCell]], int]:
        """
        Run A* on a variable-resolution RoutingGraph (0.5° ocean + 0.1° nearshore).

        Builds a RoutingGraph, converts nodes to GridCell-compatible format,
        and runs A* using graph.get_neighbors() instead of grid DIRECTIONS.
        """
        from src.optimization.routing_graph import RoutingGraph

        # Build variable-resolution graph
        graph = RoutingGraph([origin, destination])
        nodes = graph.build()

        if not nodes:
            raise ValueError("Variable resolution graph has no nodes")

        # Convert to GridCell dict for A* compatibility
        grid: Dict[Tuple[int, int], GridCell] = {}
        node_to_key: Dict[str, Tuple[int, int]] = {}
        key_to_node_id: Dict[Tuple[int, int], str] = {}

        for i, (node_id, node) in enumerate(nodes.items()):
            key = (i, 0)
            grid[key] = GridCell(lat=node.lat, lon=node.lon, row=i, col=0)
            node_to_key[node_id] = key
            key_to_node_id[key] = node_id

        # Build neighbor lookup: key → list of neighbor keys
        self._vr_neighbors: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for node_id, node in nodes.items():
            key = node_to_key[node_id]
            self._vr_neighbors[key] = []
            for nid in node.neighbors:
                if nid in node_to_key:
                    self._vr_neighbors[key].append(node_to_key[nid])

        # Find start/end cells
        start_node = graph.get_nearest_node(origin[0], origin[1])
        end_node = graph.get_nearest_node(destination[0], destination[1])

        if start_node is None or end_node is None:
            raise ValueError("Origin or destination not found in variable resolution graph")

        start_key = node_to_key[start_node.id]
        end_key = node_to_key[end_node.id]
        start_cell = grid[start_key]
        end_cell = grid[end_key]

        # Run A* with variable resolution neighbor lookup
        self._use_vr_neighbors = True
        try:
            path, cells_explored = self._astar_search(
                start_cell=start_cell,
                end_cell=end_cell,
                grid=grid,
                departure_time=departure_time,
                calm_speed_kts=calm_speed_kts,
                is_laden=is_laden,
                max_cells=max_cells,
            )
        finally:
            self._use_vr_neighbors = False

        return path, cells_explored

    def _build_grid(
        self,
        corridor_waypoints: List[Tuple[float, float]],
        margin_deg: float = 5.0,
        filter_land: bool = True,
    ) -> Dict[Tuple[int, int], GridCell]:
        """
        Build routing grid covering the corridor defined by waypoints.

        Computes a bounding box around ALL waypoints with margin, so A*
        can explore the full area the user intended.
        Filters out land cells if filter_land=True.
        """
        lats = [wp[0] for wp in corridor_waypoints]
        lons = [wp[1] for wp in corridor_waypoints]

        # Calculate bounding box with margin around all waypoints
        lat_min = min(lats) - margin_deg
        lat_max = max(lats) + margin_deg
        lon_min = min(lons) - margin_deg
        lon_max = max(lons) + margin_deg

        # Expand bbox to include nearby strait waypoints so detour paths
        # (e.g. Rotterdam→Augusta via Gibraltar) are reachable.  Only include
        # straits whose waypoints are close to the original bbox in BOTH
        # dimensions — otherwise far-away straits (Bosporus, Messina) bloat
        # the grid for routes that don't need them.
        from src.data.strait_waypoints import STRAITS
        expand_margin = margin_deg  # how far outside the bbox a strait can be
        for strait in STRAITS:
            for wlat, wlon in strait.waypoints:
                if (lat_min - expand_margin <= wlat <= lat_max + expand_margin
                        and lon_min - expand_margin <= wlon <= lon_max + expand_margin):
                    lon_min = min(lon_min, wlon - 1.0)
                    lon_max = max(lon_max, wlon + 1.0)
                    lat_min = min(lat_min, wlat - 1.0)
                    lat_max = max(lat_max, wlat + 1.0)

        # Clamp to valid ranges
        lat_min = max(lat_min, -85)
        lat_max = min(lat_max, 85)

        # Handle antimeridian crossing
        if lon_max - lon_min > 180:
            # Route crosses antimeridian - expand to cover
            lon_min, lon_max = -180, 180

        # Build grid
        grid = {}
        land_cells = 0
        row = 0
        lat = lat_min
        while lat <= lat_max:
            col = 0
            lon = lon_min
            while lon <= lon_max:
                # Check if cell is ocean (skip land cells)
                if filter_land and not is_ocean(lat, lon):
                    land_cells += 1
                else:
                    cell = GridCell(lat=lat, lon=lon, row=row, col=col)
                    grid[(row, col)] = cell
                lon += self.resolution_deg
                col += 1
            lat += self.resolution_deg
            row += 1

        total_cells = row * col if row > 0 and col > 0 else 1
        logger.info(f"Built grid: {len(grid)} ocean cells, {land_cells} land cells filtered "
                   f"({row} rows x {col} cols, {land_cells/total_cells*100:.1f}% land)")

        return grid

    def _inject_strait_edges(
        self,
        grid: Dict[Tuple[int, int], GridCell],
        threshold_deg: float = 1.0,
    ) -> int:
        """
        Inject strait waypoints into the routing grid as additional cells.

        For each strait whose waypoints fall within the grid bbox,
        adds waypoint cells and connects them to the nearest existing
        grid node within threshold_deg. Strait edges between consecutive
        waypoints are pre-validated and skip is_path_clear during A*.

        Returns the number of strait nodes injected.
        """
        if not grid:
            return 0

        # Get grid bounding box
        lats = [c.lat for c in grid.values()]
        lons = [c.lon for c in grid.values()]
        lat_min, lat_max = min(lats), max(lats)
        lon_min, lon_max = min(lons), max(lons)

        # Track highest row/col for new node IDs
        max_row = max(k[0] for k in grid)
        max_col = max(k[1] for k in grid)
        next_row = max_row + 100  # Offset to avoid collisions

        injected = 0
        self._strait_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()

        for strait in STRAITS:
            # Check if any waypoint is within grid bbox (with margin)
            in_bbox = any(
                lat_min - 1 <= wlat <= lat_max + 1 and
                lon_min - 1 <= wlon <= lon_max + 1
                for wlat, wlon in strait.waypoints
            )
            if not in_bbox:
                continue

            strait_cells = []
            for i, (wlat, wlon) in enumerate(strait.waypoints):
                # Create a new grid cell for this waypoint
                row_id = next_row + injected
                col_id = 0
                key = (row_id, col_id)

                cell = GridCell(lat=wlat, lon=wlon, row=row_id, col=col_id)
                grid[key] = cell
                strait_cells.append(cell)
                injected += 1

                # Connect to nearest existing grid node
                min_dist_sq = threshold_deg ** 2
                nearest_key = None
                for gk, gc in grid.items():
                    if gk == key:
                        continue
                    dist_sq = (gc.lat - wlat) ** 2 + (gc.lon - wlon) ** 2
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        nearest_key = gk

                if nearest_key is not None:
                    # Mark bidirectional edge (strait node ↔ grid node)
                    self._strait_edges.add((key, nearest_key))
                    self._strait_edges.add((nearest_key, key))

            # Mark edges between consecutive strait waypoints (pre-validated)
            for i in range(len(strait_cells) - 1):
                k1 = (strait_cells[i].row, strait_cells[i].col)
                k2 = (strait_cells[i + 1].row, strait_cells[i + 1].col)
                self._strait_edges.add((k1, k2))
                if strait.bidirectional:
                    self._strait_edges.add((k2, k1))

        # Build adjacency map for O(1) neighbor lookup during A*
        self._strait_neighbor_map: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for from_k, to_k in self._strait_edges:
            self._strait_neighbor_map.setdefault(from_k, []).append(to_k)

        if injected > 0:
            logger.info(f"Injected {injected} strait waypoints ({len(self._strait_edges)} edges)")

        return injected

    def _is_strait_edge(self, from_key: Tuple[int, int], to_key: Tuple[int, int]) -> bool:
        """Check if an edge is a pre-validated strait edge (skips is_path_clear)."""
        return hasattr(self, '_strait_edges') and (from_key, to_key) in self._strait_edges

    def _get_cell(
        self,
        lat: float,
        lon: float,
        grid: Dict[Tuple[int, int], GridCell],
    ) -> Optional[GridCell]:
        """Find grid cell containing a point."""
        # Get any cell to find grid bounds
        sample_cell = next(iter(grid.values()))

        # Find row and column
        for (r, c), cell in grid.items():
            if (abs(cell.lat - lat) < self.resolution_deg / 2 and
                abs(cell.lon - lon) < self.resolution_deg / 2):
                return cell

        # Find closest cell
        min_dist = float('inf')
        closest = None
        for cell in grid.values():
            dist = (cell.lat - lat) ** 2 + (cell.lon - lon) ** 2
            if dist < min_dist:
                min_dist = dist
                closest = cell

        return closest

    def _astar_search(
        self,
        start_cell: GridCell,
        end_cell: GridCell,
        grid: Dict[Tuple[int, int], GridCell],
        departure_time: datetime,
        calm_speed_kts: float,
        is_laden: bool,
        max_cells: int,
    ) -> Tuple[Optional[List[GridCell]], int]:
        """
        A* search for optimal route.

        Returns:
            Tuple of (path as list of cells, number of cells explored)
        """
        # Priority queue: (f_score, node)
        open_set = []

        # Start node
        start_node = SearchNode(
            f_score=self._heuristic(start_cell, end_cell),
            cell=start_cell,
            g_score=0.0,
            arrival_time=departure_time,
            parent=None,
        )
        heapq.heappush(open_set, start_node)

        # Best g_score for each cell
        g_scores: Dict[Tuple[int, int], float] = {
            (start_cell.row, start_cell.col): 0.0
        }

        # Cells already fully explored
        closed_set: Set[Tuple[int, int]] = set()

        cells_explored = 0

        while open_set and cells_explored < max_cells:
            current = heapq.heappop(open_set)
            current_key = (current.cell.row, current.cell.col)

            # Skip if already explored with better score
            if current_key in closed_set:
                continue

            closed_set.add(current_key)
            cells_explored += 1

            # Check if reached destination
            if current.cell == end_cell:
                # Reconstruct path
                path = []
                node = current
                while node is not None:
                    path.append(node.cell)
                    node = node.parent
                path.reverse()
                return path, cells_explored

            # Explore neighbors: variable resolution graph OR grid directions + strait edges
            neighbor_keys = []
            if getattr(self, '_use_vr_neighbors', False) and hasattr(self, '_vr_neighbors'):
                # Variable resolution mode: use graph-based neighbors
                neighbor_keys = list(self._vr_neighbors.get(current_key, []))
            else:
                # Uniform grid mode: 16 directions
                for dr, dc in self.DIRECTIONS:
                    neighbor_keys.append((current.cell.row + dr, current.cell.col + dc))

                # Add strait edge neighbors (pre-indexed for O(1) lookup)
                if hasattr(self, '_strait_neighbor_map') and current_key in self._strait_neighbor_map:
                    neighbor_set = set(neighbor_keys)
                    for to_k in self._strait_neighbor_map[current_key]:
                        if to_k not in neighbor_set:
                            neighbor_keys.append(to_k)

            # Compute previous bearing once per expansion (incoming heading to current node)
            prev_bearing = None
            if current.parent is not None:
                prev_bearing = self.bearing(
                    current.parent.cell.lat, current.parent.cell.lon,
                    current.cell.lat, current.cell.lon,
                )

            for neighbor_key in neighbor_keys:
                if neighbor_key not in grid:
                    continue
                if neighbor_key in closed_set:
                    continue

                neighbor_cell = grid[neighbor_key]

                # Strait edges skip is_path_clear (pre-validated)
                is_strait = self._is_strait_edge(current_key, neighbor_key)
                if not is_strait:
                    if not is_path_clear(current.cell.lat, current.cell.lon,
                                         neighbor_cell.lat, neighbor_cell.lon):
                        continue

                # Calculate cost to move to neighbor
                move_cost, travel_time = self._calculate_move_cost(
                    from_cell=current.cell,
                    to_cell=neighbor_cell,
                    departure_time=current.arrival_time,
                    calm_speed_kts=calm_speed_kts,
                    is_laden=is_laden,
                    prev_bearing=prev_bearing,
                )

                if move_cost == float('inf'):
                    continue  # Impassable (land or extreme weather)

                tentative_g = current.g_score + move_cost

                # Check if this is a better path
                if tentative_g < g_scores.get(neighbor_key, float('inf')):
                    g_scores[neighbor_key] = tentative_g

                    neighbor_node = SearchNode(
                        f_score=tentative_g + self._heuristic(neighbor_cell, end_cell),
                        cell=neighbor_cell,
                        g_score=tentative_g,
                        arrival_time=current.arrival_time + timedelta(hours=travel_time),
                        parent=current,
                    )
                    heapq.heappush(open_set, neighbor_node)

        # No path found
        return None, cells_explored

    def _heuristic(self, cell: GridCell, goal: GridCell) -> float:
        """
        A* heuristic: estimated cost from cell to goal.

        Uses great circle distance with best-case fuel consumption.
        Must be admissible (never overestimate actual cost).
        """
        # Great circle distance
        distance_nm = self.haversine(
            cell.lat, cell.lon, goal.lat, goal.lon
        )

        if self.optimization_target == "time":
            # Best case: calm water speed
            return distance_nm / self.vessel_model.specs.service_speed_laden
        else:
            # Best case: calm water fuel consumption (underestimate)
            service_speed = self.vessel_model.specs.service_speed_laden
            result = self.vessel_model.calculate_fuel_consumption(
                speed_kts=service_speed,
                is_laden=True,
                weather=None,
                distance_nm=distance_nm,
            )
            fuel_heuristic = result['fuel_mt'] * 0.8  # Underestimate for admissibility

            # Time component: generous speed estimate ensures underestimate
            time_heuristic = distance_nm / (service_speed + 2.0)
            return fuel_heuristic + self._lambda_time * time_heuristic

    # Course-change penalty: penalises heading changes between consecutive
    # edges to produce smoother, more realistic tracks.  The penalty is
    # additive (fuel-equivalent cost in MT) scaled by the turn angle.
    #   0-15°  → 0        (straight or near-straight — no penalty)
    #  15-45°  → linear ramp 0 → 0.02 MT
    #  45-90°  → linear ramp 0.02 → 0.08 MT
    #  90-180° → linear ramp 0.08 → 0.20 MT
    # Values calibrated so a 90° turn ≈ 5% of a typical cell transit cost.
    COURSE_CHANGE_PENALTY_MT = 0.20   # Max penalty at 180° reversal

    @staticmethod
    def _course_change_penalty(prev_bearing: float, cur_bearing: float) -> float:
        """Return additive fuel-equivalent penalty (MT) for a heading change."""
        turn = abs(((cur_bearing - prev_bearing) + 180) % 360 - 180)
        if turn <= 15.0:
            return 0.0
        if turn <= 45.0:
            return 0.02 * (turn - 15.0) / 30.0
        if turn <= 90.0:
            return 0.02 + 0.06 * (turn - 45.0) / 45.0
        return 0.08 + 0.12 * (turn - 90.0) / 90.0

    def _calculate_move_cost(
        self,
        from_cell: GridCell,
        to_cell: GridCell,
        departure_time: datetime,
        calm_speed_kts: float,
        is_laden: bool,
        prev_bearing: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Calculate cost to move from one cell to another.

        Args:
            prev_bearing: Heading of the previous edge (degrees). If provided,
                          a course-change penalty is added for sharp turns.

        Returns:
            Tuple of (cost, travel_time_hours)
        """
        # Calculate distance
        distance_nm = self.haversine(
            from_cell.lat, from_cell.lon, to_cell.lat, to_cell.lon
        )

        # Get weather at midpoint
        mid_lat = (from_cell.lat + to_cell.lat) / 2
        mid_lon = (from_cell.lon + to_cell.lon) / 2
        mid_time = departure_time + timedelta(hours=distance_nm / calm_speed_kts / 2)

        try:
            weather = self.weather_provider(mid_lat, mid_lon, mid_time)
        except Exception as e:
            logger.warning(f"Weather fetch failed at ({mid_lat}, {mid_lon}): {e}")
            weather = LegWeather()  # Calm conditions fallback

        # Calculate bearing
        bearing = self.bearing(
            from_cell.lat, from_cell.lon, to_cell.lat, to_cell.lon
        )

        # SPEC-P1: Ice exclusion and penalty zones
        ICE_EXCLUSION_THRESHOLD = 0.15  # IMO Polar Code limit
        ICE_PENALTY_THRESHOLD = 0.05   # Caution zone
        if weather.ice_concentration >= ICE_EXCLUSION_THRESHOLD:
            return float('inf'), float('inf')
        ice_cost_factor = 2.0 if weather.ice_concentration >= ICE_PENALTY_THRESHOLD else 1.0

        # Build weather dict for vessel model
        weather_dict = {
            'wind_speed_ms': weather.wind_speed_ms,
            'wind_dir_deg': weather.wind_dir_deg,
            'heading_deg': bearing,
            'sig_wave_height_m': weather.sig_wave_height_m,
            'wave_dir_deg': weather.wave_dir_deg,
        }

        # SPEC-P1: Visibility speed cap — IMO COLREG Rule 6
        effective_speed_kts = calm_speed_kts
        effective_speed_kts = apply_visibility_cap(effective_speed_kts, weather.visibility_km * 1000.0)

        # Calculate fuel consumption
        result = self.vessel_model.calculate_fuel_consumption(
            speed_kts=effective_speed_kts,
            is_laden=is_laden,
            weather=weather_dict,
            distance_nm=distance_nm,
        )

        # Calculate actual travel time considering current (SOG = STW + current projection)
        current_effect = self.current_effect(
            heading_deg=bearing,
            current_speed_ms=weather.current_speed_ms,
            current_dir_deg=weather.current_dir_deg,
        )

        # SPEC-P1: Cross-current drift correction
        # Compute lateral current component for drift penalty
        relative_angle_rad = math.radians(
            abs(((weather.current_dir_deg - bearing) + 180) % 360 - 180)
        )
        current_kts = weather.current_speed_ms * 1.94384
        cross_current_kts = abs(current_kts * math.sin(relative_angle_rad))
        # Drift penalty: extra distance needed to compensate for lateral set
        drift_factor = 1.0
        if cross_current_kts > 0.5 and effective_speed_kts > 0:
            drift_ratio = cross_current_kts / effective_speed_kts
            drift_factor = 1.0 / max(math.sqrt(1.0 - min(drift_ratio, 0.95) ** 2), 0.1)

        sog_kts = effective_speed_kts + current_effect
        if sog_kts <= 0:
            return float('inf'), float('inf')  # Can't make headway

        travel_time_hours = (distance_nm * drift_factor) / sog_kts

        # Apply safety constraints
        safety_factor = 1.0
        if self.enforce_safety and weather.sig_wave_height_m > 0:
            # Use actual wave period from data, fallback to estimate if not available
            wave_period_s = weather.wave_period_s if weather.wave_period_s > 0 else (5.0 + weather.sig_wave_height_m)
            safety_factor = self.safety_constraints.get_safety_cost_factor(
                wave_height_m=weather.sig_wave_height_m,
                wave_period_s=wave_period_s,
                wave_dir_deg=weather.wave_dir_deg,
                heading_deg=bearing,
                speed_kts=calm_speed_kts,
                is_laden=is_laden,
                wind_speed_kts=weather.wind_speed_ms * 1.9438,
                skip_hard_limits=self.skip_hard_limits,
            )
            if safety_factor == float('inf'):
                return float('inf'), float('inf')  # Dangerous - forbidden

        # Apply regulatory zone penalties
        zone_factor = 1.0
        if self.enforce_zones:
            zone_penalty, _ = self.zone_checker.get_path_penalty(
                from_cell.lat, from_cell.lon, to_cell.lat, to_cell.lon
            )
            if zone_penalty == float('inf'):
                return float('inf'), float('inf')  # Exclusion zone - forbidden
            zone_factor = zone_penalty

        # Dampen safety factor by safety_weight
        if safety_factor == float('inf'):
            dampened_sf = float('inf')  # hard constraint always applies
        elif self.safety_weight > 0 and safety_factor > 1.0:
            dampened_sf = safety_factor ** self.safety_weight
        else:
            dampened_sf = 1.0

        # Combined cost factor (includes ice caution zone penalty)
        total_factor = dampened_sf * zone_factor * ice_cost_factor

        # Course-change penalty (additive, fuel-equivalent MT)
        turn_penalty = 0.0
        if prev_bearing is not None:
            turn_penalty = self._course_change_penalty(prev_bearing, bearing)

        # Return cost based on optimization target
        if self.optimization_target == "time":
            return travel_time_hours * total_factor + turn_penalty, travel_time_hours
        else:
            # Time-constrained fuel minimization:
            # fuel cost + time penalty prevents detours that save marginal fuel
            fuel_cost = result['fuel_mt'] * total_factor
            time_penalty = self._lambda_time * travel_time_hours
            return fuel_cost + time_penalty + turn_penalty, travel_time_hours

    def _find_optimal_speed(
        self,
        distance_nm: float,
        weather: LegWeather,
        bearing_deg: float,
        is_laden: bool,
        target_time_hours: Optional[float] = None,
        calm_speed_kts: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """
        Find optimal speed for a leg considering weather conditions.

        For fuel optimization: finds speed that minimizes fuel per mile
        For time-constrained: finds speed that meets ETA with minimum fuel

        Args:
            distance_nm: Leg distance
            weather: Weather conditions
            bearing_deg: Vessel heading
            is_laden: Loading condition
            target_time_hours: Optional target time for this leg
            calm_speed_kts: User's calm-water speed (caps max STW)

        Returns:
            Tuple of (optimal_speed_kts, fuel_mt, time_hours)
        """
        min_speed, max_speed = self.SPEED_RANGE_KTS

        # Cap at user's calm speed — optimizer finds better paths, not faster engines
        if calm_speed_kts is not None:
            max_speed = min(max_speed, calm_speed_kts)
        else:
            if is_laden:
                max_speed = min(max_speed, self.vessel_model.specs.service_speed_laden)
            else:
                max_speed = min(max_speed, self.vessel_model.specs.service_speed_ballast)

        # Build weather dict
        weather_dict = {
            'wind_speed_ms': weather.wind_speed_ms,
            'wind_dir_deg': weather.wind_dir_deg,
            'heading_deg': bearing_deg,
            'sig_wave_height_m': weather.sig_wave_height_m,
            'wave_dir_deg': weather.wave_dir_deg,
        }

        # Calculate current effect (constant for all speeds)
        current_effect = self.current_effect(
            heading_deg=bearing_deg,
            current_speed_ms=weather.current_speed_ms,
            current_dir_deg=weather.current_dir_deg,
        )

        # Test speeds and find optimal
        speeds_to_test = np.linspace(min_speed, max_speed, self.SPEED_STEPS)
        best_speed = min_speed
        best_fuel = float('inf')
        best_time = float('inf')
        best_score = float('inf')

        results = []

        for speed_kts in speeds_to_test:
            # Calculate fuel consumption at this speed
            result = self.vessel_model.calculate_fuel_consumption(
                speed_kts=speed_kts,
                is_laden=is_laden,
                weather=weather_dict,
                distance_nm=distance_nm,
            )

            # Calculate SOG and time
            sog_kts = speed_kts + current_effect
            if sog_kts <= 1.0:
                continue  # Can't make meaningful progress

            time_hours = distance_nm / sog_kts
            fuel_mt = result['fuel_mt']

            # Calculate score based on optimization target
            if self.optimization_target == "time":
                # Minimize time, but check fuel penalty
                score = time_hours
            else:
                # Time-constrained fuel minimization: trade off fuel vs time
                # This picks 12-14 kts typically, not the slowest possible speed
                score = fuel_mt + self._lambda_time * time_hours

            # Apply safety penalty for high speeds in heavy weather
            if self.enforce_safety and weather.sig_wave_height_m > 2.0:
                wave_period_s = weather.wave_period_s if weather.wave_period_s > 0 else (5.0 + weather.sig_wave_height_m)
                safety_factor = self.safety_constraints.get_safety_cost_factor(
                    wave_height_m=weather.sig_wave_height_m,
                    wave_period_s=wave_period_s,
                    wave_dir_deg=weather.wave_dir_deg,
                    heading_deg=bearing_deg,
                    speed_kts=speed_kts,
                    is_laden=is_laden,
                    wind_speed_kts=weather.wind_speed_ms * 1.9438,
                    skip_hard_limits=self.skip_hard_limits,
                )
                if safety_factor == float('inf'):
                    continue  # Skip dangerous speeds
                # Dampen safety penalty by safety_weight
                if self.safety_weight > 0 and safety_factor > 1.0:
                    score *= safety_factor ** self.safety_weight
                elif self.safety_weight <= 0:
                    pass  # no penalty
                else:
                    score *= safety_factor

            results.append((speed_kts, fuel_mt, time_hours, score))

            if score < best_score:
                best_score = score
                best_speed = speed_kts
                best_fuel = fuel_mt
                best_time = time_hours

        # Fallback if no valid speed found (all skipped by safety or SOG)
        if not results or best_time == float('inf'):
            fallback_sog = max(min_speed + current_effect, 0.5)
            # Estimate fuel at min speed (calm conditions) rather than returning 0.0
            fallback_result = self.vessel_model.calculate_fuel_consumption(
                speed_kts=min_speed, is_laden=is_laden,
                weather=weather_dict, distance_nm=distance_nm,
            )
            return min_speed, fallback_result['fuel_mt'], distance_nm / fallback_sog

        # If we have a target time constraint, adjust speed
        if target_time_hours is not None and best_time > target_time_hours:
            # Need to go faster - find minimum speed that meets target
            for speed_kts, fuel_mt, time_hours, _ in sorted(results, key=lambda x: x[2]):
                if time_hours <= target_time_hours:
                    return speed_kts, fuel_mt, time_hours
            # Can't meet target - return fastest safe option
            fastest = max(results, key=lambda x: x[0]) if results else (max_speed, best_fuel, best_time)
            return fastest[0], fastest[1], fastest[2]

        return best_speed, best_fuel, best_time

    def _smooth_path(
        self,
        waypoints: List[Tuple[float, float]],
        tolerance_nm: float = None,
        check_land: bool = True,
    ) -> List[Tuple[float, float]]:
        """
        Smooth path using Douglas-Peucker algorithm.

        Removes unnecessary waypoints while keeping path shape.
        Ensures simplified path doesn't cross land.

        Default tolerance is half a grid cell (~15nm at 0.5°).
        """
        if tolerance_nm is None:
            tolerance_nm = self.resolution_deg * 60 * 0.5  # half cell width
        if len(waypoints) <= 2:
            return waypoints

        # Douglas-Peucker simplification
        def perpendicular_distance(point, line_start, line_end):
            """Distance from point to line segment."""
            x0, y0 = point
            x1, y1 = line_start
            x2, y2 = line_end

            dx = x2 - x1
            dy = y2 - y1

            if dx == 0 and dy == 0:
                return math.sqrt((x0 - x1)**2 + (y0 - y1)**2) * 60  # Convert to nm

            t = max(0, min(1, ((x0 - x1) * dx + (y0 - y1) * dy) / (dx*dx + dy*dy)))

            proj_x = x1 + t * dx
            proj_y = y1 + t * dy

            # Approximate distance in nm
            return math.sqrt((x0 - proj_x)**2 + (y0 - proj_y)**2) * 60

        def simplify(points, epsilon):
            if len(points) <= 2:
                return points

            # Find point with maximum distance
            max_dist = 0
            max_idx = 0
            for i in range(1, len(points) - 1):
                dist = perpendicular_distance(points[i], points[0], points[-1])
                if dist > max_dist:
                    max_dist = dist
                    max_idx = i

            # If max distance > epsilon, recursively simplify
            if max_dist > epsilon:
                left = simplify(points[:max_idx + 1], epsilon)
                right = simplify(points[max_idx:], epsilon)
                return left[:-1] + right
            else:
                # Before simplifying to just endpoints, check for land crossing
                if check_land:
                    start = points[0]
                    end = points[-1]
                    if not is_path_clear(start[0], start[1], end[0], end[1]):
                        # Can't simplify - path would cross land
                        # Keep the midpoint
                        mid_idx = len(points) // 2
                        left = simplify(points[:mid_idx + 1], epsilon)
                        right = simplify(points[mid_idx:], epsilon)
                        return left[:-1] + right

                return [points[0], points[-1]]

        smoothed = simplify(waypoints, tolerance_nm)

        # Second pass: remove waypoints that create insignificant course changes
        # (grid staircase artifacts that Douglas-Peucker keeps)
        smoothed = self._remove_small_turns(smoothed, min_turn_deg=15.0, check_land=check_land)

        # Third pass: subdivide long segments to prevent Mercator rendering
        # from crossing land (straight lines diverge from geographic path)
        max_seg_nm = 120.0
        result = [smoothed[0]]
        for i in range(1, len(smoothed)):
            prev = result[-1]
            cur = smoothed[i]
            seg_nm = self.haversine(prev[0], prev[1], cur[0], cur[1])
            if seg_nm > max_seg_nm:
                n_sub = int(math.ceil(seg_nm / max_seg_nm))
                for j in range(1, n_sub):
                    t = j / n_sub
                    mid_lat = prev[0] + t * (cur[0] - prev[0])
                    mid_lon = prev[1] + t * (cur[1] - prev[1])
                    result.append((mid_lat, mid_lon))
            result.append(cur)
        return result

    def _remove_small_turns(
        self,
        waypoints: List[Tuple[float, float]],
        min_turn_deg: float = 15.0,
        check_land: bool = True,
    ) -> List[Tuple[float, float]]:
        """
        Remove waypoints that don't contribute a significant course change.

        Iterates through waypoints; if the turn angle at a waypoint is below
        min_turn_deg AND the direct path from the previous kept waypoint to the
        next one is clear of land, drop it.
        """
        if len(waypoints) <= 2:
            return waypoints

        result = [waypoints[0]]

        for i in range(1, len(waypoints) - 1):
            bearing_in = self.bearing(
                result[-1][0], result[-1][1], waypoints[i][0], waypoints[i][1]
            )
            bearing_out = self.bearing(
                waypoints[i][0], waypoints[i][1], waypoints[i + 1][0], waypoints[i + 1][1]
            )
            turn = abs(((bearing_out - bearing_in) + 180) % 360 - 180)

            if turn < min_turn_deg:
                # Small turn — check if we can skip this waypoint
                if check_land and not is_path_clear(
                    result[-1][0], result[-1][1], waypoints[i + 1][0], waypoints[i + 1][1]
                ):
                    result.append(waypoints[i])  # Keep — removing would cross land
                # else: skip this waypoint (insignificant turn, path clear)
            else:
                result.append(waypoints[i])  # Keep — genuine course change

        result.append(waypoints[-1])
        logger.info(f"Turn-angle filter: {len(waypoints)} → {len(result)} waypoints (min turn {min_turn_deg}°)")
        return result

    def _calculate_route_stats(
        self,
        waypoints: List[Tuple[float, float]],
        departure_time: datetime,
        calm_speed_kts: float,
        is_laden: bool,
        use_variable_speed: bool = None,
    ) -> Tuple[float, float, float, List[Dict], Dict, List[float]]:
        """
        Calculate total fuel, time, and distance for a route.

        Args:
            waypoints: Route waypoints
            departure_time: Departure time
            calm_speed_kts: Base speed (used if variable speed disabled)
            is_laden: Loading condition
            use_variable_speed: Override variable speed setting

        Returns:
            Tuple of (total_fuel_mt, total_time_hours, total_distance_nm, leg_details, safety_summary, speed_profile)
        """
        use_var_speed = use_variable_speed if use_variable_speed is not None else self.variable_speed

        total_fuel = 0.0
        total_time = 0.0
        total_distance = 0.0
        leg_details = []
        speed_profile = []

        # Safety tracking
        max_roll = 0.0
        max_pitch = 0.0
        max_accel = 0.0
        all_warnings = []
        worst_safety_status = SafetyStatus.SAFE

        current_time = departure_time

        for i in range(len(waypoints) - 1):
            from_wp = waypoints[i]
            to_wp = waypoints[i + 1]

            distance = self.haversine(from_wp[0], from_wp[1], to_wp[0], to_wp[1])
            bearing = self.bearing(from_wp[0], from_wp[1], to_wp[0], to_wp[1])

            # Get weather at midpoint
            mid_lat = (from_wp[0] + to_wp[0]) / 2
            mid_lon = (from_wp[1] + to_wp[1]) / 2
            mid_time = current_time + timedelta(hours=distance / calm_speed_kts / 2)

            try:
                weather = self.weather_provider(mid_lat, mid_lon, mid_time)
            except Exception:
                weather = LegWeather()

            # Variable speed optimization
            if use_var_speed:
                optimal_speed, fuel_mt, time_hours = self._find_optimal_speed(
                    distance_nm=distance,
                    weather=weather,
                    bearing_deg=bearing,
                    is_laden=is_laden,
                    calm_speed_kts=calm_speed_kts,
                )
                leg_speed = optimal_speed
            else:
                leg_speed = calm_speed_kts
                weather_dict = {
                    'wind_speed_ms': weather.wind_speed_ms,
                    'wind_dir_deg': weather.wind_dir_deg,
                    'heading_deg': bearing,
                    'sig_wave_height_m': weather.sig_wave_height_m,
                    'wave_dir_deg': weather.wave_dir_deg,
                }

                result = self.vessel_model.calculate_fuel_consumption(
                    speed_kts=calm_speed_kts,
                    is_laden=is_laden,
                    weather=weather_dict,
                    distance_nm=distance,
                )
                fuel_mt = result['fuel_mt']

                current_effect = self.current_effect(
                    bearing, weather.current_speed_ms, weather.current_dir_deg
                )
                sog = max(calm_speed_kts + current_effect, 0.1)
                time_hours = distance / sog

            speed_profile.append(leg_speed)

            # Calculate SOG for this leg
            current_effect = self.current_effect(
                bearing, weather.current_speed_ms, weather.current_dir_deg
            )
            sog = max(leg_speed + current_effect, 0.1)

            total_fuel += fuel_mt
            total_time += time_hours
            total_distance += distance

            # Safety assessment for this leg
            leg_safety = None
            if weather.sig_wave_height_m > 0:
                # Use actual wave period from data, fallback to estimate if not available
                wave_period_s = weather.wave_period_s if weather.wave_period_s > 0 else (5.0 + weather.sig_wave_height_m)
                leg_safety = self.safety_constraints.assess_safety(
                    wave_height_m=weather.sig_wave_height_m,
                    wave_period_s=wave_period_s,
                    wave_dir_deg=weather.wave_dir_deg,
                    heading_deg=bearing,
                    speed_kts=leg_speed,
                    is_laden=is_laden,
                )

                # Track worst values
                max_roll = max(max_roll, leg_safety.motions.roll_amplitude_deg)
                max_pitch = max(max_pitch, leg_safety.motions.pitch_amplitude_deg)
                max_accel = max(max_accel, leg_safety.motions.bridge_accel_ms2)

                # Collect warnings
                for warning in leg_safety.warnings:
                    if warning not in all_warnings:
                        all_warnings.append(warning)

                # Track worst status
                if leg_safety.status == SafetyStatus.DANGEROUS:
                    worst_safety_status = SafetyStatus.DANGEROUS
                elif leg_safety.status == SafetyStatus.MARGINAL and worst_safety_status != SafetyStatus.DANGEROUS:
                    worst_safety_status = SafetyStatus.MARGINAL

            leg_details.append({
                'from': from_wp,
                'to': to_wp,
                'distance_nm': distance,
                'bearing_deg': bearing,
                'fuel_mt': fuel_mt,
                'time_hours': time_hours,
                'sog_kts': sog,
                'stw_kts': leg_speed,  # Speed through water (optimized)
                'wind_speed_ms': weather.wind_speed_ms,
                'wave_height_m': weather.sig_wave_height_m,
                'safety_status': leg_safety.status.value if leg_safety else 'safe',
                'roll_deg': leg_safety.motions.roll_amplitude_deg if leg_safety else 0.0,
                'pitch_deg': leg_safety.motions.pitch_amplitude_deg if leg_safety else 0.0,
                # Extended fields (SPEC-P1)
                'swell_hs_m': weather.swell_height_m,
                'windsea_hs_m': weather.windwave_height_m,
                'current_effect_kts': current_effect,
                'visibility_m': weather.visibility_km * 1000.0,
                'sst_celsius': weather.sst_celsius,
                'ice_concentration': weather.ice_concentration,
            })

            current_time += timedelta(hours=time_hours)

        safety_summary = {
            'status': worst_safety_status.value,
            'warnings': all_warnings,
            'max_roll_deg': max_roll,
            'max_pitch_deg': max_pitch,
            'max_accel_ms2': max_accel,
        }

        return total_fuel, total_time, total_distance, leg_details, safety_summary, speed_profile

    def _calculate_route_stats_time_constrained(
        self,
        waypoints: List[Tuple[float, float]],
        departure_time: datetime,
        calm_speed_kts: float,
        is_laden: bool,
        max_time_hours: float,
    ) -> Tuple[float, float, float, List[Dict], Dict, List[float]]:
        """
        Calculate route stats with variable speed under a total time budget.

        Distributes the time budget proportionally across legs (by distance),
        then finds the minimum-fuel speed for each leg that meets its time target.
        """
        # First pass: compute leg distances and weather
        legs_info = []
        total_distance = 0.0
        current_time = departure_time

        for i in range(len(waypoints) - 1):
            from_wp = waypoints[i]
            to_wp = waypoints[i + 1]
            distance = self.haversine(from_wp[0], from_wp[1], to_wp[0], to_wp[1])
            bearing = self.bearing(from_wp[0], from_wp[1], to_wp[0], to_wp[1])

            mid_lat = (from_wp[0] + to_wp[0]) / 2
            mid_lon = (from_wp[1] + to_wp[1]) / 2
            mid_time = current_time + timedelta(hours=distance / calm_speed_kts / 2)

            try:
                weather = self.weather_provider(mid_lat, mid_lon, mid_time)
            except Exception:
                weather = LegWeather()

            legs_info.append({
                'from_wp': from_wp, 'to_wp': to_wp,
                'distance': distance, 'bearing': bearing, 'weather': weather,
            })
            total_distance += distance
            current_time += timedelta(hours=distance / calm_speed_kts)

        # Distribute time budget proportionally by distance
        total_fuel = 0.0
        total_time = 0.0
        leg_details = []
        speed_profile = []
        max_roll = 0.0
        max_pitch = 0.0
        max_accel = 0.0
        all_warnings = []
        worst_safety_status = SafetyStatus.SAFE
        current_time = departure_time

        for info in legs_info:
            distance = info['distance']
            bearing = info['bearing']
            weather = info['weather']

            # Per-leg time target proportional to distance share
            leg_target_time = max_time_hours * (distance / total_distance) if total_distance > 0 else 1.0

            # Find optimal speed that meets the time target
            leg_speed, fuel_mt, time_hours = self._find_optimal_speed(
                distance_nm=distance,
                weather=weather,
                bearing_deg=bearing,
                is_laden=is_laden,
                target_time_hours=leg_target_time,
                calm_speed_kts=calm_speed_kts,
            )

            speed_profile.append(leg_speed)

            # SOG for this leg
            current_effect = self.current_effect(
                bearing, weather.current_speed_ms, weather.current_dir_deg
            )
            sog = max(leg_speed + current_effect, 0.1)

            total_fuel += fuel_mt
            total_time += time_hours

            # Safety assessment
            leg_safety = None
            if weather.sig_wave_height_m > 0:
                wave_period_s = weather.wave_period_s if weather.wave_period_s > 0 else (5.0 + weather.sig_wave_height_m)
                leg_safety = self.safety_constraints.assess_safety(
                    wave_height_m=weather.sig_wave_height_m,
                    wave_period_s=wave_period_s,
                    wave_dir_deg=weather.wave_dir_deg,
                    heading_deg=bearing,
                    speed_kts=leg_speed,
                    is_laden=is_laden,
                )
                max_roll = max(max_roll, leg_safety.motions.roll_amplitude_deg)
                max_pitch = max(max_pitch, leg_safety.motions.pitch_amplitude_deg)
                max_accel = max(max_accel, leg_safety.motions.bridge_accel_ms2)
                if leg_safety.status.value != 'safe':
                    worst_safety_status = max(worst_safety_status, leg_safety.status, key=lambda s: ['safe', 'marginal', 'dangerous'].index(s.value) if hasattr(s, 'value') else 0)
                for w in (leg_safety.warnings if leg_safety else []):
                    if w not in all_warnings:
                        all_warnings.append(w)

            leg_details.append({
                'from': info['from_wp'],
                'to': info['to_wp'],
                'distance_nm': distance,
                'bearing_deg': bearing,
                'fuel_mt': fuel_mt,
                'time_hours': time_hours,
                'sog_kts': sog,
                'stw_kts': leg_speed,
                'wind_speed_ms': weather.wind_speed_ms,
                'wave_height_m': weather.sig_wave_height_m,
                'safety_status': leg_safety.status.value if leg_safety else 'safe',
                'roll_deg': leg_safety.motions.roll_amplitude_deg if leg_safety else 0.0,
                'pitch_deg': leg_safety.motions.pitch_amplitude_deg if leg_safety else 0.0,
                # Extended fields (SPEC-P1)
                'swell_hs_m': weather.swell_height_m,
                'windsea_hs_m': weather.windwave_height_m,
                'current_effect_kts': current_effect,
                'visibility_m': weather.visibility_km * 1000.0,
                'sst_celsius': weather.sst_celsius,
                'ice_concentration': weather.ice_concentration,
            })

            current_time += timedelta(hours=time_hours)

        safety_summary = {
            'status': worst_safety_status.value if hasattr(worst_safety_status, 'value') else 'safe',
            'warnings': all_warnings,
            'max_roll_deg': max_roll,
            'max_pitch_deg': max_pitch,
            'max_accel_ms2': max_accel,
        }

        logger.info(f"Time-constrained recalc: {total_time:.1f}h (budget={max_time_hours:.1f}h), fuel={total_fuel:.1f}mt")
        return total_fuel, total_time, total_distance, leg_details, safety_summary, speed_profile


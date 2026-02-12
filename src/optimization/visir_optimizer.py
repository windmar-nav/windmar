"""
VISIR-style route optimizer for WINDMAR.

Implements a graph-based shortest-path algorithm inspired by the VISIR
(VIrtual Ship Route) model developed by CMCC.  Key characteristics:

* **Time-expanded graph** – the ocean is discretised into (lat, lon, time)
  nodes so the vessel encounters weather *as it evolves*.
* **Dijkstra on fuel cost** – unlike the A* engine which uses a heuristic,
  VISIR builds isochrone shells (equal-time fronts) and picks the
  minimum-cost path through them using Dijkstra's algorithm.
* **Voluntary speed reduction (VSR)** – in heavy seas the engine
  automatically lowers speed to keep motions within safety limits,
  faithfully modelling real-world slow-steaming in bad weather.
* **Multi-step speed discretisation** – each edge is evaluated at several
  candidate speeds; the most fuel-efficient safe option is chosen.

The public interface matches ``BaseOptimizer.optimize_route`` so it can
be used as a drop-in replacement for (or alongside) the A* engine.
"""

import heapq
import logging
import math
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.optimization.base_optimizer import BaseOptimizer, OptimizedRoute
from src.optimization.vessel_model import VesselModel, VesselSpecs
from src.optimization.voyage import LegWeather
from src.optimization.seakeeping import (
    SafetyConstraints,
    SafetyStatus,
    create_default_safety_constraints,
)
from src.data.land_mask import is_ocean, is_path_clear
from src.data.regulatory_zones import get_zone_checker, ZoneChecker
from src.optimization.route_optimizer import apply_visibility_cap

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class _GraphNode:
    """Node in the time-expanded routing graph."""

    lat: float
    lon: float
    time: datetime
    # Grid indices for hashing
    row: int
    col: int
    time_step: int

    def key(self) -> Tuple[int, int, int]:
        return (self.row, self.col, self.time_step)

    def __hash__(self):
        return hash(self.key())

    def __eq__(self, other):
        return self.key() == other.key()


@dataclass(order=True)
class _QueueEntry:
    cost: float
    node: _GraphNode = field(compare=False)
    speed_kts: float = field(compare=False, default=0.0)
    parent_key: Optional[Tuple[int, int, int]] = field(compare=False, default=None)


# ---------------------------------------------------------------------------
# VISIR Optimizer
# ---------------------------------------------------------------------------

class VisirOptimizer(BaseOptimizer):
    """
    Graph-based Dijkstra optimizer with isochrone expansion.

    This is the VISIR-style engine.  It builds a time-expanded graph and
    finds the shortest (cheapest) path using Dijkstra, with voluntary
    speed reduction in heavy weather.
    """

    # Defaults
    DEFAULT_RESOLUTION_DEG = 0.25
    DEFAULT_TIME_STEP_HOURS = 3.0   # temporal resolution of the graph
    DEFAULT_MAX_NODES = 200_000
    SPEED_RANGE_KTS = (10.0, 18.0)  # practical speed range for graph exploration
    SPEED_STEPS = 3                  # candidate speeds per edge

    # 8-connected grid directions (row_delta, col_delta)
    # Diagonals are checked first to prefer direct-line progress
    DIRECTIONS = [
        (-1, 1), (1, 1), (-1, -1), (1, -1),
        (-1, 0), (0, 1), (1, 0), (0, -1),
    ]

    def __init__(
        self,
        vessel_model: Optional[VesselModel] = None,
        resolution_deg: float = DEFAULT_RESOLUTION_DEG,
        time_step_hours: float = DEFAULT_TIME_STEP_HOURS,
        optimization_target: str = "fuel",
        safety_constraints: Optional[SafetyConstraints] = None,
        enforce_safety: bool = True,
        zone_checker: Optional[ZoneChecker] = None,
        enforce_zones: bool = True,
    ):
        super().__init__(vessel_model=vessel_model)
        self.resolution_deg = resolution_deg
        self.time_step_hours = time_step_hours
        self.optimization_target = optimization_target
        self.enforce_safety = enforce_safety
        self.enforce_zones = enforce_zones
        self.safety_weight: float = 0.0  # 0=pure fuel, 1=full safety penalties

        self.safety_constraints = safety_constraints or create_default_safety_constraints(
            lpp=self.vessel_model.specs.lpp,
            beam=self.vessel_model.specs.beam,
        )
        self.zone_checker = zone_checker or get_zone_checker()

    # ------------------------------------------------------------------
    # Public API (BaseOptimizer contract)
    # ------------------------------------------------------------------

    def optimize_route(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        departure_time: datetime,
        calm_speed_kts: float,
        is_laden: bool,
        weather_provider: Callable[[float, float, datetime], LegWeather],
        max_cells: int = DEFAULT_MAX_NODES,
        avoid_land: bool = True,
        max_time_factor: float = 1.30,
    ) -> OptimizedRoute:
        t0 = _time.time()

        # 1. Build spatial grid (also stores grid bounds for O(1) cell lookup)
        grid, grid_bounds = self._build_spatial_grid(origin, destination, filter_land=avoid_land)

        # 2. Locate start / end cells
        start_rc = self._nearest_cell(origin, grid, grid_bounds)
        end_rc = self._nearest_cell(destination, grid, grid_bounds)
        if start_rc is None or end_rc is None:
            raise ValueError("Origin or destination outside grid bounds")

        logger.info(f"VISIR start_rc={start_rc} -> {grid[start_rc]}, end_rc={end_rc} -> {grid[end_rc]}")

        # 3. Estimate max time steps needed
        #    Chebyshev distance × 2 to account for routing around obstacles
        chebyshev = max(abs(start_rc[0] - end_rc[0]), abs(start_rc[1] - end_rc[1]))
        gc_dist = self.haversine(origin[0], origin[1], destination[0], destination[1])
        min_speed = self.SPEED_RANGE_KTS[0]
        gc_time_steps = int(math.ceil((gc_dist / min_speed) * 1.5 / self.time_step_hours))
        max_time_steps = max(chebyshev * 2, gc_time_steps, 8)
        logger.info(f"VISIR gc_dist={gc_dist:.0f}nm, max_time_steps={max_time_steps}, "
                     f"max_voyage_hours={gc_dist / max(calm_speed_kts, 0.1) * max_time_factor:.1f}h")

        # 4. Compute time budget: direct time at calm speed × max_time_factor
        direct_time_hours = gc_dist / max(calm_speed_kts, 0.1)
        max_voyage_hours = direct_time_hours * max_time_factor

        # 5. Run Dijkstra on the time-expanded graph
        path, explored = self._dijkstra(
            grid=grid,
            start_rc=start_rc,
            end_rc=end_rc,
            departure_time=departure_time,
            calm_speed_kts=calm_speed_kts,
            is_laden=is_laden,
            weather_provider=weather_provider,
            max_time_steps=max_time_steps,
            max_nodes=max_cells,
            max_voyage_hours=max_voyage_hours,
        )

        if path is None:
            raise ValueError(
                f"VISIR: no route found after exploring {explored} nodes"
            )

        # 5. Extract waypoints
        waypoints = [(n.lat, n.lon) for n in path]
        waypoints = self.smooth_path(waypoints)

        # Pin endpoints to actual origin/destination (grid cells may be offset)
        waypoints[0] = origin
        waypoints[-1] = destination

        # 6. Compute detailed leg stats using shared base method.
        #    For stats, cap STW at calm_speed_kts — the Dijkstra may have used
        #    higher speeds internally to traverse the graph, but the reported
        #    speed should not exceed the user's setting.
        def find_speed(dist, weather, bearing, is_laden):
            # Use calm_speed as max STW for stats
            stw = min(calm_speed_kts, self.vessel_model.specs.service_speed_laden + 2
                      if is_laden else self.vessel_model.specs.service_speed_ballast + 2)
            weather_dict = {
                'wind_speed_ms': weather.wind_speed_ms,
                'wind_dir_deg': weather.wind_dir_deg,
                'heading_deg': bearing,
                'sig_wave_height_m': weather.sig_wave_height_m,
                'wave_dir_deg': weather.wave_dir_deg,
            }
            res = self.vessel_model.calculate_fuel_consumption(
                speed_kts=stw, is_laden=is_laden,
                weather=weather_dict, distance_nm=dist,
            )
            ce = self.current_effect(bearing, weather.current_speed_ms, weather.current_dir_deg)
            sog = max(stw + ce, 0.1)
            return stw, res['fuel_mt'], dist / sog

        (
            total_fuel, total_time, total_dist,
            leg_details, safety_summary, speed_profile,
        ) = self.calculate_route_stats(
            waypoints, departure_time, calm_speed_kts, is_laden,
            weather_provider=weather_provider,
            safety_constraints=self.safety_constraints,
            find_optimal_speed=find_speed,
        )

        # 7. Direct-route comparison
        (
            direct_fuel, direct_time, direct_dist, _, _, _,
        ) = self.calculate_route_stats(
            [origin, destination], departure_time, calm_speed_kts, is_laden,
            weather_provider=weather_provider,
            safety_constraints=self.safety_constraints,
            find_optimal_speed=find_speed,
        )

        elapsed_ms = (_time.time() - t0) * 1000
        fuel_sav = ((direct_fuel - total_fuel) / direct_fuel * 100) if direct_fuel > 0 else 0
        time_sav = ((direct_time - total_time) / direct_time * 100) if direct_time > 0 else 0
        avg_speed = total_dist / total_time if total_time > 0 else calm_speed_kts

        return OptimizedRoute(
            waypoints=waypoints,
            total_fuel_mt=total_fuel,
            total_time_hours=total_time,
            total_distance_nm=total_dist,
            direct_fuel_mt=direct_fuel,
            direct_time_hours=direct_time,
            fuel_savings_pct=fuel_sav,
            time_savings_pct=time_sav,
            leg_details=leg_details,
            speed_profile=speed_profile,
            avg_speed_kts=avg_speed,
            safety_status=safety_summary["status"],
            safety_warnings=safety_summary["warnings"],
            max_roll_deg=safety_summary["max_roll_deg"],
            max_pitch_deg=safety_summary["max_pitch_deg"],
            max_accel_ms2=safety_summary["max_accel_ms2"],
            grid_resolution_deg=self.resolution_deg,
            cells_explored=explored,
            optimization_time_ms=elapsed_ms,
            variable_speed_enabled=True,
        )

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_spatial_grid(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        margin_deg: float = 5.0,
        filter_land: bool = True,
    ) -> Tuple[Dict[Tuple[int, int], Tuple[float, float]], Dict[str, float]]:
        """
        Build a 2-D (row, col) -> (lat, lon) ocean grid.

        Returns the grid dict and grid_bounds dict with lat_min, lon_min,
        num_rows, num_cols for O(1) index calculation and time-step sizing.
        """
        lat_min = min(origin[0], destination[0]) - margin_deg
        lat_max = max(origin[0], destination[0]) + margin_deg
        lon_min = min(origin[1], destination[1]) - margin_deg
        lon_max = max(origin[1], destination[1]) + margin_deg
        lat_min, lat_max = max(lat_min, -85), min(lat_max, 85)

        grid: Dict[Tuple[int, int], Tuple[float, float]] = {}
        num_rows = 0
        num_cols = 0
        lat = lat_min
        while lat <= lat_max:
            col = 0
            lon = lon_min
            while lon <= lon_max:
                if not filter_land or is_ocean(lat, lon):
                    grid[(num_rows, col)] = (lat, lon)
                lon += self.resolution_deg
                col += 1
            num_cols = max(num_cols, col)
            lat += self.resolution_deg
            num_rows += 1

        grid_bounds = {
            "lat_min": lat_min,
            "lon_min": lon_min,
            "num_rows": num_rows,
            "num_cols": num_cols,
        }
        logger.info(f"VISIR grid: {len(grid)} ocean cells ({num_rows}x{num_cols})")
        return grid, grid_bounds

    def _nearest_cell(
        self,
        point: Tuple[float, float],
        grid: Dict[Tuple[int, int], Tuple[float, float]],
        grid_bounds: Dict[str, float],
    ) -> Optional[Tuple[int, int]]:
        """
        Find the grid cell nearest to *point* using O(1) index calculation.

        Falls back to searching the 3x3 neighbourhood if the direct cell
        was filtered out (land).
        """
        lat, lon = point
        row = round((lat - grid_bounds["lat_min"]) / self.resolution_deg)
        col = round((lon - grid_bounds["lon_min"]) / self.resolution_deg)

        # Direct hit
        if (row, col) in grid:
            return (row, col)

        # Search 3x3 neighbourhood for nearest ocean cell
        best_key = None
        best_dist = float("inf")
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rc = (row + dr, col + dc)
                if rc in grid:
                    g_lat, g_lon = grid[rc]
                    d = (g_lat - lat) ** 2 + (g_lon - lon) ** 2
                    if d < best_dist:
                        best_dist = d
                        best_key = rc

        # Wider fallback (rare — only if point is deep inland)
        if best_key is None:
            for key, (g_lat, g_lon) in grid.items():
                d = (g_lat - lat) ** 2 + (g_lon - lon) ** 2
                if d < best_dist:
                    best_dist = d
                    best_key = key

        return best_key

    # ------------------------------------------------------------------
    # Dijkstra on time-expanded graph
    # ------------------------------------------------------------------

    def _dijkstra(
        self,
        grid: Dict[Tuple[int, int], Tuple[float, float]],
        start_rc: Tuple[int, int],
        end_rc: Tuple[int, int],
        departure_time: datetime,
        calm_speed_kts: float,
        is_laden: bool,
        weather_provider: Callable[[float, float, datetime], LegWeather],
        max_time_steps: int,
        max_nodes: int,
        max_voyage_hours: float = float("inf"),
    ) -> Tuple[Optional[List[_GraphNode]], int]:
        """
        A*-guided Dijkstra over the (row, col, time_step) graph.

        Uses an admissible heuristic (minimum fuel to reach destination in
        calm conditions) to focus exploration toward the goal, dramatically
        reducing the number of nodes explored on long routes.

        At each expansion, for every spatial neighbour we try multiple
        candidate speeds and pick the cheapest safe option.  The time-step
        of the neighbour is determined by the travel time at the chosen
        speed, giving the algorithm its *isochrone* character.
        """
        start_lat, start_lon = grid[start_rc]
        end_lat, end_lon = grid[end_rc]
        start_node = _GraphNode(
            lat=start_lat, lon=start_lon, time=departure_time,
            row=start_rc[0], col=start_rc[1], time_step=0,
        )

        # Time-value penalty: each extra hour "costs" the fuel of 1 hour at
        # calm speed.  This prevents slow-steaming by making detour time
        # expensive, the same approach used by the A* engine.
        service_fuel_res = self.vessel_model.calculate_fuel_consumption(
            speed_kts=calm_speed_kts, is_laden=is_laden,
            weather=None, distance_nm=calm_speed_kts,  # 1 hour
        )
        lambda_time = service_fuel_res["fuel_mt"]

        # Compute admissible heuristic: min (fuel + time penalty) per nm
        # in calm conditions across all candidate speeds
        min_spd, max_spd = self.SPEED_RANGE_KTS
        min_cost_per_nm = float("inf")
        for speed_kts in np.linspace(min_spd, max_spd, self.SPEED_STEPS * 2):
            res = self.vessel_model.calculate_fuel_consumption(
                speed_kts=float(speed_kts), is_laden=is_laden,
                weather=None, distance_nm=100.0,
            )
            fuel_per_nm = res["fuel_mt"] / 100.0
            time_per_nm = 1.0 / float(speed_kts)  # hours per nm
            cost_per_nm = fuel_per_nm + lambda_time * time_per_nm
            min_cost_per_nm = min(min_cost_per_nm, cost_per_nm)

        # Cache heuristic per spatial cell (haversine to dest × min cost rate)
        h_cache: Dict[Tuple[int, int], float] = {}

        def heuristic(rc: Tuple[int, int]) -> float:
            if rc not in h_cache:
                lat, lon = grid[rc]
                dist = self.haversine(lat, lon, end_lat, end_lon)
                h_cache[rc] = dist * min_cost_per_nm
            return h_cache[rc]

        cost_so_far: Dict[Tuple[int, int, int], float] = {start_node.key(): 0.0}
        parent: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
        node_map: Dict[Tuple[int, int, int], _GraphNode] = {start_node.key(): start_node}

        h0 = heuristic(start_rc)
        pq: List[_QueueEntry] = [_QueueEntry(cost=h0, node=start_node)]
        explored = 0

        # Cache zone penalties per spatial edge (row,col)->(row,col) to avoid
        # redundant polygon intersection tests across time steps
        zone_cache: Dict[Tuple[Tuple[int, int], Tuple[int, int]], float] = {}

        while pq and explored < max_nodes:
            entry = heapq.heappop(pq)
            cur = entry.node
            cur_key = cur.key()

            # Stale entry? (compare against g-cost, not f-cost)
            g_cur = cost_so_far.get(cur_key, float("inf"))
            h_cur = heuristic((cur.row, cur.col))
            if entry.cost > g_cur + h_cur + 1e-9:
                continue

            explored += 1

            # Reached destination?
            if cur.row == end_rc[0] and cur.col == end_rc[1]:
                path = self._reconstruct(cur_key, parent, node_map)
                return path, explored

            # Expand spatial neighbours
            for dr, dc in self.DIRECTIONS:
                nb_rc = (cur.row + dr, cur.col + dc)
                if nb_rc not in grid:
                    continue

                nb_lat, nb_lon = grid[nb_rc]

                # Edge land check — at 0.25° the grid is fine enough that
                # is_path_clear won't disconnect valid passages.
                if not is_path_clear(cur.lat, cur.lon, nb_lat, nb_lon):
                    continue

                spatial_edge = ((cur.row, cur.col), nb_rc)
                if spatial_edge not in zone_cache:
                    if self.enforce_zones:
                        zp, _ = self.zone_checker.get_path_penalty(
                            cur.lat, cur.lon, nb_lat, nb_lon,
                        )
                        zone_cache[spatial_edge] = zp
                    else:
                        zone_cache[spatial_edge] = 1.0
                if zone_cache[spatial_edge] == float("inf"):
                    continue  # land crossing or exclusion zone

                dist_nm = self.haversine(cur.lat, cur.lon, nb_lat, nb_lon)
                brg = self.bearing(cur.lat, cur.lon, nb_lat, nb_lon)

                # Get weather at midpoint
                mid_lat = (cur.lat + nb_lat) / 2
                mid_lon = (cur.lon + nb_lon) / 2
                mid_time = cur.time + timedelta(hours=dist_nm / calm_speed_kts / 2)
                try:
                    weather = weather_provider(mid_lat, mid_lon, mid_time)
                except Exception:
                    weather = LegWeather()

                # --- Try candidate speeds (voluntary speed reduction) ---
                cached_zone_factor = zone_cache[spatial_edge]

                best_edge = self._best_edge(
                    dist_nm, brg, weather, calm_speed_kts, is_laden,
                    zone_factor=cached_zone_factor,
                    lambda_time=lambda_time,
                )
                if best_edge is None:
                    continue  # all speeds unsafe

                edge_cost, travel_hours, chosen_speed = best_edge

                # Map travel time to the next discrete time step
                nb_time = cur.time + timedelta(hours=travel_hours)

                nb_ts = cur.time_step + max(1, round(travel_hours / self.time_step_hours))
                if nb_ts > max_time_steps:
                    continue  # exceeds time horizon

                nb_node = _GraphNode(
                    lat=nb_lat, lon=nb_lon, time=nb_time,
                    row=nb_rc[0], col=nb_rc[1], time_step=nb_ts,
                )
                nb_key = nb_node.key()

                tentative_g = cost_so_far[cur_key] + edge_cost
                if tentative_g < cost_so_far.get(nb_key, float("inf")):
                    cost_so_far[nb_key] = tentative_g
                    parent[nb_key] = cur_key
                    node_map[nb_key] = nb_node
                    f_score = tentative_g + heuristic(nb_rc)
                    heapq.heappush(pq, _QueueEntry(cost=f_score, node=nb_node, speed_kts=chosen_speed))

        reason = "max_nodes" if explored >= max_nodes else "pq_empty"
        logger.warning(f"VISIR search failed: reason={reason}, explored={explored}, "
                       f"pq_size={len(pq)}, cost_so_far_size={len(cost_so_far)}, "
                       f"max_time_steps={max_time_steps}")
        return None, explored

    # ------------------------------------------------------------------
    # Edge cost with voluntary speed reduction + zone enforcement
    # ------------------------------------------------------------------

    def _best_edge(
        self,
        dist_nm: float,
        bearing_deg: float,
        weather: LegWeather,
        calm_speed_kts: float,
        is_laden: bool,
        zone_factor: float = 1.0,
        lambda_time: float = 0.0,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Evaluate candidate speeds and return the cheapest safe option.

        Parameters
        ----------
        zone_factor : pre-computed zone penalty for this edge (from cache).
        lambda_time : time-value penalty (fuel-equivalent cost per hour).
            Prevents slow-steaming by making extra hours expensive.

        Returns ``(cost, travel_hours, chosen_speed)`` or *None* if
        no safe speed exists.
        """
        # SPEC-P1: Ice exclusion and penalty zones
        ICE_EXCLUSION_THRESHOLD = 0.15  # IMO Polar Code limit
        ICE_PENALTY_THRESHOLD = 0.05   # Caution zone
        if weather.ice_concentration >= ICE_EXCLUSION_THRESHOLD:
            return None
        ice_cost_factor = 2.0 if weather.ice_concentration >= ICE_PENALTY_THRESHOLD else 1.0

        min_spd, max_spd = self.SPEED_RANGE_KTS
        if is_laden:
            max_spd = min(max_spd, self.vessel_model.specs.service_speed_laden + 2)
        else:
            max_spd = min(max_spd, self.vessel_model.specs.service_speed_ballast + 2)

        # SPEC-P1: Visibility speed cap (IMO COLREG Rule 6)
        max_spd = apply_visibility_cap(max_spd, weather.visibility_km * 1000.0)

        weather_dict = {
            "wind_speed_ms": weather.wind_speed_ms,
            "wind_dir_deg": weather.wind_dir_deg,
            "heading_deg": bearing_deg,
            "sig_wave_height_m": weather.sig_wave_height_m,
            "wave_dir_deg": weather.wave_dir_deg,
        }

        ce = self.current_effect(
            bearing_deg, weather.current_speed_ms, weather.current_dir_deg,
        )

        best: Optional[Tuple[float, float, float]] = None

        for speed_kts in np.linspace(min_spd, max_spd, self.SPEED_STEPS):
            # Safety gate (voluntary speed reduction)
            safety_cost = 1.0
            if self.enforce_safety and weather.sig_wave_height_m > 0:
                wp = self.estimate_wave_period(weather)
                sf = self.safety_constraints.get_safety_cost_factor(
                    wave_height_m=weather.sig_wave_height_m,
                    wave_period_s=wp,
                    wave_dir_deg=weather.wave_dir_deg,
                    heading_deg=bearing_deg,
                    speed_kts=speed_kts,
                    is_laden=is_laden,
                )
                if sf == float("inf"):
                    continue  # unsafe at this speed — hard constraint always
                # Soft safety penalty dampened by safety_weight
                if self.safety_weight > 0 and sf > 1.0:
                    safety_cost = sf ** self.safety_weight

            sog = speed_kts + ce
            if sog <= 0.5:
                continue

            hours = dist_nm / sog

            result = self.vessel_model.calculate_fuel_consumption(
                speed_kts=speed_kts,
                is_laden=is_laden,
                weather=weather_dict,
                distance_nm=dist_nm,
            )
            fuel = result["fuel_mt"]

            if self.optimization_target == "fuel":
                score = (fuel + lambda_time * hours) * zone_factor * safety_cost * ice_cost_factor
            else:
                score = hours * zone_factor * safety_cost * ice_cost_factor

            if best is None or score < best[0]:
                best = (score, hours, float(speed_kts))

        return best

    # ------------------------------------------------------------------
    # Path reconstruction
    # ------------------------------------------------------------------

    def _reconstruct(
        self,
        end_key: Tuple[int, int, int],
        parent: Dict[Tuple[int, int, int], Tuple[int, int, int]],
        node_map: Dict[Tuple[int, int, int], _GraphNode],
    ) -> List[_GraphNode]:
        path: List[_GraphNode] = []
        key = end_key
        while key is not None:
            path.append(node_map[key])
            key = parent.get(key)
        path.reverse()
        return path

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
from src.data.land_mask import is_ocean, is_path_clear, get_land_mask_status

logger = logging.getLogger(__name__)


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
        return self.row == other.row and self.col == other.col


@dataclass(order=True)
class SearchNode:
    """Node in A* search priority queue."""
    f_score: float  # g + h (total estimated cost)
    cell: GridCell = field(compare=False)
    g_score: float = field(compare=False)  # Cost from start
    arrival_time: datetime = field(compare=False)
    parent: Optional['SearchNode'] = field(compare=False, default=None)


@dataclass
class OptimizedRoute:
    """Result of route optimization."""
    waypoints: List[Tuple[float, float]]  # (lat, lon) pairs
    total_fuel_mt: float
    total_time_hours: float
    total_distance_nm: float

    # Comparison with direct route
    direct_fuel_mt: float
    direct_time_hours: float
    fuel_savings_pct: float
    time_savings_pct: float

    # Per-leg details
    leg_details: List[Dict]

    # Metadata
    grid_resolution_deg: float
    cells_explored: int
    optimization_time_ms: float


class RouteOptimizer:
    """
    A* based route optimizer.

    Finds minimum-fuel (or minimum-time) routes through weather.
    """

    # Default grid settings
    DEFAULT_RESOLUTION_DEG = 0.5  # Grid cell size in degrees (~30nm at equator)
    DEFAULT_MAX_CELLS = 50000  # Maximum cells to explore before giving up

    # Neighbor directions (8-connected grid)
    # (row_delta, col_delta) for N, NE, E, SE, S, SW, W, NW
    DIRECTIONS = [
        (-1, 0), (-1, 1), (0, 1), (1, 1),
        (1, 0), (1, -1), (0, -1), (-1, -1)
    ]

    def __init__(
        self,
        vessel_model: Optional[VesselModel] = None,
        resolution_deg: float = DEFAULT_RESOLUTION_DEG,
        optimization_target: str = "fuel",  # "fuel" or "time"
    ):
        """
        Initialize route optimizer.

        Args:
            vessel_model: Vessel performance model
            resolution_deg: Grid resolution in degrees
            optimization_target: What to minimize ("fuel" or "time")
        """
        self.vessel_model = vessel_model or VesselModel()
        self.resolution_deg = resolution_deg
        self.optimization_target = optimization_target

        # Weather provider function (set before optimization)
        self.weather_provider: Optional[Callable] = None

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
    ) -> OptimizedRoute:
        """
        Find optimal route from origin to destination.

        Args:
            origin: (lat, lon) starting point
            destination: (lat, lon) ending point
            departure_time: When voyage starts
            calm_speed_kts: Calm water speed
            is_laden: Vessel loading condition
            weather_provider: Function(lat, lon, time) -> LegWeather
            max_cells: Maximum cells to explore
            avoid_land: Whether to avoid land masses

        Returns:
            OptimizedRoute with waypoints and statistics
        """
        import time
        start_time = time.time()

        self.weather_provider = weather_provider

        # Build grid around origin-destination corridor
        grid = self._build_grid(origin, destination)

        # Get start and end cells
        start_cell = self._get_cell(origin[0], origin[1], grid)
        end_cell = self._get_cell(destination[0], destination[1], grid)

        if start_cell is None or end_cell is None:
            raise ValueError("Origin or destination outside grid bounds")

        # Run A* search
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

        # Convert path to waypoints
        waypoints = [(cell.lat, cell.lon) for cell in path]

        # Smooth path to reduce unnecessary waypoints
        waypoints = self._smooth_path(waypoints)

        # Calculate route statistics
        opt_fuel, opt_time, opt_distance, leg_details = self._calculate_route_stats(
            waypoints, departure_time, calm_speed_kts, is_laden
        )

        # Calculate direct route for comparison
        direct_fuel, direct_time, direct_distance, _ = self._calculate_route_stats(
            [origin, destination], departure_time, calm_speed_kts, is_laden
        )

        optimization_time_ms = (time.time() - start_time) * 1000

        # Calculate savings
        fuel_savings = ((direct_fuel - opt_fuel) / direct_fuel * 100) if direct_fuel > 0 else 0
        time_savings = ((direct_time - opt_time) / direct_time * 100) if direct_time > 0 else 0

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
            grid_resolution_deg=self.resolution_deg,
            cells_explored=cells_explored,
            optimization_time_ms=optimization_time_ms,
        )

    def _build_grid(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        margin_deg: float = 5.0,
        filter_land: bool = True,
    ) -> Dict[Tuple[int, int], GridCell]:
        """
        Build routing grid around origin-destination corridor.

        Adds margin around the direct path to allow for deviations.
        Filters out land cells if filter_land=True.
        """
        lat1, lon1 = origin
        lat2, lon2 = destination

        # Calculate bounding box with margin
        lat_min = min(lat1, lat2) - margin_deg
        lat_max = max(lat1, lat2) + margin_deg
        lon_min = min(lon1, lon2) - margin_deg
        lon_max = max(lon1, lon2) + margin_deg

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

        total_cells = row * col
        logger.info(f"Built grid: {len(grid)} ocean cells, {land_cells} land cells filtered "
                   f"({row} rows x {col} cols, {land_cells/total_cells*100:.1f}% land)")

        return grid

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

            # Explore neighbors
            for dr, dc in self.DIRECTIONS:
                neighbor_key = (current.cell.row + dr, current.cell.col + dc)

                if neighbor_key not in grid:
                    continue
                if neighbor_key in closed_set:
                    continue

                neighbor_cell = grid[neighbor_key]

                # Calculate cost to move to neighbor
                move_cost, travel_time = self._calculate_move_cost(
                    from_cell=current.cell,
                    to_cell=neighbor_cell,
                    departure_time=current.arrival_time,
                    calm_speed_kts=calm_speed_kts,
                    is_laden=is_laden,
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
        distance_nm = self._haversine_distance(
            cell.lat, cell.lon, goal.lat, goal.lon
        )

        if self.optimization_target == "time":
            # Best case: calm water speed
            return distance_nm / self.vessel_model.specs.service_speed_laden
        else:
            # Best case: calm water fuel consumption
            # Use a conservative (low) estimate to ensure admissibility
            result = self.vessel_model.calculate_fuel_consumption(
                speed_kts=self.vessel_model.specs.service_speed_laden,
                is_laden=True,
                weather=None,
                distance_nm=distance_nm,
            )
            return result['fuel_mt'] * 0.8  # Underestimate for admissibility

    def _calculate_move_cost(
        self,
        from_cell: GridCell,
        to_cell: GridCell,
        departure_time: datetime,
        calm_speed_kts: float,
        is_laden: bool,
    ) -> Tuple[float, float]:
        """
        Calculate cost to move from one cell to another.

        Returns:
            Tuple of (cost, travel_time_hours)
        """
        # Calculate distance
        distance_nm = self._haversine_distance(
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
        bearing = self._calculate_bearing(
            from_cell.lat, from_cell.lon, to_cell.lat, to_cell.lon
        )

        # Build weather dict for vessel model
        weather_dict = {
            'wind_speed_ms': weather.wind_speed_ms,
            'wind_dir_deg': weather.wind_dir_deg,
            'heading_deg': bearing,
            'sig_wave_height_m': weather.sig_wave_height_m,
            'wave_dir_deg': weather.wave_dir_deg,
        }

        # Calculate fuel consumption
        result = self.vessel_model.calculate_fuel_consumption(
            speed_kts=calm_speed_kts,
            is_laden=is_laden,
            weather=weather_dict,
            distance_nm=distance_nm,
        )

        # Calculate actual travel time considering current
        current_effect = self._calculate_current_effect(
            vessel_speed_kts=calm_speed_kts,
            heading_deg=bearing,
            current_speed_ms=weather.current_speed_ms,
            current_dir_deg=weather.current_dir_deg,
        )

        sog_kts = calm_speed_kts + current_effect
        if sog_kts <= 0:
            return float('inf'), float('inf')  # Can't make headway

        travel_time_hours = distance_nm / sog_kts

        # Return cost based on optimization target
        if self.optimization_target == "time":
            return travel_time_hours, travel_time_hours
        else:
            return result['fuel_mt'], travel_time_hours

    def _calculate_current_effect(
        self,
        vessel_speed_kts: float,
        heading_deg: float,
        current_speed_ms: float,
        current_dir_deg: float,
    ) -> float:
        """
        Calculate effect of current on speed over ground.

        Returns speed adjustment in knots (positive = favorable, negative = adverse).
        """
        if current_speed_ms <= 0:
            return 0.0

        current_kts = current_speed_ms * 1.94384

        # Relative angle between heading and current
        relative_angle = abs(((current_dir_deg - heading_deg) + 180) % 360 - 180)
        relative_rad = math.radians(relative_angle)

        # Component of current in direction of travel
        current_effect = current_kts * math.cos(relative_rad)

        return current_effect

    def _smooth_path(
        self,
        waypoints: List[Tuple[float, float]],
        tolerance_nm: float = 5.0,
        check_land: bool = True,
    ) -> List[Tuple[float, float]]:
        """
        Smooth path using Douglas-Peucker algorithm.

        Removes unnecessary waypoints while keeping path shape.
        Ensures simplified path doesn't cross land.
        """
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

        return simplify(waypoints, tolerance_nm)

    def _calculate_route_stats(
        self,
        waypoints: List[Tuple[float, float]],
        departure_time: datetime,
        calm_speed_kts: float,
        is_laden: bool,
    ) -> Tuple[float, float, float, List[Dict]]:
        """
        Calculate total fuel, time, and distance for a route.

        Returns:
            Tuple of (total_fuel_mt, total_time_hours, total_distance_nm, leg_details)
        """
        total_fuel = 0.0
        total_time = 0.0
        total_distance = 0.0
        leg_details = []

        current_time = departure_time

        for i in range(len(waypoints) - 1):
            from_wp = waypoints[i]
            to_wp = waypoints[i + 1]

            distance = self._haversine_distance(from_wp[0], from_wp[1], to_wp[0], to_wp[1])
            bearing = self._calculate_bearing(from_wp[0], from_wp[1], to_wp[0], to_wp[1])

            # Get weather at midpoint
            mid_lat = (from_wp[0] + to_wp[0]) / 2
            mid_lon = (from_wp[1] + to_wp[1]) / 2
            mid_time = current_time + timedelta(hours=distance / calm_speed_kts / 2)

            try:
                weather = self.weather_provider(mid_lat, mid_lon, mid_time)
            except Exception:
                weather = LegWeather()

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

            current_effect = self._calculate_current_effect(
                calm_speed_kts, bearing, weather.current_speed_ms, weather.current_dir_deg
            )
            sog = max(calm_speed_kts + current_effect, 0.1)
            time_hours = distance / sog

            total_fuel += result['fuel_mt']
            total_time += time_hours
            total_distance += distance

            leg_details.append({
                'from': from_wp,
                'to': to_wp,
                'distance_nm': distance,
                'bearing_deg': bearing,
                'fuel_mt': result['fuel_mt'],
                'time_hours': time_hours,
                'sog_kts': sog,
                'wind_speed_ms': weather.wind_speed_ms,
                'wave_height_m': weather.sig_wave_height_m,
            })

            current_time += timedelta(hours=time_hours)

        return total_fuel, total_time, total_distance, leg_details

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great circle distance in nautical miles."""
        R = 3440.065  # Earth radius in nm

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))

        return R * c

    @staticmethod
    def _calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate initial bearing from point 1 to point 2."""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon_rad = math.radians(lon2 - lon1)

        x = math.sin(dlon_rad) * math.cos(lat2_rad)
        y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad)

        bearing = math.degrees(math.atan2(x, y))
        return (bearing + 360) % 360

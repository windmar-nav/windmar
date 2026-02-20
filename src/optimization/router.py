"""
Maritime route optimization using A* algorithm.

Finds fuel-optimal routes considering weather, waves, and maritime constraints.
"""

import heapq
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..validation import validate_coordinates, validate_speed
from .vessel_model import VesselModel


logger = logging.getLogger(__name__)


@dataclass
class RouteConstraints:
    """Constraints for route optimization."""

    # Under Keel Clearance (UKC) requirements
    min_ukc_m: float = 2.0  # Minimum UKC in meters
    ukc_safety_factor: float = 1.3  # Safety factor for UKC

    # Weather limits
    max_wind_speed_ms: float = 25.0  # Maximum wind speed (m/s)
    max_wave_height_m: float = 5.0  # Maximum significant wave height (m)

    # Operational limits
    max_speed_reduction_pct: float = 30.0  # Max speed reduction in bad weather (%)
    min_speed_kts: float = 8.0  # Minimum safe speed (knots)

    # ECA zones (simplified - in practice would use polygon boundaries)
    avoid_eca: bool = False  # Whether to avoid ECA zones

    # Grid resolution for pathfinding
    grid_resolution_deg: float = 0.5  # Grid cell size in degrees


class Node:
    """Node in the A* search graph."""

    def __init__(
        self,
        lat: float,
        lon: float,
        g_cost: float = float("inf"),
        h_cost: float = 0.0,
        parent: Optional["Node"] = None,
        time: Optional[datetime] = None,
    ):
        self.lat = lat
        self.lon = lon
        self.g_cost = g_cost  # Cost from start (fuel consumed)
        self.h_cost = h_cost  # Heuristic cost to goal
        self.parent = parent
        self.time = time  # Time at this node

    @property
    def f_cost(self) -> float:
        """Total cost (g + h)."""
        return self.g_cost + self.h_cost

    def __lt__(self, other: "Node") -> bool:
        """Compare nodes by f_cost for priority queue."""
        return self.f_cost < other.f_cost

    def __hash__(self) -> int:
        """Hash by position for use in sets."""
        return hash((round(self.lat, 4), round(self.lon, 4)))

    def __eq__(self, other: object) -> bool:
        """Check equality by position."""
        if not isinstance(other, Node):
            return False
        return (
            abs(self.lat - other.lat) < 0.001
            and abs(self.lon - other.lon) < 0.001
        )


class MaritimeRouter:
    """
    A* pathfinding for maritime route optimization.

    Finds fuel-optimal routes considering:
    - Weather and wave forecasts
    - Vessel performance characteristics
    - Maritime constraints (UKC, weather limits)
    - Great circle heuristic
    """

    def __init__(
        self,
        vessel_model: VesselModel,
        grib_parser_gfs=None,
        grib_parser_wave=None,
        constraints: Optional[RouteConstraints] = None,
    ):
        """
        Initialize maritime router.

        Args:
            vessel_model: Vessel performance model
            grib_parser_gfs: GFS weather parser
            grib_parser_wave: WaveWatch III parser
            constraints: Route constraints
        """
        self.vessel_model = vessel_model
        self.grib_parser_gfs = grib_parser_gfs
        self.grib_parser_wave = grib_parser_wave
        self.constraints = constraints or RouteConstraints()

    def find_optimal_route(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        departure_time: datetime,
        is_laden: bool,
        target_speed_kts: Optional[float] = None,
    ) -> Dict:
        """
        Find fuel-optimal route using A* algorithm.

        Args:
            start_pos: Starting position (lat, lon)
            end_pos: Destination position (lat, lon)
            departure_time: Departure time (UTC)
            is_laden: Loading condition
            target_speed_kts: Target speed (defaults to service speed)

        Returns:
            Dictionary with:
                - waypoints: List of (lat, lon) tuples
                - total_fuel_mt: Total fuel consumption
                - total_distance_nm: Total distance
                - total_time_hours: Total time
                - weather_along_route: Weather at each waypoint
        """
        # Validate inputs
        validate_coordinates(
            start_pos[0], start_pos[1],
            "start_position.latitude", "start_position.longitude",
        )
        validate_coordinates(
            end_pos[0], end_pos[1],
            "end_position.latitude", "end_position.longitude",
        )
        if target_speed_kts is not None:
            validate_speed(target_speed_kts, "target_speed_kts")

        if target_speed_kts is None:
            target_speed_kts = (
                self.vessel_model.specs.service_speed_laden
                if is_laden
                else self.vessel_model.specs.service_speed_ballast
            )

        logger.info(
            f"Starting A* route optimization from {start_pos} to {end_pos}"
        )

        # Initialize start and goal nodes
        start_node = Node(
            start_pos[0], start_pos[1], g_cost=0.0, time=departure_time
        )
        goal_node = Node(end_pos[0], end_pos[1])

        # Calculate heuristic for start node
        start_node.h_cost = self._heuristic(start_node, goal_node, is_laden)

        # A* data structures
        open_set: List[Node] = [start_node]
        closed_set: Set[Node] = set()
        heapq.heapify(open_set)

        iterations = 0
        max_iterations = 10000

        while open_set and iterations < max_iterations:
            iterations += 1

            # Get node with lowest f_cost
            current = heapq.heappop(open_set)

            # Check if goal reached
            if self._distance_nm(current.lat, current.lon, goal_node.lat, goal_node.lon) < 10:
                logger.info(f"Route found after {iterations} iterations")
                return self._reconstruct_route(
                    current, goal_node, is_laden, target_speed_kts
                )

            closed_set.add(current)

            # Explore neighbors
            for neighbor in self._get_neighbors(current, goal_node):
                if neighbor in closed_set:
                    continue

                # Calculate travel cost to neighbor
                cost_result = self._calculate_edge_cost(
                    current, neighbor, is_laden, target_speed_kts
                )

                if cost_result is None:
                    # Path not feasible (violates constraints)
                    continue

                tentative_g_cost = current.g_cost + cost_result["fuel_mt"]

                # Check if this path to neighbor is better
                if tentative_g_cost < neighbor.g_cost:
                    neighbor.parent = current
                    neighbor.g_cost = tentative_g_cost
                    neighbor.h_cost = self._heuristic(neighbor, goal_node, is_laden)
                    neighbor.time = current.time + timedelta(
                        hours=cost_result["time_hours"]
                    )

                    # Add to open set if not already there
                    if neighbor not in open_set:
                        heapq.heappush(open_set, neighbor)

            if iterations % 100 == 0:
                logger.debug(
                    f"A* iteration {iterations}, open set size: {len(open_set)}"
                )

        logger.warning("A* search did not find a route, using great circle")
        return self._great_circle_fallback(
            start_pos, end_pos, departure_time, is_laden, target_speed_kts
        )

    def _get_neighbors(
        self, current: Node, goal: Node
    ) -> List[Node]:
        """
        Get neighboring nodes for A* expansion.

        Uses adaptive grid resolution - finer near goal.

        Args:
            current: Current node
            goal: Goal node

        Returns:
            List of neighbor nodes
        """
        # Distance to goal
        dist_to_goal = self._distance_nm(
            current.lat, current.lon, goal.lat, goal.lon
        )

        # Adaptive resolution (finer near goal)
        if dist_to_goal < 100:
            resolution = self.constraints.grid_resolution_deg * 0.5
        elif dist_to_goal < 500:
            resolution = self.constraints.grid_resolution_deg
        else:
            resolution = self.constraints.grid_resolution_deg * 1.5

        # Generate neighbors in 8 directions
        neighbors = []
        for dlat in [-resolution, 0, resolution]:
            for dlon in [-resolution, 0, resolution]:
                if dlat == 0 and dlon == 0:
                    continue

                new_lat = current.lat + dlat
                new_lon = current.lon + dlon

                # Keep latitude in bounds
                if abs(new_lat) > 85:
                    continue

                # Normalize longitude
                new_lon = ((new_lon + 180) % 360) - 180

                neighbor = Node(new_lat, new_lon)
                neighbors.append(neighbor)

        return neighbors

    def _calculate_edge_cost(
        self,
        from_node: Node,
        to_node: Node,
        is_laden: bool,
        target_speed_kts: float,
    ) -> Optional[Dict]:
        """
        Calculate cost (fuel) to travel between nodes.

        Args:
            from_node: Starting node
            to_node: Destination node
            is_laden: Loading condition
            target_speed_kts: Target speed

        Returns:
            Cost dictionary or None if path violates constraints
        """
        # Calculate distance
        distance_nm = self._distance_nm(
            from_node.lat, from_node.lon, to_node.lat, to_node.lon
        )

        # Get heading
        heading = self._bearing(
            from_node.lat, from_node.lon, to_node.lat, to_node.lon
        )

        # Get weather at midpoint and time
        mid_lat = (from_node.lat + to_node.lat) / 2
        mid_lon = (from_node.lon + to_node.lon) / 2

        weather = self._get_weather_at_point(mid_lat, mid_lon, from_node.time)

        # Check weather constraints
        if weather:
            if weather.get("wind_speed_ms", 0) > self.constraints.max_wind_speed_ms:
                return None
            if (
                weather.get("sig_wave_height_m", 0)
                > self.constraints.max_wave_height_m
            ):
                return None

            # Add heading to weather dict
            weather["heading_deg"] = heading

        # Calculate fuel consumption
        fuel_result = self.vessel_model.calculate_fuel_consumption(
            target_speed_kts, is_laden, weather, distance_nm
        )

        return fuel_result

    def _get_weather_at_point(
        self, lat: float, lon: float, time: datetime
    ) -> Optional[Dict[str, float]]:
        """Get weather conditions at a point and time."""
        if self.grib_parser_gfs is None:
            return None

        try:
            weather = self.grib_parser_gfs.get_weather_at_point(lat, lon, time)

            if self.grib_parser_wave:
                waves = self.grib_parser_wave.get_waves_at_point(lat, lon, time)
                weather.update(waves)

            return weather
        except Exception as e:
            logger.debug(f"Could not get weather: {e}")
            return None

    def _heuristic(
        self, node: Node, goal: Node, is_laden: bool
    ) -> float:
        """
        Heuristic function for A* (admissible).

        Uses great circle distance and calm water fuel consumption.

        Args:
            node: Current node
            goal: Goal node
            is_laden: Loading condition

        Returns:
            Estimated fuel cost to goal
        """
        # Great circle distance
        distance_nm = self._distance_nm(node.lat, node.lon, goal.lat, goal.lon)

        # Estimate fuel using calm water consumption
        speed = (
            self.vessel_model.specs.service_speed_laden
            if is_laden
            else self.vessel_model.specs.service_speed_ballast
        )

        fuel_result = self.vessel_model.calculate_fuel_consumption(
            speed, is_laden, weather=None, distance_nm=distance_nm
        )

        return fuel_result["fuel_mt"]

    def _reconstruct_route(
        self,
        final_node: Node,
        goal_node: Node,
        is_laden: bool,
        target_speed_kts: float,
    ) -> Dict:
        """
        Reconstruct route from A* path.

        Args:
            final_node: Final node reached
            goal_node: Original goal
            is_laden: Loading condition
            target_speed_kts: Target speed

        Returns:
            Route dictionary
        """
        # Build path from goal to start
        path = []
        current = final_node
        while current is not None:
            path.append((current.lat, current.lon, current.time))
            current = current.parent

        path.reverse()

        # Add final goal if not exactly reached
        if len(path) > 0:
            last_lat, last_lon, last_time = path[-1]
            dist_to_goal = self._distance_nm(
                last_lat, last_lon, goal_node.lat, goal_node.lon
            )
            if dist_to_goal > 1:
                # Add final segment to goal
                time_to_goal = dist_to_goal / target_speed_kts
                final_time = last_time + timedelta(hours=time_to_goal)
                path.append((goal_node.lat, goal_node.lon, final_time))

        # Calculate total statistics
        waypoints = [(lat, lon) for lat, lon, _ in path]
        total_fuel = final_node.g_cost

        # Add final segment fuel if needed
        if len(path) > 1:
            last_seg_dist = self._distance_nm(
                path[-2][0], path[-2][1], path[-1][0], path[-1][1]
            )
            last_seg_fuel = self.vessel_model.calculate_fuel_consumption(
                target_speed_kts, is_laden, None, last_seg_dist
            )
            total_fuel += last_seg_fuel["fuel_mt"]

        total_distance = sum(
            self._distance_nm(waypoints[i][0], waypoints[i][1],
                            waypoints[i + 1][0], waypoints[i + 1][1])
            for i in range(len(waypoints) - 1)
        )

        total_time = (path[-1][2] - path[0][2]).total_seconds() / 3600

        return {
            "waypoints": waypoints,
            "total_fuel_mt": total_fuel,
            "total_distance_nm": total_distance,
            "total_time_hours": total_time,
            "departure_time": path[0][2],
            "arrival_time": path[-1][2],
        }

    def _great_circle_fallback(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        departure_time: datetime,
        is_laden: bool,
        target_speed_kts: float,
    ) -> Dict:
        """
        Fallback to simple great circle route.

        Args:
            start_pos: Start (lat, lon)
            end_pos: End (lat, lon)
            departure_time: Departure time
            is_laden: Loading condition
            target_speed_kts: Target speed

        Returns:
            Route dictionary
        """
        logger.info("Using great circle route")

        # Calculate distance
        distance_nm = self._distance_nm(
            start_pos[0], start_pos[1], end_pos[0], end_pos[1]
        )

        # Generate waypoints along great circle (every 100 nm)
        num_waypoints = max(2, int(distance_nm / 100))
        waypoints = []

        for i in range(num_waypoints + 1):
            fraction = i / num_waypoints
            lat, lon = self._interpolate_great_circle(
                start_pos[0], start_pos[1], end_pos[0], end_pos[1], fraction
            )
            waypoints.append((lat, lon))

        # Calculate fuel
        fuel_result = self.vessel_model.calculate_fuel_consumption(
            target_speed_kts, is_laden, None, distance_nm
        )

        arrival_time = departure_time + timedelta(hours=fuel_result["time_hours"])

        return {
            "waypoints": waypoints,
            "total_fuel_mt": fuel_result["fuel_mt"],
            "total_distance_nm": distance_nm,
            "total_time_hours": fuel_result["time_hours"],
            "departure_time": departure_time,
            "arrival_time": arrival_time,
        }

    @staticmethod
    def _distance_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate great circle distance in nautical miles.

        Uses Haversine formula.
        """
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlon_rad = np.radians(lon2 - lon1)
        dlat_rad = np.radians(lat2 - lat1)

        a = (
            np.sin(dlat_rad / 2) ** 2
            + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon_rad / 2) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        # Earth radius in nautical miles
        r_nm = 3440.065

        return r_nm * c

    @staticmethod
    def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate initial bearing from point 1 to point 2 (degrees)."""
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlon_rad = np.radians(lon2 - lon1)

        y = np.sin(dlon_rad) * np.cos(lat2_rad)
        x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(
            lat2_rad
        ) * np.cos(dlon_rad)

        bearing_rad = np.arctan2(y, x)
        bearing_deg = (np.degrees(bearing_rad) + 360) % 360

        return bearing_deg

    @staticmethod
    def _interpolate_great_circle(
        lat1: float, lon1: float, lat2: float, lon2: float, fraction: float
    ) -> Tuple[float, float]:
        """Interpolate point along great circle."""
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)

        d = 2 * np.arcsin(
            np.sqrt(
                np.sin((lat2_rad - lat1_rad) / 2) ** 2
                + np.cos(lat1_rad)
                * np.cos(lat2_rad)
                * np.sin((lon2_rad - lon1_rad) / 2) ** 2
            )
        )

        a = np.sin((1 - fraction) * d) / np.sin(d)
        b = np.sin(fraction * d) / np.sin(d)

        x = a * np.cos(lat1_rad) * np.cos(lon1_rad) + b * np.cos(lat2_rad) * np.cos(
            lon2_rad
        )
        y = a * np.cos(lat1_rad) * np.sin(lon1_rad) + b * np.cos(lat2_rad) * np.sin(
            lon2_rad
        )
        z = a * np.sin(lat1_rad) + b * np.sin(lat2_rad)

        lat = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
        lon = np.degrees(np.arctan2(y, x))

        return lat, lon

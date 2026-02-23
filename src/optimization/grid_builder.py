"""
Standalone grid builder for WINDMAR routing engines.

Extracted from RouteOptimizer._build_grid() and VisirOptimizer._build_spatial_grid()
to provide a shared, testable grid generation module.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.data.land_mask import is_ocean

logger = logging.getLogger(__name__)


@dataclass
class GridCell:
    """A cell in the routing grid."""
    lat: float
    lon: float
    row: int
    col: int


class GridBuilder:
    """Builds uniform or variable-resolution routing grids."""

    @staticmethod
    def build_uniform(
        corridor_waypoints: List[Tuple[float, float]],
        resolution_deg: float = 0.2,
        margin_deg: float = 5.0,
        filter_land: bool = True,
    ) -> Dict[Tuple[int, int], GridCell]:
        """
        Build a uniform-resolution routing grid covering the corridor.

        This is the logic extracted from RouteOptimizer._build_grid().
        
        Args:
            corridor_waypoints: [(lat, lon), ...] defining the routing corridor
            resolution_deg: Grid cell size in degrees
            margin_deg: Margin around corridor bounding box
            filter_land: Whether to exclude land cells
            
        Returns:
            Dict mapping (row, col) to GridCell
        """
        lats = [wp[0] for wp in corridor_waypoints]
        lons = [wp[1] for wp in corridor_waypoints]

        lat_min = min(lats) - margin_deg
        lat_max = max(lats) + margin_deg
        lon_min = min(lons) - margin_deg
        lon_max = max(lons) + margin_deg

        lat_min = max(lat_min, -85)
        lat_max = min(lat_max, 85)

        if lon_max - lon_min > 180:
            lon_min, lon_max = -180, 180

        grid = {}
        land_cells = 0
        row = 0
        lat = lat_min
        while lat <= lat_max:
            col = 0
            lon = lon_min
            while lon <= lon_max:
                if filter_land and not is_ocean(lat, lon):
                    land_cells += 1
                else:
                    cell = GridCell(lat=lat, lon=lon, row=row, col=col)
                    grid[(row, col)] = cell
                lon += resolution_deg
                col += 1
            lat += resolution_deg
            row += 1

        total_cells = row * col if row > 0 and col > 0 else 1
        logger.info(f"Built uniform grid: {len(grid)} ocean cells, {land_cells} land cells filtered "
                   f"({row} rows x {col} cols, {land_cells/total_cells*100:.1f}% land)")
        return grid

    @staticmethod
    def build_spatial(
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        resolution_deg: float = 0.25,
        margin_deg: float = 5.0,
        filter_land: bool = True,
    ) -> Tuple[Dict[Tuple[int, int], Tuple[float, float]], Dict[str, float]]:
        """
        Build a 2-D (row, col) -> (lat, lon) ocean grid with bounds metadata.

        This is the logic extracted from VisirOptimizer._build_spatial_grid().
        
        Args:
            origin: (lat, lon) start point
            destination: (lat, lon) end point
            resolution_deg: Grid cell size in degrees
            margin_deg: Margin around corridor bounding box
            filter_land: Whether to exclude land cells
            
        Returns:
            Tuple of (grid dict, grid_bounds dict)
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
                lon += resolution_deg
                col += 1
            num_cols = max(num_cols, col)
            lat += resolution_deg
            num_rows += 1

        grid_bounds = {
            "lat_min": lat_min,
            "lon_min": lon_min,
            "num_rows": num_rows,
            "num_cols": num_cols,
        }
        logger.info(f"Built spatial grid: {len(grid)} ocean cells ({num_rows}x{num_cols})")
        return grid, grid_bounds

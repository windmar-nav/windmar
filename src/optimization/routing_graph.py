"""
Variable-resolution routing graph for WINDMAR.

Two-tier grid:
- 0.5° cells in open ocean
- 0.1° cells nearshore (within ~50nm of coast)

Cross-resolution connectivity: fine cells at tier boundaries connect
to coarse cells bidirectionally. Spatial lookup via STRtree.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """A node in the variable-resolution routing graph."""
    id: str                    # "fine_234_567" or "coarse_12_34"
    lat: float
    lon: float
    resolution_deg: float
    neighbors: List[str] = field(default_factory=list)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, GraphNode):
            return self.id == other.id
        return NotImplemented


class RoutingGraph:
    """
    Two-tier variable-resolution routing graph.

    Coarse (0.5°) over open ocean, fine (0.1°) within ~50nm of coast.
    """

    FINE_RESOLUTION = 0.05
    COARSE_RESOLUTION = 0.5
    NEARSHORE_THRESHOLD_DEG = 1.5  # ~90nm at mid-latitudes — catches narrow peninsulas

    # 16-connected grid directions (shared with RouteOptimizer)
    DIRECTIONS = [
        (-1, 0), (0, 1), (1, 0), (0, -1),
        (-1, 1), (1, 1), (1, -1), (-1, -1),
        (-2, 1), (-1, 2), (1, 2), (2, 1),
        (2, -1), (1, -2), (-1, -2), (-2, -1),
    ]

    def __init__(
        self,
        corridor_waypoints: List[Tuple[float, float]],
        margin_deg: float = 5.0,
    ):
        """
        Initialize the routing graph builder.

        Args:
            corridor_waypoints: [(lat, lon), ...] defining the routing corridor
            margin_deg: Margin around corridor bbox
        """
        self.corridor_waypoints = corridor_waypoints
        self.margin_deg = margin_deg

        self._nodes: Dict[str, GraphNode] = {}
        self._strtree = None  # Lazy-built spatial index
        self._node_coords: List[Tuple[float, float]] = []  # For STRtree
        self._node_ids: List[str] = []  # Parallel to _node_coords

        # Grid index for neighbor lookup
        self._coarse_grid: Dict[Tuple[int, int], str] = {}  # (row, col) → node_id
        self._fine_grid: Dict[Tuple[int, int], str] = {}    # (row, col) → node_id

    def build(self) -> Dict[str, GraphNode]:
        """
        Build the two-tier routing graph.

        Algorithm:
        1. Create 0.5° coarse grid over corridor bbox, filter land
        2. For each coarse cell, compute distance to coast — if < 0.83° → nearshore
        3. Subdivide nearshore coarse cells into 5×5 fine (0.1°) cells, filter land
        4. Build neighbor edges: 16-connected within tier, cross-tier at boundaries
        5. Build STRtree index for spatial queries

        Returns:
            Dict of node_id → GraphNode
        """
        from src.data.land_mask import is_ocean, get_land_geometry

        # Compute bounding box
        lats = [wp[0] for wp in self.corridor_waypoints]
        lons = [wp[1] for wp in self.corridor_waypoints]
        lat_min = max(min(lats) - self.margin_deg, -85)
        lat_max = min(max(lats) + self.margin_deg, 85)
        lon_min = min(lons) - self.margin_deg
        lon_max = max(lons) + self.margin_deg

        if lon_max - lon_min > 180:
            lon_min, lon_max = -180, 180

        # Step 1: Build coarse grid
        land_geom = get_land_geometry()  # May be None if GSHHS unavailable

        coarse_cells = {}  # (row, col) → (lat, lon)
        row = 0
        lat = lat_min
        while lat <= lat_max:
            col = 0
            lon = lon_min
            while lon <= lon_max:
                if is_ocean(lat, lon):
                    coarse_cells[(row, col)] = (lat, lon)
                lon += self.COARSE_RESOLUTION
                col += 1
            lat += self.COARSE_RESOLUTION
            row += 1

        logger.info(f"Coarse grid: {len(coarse_cells)} ocean cells "
                    f"({row} rows, bbox [{lat_min:.1f},{lat_max:.1f},{lon_min:.1f},{lon_max:.1f}])")

        # Step 2: Identify nearshore cells
        nearshore_keys = set()
        if land_geom is not None:
            try:
                from shapely.geometry import Point

                for key, (lat, lon) in coarse_cells.items():
                    pt = Point(lon, lat)
                    dist = land_geom.distance(pt)
                    if dist < self.NEARSHORE_THRESHOLD_DEG:
                        nearshore_keys.add(key)
            except Exception as e:
                logger.warning(f"Distance-to-coast calculation failed: {e}. Using all coarse.")
        else:
            # Without GSHHS, use heuristic: cells near continental bounds
            logger.info("No land geometry available — skipping nearshore subdivision")

        logger.info(f"Nearshore cells: {len(nearshore_keys)} / {len(coarse_cells)}")

        # Step 3: Build nodes
        # Coarse nodes (non-nearshore)
        for key, (lat, lon) in coarse_cells.items():
            if key not in nearshore_keys:
                node_id = f"coarse_{key[0]}_{key[1]}"
                self._nodes[node_id] = GraphNode(
                    id=node_id, lat=lat, lon=lon,
                    resolution_deg=self.COARSE_RESOLUTION,
                )
                self._coarse_grid[key] = node_id

        # Fine nodes (subdivided nearshore)
        fine_offset = 10_000  # Row offset for fine grid
        for coarse_key in nearshore_keys:
            clat, clon = coarse_cells[coarse_key]
            # Subdivide into 5×5 fine cells
            for fi in range(5):
                for fj in range(5):
                    flat = clat - self.COARSE_RESOLUTION / 2 + (fi + 0.5) * self.FINE_RESOLUTION
                    flon = clon - self.COARSE_RESOLUTION / 2 + (fj + 0.5) * self.FINE_RESOLUTION
                    if is_ocean(flat, flon):
                        fine_row = fine_offset + coarse_key[0] * 5 + fi
                        fine_col = coarse_key[1] * 5 + fj
                        node_id = f"fine_{fine_row}_{fine_col}"
                        self._nodes[node_id] = GraphNode(
                            id=node_id, lat=flat, lon=flon,
                            resolution_deg=self.FINE_RESOLUTION,
                        )
                        self._fine_grid[(fine_row, fine_col)] = node_id

        logger.info(f"Graph nodes: {len(self._nodes)} "
                    f"(coarse={len(self._coarse_grid)}, fine={len(self._fine_grid)})")

        # Step 4: Build neighbor edges
        self._build_coarse_neighbors()
        self._build_fine_neighbors()
        self._build_cross_tier_neighbors(coarse_cells, nearshore_keys)

        # Step 5: Build spatial index
        self._build_strtree()

        return self._nodes

    def _build_coarse_neighbors(self):
        """Connect coarse nodes to their 16-connected coarse neighbors."""
        for (row, col), node_id in self._coarse_grid.items():
            node = self._nodes[node_id]
            for dr, dc in self.DIRECTIONS:
                nkey = (row + dr, col + dc)
                if nkey in self._coarse_grid:
                    node.neighbors.append(self._coarse_grid[nkey])

    def _build_fine_neighbors(self):
        """Connect fine nodes to their 16-connected fine neighbors."""
        for (row, col), node_id in self._fine_grid.items():
            node = self._nodes[node_id]
            for dr, dc in self.DIRECTIONS:
                nkey = (row + dr, col + dc)
                if nkey in self._fine_grid:
                    node.neighbors.append(self._fine_grid[nkey])

    def _build_cross_tier_neighbors(
        self,
        coarse_cells: Dict[Tuple[int, int], Tuple[float, float]],
        nearshore_keys: set,
    ):
        """Connect fine nodes at tier boundaries to adjacent coarse nodes."""
        # For each fine node, check if any of its cardinal+diagonal neighbors
        # would fall outside the fine grid → connect to nearest coarse node.
        # We check ALL 8 directions to ensure full connectivity at boundaries.
        for (frow, fcol), fine_id in self._fine_grid.items():
            fine_node = self._nodes[fine_id]
            for dr, dc in self.DIRECTIONS[:8]:  # Only cardinal+diagonal for cross-tier
                nkey = (frow + dr, fcol + dc)
                if nkey not in self._fine_grid:
                    # Look for a nearby coarse node
                    # Map fine coords back to approximate coarse coords
                    approx_crow = (frow - 10_000) // 5
                    approx_ccol = fcol // 5
                    # Check surrounding coarse cells
                    for cdr in range(-1, 2):
                        for cdc in range(-1, 2):
                            ckey = (approx_crow + cdr, approx_ccol + cdc)
                            if ckey in self._coarse_grid:
                                coarse_id = self._coarse_grid[ckey]
                                if coarse_id not in fine_node.neighbors:
                                    fine_node.neighbors.append(coarse_id)
                                    # Bidirectional
                                    coarse_node = self._nodes[coarse_id]
                                    if fine_id not in coarse_node.neighbors:
                                        coarse_node.neighbors.append(fine_id)

    def _build_strtree(self):
        """Build spatial index for get_nearest_node()."""
        try:
            from shapely.geometry import Point
            from shapely import strtree

            points = []
            ids = []
            for node_id, node in self._nodes.items():
                points.append(Point(node.lon, node.lat))
                ids.append(node_id)

            self._strtree = strtree.STRtree(points)
            self._node_coords = [(n.lat, n.lon) for n in self._nodes.values()]
            self._node_ids = list(self._nodes.keys())
        except Exception as e:
            logger.warning(f"STRtree build failed: {e}. Using brute-force nearest.")
            self._strtree = None

    def get_nearest_node(self, lat: float, lon: float) -> Optional[GraphNode]:
        """Find the nearest graph node to (lat, lon) using spatial index."""
        if not self._nodes:
            return None

        if self._strtree is not None:
            try:
                from shapely.geometry import Point
                pt = Point(lon, lat)
                idx = self._strtree.nearest(pt)
                node_id = self._node_ids[idx]
                return self._nodes[node_id]
            except Exception:
                pass

        # Brute-force fallback
        min_dist = float('inf')
        nearest = None
        for node in self._nodes.values():
            dist = (node.lat - lat) ** 2 + (node.lon - lon) ** 2
            if dist < min_dist:
                min_dist = dist
                nearest = node
        return nearest

    def get_neighbors(self, node_id: str) -> List[GraphNode]:
        """Return neighbor nodes for a given node."""
        node = self._nodes.get(node_id)
        if node is None:
            return []
        return [self._nodes[nid] for nid in node.neighbors if nid in self._nodes]

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def coarse_count(self) -> int:
        return len(self._coarse_grid)

    @property
    def fine_count(self) -> int:
        return len(self._fine_grid)

"""
Unit tests for Phase 3b: Variable Resolution Corridor Grid.

Tests the RoutingGraph two-tier grid construction, neighbor connectivity,
spatial lookup, and A* integration.
"""

import pytest

from src.optimization.routing_graph import RoutingGraph, GraphNode
from src.data.land_mask import get_land_mask_status


def _gshhs_available() -> bool:
    status = get_land_mask_status()
    return status["gshhs_loaded"]


needs_gshhs = pytest.mark.skipif(
    not _gshhs_available(),
    reason="GSHHS shapefiles not available"
)


# ---------------------------------------------------------------------------
# §1 – Graph construction
# ---------------------------------------------------------------------------
class TestGraphConstruction:
    """Verify graph builds correctly with two-tier resolution."""

    def test_builds_non_empty_graph(self):
        """Graph should contain nodes for an ocean corridor."""
        graph = RoutingGraph(
            corridor_waypoints=[(45.0, -30.0), (40.0, -20.0)],
            margin_deg=2.0,
        )
        nodes = graph.build()
        assert len(nodes) > 0

    def test_coarse_nodes_exist(self):
        """Open ocean corridor should have coarse (0.5°) nodes."""
        graph = RoutingGraph(
            corridor_waypoints=[(45.0, -30.0), (40.0, -20.0)],
            margin_deg=2.0,
        )
        graph.build()
        assert graph.coarse_count > 0

    @needs_gshhs
    def test_fine_nodes_near_coast(self):
        """Corridor near coastline should have fine (0.1°) nodes."""
        # English Channel → near coastline
        graph = RoutingGraph(
            corridor_waypoints=[(50.5, -1.0), (51.0, 1.5)],
            margin_deg=1.0,
        )
        graph.build()
        assert graph.fine_count > 0

    def test_no_land_nodes(self):
        """All nodes should be classified as ocean."""
        from src.data.land_mask import is_ocean

        graph = RoutingGraph(
            corridor_waypoints=[(45.0, -30.0), (40.0, -20.0)],
            margin_deg=2.0,
        )
        nodes = graph.build()
        for node_id, node in nodes.items():
            assert is_ocean(node.lat, node.lon), (
                f"Node {node_id} at ({node.lat}, {node.lon}) is land"
            )

    def test_node_resolution_fields(self):
        """Nodes should have correct resolution field."""
        graph = RoutingGraph(
            corridor_waypoints=[(45.0, -30.0), (40.0, -20.0)],
            margin_deg=2.0,
        )
        nodes = graph.build()
        for node_id, node in nodes.items():
            if node_id.startswith("coarse"):
                assert node.resolution_deg == 0.5
            elif node_id.startswith("fine"):
                assert node.resolution_deg == 0.1


# ---------------------------------------------------------------------------
# §2 – Cross-resolution connectivity
# ---------------------------------------------------------------------------
class TestCrossResolution:
    """Verify fine-to-coarse edges exist and are bidirectional."""

    @needs_gshhs
    def test_cross_tier_edges_exist(self):
        """Fine nodes at boundaries should connect to coarse nodes."""
        graph = RoutingGraph(
            corridor_waypoints=[(50.5, -1.0), (51.0, 1.5)],
            margin_deg=1.0,
        )
        nodes = graph.build()

        # Look for fine nodes with coarse neighbors
        cross_tier_found = False
        for node_id, node in nodes.items():
            if node_id.startswith("fine"):
                for nid in node.neighbors:
                    if nid.startswith("coarse"):
                        cross_tier_found = True
                        break
            if cross_tier_found:
                break

        assert cross_tier_found, "No cross-tier edges found"

    @needs_gshhs
    def test_cross_tier_bidirectional(self):
        """Cross-tier edges should be bidirectional."""
        graph = RoutingGraph(
            corridor_waypoints=[(50.5, -1.0), (51.0, 1.5)],
            margin_deg=1.0,
        )
        nodes = graph.build()

        for node_id, node in nodes.items():
            for nid in node.neighbors:
                if nid in nodes:
                    neighbor = nodes[nid]
                    # If fine→coarse, then coarse→fine should exist
                    if (node_id.startswith("fine") and nid.startswith("coarse")) or \
                       (node_id.startswith("coarse") and nid.startswith("fine")):
                        assert node_id in neighbor.neighbors, (
                            f"Edge {node_id} → {nid} exists but reverse doesn't"
                        )

    def test_no_orphan_nodes(self):
        """Every node should have at least one neighbor."""
        graph = RoutingGraph(
            corridor_waypoints=[(45.0, -30.0), (40.0, -20.0)],
            margin_deg=2.0,
        )
        nodes = graph.build()

        orphans = [nid for nid, n in nodes.items() if len(n.neighbors) == 0]
        # Allow small number of orphans at grid edges
        assert len(orphans) < len(nodes) * 0.05, (
            f"Too many orphans: {len(orphans)} / {len(nodes)}"
        )


# ---------------------------------------------------------------------------
# §3 – Spatial lookup
# ---------------------------------------------------------------------------
class TestSpatialLookup:
    """Test get_nearest_node() accuracy."""

    def test_nearest_node_accuracy(self):
        """Nearest node should be within 1° of query point."""
        graph = RoutingGraph(
            corridor_waypoints=[(45.0, -30.0), (40.0, -20.0)],
            margin_deg=2.0,
        )
        graph.build()

        query = (43.0, -25.0)
        nearest = graph.get_nearest_node(*query)
        assert nearest is not None

        dist_deg = ((nearest.lat - query[0]) ** 2 + (nearest.lon - query[1]) ** 2) ** 0.5
        assert dist_deg < 1.0, f"Nearest node is {dist_deg:.2f}° away"

    def test_handles_land_point(self):
        """Query for a land point should return nearest ocean node (not crash)."""
        graph = RoutingGraph(
            corridor_waypoints=[(45.0, -30.0), (40.0, -20.0)],
            margin_deg=2.0,
        )
        graph.build()

        # Land point (but graph was built with ocean corridor)
        nearest = graph.get_nearest_node(51.5, -0.1)  # London
        # Should return some node (nearest ocean point)
        assert nearest is not None

    def test_empty_graph_returns_none(self):
        """Empty graph returns None for nearest."""
        graph = RoutingGraph(corridor_waypoints=[(0, 0)], margin_deg=0.01)
        # Don't build — empty
        result = graph.get_nearest_node(0, 0)
        assert result is None


# ---------------------------------------------------------------------------
# §4 – A* integration
# ---------------------------------------------------------------------------
class TestAStarIntegration:
    """Test that A* works with variable resolution grid."""

    def test_variable_resolution_optimizer_creates(self):
        """RouteOptimizer with variable_resolution=True should instantiate."""
        from src.optimization.route_optimizer import RouteOptimizer
        opt = RouteOptimizer(variable_resolution=True)
        assert opt.variable_resolution is True

    def test_default_variable_resolution_true(self):
        """Default variable_resolution is True (v0.1.1: enabled by default)."""
        from src.optimization.route_optimizer import RouteOptimizer
        opt = RouteOptimizer()
        assert opt.variable_resolution is True


# ---------------------------------------------------------------------------
# §5 – GraphNode dataclass
# ---------------------------------------------------------------------------
class TestGraphNode:
    """Verify GraphNode equality and hashing."""

    def test_hash_by_id(self):
        a = GraphNode(id="coarse_1_2", lat=45.0, lon=-30.0, resolution_deg=0.5)
        b = GraphNode(id="coarse_1_2", lat=45.0, lon=-30.0, resolution_deg=0.5)
        assert hash(a) == hash(b)
        assert a == b

    def test_different_ids_not_equal(self):
        a = GraphNode(id="coarse_1_2", lat=45.0, lon=-30.0, resolution_deg=0.5)
        b = GraphNode(id="coarse_1_3", lat=45.0, lon=-29.5, resolution_deg=0.5)
        assert a != b

    def test_neighbors_default_empty(self):
        n = GraphNode(id="test", lat=0, lon=0, resolution_deg=0.5)
        assert n.neighbors == []

    def test_resolution_field(self):
        n = GraphNode(id="fine_1_2", lat=50.0, lon=1.0, resolution_deg=0.1)
        assert n.resolution_deg == 0.1

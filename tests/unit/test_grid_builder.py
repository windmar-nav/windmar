"""
Unit tests for GridBuilder module.

Tests grid generation, land filtering, and cross-resolution connectivity.
"""

import pytest

from src.optimization.grid_builder import GridBuilder, GridCell


class TestBuildUniform:
    """Tests for GridBuilder.build_uniform()."""

    def test_returns_dict_of_grid_cells(self):
        """Grid should be a dict mapping (row, col) to GridCell."""
        grid = GridBuilder.build_uniform(
            corridor_waypoints=[(45.0, -30.0), (40.0, -20.0)],
            resolution_deg=2.0,
            margin_deg=2.0,
            filter_land=False,
        )
        assert isinstance(grid, dict)
        assert len(grid) > 0
        first_key = next(iter(grid))
        assert isinstance(first_key, tuple)
        assert len(first_key) == 2
        cell = grid[first_key]
        assert isinstance(cell, GridCell)
        assert hasattr(cell, 'lat')
        assert hasattr(cell, 'lon')
        assert hasattr(cell, 'row')
        assert hasattr(cell, 'col')

    def test_land_filtering_reduces_cells(self):
        """With filter_land=True, grid should have fewer cells than without."""
        wps = [(45.0, 0.0), (50.0, 5.0)]  # Near Europe — lots of land
        grid_no_filter = GridBuilder.build_uniform(
            corridor_waypoints=wps, resolution_deg=1.0, margin_deg=2.0, filter_land=False,
        )
        grid_with_filter = GridBuilder.build_uniform(
            corridor_waypoints=wps, resolution_deg=1.0, margin_deg=2.0, filter_land=True,
        )
        assert len(grid_with_filter) <= len(grid_no_filter)

    def test_open_ocean_no_filtering(self):
        """In open ocean, filter_land=True should remove very few cells."""
        grid = GridBuilder.build_uniform(
            corridor_waypoints=[(45.0, -30.0), (40.0, -20.0)],
            resolution_deg=1.0, margin_deg=2.0, filter_land=True,
        )
        # Mid-Atlantic — should have many ocean cells
        assert len(grid) > 50

    def test_resolution_affects_cell_count(self):
        """Finer resolution should produce more cells."""
        wps = [(45.0, -30.0), (40.0, -20.0)]
        grid_coarse = GridBuilder.build_uniform(
            corridor_waypoints=wps, resolution_deg=2.0, margin_deg=2.0, filter_land=False,
        )
        grid_fine = GridBuilder.build_uniform(
            corridor_waypoints=wps, resolution_deg=1.0, margin_deg=2.0, filter_land=False,
        )
        assert len(grid_fine) > len(grid_coarse)

    def test_cells_within_bounds(self):
        """All cells should be within the expected bounding box."""
        wps = [(45.0, -30.0), (40.0, -20.0)]
        margin = 2.0
        grid = GridBuilder.build_uniform(
            corridor_waypoints=wps, resolution_deg=1.0, margin_deg=margin, filter_land=False,
        )
        for (row, col), cell in grid.items():
            assert cell.lat >= 40.0 - margin - 0.01
            assert cell.lat <= 45.0 + margin + 0.01
            assert cell.lon >= -30.0 - margin - 0.01
            assert cell.lon <= -20.0 + margin + 0.01


class TestBuildSpatial:
    """Tests for GridBuilder.build_spatial()."""

    def test_returns_grid_and_bounds(self):
        """Should return (grid_dict, grid_bounds_dict)."""
        grid, bounds = GridBuilder.build_spatial(
            origin=(45.0, -5.0), destination=(50.0, 5.0),
            resolution_deg=1.0, margin_deg=2.0, filter_land=False,
        )
        assert isinstance(grid, dict)
        assert isinstance(bounds, dict)
        assert "lat_min" in bounds
        assert "lon_min" in bounds
        assert "num_rows" in bounds
        assert "num_cols" in bounds

    def test_grid_values_are_tuples(self):
        """Grid values should be (lat, lon) tuples."""
        grid, _ = GridBuilder.build_spatial(
            origin=(45.0, -5.0), destination=(50.0, 5.0),
            resolution_deg=1.0, margin_deg=2.0, filter_land=False,
        )
        first_val = next(iter(grid.values()))
        assert isinstance(first_val, tuple)
        assert len(first_val) == 2

    def test_bounds_consistency(self):
        """Grid bounds should be consistent with actual grid cells."""
        grid, bounds = GridBuilder.build_spatial(
            origin=(45.0, -5.0), destination=(50.0, 5.0),
            resolution_deg=1.0, margin_deg=2.0, filter_land=False,
        )
        assert bounds["num_rows"] > 0
        assert bounds["num_cols"] > 0
        # All grid keys should be within bounds
        for (row, col) in grid:
            assert 0 <= row < bounds["num_rows"]
            assert 0 <= col < bounds["num_cols"]

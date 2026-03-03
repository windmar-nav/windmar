"""Tests for api.weather.ocean_mask — NaN-based masking."""

import numpy as np
import pytest

from api.weather.grid_processor import SubsampledGrid
from api.weather.ocean_mask import (
    build_ocean_mask_from_data,
    mask_velocity_with_nan,
)


class TestBuildOceanMaskFromData:
    def test_all_nan_is_land(self):
        """All-NaN grid → all-False (land) mask."""
        full_lats = np.arange(10.0)
        full_lons = np.arange(10.0)
        grid = SubsampledGrid(
            lats=full_lats, lons=full_lons, step=1, ny=10, nx=10,
        )
        grids = {"param": {0: (full_lats, full_lons, np.full((10, 10), np.nan))}}
        mask = build_ocean_mask_from_data(grids, "param", [0], grid, full_lats, full_lons)
        assert not any(any(row) for row in mask)

    def test_all_finite_is_ocean(self):
        """All-finite grid → all-True (ocean) mask."""
        full_lats = np.arange(10.0)
        full_lons = np.arange(10.0)
        grid = SubsampledGrid(
            lats=full_lats, lons=full_lons, step=1, ny=10, nx=10,
        )
        grids = {"param": {0: (full_lats, full_lons, np.ones((10, 10)))}}
        mask = build_ocean_mask_from_data(grids, "param", [0], grid, full_lats, full_lons)
        assert all(all(row) for row in mask)

    def test_union_across_hours(self):
        """Ocean mask is union (OR) of all forecast hours."""
        full_lats = np.arange(4.0)
        full_lons = np.arange(4.0)
        grid = SubsampledGrid(
            lats=full_lats, lons=full_lons, step=1, ny=4, nx=4,
        )
        # Hour 0: only top-left is finite
        d0 = np.full((4, 4), np.nan)
        d0[0, 0] = 1.0
        # Hour 3: only bottom-right is finite
        d3 = np.full((4, 4), np.nan)
        d3[3, 3] = 2.0
        grids = {
            "param": {
                0: (full_lats, full_lons, d0),
                3: (full_lats, full_lons, d3),
            }
        }
        mask = build_ocean_mask_from_data(grids, "param", [0, 3], grid, full_lats, full_lons)
        assert mask[0][0] is True  # from hour 0
        assert mask[3][3] is True  # from hour 3
        assert mask[1][1] is False  # never finite

    def test_subsampled_shape_match(self):
        """Mask shape matches grid.ny x grid.nx after subsampling."""
        full_lats = np.arange(100.0)
        full_lons = np.arange(100.0)
        grid = SubsampledGrid(
            lats=full_lats[::5], lons=full_lons[::5], step=5, ny=20, nx=20,
        )
        grids = {"p": {0: (full_lats, full_lons, np.ones((100, 100)))}}
        mask = build_ocean_mask_from_data(grids, "p", [0], grid, full_lats, full_lons)
        assert len(mask) == grid.ny
        assert len(mask[0]) == grid.nx


class TestMaskVelocityWithNan:
    def test_erodes_coastal_cells(self):
        """1-cell erosion: an ocean cell next to land is zeroed."""
        # 5x5 grid: center 3x3 is ocean, border is land
        u = np.ones((5, 5))
        v = np.ones((5, 5))
        lats = np.arange(5.0)
        lons = np.arange(5.0)
        grid = SubsampledGrid(lats=lats, lons=lons, step=1, ny=5, nx=5)

        # Mark border as NaN → land, center as finite → ocean
        u_in = np.full((5, 5), np.nan)
        v_in = np.full((5, 5), np.nan)
        u_in[1:4, 1:4] = 1.0
        v_in[1:4, 1:4] = 1.0

        u_m, v_m = mask_velocity_with_nan(u_in, v_in, grid)
        # Center cell (2,2) should survive; edge ocean cells (1,1) eroded
        assert u_m[2, 2] == 1.0
        assert u_m[1, 1] == 0.0  # eroded

    def test_all_ocean_preserves(self):
        """All-finite grid: no erosion, values preserved."""
        u = np.full((5, 5), 2.0)
        v = np.full((5, 5), 3.0)
        lats = np.arange(5.0)
        lons = np.arange(5.0)
        grid = SubsampledGrid(lats=lats, lons=lons, step=1, ny=5, nx=5)
        u_m, v_m = mask_velocity_with_nan(u, v, grid)
        np.testing.assert_array_equal(u_m, u)
        np.testing.assert_array_equal(v_m, v)

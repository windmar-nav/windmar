"""Tests for api.weather.grid_processor — shape alignment guarantees."""

import numpy as np
import pytest

from api.weather.grid_processor import (
    SubsampledGrid,
    compute_step,
    make_grid,
    subsample_2d,
    subsample_2d_raw,
    clamp_bbox,
)
from api.weather_fields import get_field


class TestSubsampledGrid:
    def test_valid_construction(self):
        lats = np.array([30, 31, 32, 33, 34])
        lons = np.array([10, 11, 12])
        g = SubsampledGrid(lats=lats, lons=lons, step=1, ny=5, nx=3)
        assert g.ny == 5
        assert g.nx == 3
        assert g.step == 1

    def test_dimension_mismatch_raises(self):
        lats = np.array([30, 31, 32])
        lons = np.array([10, 11])
        with pytest.raises(ValueError, match="dimension mismatch"):
            SubsampledGrid(lats=lats, lons=lons, step=1, ny=5, nx=2)

    def test_frozen(self):
        lats = np.array([30, 31])
        lons = np.array([10, 11])
        g = SubsampledGrid(lats=lats, lons=lons, step=1, ny=2, nx=2)
        with pytest.raises(AttributeError):
            g.step = 2


class TestComputeStep:
    def test_small_grid_step_1(self):
        lats = np.arange(100)
        lons = np.arange(80)
        assert compute_step(lats, lons, 200) == 1

    def test_large_grid_subsampled(self):
        lats = np.arange(500)
        lons = np.arange(500)
        step = compute_step(lats, lons, 200)
        assert step == 3  # ceil(500/200) = 3

    def test_minimum_step_is_1(self):
        lats = np.arange(10)
        lons = np.arange(10)
        assert compute_step(lats, lons, 1000) == 1


class TestMakeGrid:
    def test_grid_shape_consistency(self):
        lats = np.linspace(30, 60, 361)
        lons = np.linspace(-30, 45, 901)
        cfg = get_field("wind")
        grid = make_grid(lats, lons, cfg)
        assert grid.ny == len(grid.lats)
        assert grid.nx == len(grid.lons)

    def test_all_fields_produce_valid_grid(self):
        lats = np.linspace(20, 65, 541)
        lons = np.linspace(-35, 45, 961)
        for field_name in ("wind", "waves", "swell", "currents", "sst", "visibility", "ice"):
            cfg = get_field(field_name)
            grid = make_grid(lats, lons, cfg)
            assert grid.ny == len(grid.lats)
            assert grid.nx == len(grid.lons)
            assert grid.step >= 1

    def test_ocean_mask_shape_matches_data(self):
        """The critical invariant: data subsampled at grid.step must match grid.ny x grid.nx."""
        lats = np.linspace(20, 65, 541)
        lons = np.linspace(-35, 45, 961)
        cfg = get_field("waves")
        grid = make_grid(lats, lons, cfg)

        # Simulate subsampling a data array
        data = np.random.rand(541, 961)
        sub = data[::grid.step, ::grid.step]
        assert sub.shape == (grid.ny, grid.nx), (
            f"Shape mismatch: data {sub.shape} vs grid ({grid.ny},{grid.nx})"
        )


class TestSubsample2d:
    def test_basic_subsample(self):
        arr = np.ones((10, 10))
        result = subsample_2d(arr, step=2)
        assert len(result) == 5
        assert len(result[0]) == 5

    def test_nan_fill(self):
        arr = np.array([[1.0, np.nan], [np.nan, 2.0]])
        result = subsample_2d(arr, step=1, nan_fill=-999.0)
        assert result[0][1] == -999.0

    def test_none_input(self):
        assert subsample_2d(None, step=1) is None

    def test_raw_returns_ndarray(self):
        arr = np.ones((10, 10))
        result = subsample_2d_raw(arr, step=2)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5, 5)

    def test_raw_none_input(self):
        assert subsample_2d_raw(None, step=1) is None


class TestClampBbox:
    def test_within_limits(self):
        result = clamp_bbox(30, 60, -30, 40)
        assert result == (30, 60, -30, 40)

    def test_exceeding_limits(self):
        lat_min, lat_max, lon_min, lon_max = clamp_bbox(-90, 90, -180, 180, 50, 80)
        assert (lat_max - lat_min) <= 50
        assert (lon_max - lon_min) <= 80

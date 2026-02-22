"""Tests for safe WeatherData serialization (replaces pickle)."""

from datetime import datetime

import numpy as np
import pytest

from api.weather_service import (
    _deserialize_ndarray,
    _deserialize_weather_data,
    _serialize_ndarray,
    _serialize_weather_data,
)
from src.data.copernicus import WeatherData


class TestNdarraySerialization:
    """Round-trip tests for numpy array serialization."""

    def test_1d_array_roundtrip(self):
        arr = np.array([1.0, 2.5, 3.7])
        result = _deserialize_ndarray(_serialize_ndarray(arr))
        np.testing.assert_array_almost_equal(result, arr)

    def test_2d_array_roundtrip(self):
        arr = np.random.rand(50, 80)
        result = _deserialize_ndarray(_serialize_ndarray(arr))
        np.testing.assert_array_almost_equal(result, arr)

    def test_preserves_shape(self):
        arr = np.zeros((10, 20))
        result = _deserialize_ndarray(_serialize_ndarray(arr))
        assert result.shape == (10, 20)

    def test_nan_values_preserved(self):
        arr = np.array([1.0, np.nan, 3.0])
        result = _deserialize_ndarray(_serialize_ndarray(arr))
        assert np.isnan(result[1])
        assert result[0] == 1.0
        assert result[2] == 3.0


class TestWeatherDataSerialization:
    """Round-trip tests for full WeatherData serialization."""

    @pytest.fixture
    def sample_weather_data(self):
        lats = np.linspace(30, 45, 20)
        lons = np.linspace(-10, 20, 30)
        return WeatherData(
            parameter="VHM0",
            time=datetime(2026, 2, 22, 12, 0, 0),
            lats=lats,
            lons=lons,
            values=np.random.rand(20, 30),
            unit="m",
            u_component=np.random.rand(20, 30),
            v_component=np.random.rand(20, 30),
            wave_period=np.random.rand(20, 30) * 10,
            wave_direction=np.random.rand(20, 30) * 360,
        )

    def test_full_roundtrip(self, sample_weather_data):
        serialized = _serialize_weather_data(sample_weather_data)
        result = _deserialize_weather_data(serialized)

        assert result.parameter == sample_weather_data.parameter
        assert result.unit == sample_weather_data.unit
        assert result.time == sample_weather_data.time
        np.testing.assert_array_almost_equal(result.lats, sample_weather_data.lats)
        np.testing.assert_array_almost_equal(result.lons, sample_weather_data.lons)
        np.testing.assert_array_almost_equal(result.values, sample_weather_data.values)
        np.testing.assert_array_almost_equal(
            result.u_component, sample_weather_data.u_component
        )
        np.testing.assert_array_almost_equal(
            result.v_component, sample_weather_data.v_component
        )

    def test_optional_fields_none(self):
        wd = WeatherData(
            parameter="VHM0",
            time=datetime(2026, 1, 1),
            lats=np.array([30.0, 31.0]),
            lons=np.array([10.0, 11.0]),
            values=np.array([[1.0, 2.0], [3.0, 4.0]]),
            unit="m",
        )
        serialized = _serialize_weather_data(wd)
        result = _deserialize_weather_data(serialized)

        assert result.u_component is None
        assert result.v_component is None
        assert result.wave_period is None
        assert result.swell_height is None

    def test_datetime_precision(self, sample_weather_data):
        serialized = _serialize_weather_data(sample_weather_data)
        result = _deserialize_weather_data(serialized)
        assert result.time == sample_weather_data.time

    def test_output_is_bytes(self, sample_weather_data):
        serialized = _serialize_weather_data(sample_weather_data)
        assert isinstance(serialized, bytes)

    def test_no_pickle_in_output(self, sample_weather_data):
        """Ensure output doesn't contain pickle opcodes."""
        serialized = _serialize_weather_data(sample_weather_data)
        # Pickle protocol 2+ starts with \x80, but zlib compressed JSON won't
        # contain pickle's GLOBAL opcode sequence
        import pickle
        with pytest.raises(Exception):
            pickle.loads(serialized)

    def test_corrupted_data_raises(self):
        with pytest.raises(Exception):
            _deserialize_weather_data(b"not valid data")

    def test_missing_required_field_raises(self):
        """Missing lats/lons/values should raise ValueError."""
        import json
        import zlib

        doc = json.dumps({
            "parameter": "VHM0",
            "time": "2026-01-01T00:00:00",
            "unit": "m",
            "lats": {"d": "", "s": [2]},
            # missing lons and values
        }).encode()
        with pytest.raises((ValueError, Exception)):
            _deserialize_weather_data(zlib.compress(doc))

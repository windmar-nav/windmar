"""Verify ice latitude filter has been removed from CopernicusDataProvider."""

import inspect

from src.data.copernicus import CopernicusDataProvider


class TestIceLatitudeFilter:
    """Ensure fetch_ice_data and fetch_ice_forecast no longer gate on latitude."""

    def test_fetch_ice_data_no_latitude_gate(self):
        source = inspect.getsource(CopernicusDataProvider.fetch_ice_data)
        assert "< 55" not in source, "fetch_ice_data still contains latitude gate"

    def test_fetch_ice_forecast_no_latitude_gate(self):
        source = inspect.getsource(CopernicusDataProvider.fetch_ice_forecast)
        assert "< 55" not in source, "fetch_ice_forecast still contains latitude gate"

    def test_fetch_ice_data_has_debug_log(self):
        source = inspect.getsource(CopernicusDataProvider.fetch_ice_data)
        assert "Ice fetch for bbox" in source, "fetch_ice_data missing debug log"

    def test_fetch_ice_forecast_has_debug_log(self):
        source = inspect.getsource(CopernicusDataProvider.fetch_ice_forecast)
        assert "Ice forecast fetch for bbox" in source, "fetch_ice_forecast missing debug log"

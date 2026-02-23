"""Tests for input validation bounds and NaN/Inf sanitization."""

import pytest
from pydantic import ValidationError

from api.schemas.vessel import VesselConfig


class TestVesselConfigBounds:
    """VesselConfig must reject zero, negative, and extreme values."""

    def test_defaults_valid(self):
        vc = VesselConfig()
        assert vc.mcr_kw > 0
        assert vc.draft_laden > 0

    def test_zero_mcr_rejected(self):
        with pytest.raises(ValidationError):
            VesselConfig(mcr_kw=0)

    def test_negative_dwt_rejected(self):
        with pytest.raises(ValidationError):
            VesselConfig(dwt=-1000)

    def test_zero_draft_rejected(self):
        with pytest.raises(ValidationError):
            VesselConfig(draft_laden=0)

    def test_extreme_dwt_rejected(self):
        with pytest.raises(ValidationError):
            VesselConfig(dwt=999999)

    def test_extreme_beam_rejected(self):
        with pytest.raises(ValidationError):
            VesselConfig(beam=200)

    def test_valid_custom_values(self):
        vc = VesselConfig(dwt=30000, mcr_kw=6000, beam=28)
        assert vc.dwt == 30000

    def test_nan_mcr_rejected(self):
        with pytest.raises(ValidationError):
            VesselConfig(mcr_kw=float('nan'))

    def test_inf_speed_rejected(self):
        with pytest.raises(ValidationError):
            VesselConfig(service_speed_laden=float('inf'))



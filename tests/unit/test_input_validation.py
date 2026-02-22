"""Tests for input validation bounds and NaN/Inf sanitization."""

import math

import pytest
from pydantic import ValidationError

from api.routers.optimization import _safe_round
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


class TestSafeRound:
    """_safe_round must clamp NaN/Inf to fallback."""

    def test_normal_value(self):
        assert _safe_round(3.14159, 2) == 3.14

    def test_nan_returns_fallback(self):
        assert _safe_round(float('nan'), 2) == 0.0

    def test_inf_returns_fallback(self):
        assert _safe_round(float('inf'), 2) == 0.0

    def test_neg_inf_returns_fallback(self):
        assert _safe_round(float('-inf'), 2) == 0.0

    def test_custom_fallback(self):
        assert _safe_round(float('nan'), 2, fallback=-1.0) == -1.0

    def test_zero(self):
        assert _safe_round(0.0, 2) == 0.0

    def test_negative_value(self):
        assert _safe_round(-3.456, 1) == -3.5

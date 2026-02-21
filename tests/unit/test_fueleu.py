"""Tests for FuelEU Maritime compliance module."""

import pytest

from src.compliance.fueleu import (
    FuelEUCalculator,
    REFERENCE_GHG,
    LCV,
    WTT_FACTORS,
    TTW_FACTORS,
    REDUCTION_TARGETS,
)


@pytest.fixture
def calc():
    return FuelEUCalculator()


# =============================================================================
# GHG Intensity
# =============================================================================

class TestGHGIntensity:
    def test_hfo_only(self, calc):
        """HFO-only: WtW = 13.5 + 78.24 = 91.74 gCO2eq/MJ."""
        result = calc.calculate_ghg_intensity({"hfo": 5000})
        assert result.ghg_intensity == pytest.approx(91.74, abs=0.01)
        assert result.total_energy_mj > 0
        assert len(result.fuel_breakdown) == 1
        assert result.fuel_breakdown[0].fuel_type == "hfo"

    def test_vlsfo_only(self, calc):
        """VLSFO-only: WtW = 13.2 + 78.19 = 91.39."""
        result = calc.calculate_ghg_intensity({"vlsfo": 3000})
        assert result.ghg_intensity == pytest.approx(91.39, abs=0.01)

    def test_lng_only(self, calc):
        """LNG-only: WtW = 18.5 + 70.70 = 89.20."""
        result = calc.calculate_ghg_intensity({"lng": 2000})
        assert result.ghg_intensity == pytest.approx(89.20, abs=0.01)

    def test_mixed_fuels(self, calc):
        """Mixed HFO+LNG should be between the two pure intensities."""
        result = calc.calculate_ghg_intensity({"hfo": 3000, "lng": 2000})
        assert 89.20 < result.ghg_intensity < 91.74
        assert len(result.fuel_breakdown) == 2

    def test_zero_fuel(self, calc):
        """Zero fuel returns zero intensity."""
        result = calc.calculate_ghg_intensity({"hfo": 0})
        assert result.ghg_intensity == 0.0
        assert result.total_energy_mj == 0.0

    def test_unknown_fuel_skipped(self, calc):
        """Unknown fuel type is ignored, doesn't crash."""
        result = calc.calculate_ghg_intensity({"hfo": 1000, "nuclear": 500})
        assert result.ghg_intensity == pytest.approx(91.74, abs=0.01)
        assert len(result.fuel_breakdown) == 1

    def test_energy_calculation(self, calc):
        """Verify energy = mass * 1e6 * LCV."""
        result = calc.calculate_ghg_intensity({"mgo": 1000})
        expected_energy = 1000 * 1_000_000 * LCV["mgo"]
        assert result.total_energy_mj == pytest.approx(expected_energy, rel=1e-6)


# =============================================================================
# Compliance Balance
# =============================================================================

class TestComplianceBalance:
    def test_hfo_2025_deficit(self, calc):
        """HFO at 2025 (limit 89.34): 91.74 > 89.34 → deficit."""
        result = calc.calculate_compliance_balance({"hfo": 5000}, 2025)
        assert result.status == "deficit"
        assert result.compliance_balance_gco2eq < 0
        assert result.ghg_limit == pytest.approx(89.34, abs=0.01)
        assert result.reduction_target_pct == 2.0

    def test_lng_2025_compliant(self, calc):
        """LNG at 2025: 89.20 < 89.34 → compliant."""
        result = calc.calculate_compliance_balance({"lng": 5000}, 2025)
        assert result.status == "compliant"
        assert result.compliance_balance_gco2eq > 0

    def test_reduction_target_interpolation(self, calc):
        """Year 2028 should interpolate between 2025 (2%) and 2030 (6%)."""
        result = calc.calculate_compliance_balance({"hfo": 1000}, 2028)
        expected_pct = 2.0 + (6.0 - 2.0) * (2028 - 2025) / (2030 - 2025)
        assert result.reduction_target_pct == pytest.approx(expected_pct, abs=0.01)

    def test_exact_target_year(self, calc):
        """Exact target year uses exact reduction percentage."""
        result = calc.calculate_compliance_balance({"hfo": 1000}, 2030)
        assert result.reduction_target_pct == 6.0

    def test_before_2025(self, calc):
        """Before 2025, reduction target = 0%."""
        result = calc.calculate_compliance_balance({"hfo": 1000}, 2024)
        assert result.reduction_target_pct == 0.0


# =============================================================================
# Penalty
# =============================================================================

class TestPenalty:
    def test_compliant_no_penalty(self, calc):
        """Compliant vessel pays zero penalty."""
        result = calc.calculate_penalty({"lng": 5000}, 2025)
        assert result.penalty_eur == 0.0
        assert result.vlsfo_equivalent_mt == 0.0

    def test_deficit_positive_penalty(self, calc):
        """Deficit vessel gets a positive penalty."""
        result = calc.calculate_penalty({"hfo": 5000}, 2025)
        assert result.penalty_eur > 0
        assert result.vlsfo_equivalent_mt > 0
        assert result.non_compliant_energy_mj > 0

    def test_consecutive_escalation(self, calc):
        """Consecutive deficit years escalate penalty by 10% each."""
        base = calc.calculate_penalty({"hfo": 5000}, 2025, consecutive_deficit_years=0)
        escalated = calc.calculate_penalty({"hfo": 5000}, 2025, consecutive_deficit_years=2)
        assert escalated.penalty_eur == pytest.approx(base.penalty_eur * 1.2, rel=1e-4)

    def test_penalty_spread(self, calc):
        """Penalty per MT fuel is penalty / total fuel."""
        result = calc.calculate_penalty({"hfo": 5000}, 2025)
        if result.penalty_eur > 0:
            assert result.penalty_per_mt_fuel == pytest.approx(
                result.penalty_eur / 5000, rel=1e-4,
            )


# =============================================================================
# Pooling
# =============================================================================

class TestPooling:
    def test_single_vessel_pooling(self, calc):
        """Single vessel pool = individual result."""
        vessels = [{"name": "Vessel A", "fuel_mt": {"hfo": 5000}}]
        result = calc.simulate_pooling(vessels, 2025)
        assert len(result.per_vessel) == 1
        assert result.fleet_ghg_intensity == result.per_vessel[0].ghg_intensity

    def test_mixed_pool(self, calc):
        """Pool with HFO + LNG vessel: fleet intensity between the two."""
        vessels = [
            {"name": "HFO Ship", "fuel_mt": {"hfo": 5000}},
            {"name": "LNG Ship", "fuel_mt": {"lng": 5000}},
        ]
        result = calc.simulate_pooling(vessels, 2025)
        assert len(result.per_vessel) == 2
        hfo_intensity = result.per_vessel[0].ghg_intensity
        lng_intensity = result.per_vessel[1].ghg_intensity
        assert lng_intensity < result.fleet_ghg_intensity < hfo_intensity

    def test_pooling_helps_deficit(self, calc):
        """Adding a compliant vessel to a pool can turn deficit into surplus."""
        hfo_only = calc.simulate_pooling(
            [{"name": "HFO", "fuel_mt": {"hfo": 3000}}], 2025,
        )
        assert hfo_only.status == "deficit"

        mixed = calc.simulate_pooling(
            [
                {"name": "HFO", "fuel_mt": {"hfo": 1000}},
                {"name": "Methanol", "fuel_mt": {"methanol": 10000}},
            ],
            2025,
        )
        # Methanol has much higher volume but lower intensity
        # Fleet intensity should be pulled down


# =============================================================================
# Limits
# =============================================================================

class TestLimits:
    def test_limits_count(self, calc):
        """Should return one entry per target year."""
        limits = calc.get_limits_by_year()
        assert len(limits) == len(REDUCTION_TARGETS)

    def test_limits_2025(self, calc):
        """2025 limit: 91.16 * (1 - 0.02) = 89.3368."""
        limits = calc.get_limits_by_year()
        y2025 = next(l for l in limits if l["year"] == 2025)
        assert y2025["ghg_limit"] == pytest.approx(89.34, abs=0.01)

    def test_limits_2050(self, calc):
        """2050 limit: 91.16 * (1 - 0.80) = 18.232."""
        limits = calc.get_limits_by_year()
        y2050 = next(l for l in limits if l["year"] == 2050)
        assert y2050["ghg_limit"] == pytest.approx(18.23, abs=0.01)

    def test_limits_monotonically_decreasing(self, calc):
        """GHG limits must decrease over time."""
        limits = calc.get_limits_by_year()
        for i in range(1, len(limits)):
            assert limits[i]["ghg_limit"] < limits[i - 1]["ghg_limit"]


# =============================================================================
# Projection
# =============================================================================

class TestProjection:
    def test_projection_length(self, calc):
        """Projection should return one entry per year."""
        projections = calc.project_compliance({"hfo": 5000}, 2025, 2030)
        assert len(projections) == 6  # 2025..2030 inclusive

    def test_flat_intensity_tightening_limits(self, calc):
        """With flat fuel, limits tighten → eventually deficit."""
        projections = calc.project_compliance({"hfo": 5000}, 2025, 2050)
        # 2025: HFO intensity > 2025 limit → deficit from start
        assert projections[0].status == "deficit"
        # Deficit should worsen over time (more negative balance)
        assert projections[-1].compliance_balance_gco2eq < projections[0].compliance_balance_gco2eq

    def test_efficiency_improvement(self, calc):
        """Annual efficiency improvement reduces fuel and emissions."""
        flat = calc.project_compliance({"hfo": 5000}, 2025, 2030)
        improved = calc.project_compliance({"hfo": 5000}, 2025, 2030, annual_efficiency_improvement_pct=5.0)
        # Improved should have less penalty by 2030
        assert improved[-1].penalty_eur <= flat[-1].penalty_eur


# =============================================================================
# Fuel Info
# =============================================================================

class TestFuelInfo:
    def test_fuel_info_count(self, calc):
        """Should return one entry per fuel type."""
        fuels = calc.get_fuel_info()
        assert len(fuels) == len(LCV)

    def test_fuel_info_wtw(self, calc):
        """WtW = WtT + TtW for each fuel."""
        for fuel in calc.get_fuel_info():
            expected_wtw = WTT_FACTORS[fuel["id"]] + TTW_FACTORS[fuel["id"]]
            assert fuel["wtw_gco2eq_per_mj"] == pytest.approx(expected_wtw, abs=0.01)


# =============================================================================
# API Endpoint Tests (TestClient)
# =============================================================================

class TestFuelEUEndpoints:
    @pytest.fixture(autouse=True)
    def setup_client(self):
        """Create test client from the WINDMAR app."""
        try:
            from fastapi.testclient import TestClient
            from api.routers.fueleu import router
            from fastapi import FastAPI

            test_app = FastAPI()
            test_app.include_router(router)
            self.client = TestClient(test_app)
        except ImportError:
            pytest.skip("FastAPI test client not available")

    def test_get_fuel_types(self):
        resp = self.client.get("/api/fueleu/fuel-types")
        assert resp.status_code == 200
        data = resp.json()
        assert "fuel_types" in data
        assert len(data["fuel_types"]) == len(LCV)

    def test_get_limits(self):
        resp = self.client.get("/api/fueleu/limits")
        assert resp.status_code == 200
        data = resp.json()
        assert "limits" in data
        assert data["reference_ghg"] == REFERENCE_GHG

    def test_calculate(self):
        resp = self.client.post("/api/fueleu/calculate", json={
            "fuel_consumption_mt": {"hfo": 5000},
            "year": 2025,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["ghg_intensity"] == pytest.approx(91.74, abs=0.01)

    def test_compliance(self):
        resp = self.client.post("/api/fueleu/compliance", json={
            "fuel_consumption_mt": {"hfo": 5000},
            "year": 2025,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "deficit"

    def test_penalty(self):
        resp = self.client.post("/api/fueleu/penalty", json={
            "fuel_consumption_mt": {"hfo": 5000},
            "year": 2025,
            "consecutive_deficit_years": 0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["penalty_eur"] > 0

    def test_pooling(self):
        resp = self.client.post("/api/fueleu/pooling", json={
            "vessels": [
                {"name": "Ship A", "fuel_mt": {"hfo": 5000}},
                {"name": "Ship B", "fuel_mt": {"lng": 5000}},
            ],
            "year": 2025,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["per_vessel"]) == 2

    def test_project(self):
        resp = self.client.post("/api/fueleu/project", json={
            "fuel_consumption_mt": {"hfo": 5000},
            "start_year": 2025,
            "end_year": 2030,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["projections"]) == 6

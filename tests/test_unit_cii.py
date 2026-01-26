"""
Unit tests for CII (Carbon Intensity Indicator) Calculator.

Tests compliance calculations according to IMO MEPC.339(76).
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.compliance.cii import (
    CIICalculator,
    CIIRating,
    CIIResult,
    VesselType,
    estimate_cii_from_route,
    annualize_voyage_cii,
)


class TestCIICalculator:
    """Unit tests for CIICalculator class."""

    @pytest.fixture
    def tanker_calculator(self):
        """Create a tanker calculator for testing."""
        return CIICalculator(
            vessel_type=VesselType.TANKER,
            dwt=49000,
            year=2024
        )

    @pytest.fixture
    def bulk_calculator(self):
        """Create a bulk carrier calculator for testing."""
        return CIICalculator(
            vessel_type=VesselType.BULK_CARRIER,
            dwt=75000,
            year=2024
        )

    def test_calculator_initialization(self, tanker_calculator):
        """Test calculator initializes correctly."""
        assert tanker_calculator.vessel_type == VesselType.TANKER
        assert tanker_calculator.dwt == 49000
        assert tanker_calculator.year == 2024
        assert tanker_calculator.capacity == 49000

    def test_calculator_gt_required_for_passenger(self):
        """Test GT is required for passenger vessels."""
        with pytest.raises(ValueError, match="GT required"):
            CIICalculator(
                vessel_type=VesselType.CRUISE_PASSENGER,
                dwt=50000,
                year=2024
            )

    def test_calculator_gt_used_for_passenger(self):
        """Test GT is used for passenger vessel capacity."""
        calc = CIICalculator(
            vessel_type=VesselType.CRUISE_PASSENGER,
            dwt=50000,
            gt=100000,
            year=2024
        )
        assert calc.capacity == 100000

    def test_co2_factor_hfo(self, tanker_calculator):
        """Test CO2 emission factor for HFO."""
        assert tanker_calculator.CO2_FACTORS["hfo"] == 3.114

    def test_co2_factor_vlsfo(self, tanker_calculator):
        """Test CO2 emission factor for VLSFO."""
        assert tanker_calculator.CO2_FACTORS["vlsfo"] == 3.114

    def test_co2_factor_lng(self, tanker_calculator):
        """Test CO2 emission factor for LNG."""
        assert tanker_calculator.CO2_FACTORS["lng"] == 2.750

    def test_co2_factor_mgo(self, tanker_calculator):
        """Test CO2 emission factor for MGO."""
        assert tanker_calculator.CO2_FACTORS["mgo"] == 3.206

    def test_calculate_co2_emissions(self, tanker_calculator):
        """Test CO2 emissions calculation."""
        fuel_mt = {"hfo": 1000, "mgo": 100}
        co2 = tanker_calculator._calculate_co2_emissions(fuel_mt)

        expected = 1000 * 3.114 + 100 * 3.206
        assert abs(co2 - expected) < 0.1

    def test_calculate_reference_cii_tanker(self, tanker_calculator):
        """Test reference CII calculation for tanker."""
        ref = tanker_calculator._calculate_reference_cii()

        # Reference formula: CII_ref = a × Capacity^(-c)
        # For tanker: a=5247, c=0.610
        expected = 5247 * (49000 ** -0.610)
        assert abs(ref - expected) < 0.01

    def test_reduction_factor_2024(self, tanker_calculator):
        """Test reduction factor for 2024."""
        factor = tanker_calculator._get_reduction_factor(2024)
        assert factor == 7.0

    def test_reduction_factor_2023(self, tanker_calculator):
        """Test reduction factor for 2023."""
        factor = tanker_calculator._get_reduction_factor(2023)
        assert factor == 5.0

    def test_reduction_factor_2030(self, tanker_calculator):
        """Test reduction factor for 2030."""
        factor = tanker_calculator._get_reduction_factor(2030)
        assert factor == 19.0

    def test_reduction_factor_extrapolation(self, tanker_calculator):
        """Test reduction factor extrapolation beyond 2030."""
        factor = tanker_calculator._get_reduction_factor(2032)
        # 2030 factor (19) + 2 years × 2% per year
        assert factor == 23.0

    def test_calculate_basic(self, tanker_calculator):
        """Test basic CII calculation."""
        result = tanker_calculator.calculate(
            total_fuel_mt={"vlsfo": 5000},
            total_distance_nm=50000
        )

        assert isinstance(result, CIIResult)
        assert result.attained_cii > 0
        assert result.required_cii > 0
        assert result.rating in CIIRating
        assert result.year == 2024
        assert result.vessel_type == VesselType.TANKER

    def test_calculate_returns_all_fields(self, tanker_calculator):
        """Test calculation returns all required fields."""
        result = tanker_calculator.calculate(
            total_fuel_mt={"vlsfo": 5000},
            total_distance_nm=50000
        )

        assert result.total_co2_mt > 0
        assert result.total_distance_nm == 50000
        assert result.capacity == 49000
        assert "A_upper" in result.rating_boundaries
        assert "B_upper" in result.rating_boundaries
        assert "C_upper" in result.rating_boundaries
        assert "D_upper" in result.rating_boundaries

    def test_rating_a_for_low_emissions(self, tanker_calculator):
        """Test A rating for very low emissions."""
        result = tanker_calculator.calculate(
            total_fuel_mt={"vlsfo": 1000},  # Very low fuel
            total_distance_nm=100000  # Long distance
        )

        assert result.rating == CIIRating.A

    def test_rating_e_for_high_emissions(self, tanker_calculator):
        """Test E rating for very high emissions."""
        result = tanker_calculator.calculate(
            total_fuel_mt={"hfo": 20000},  # Very high fuel
            total_distance_nm=10000  # Short distance
        )

        assert result.rating == CIIRating.E

    def test_calculate_from_voyages(self, tanker_calculator):
        """Test calculation from voyage list."""
        voyages = [
            {"fuel_mt": {"vlsfo": 1000}, "distance_nm": 10000},
            {"fuel_mt": {"vlsfo": 1500}, "distance_nm": 15000},
            {"fuel_mt": {"vlsfo": 2500}, "distance_nm": 25000},
        ]

        result = tanker_calculator.calculate_from_voyages(voyages)

        assert result.total_distance_nm == 50000
        assert result.total_co2_mt > 0

    def test_project_rating(self, tanker_calculator):
        """Test multi-year rating projection."""
        projections = tanker_calculator.project_rating(
            annual_fuel_mt={"vlsfo": 5000},
            annual_distance_nm=50000,
            years=[2024, 2025, 2026]
        )

        assert len(projections) == 3
        assert projections[0].year == 2024
        assert projections[1].year == 2025
        assert projections[2].year == 2026

        # Ratings should typically worsen over time due to tightening requirements
        assert projections[0].required_cii > projections[2].required_cii

    def test_project_rating_with_fuel_reduction(self, tanker_calculator):
        """Test projection with fuel efficiency improvements."""
        projections_no_reduction = tanker_calculator.project_rating(
            annual_fuel_mt={"vlsfo": 5000},
            annual_distance_nm=50000,
            years=[2024, 2026],
            fuel_reduction_rate=0.0
        )

        projections_with_reduction = tanker_calculator.project_rating(
            annual_fuel_mt={"vlsfo": 5000},
            annual_distance_nm=50000,
            years=[2024, 2026],
            fuel_reduction_rate=5.0  # 5% per year
        )

        # With reduction, 2026 should have lower attained CII
        assert projections_with_reduction[1].attained_cii < projections_no_reduction[1].attained_cii

    def test_calculate_required_reduction(self, tanker_calculator):
        """Test required reduction calculation."""
        reduction = tanker_calculator.calculate_required_reduction(
            current_fuel_mt={"vlsfo": 10000},
            current_distance_nm=50000,
            target_rating=CIIRating.C,
            target_year=2026
        )

        assert "reduction_needed_pct" in reduction
        assert "current_cii" in reduction
        assert "target_cii" in reduction
        assert "fuel_savings_mt" in reduction

    def test_rating_boundaries_ordered(self, tanker_calculator):
        """Test rating boundaries are properly ordered."""
        result = tanker_calculator.calculate(
            total_fuel_mt={"vlsfo": 5000},
            total_distance_nm=50000
        )

        bounds = result.rating_boundaries
        assert bounds["A_upper"] < bounds["B_upper"]
        assert bounds["B_upper"] < bounds["C_upper"]
        assert bounds["C_upper"] < bounds["D_upper"]


class TestVesselTypes:
    """Test different vessel type configurations."""

    @pytest.mark.parametrize("vessel_type", [
        VesselType.BULK_CARRIER,
        VesselType.GAS_CARRIER,
        VesselType.TANKER,
        VesselType.CONTAINER,
        VesselType.GENERAL_CARGO,
        VesselType.LNG_CARRIER,
        VesselType.RO_RO_CARGO,
    ])
    def test_vessel_type_calculations(self, vessel_type):
        """Test CII calculation works for all vessel types."""
        calc = CIICalculator(
            vessel_type=vessel_type,
            dwt=50000,
            year=2024
        )

        result = calc.calculate(
            total_fuel_mt={"vlsfo": 5000},
            total_distance_nm=50000
        )

        assert result.attained_cii > 0
        assert result.required_cii > 0
        assert result.rating in CIIRating


class TestHelperFunctions:
    """Test helper functions."""

    def test_estimate_cii_from_route(self):
        """Test quick CII estimation from route."""
        result = estimate_cii_from_route(
            fuel_mt=500,
            distance_nm=1000,
            dwt=49000,
            fuel_type="vlsfo",
            vessel_type=VesselType.TANKER,
            year=2024
        )

        assert isinstance(result, CIIResult)
        assert result.total_distance_nm == 1000

    def test_annualize_voyage_cii(self):
        """Test voyage annualization."""
        result = annualize_voyage_cii(
            voyage_fuel_mt=500,
            voyage_distance_nm=1000,
            voyages_per_year=50,
            dwt=49000,
            fuel_type="vlsfo",
            vessel_type=VesselType.TANKER,
            year=2024
        )

        assert result.total_distance_nm == 50000
        assert result.total_co2_mt == pytest.approx(500 * 50 * 3.114, rel=0.01)


class TestCIIRatingEnum:
    """Test CII Rating enumeration."""

    def test_rating_values(self):
        """Test rating values are correct."""
        assert CIIRating.A.value == "A"
        assert CIIRating.B.value == "B"
        assert CIIRating.C.value == "C"
        assert CIIRating.D.value == "D"
        assert CIIRating.E.value == "E"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

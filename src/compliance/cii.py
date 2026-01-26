"""
Carbon Intensity Indicator (CII) Calculator.

Implements IMO MEPC.339(76) and MEPC.352(78) regulations for:
- CII calculation (Attained CII)
- Reference line calculation (Required CII)
- Rating determination (A, B, C, D, E)
- Annual projections with tightening thresholds

Reference: IMO Resolution MEPC.339(76) - 2021 Guidelines on the operational
carbon intensity rating of ships (CII rating, G1 Guidelines)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


logger = logging.getLogger(__name__)


class VesselType(Enum):
    """IMO ship type categories for CII reference lines."""
    BULK_CARRIER = "bulk_carrier"
    GAS_CARRIER = "gas_carrier"
    TANKER = "tanker"
    CONTAINER = "container"
    GENERAL_CARGO = "general_cargo"
    REFRIGERATED_CARGO = "refrigerated_cargo"
    COMBINATION_CARRIER = "combination_carrier"
    LNG_CARRIER = "lng_carrier"
    RO_RO_CARGO = "roro_cargo"
    RO_RO_PASSENGER = "roro_passenger"
    CRUISE_PASSENGER = "cruise_passenger"


class CIIRating(Enum):
    """CII Rating grades (A = best, E = worst)."""
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"


@dataclass
class CIIResult:
    """Result of CII calculation."""
    attained_cii: float  # Actual carbon intensity (g CO2 / dwt·nm)
    required_cii: float  # Reference line value
    rating: CIIRating
    rating_boundaries: Dict[str, float]  # A/B/C/D/E boundaries
    reduction_factor: float  # Annual reduction factor applied
    year: int
    vessel_type: VesselType

    # Input values for transparency
    total_co2_mt: float
    total_distance_nm: float
    capacity: float  # DWT or GT depending on vessel type

    # Margin to next rating
    margin_to_downgrade: float  # % above current rating upper bound
    margin_to_upgrade: float  # % below current rating lower bound


@dataclass
class CIIProjection:
    """Multi-year CII projection result."""
    year: int
    attained_cii: float
    required_cii: float
    rating: CIIRating
    reduction_factor: float
    status: str  # "compliant", "at_risk", "non_compliant"


class CIICalculator:
    """
    Calculator for IMO Carbon Intensity Indicator (CII).

    Implements MEPC.339(76) guidelines for operational carbon intensity
    rating of ships.

    Example usage:
        calculator = CIICalculator(
            vessel_type=VesselType.TANKER,
            dwt=49000,
            year=2024
        )
        result = calculator.calculate(
            total_fuel_mt={"hfo": 5000, "vlsfo": 2000},
            total_distance_nm=50000
        )
        print(f"Rating: {result.rating.value}")
    """

    # CO2 emission factors (g CO2 / g fuel) - IMO MEPC.308(73)
    CO2_FACTORS = {
        "hfo": 3.114,      # Heavy Fuel Oil
        "lfo": 3.114,      # Light Fuel Oil
        "vlsfo": 3.114,    # Very Low Sulphur Fuel Oil
        "mdo": 3.206,      # Marine Diesel Oil
        "mgo": 3.206,      # Marine Gas Oil
        "lng": 2.750,      # LNG (tank-to-wake)
        "lpg_propane": 3.000,
        "lpg_butane": 3.030,
        "methanol": 1.375,
        "ethanol": 1.913,
    }

    # Reference line parameters: CII_ref = a × Capacity^(-c)
    # From MEPC.339(76) Table 1
    REFERENCE_PARAMS = {
        VesselType.BULK_CARRIER: {"a": 4745, "c": 0.622},
        VesselType.GAS_CARRIER: {"a": 14405, "c": 0.901},
        VesselType.TANKER: {"a": 5247, "c": 0.610},
        VesselType.CONTAINER: {"a": 1984, "c": 0.489},
        VesselType.GENERAL_CARGO: {"a": 588, "c": 0.3885},
        VesselType.REFRIGERATED_CARGO: {"a": 4600, "c": 0.557},
        VesselType.COMBINATION_CARRIER: {"a": 5119, "c": 0.622},
        VesselType.LNG_CARRIER: {"a": 9827, "c": 0.827},
        VesselType.RO_RO_CARGO: {"a": 10952, "c": 0.637},
        VesselType.RO_RO_PASSENGER: {"a": 7540, "c": 0.587},
        VesselType.CRUISE_PASSENGER: {"a": 930, "c": 0.383},
    }

    # Rating boundary vectors (dd1, dd2, dd3, dd4) from MEPC.339(76) Table 4
    # Boundaries relative to reference line: d1=exp(dd1), d2=exp(dd2), etc.
    RATING_VECTORS = {
        VesselType.BULK_CARRIER: {"dd1": 0.86, "dd2": 0.94, "dd3": 1.06, "dd4": 1.18},
        VesselType.GAS_CARRIER: {"dd1": 0.81, "dd2": 0.91, "dd3": 1.12, "dd4": 1.44},
        VesselType.TANKER: {"dd1": 0.82, "dd2": 0.93, "dd3": 1.08, "dd4": 1.28},
        VesselType.CONTAINER: {"dd1": 0.83, "dd2": 0.94, "dd3": 1.07, "dd4": 1.19},
        VesselType.GENERAL_CARGO: {"dd1": 0.83, "dd2": 0.94, "dd3": 1.06, "dd4": 1.19},
        VesselType.REFRIGERATED_CARGO: {"dd1": 0.78, "dd2": 0.91, "dd3": 1.07, "dd4": 1.20},
        VesselType.COMBINATION_CARRIER: {"dd1": 0.87, "dd2": 0.96, "dd3": 1.06, "dd4": 1.14},
        VesselType.LNG_CARRIER: {"dd1": 0.89, "dd2": 0.98, "dd3": 1.06, "dd4": 1.13},
        VesselType.RO_RO_CARGO: {"dd1": 0.66, "dd2": 0.90, "dd3": 1.11, "dd4": 1.37},
        VesselType.RO_RO_PASSENGER: {"dd1": 0.72, "dd2": 0.90, "dd3": 1.12, "dd4": 1.41},
        VesselType.CRUISE_PASSENGER: {"dd1": 0.87, "dd2": 0.95, "dd3": 1.06, "dd4": 1.16},
    }

    # Annual reduction factors (Z%) from MEPC.338(76)
    # Required CII = Reference × (1 - Z/100)
    REDUCTION_FACTORS = {
        2019: 0.0,   # Baseline year
        2020: 1.0,
        2021: 2.0,
        2022: 3.0,
        2023: 5.0,   # CII became mandatory
        2024: 7.0,
        2025: 9.0,
        2026: 11.0,
        2027: 13.0,  # Projected (subject to IMO review)
        2028: 15.0,
        2029: 17.0,
        2030: 19.0,  # IMO 2030 target: 40% reduction vs 2008
    }

    def __init__(
        self,
        vessel_type: VesselType,
        dwt: float,
        year: int = 2024,
        gt: Optional[float] = None,
    ):
        """
        Initialize CII calculator.

        Args:
            vessel_type: IMO vessel type category
            dwt: Deadweight tonnage (used for most ship types)
            year: Calculation year (affects reduction factor)
            gt: Gross tonnage (used for cruise/ro-ro passenger ships)
        """
        self.vessel_type = vessel_type
        self.dwt = dwt
        self.gt = gt
        self.year = year

        # Determine capacity metric
        if vessel_type in [VesselType.CRUISE_PASSENGER, VesselType.RO_RO_PASSENGER]:
            if gt is None:
                raise ValueError(f"GT required for {vessel_type.value}")
            self.capacity = gt
        else:
            self.capacity = dwt

    def calculate(
        self,
        total_fuel_mt: Dict[str, float],
        total_distance_nm: float,
        year: Optional[int] = None,
    ) -> CIIResult:
        """
        Calculate CII and determine rating.

        Args:
            total_fuel_mt: Dict of fuel type -> consumption in metric tons
                          e.g., {"hfo": 5000, "vlsfo": 2000}
            total_distance_nm: Total distance sailed in nautical miles
            year: Override year for calculation (uses instance year if None)

        Returns:
            CIIResult with attained CII, required CII, and rating
        """
        calc_year = year or self.year

        # Calculate total CO2 emissions
        total_co2_mt = self._calculate_co2_emissions(total_fuel_mt)

        # Calculate attained CII (g CO2 / capacity·nm)
        # Convert MT to grams
        total_co2_g = total_co2_mt * 1_000_000
        attained_cii = total_co2_g / (self.capacity * total_distance_nm)

        # Calculate reference line (CII_ref)
        reference_cii = self._calculate_reference_cii()

        # Apply reduction factor for the year
        reduction_factor = self._get_reduction_factor(calc_year)
        required_cii = reference_cii * (1 - reduction_factor / 100)

        # Calculate rating boundaries
        boundaries = self._calculate_rating_boundaries(required_cii)

        # Determine rating
        rating = self._determine_rating(attained_cii, boundaries)

        # Calculate margins
        margin_to_downgrade, margin_to_upgrade = self._calculate_margins(
            attained_cii, rating, boundaries
        )

        return CIIResult(
            attained_cii=round(attained_cii, 4),
            required_cii=round(required_cii, 4),
            rating=rating,
            rating_boundaries=boundaries,
            reduction_factor=reduction_factor,
            year=calc_year,
            vessel_type=self.vessel_type,
            total_co2_mt=round(total_co2_mt, 2),
            total_distance_nm=round(total_distance_nm, 2),
            capacity=self.capacity,
            margin_to_downgrade=round(margin_to_downgrade, 2),
            margin_to_upgrade=round(margin_to_upgrade, 2),
        )

    def calculate_from_voyages(
        self,
        voyages: List[Dict],
        year: Optional[int] = None,
    ) -> CIIResult:
        """
        Calculate CII from a list of voyage records.

        Args:
            voyages: List of voyage dicts with:
                     - fuel_mt: Dict of fuel consumption by type
                     - distance_nm: Distance sailed
            year: Calculation year

        Returns:
            CIIResult for aggregated voyages
        """
        total_fuel = {}
        total_distance = 0.0

        for voyage in voyages:
            for fuel_type, amount in voyage.get("fuel_mt", {}).items():
                total_fuel[fuel_type] = total_fuel.get(fuel_type, 0) + amount
            total_distance += voyage.get("distance_nm", 0)

        return self.calculate(total_fuel, total_distance, year)

    def project_rating(
        self,
        annual_fuel_mt: Dict[str, float],
        annual_distance_nm: float,
        years: Optional[List[int]] = None,
        fuel_reduction_rate: float = 0.0,
    ) -> List[CIIProjection]:
        """
        Project CII rating for future years.

        Args:
            annual_fuel_mt: Current annual fuel consumption
            annual_distance_nm: Current annual distance
            years: Years to project (defaults to 2024-2030)
            fuel_reduction_rate: Annual fuel reduction % (efficiency improvements)

        Returns:
            List of CIIProjection for each year
        """
        if years is None:
            years = list(range(2024, 2031))

        projections = []
        current_fuel = annual_fuel_mt.copy()

        for year in years:
            # Apply fuel reduction
            year_offset = year - years[0]
            efficiency_factor = (1 - fuel_reduction_rate / 100) ** year_offset

            adjusted_fuel = {
                fuel_type: amount * efficiency_factor
                for fuel_type, amount in current_fuel.items()
            }

            # Calculate CII for this year
            result = self.calculate(adjusted_fuel, annual_distance_nm, year)

            # Determine compliance status
            if result.rating in [CIIRating.A, CIIRating.B]:
                status = "compliant"
            elif result.rating == CIIRating.C:
                status = "at_risk"
            else:
                status = "non_compliant"

            projections.append(CIIProjection(
                year=year,
                attained_cii=result.attained_cii,
                required_cii=result.required_cii,
                rating=result.rating,
                reduction_factor=result.reduction_factor,
                status=status,
            ))

        return projections

    def calculate_required_reduction(
        self,
        current_fuel_mt: Dict[str, float],
        current_distance_nm: float,
        target_rating: CIIRating = CIIRating.C,
        target_year: int = 2026,
    ) -> Dict[str, float]:
        """
        Calculate fuel reduction needed to achieve target rating.

        Args:
            current_fuel_mt: Current fuel consumption
            current_distance_nm: Current distance
            target_rating: Desired rating (default C = compliant)
            target_year: Year to achieve target

        Returns:
            Dict with reduction requirements
        """
        # Get current CII
        current_result = self.calculate(current_fuel_mt, current_distance_nm, target_year)

        # Get boundary for target rating
        boundaries = current_result.rating_boundaries

        if target_rating == CIIRating.A:
            target_cii = boundaries["A_upper"]
        elif target_rating == CIIRating.B:
            target_cii = boundaries["B_upper"]
        elif target_rating == CIIRating.C:
            target_cii = boundaries["C_upper"]
        elif target_rating == CIIRating.D:
            target_cii = boundaries["D_upper"]
        else:
            # E rating - no reduction needed
            return {
                "reduction_needed_pct": 0,
                "current_cii": current_result.attained_cii,
                "target_cii": current_result.attained_cii,
                "fuel_savings_mt": 0,
                "message": "Already at E rating or better not specified",
            }

        # Calculate reduction needed
        if current_result.attained_cii <= target_cii:
            reduction_pct = 0
            message = f"Already at {target_rating.value} rating or better"
        else:
            reduction_pct = (1 - target_cii / current_result.attained_cii) * 100
            message = f"Reduce fuel by {reduction_pct:.1f}% to achieve {target_rating.value} rating"

        # Calculate absolute fuel savings
        total_current_fuel = sum(current_fuel_mt.values())
        fuel_savings = total_current_fuel * (reduction_pct / 100)

        return {
            "reduction_needed_pct": round(reduction_pct, 2),
            "current_cii": current_result.attained_cii,
            "target_cii": round(target_cii, 4),
            "current_rating": current_result.rating.value,
            "target_rating": target_rating.value,
            "fuel_savings_mt": round(fuel_savings, 2),
            "message": message,
        }

    def _calculate_co2_emissions(self, fuel_mt: Dict[str, float]) -> float:
        """Calculate total CO2 emissions from fuel consumption."""
        total_co2 = 0.0

        for fuel_type, amount_mt in fuel_mt.items():
            fuel_key = fuel_type.lower().replace(" ", "_")

            if fuel_key not in self.CO2_FACTORS:
                logger.warning(f"Unknown fuel type: {fuel_type}, using HFO factor")
                factor = self.CO2_FACTORS["hfo"]
            else:
                factor = self.CO2_FACTORS[fuel_key]

            # Convert MT fuel to MT CO2
            co2_mt = amount_mt * factor
            total_co2 += co2_mt

        return total_co2

    def _calculate_reference_cii(self) -> float:
        """Calculate CII reference line value."""
        params = self.REFERENCE_PARAMS[self.vessel_type]
        return params["a"] * (self.capacity ** (-params["c"]))

    def _get_reduction_factor(self, year: int) -> float:
        """Get reduction factor for given year."""
        if year in self.REDUCTION_FACTORS:
            return self.REDUCTION_FACTORS[year]
        elif year < 2019:
            return 0.0
        elif year > 2030:
            # Extrapolate beyond 2030 (2% per year)
            return self.REDUCTION_FACTORS[2030] + (year - 2030) * 2
        else:
            # Interpolate for any missing years
            years = sorted(self.REDUCTION_FACTORS.keys())
            for i, y in enumerate(years):
                if y > year:
                    prev_year = years[i - 1]
                    next_year = y
                    prev_factor = self.REDUCTION_FACTORS[prev_year]
                    next_factor = self.REDUCTION_FACTORS[next_year]
                    ratio = (year - prev_year) / (next_year - prev_year)
                    return prev_factor + ratio * (next_factor - prev_factor)
            return self.REDUCTION_FACTORS[2030]

    def _calculate_rating_boundaries(self, required_cii: float) -> Dict[str, float]:
        """Calculate rating boundary values."""
        vectors = self.RATING_VECTORS[self.vessel_type]

        # Boundaries are multiplicative factors on required CII
        return {
            "A_upper": required_cii * vectors["dd1"],
            "B_upper": required_cii * vectors["dd2"],
            "C_upper": required_cii * vectors["dd3"],
            "D_upper": required_cii * vectors["dd4"],
            "reference": required_cii,
        }

    def _determine_rating(
        self,
        attained_cii: float,
        boundaries: Dict[str, float],
    ) -> CIIRating:
        """Determine CII rating from attained CII and boundaries."""
        if attained_cii <= boundaries["A_upper"]:
            return CIIRating.A
        elif attained_cii <= boundaries["B_upper"]:
            return CIIRating.B
        elif attained_cii <= boundaries["C_upper"]:
            return CIIRating.C
        elif attained_cii <= boundaries["D_upper"]:
            return CIIRating.D
        else:
            return CIIRating.E

    def _calculate_margins(
        self,
        attained_cii: float,
        rating: CIIRating,
        boundaries: Dict[str, float],
    ) -> Tuple[float, float]:
        """Calculate margins to adjacent ratings."""
        if rating == CIIRating.A:
            margin_up = 100  # Already best
            margin_down = ((boundaries["B_upper"] - attained_cii) / attained_cii) * 100
        elif rating == CIIRating.B:
            margin_up = ((attained_cii - boundaries["A_upper"]) / attained_cii) * 100
            margin_down = ((boundaries["C_upper"] - attained_cii) / attained_cii) * 100
        elif rating == CIIRating.C:
            margin_up = ((attained_cii - boundaries["B_upper"]) / attained_cii) * 100
            margin_down = ((boundaries["D_upper"] - attained_cii) / attained_cii) * 100
        elif rating == CIIRating.D:
            margin_up = ((attained_cii - boundaries["C_upper"]) / attained_cii) * 100
            margin_down = ((boundaries["D_upper"] - attained_cii) / attained_cii) * 100
        else:  # E
            margin_up = ((attained_cii - boundaries["D_upper"]) / attained_cii) * 100
            margin_down = 0  # Already worst

        return max(0, margin_down), max(0, margin_up)


def estimate_cii_from_route(
    fuel_mt: float,
    distance_nm: float,
    dwt: float,
    fuel_type: str = "vlsfo",
    vessel_type: VesselType = VesselType.TANKER,
    year: int = 2024,
) -> CIIResult:
    """
    Quick CII estimation from a single route.

    Useful for route optimization to show CII impact.

    Args:
        fuel_mt: Fuel consumption for the route
        distance_nm: Route distance
        dwt: Vessel deadweight
        fuel_type: Fuel type used
        vessel_type: IMO vessel category
        year: Calculation year

    Returns:
        CIIResult (note: this is per-voyage, not annual)
    """
    calculator = CIICalculator(vessel_type=vessel_type, dwt=dwt, year=year)
    return calculator.calculate({fuel_type: fuel_mt}, distance_nm)


def annualize_voyage_cii(
    voyage_fuel_mt: float,
    voyage_distance_nm: float,
    voyages_per_year: int,
    dwt: float,
    fuel_type: str = "vlsfo",
    vessel_type: VesselType = VesselType.TANKER,
    year: int = 2024,
) -> CIIResult:
    """
    Estimate annual CII from typical voyage data.

    Args:
        voyage_fuel_mt: Fuel per voyage
        voyage_distance_nm: Distance per voyage
        voyages_per_year: Expected annual voyages
        dwt: Vessel deadweight
        fuel_type: Fuel type
        vessel_type: IMO vessel category
        year: Calculation year

    Returns:
        CIIResult for annualized operations
    """
    annual_fuel = voyage_fuel_mt * voyages_per_year
    annual_distance = voyage_distance_nm * voyages_per_year

    calculator = CIICalculator(vessel_type=vessel_type, dwt=dwt, year=year)
    return calculator.calculate({fuel_type: annual_fuel}, annual_distance)

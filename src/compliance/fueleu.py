"""
FuelEU Maritime (EU 2023/1805) GHG Intensity Calculator.

Implements the EU's Well-to-Wake GHG intensity framework:
- GHG intensity calculation (gCO2eq/MJ)
- Compliance balance (surplus/deficit vs annual limit)
- Penalty exposure estimation
- Fleet pooling simulation
- Multi-year compliance projection

Reference: EU Regulation 2023/1805 (FuelEU Maritime)
Baseline: 91.16 gCO2eq/MJ (2020 EU MRV reference)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Emission Factor Data (Annex II defaults)
# =============================================================================

# Lower Calorific Values (MJ/g fuel)
LCV = {
    "hfo": 0.0405,
    "lfo": 0.0410,
    "vlsfo": 0.0410,
    "mdo": 0.0427,
    "mgo": 0.0427,
    "lng": 0.0491,
    "lpg_propane": 0.0460,
    "lpg_butane": 0.0460,
    "methanol": 0.0199,
    "ethanol": 0.0268,
}

# Well-to-Tank emission factors (gCO2eq/MJ)
WTT_FACTORS = {
    "hfo": 13.5,
    "lfo": 13.2,
    "vlsfo": 13.2,
    "mdo": 14.4,
    "mgo": 14.4,
    "lng": 18.5,
    "lpg_propane": 7.8,
    "lpg_butane": 7.8,
    "methanol": 31.3,
    "ethanol": 31.3,
}

# Tank-to-Wake CO2eq factors (gCO2eq/MJ) — includes CO2, CH4, N2O
TTW_FACTORS = {
    "hfo": 78.24,
    "lfo": 78.19,
    "vlsfo": 78.19,
    "mdo": 76.37,
    "mgo": 76.37,
    "lng": 70.70,
    "lpg_propane": 65.22,
    "lpg_butane": 65.87,
    "methanol": 69.08,
    "ethanol": 69.08,
}

# GHG intensity reference and reduction targets
REFERENCE_GHG = 91.16  # gCO2eq/MJ (2020 baseline)

REDUCTION_TARGETS = {
    2025: 2.0,
    2030: 6.0,
    2035: 14.5,
    2040: 31.0,
    2045: 62.0,
    2050: 80.0,
}

# Penalty: €2,400 per MT VLSFO equivalent
PENALTY_EUR_PER_MT_VLSFO = 2400.0
CONSECUTIVE_YEAR_ESCALATION = 0.10  # 10% escalation


# =============================================================================
# Result Dataclasses
# =============================================================================

@dataclass
class FuelBreakdown:
    """Per-fuel breakdown of energy and emissions."""
    fuel_type: str
    mass_mt: float
    energy_mj: float
    wtt_gco2eq: float
    ttw_gco2eq: float
    wtw_gco2eq: float
    wtw_intensity: float  # gCO2eq/MJ for this fuel


@dataclass
class FuelEUResult:
    """Result of GHG intensity calculation."""
    ghg_intensity: float  # gCO2eq/MJ (WtW)
    total_energy_mj: float
    total_co2eq_g: float
    fuel_breakdown: List[FuelBreakdown] = field(default_factory=list)


@dataclass
class FuelEUComplianceResult:
    """Result of compliance balance calculation."""
    year: int
    ghg_intensity: float  # gCO2eq/MJ
    ghg_limit: float  # gCO2eq/MJ
    reduction_target_pct: float
    compliance_balance_gco2eq: float  # positive=surplus, negative=deficit
    total_energy_mj: float
    status: str  # "compliant" | "deficit"


@dataclass
class FuelEUPenaltyResult:
    """Result of penalty exposure calculation."""
    compliance_balance_gco2eq: float
    non_compliant_energy_mj: float
    vlsfo_equivalent_mt: float
    penalty_eur: float
    penalty_per_mt_fuel: float  # spread across total fuel consumed


@dataclass
class FuelEUPoolingVesselResult:
    """Individual vessel result within a pooling scenario."""
    name: str
    ghg_intensity: float
    total_energy_mj: float
    total_co2eq_g: float
    individual_balance_gco2eq: float
    status: str


@dataclass
class FuelEUPoolingResult:
    """Result of fleet pooling simulation."""
    fleet_ghg_intensity: float
    fleet_total_energy_mj: float
    fleet_total_co2eq_g: float
    fleet_balance_gco2eq: float
    per_vessel: List[FuelEUPoolingVesselResult] = field(default_factory=list)
    status: str = "compliant"


@dataclass
class FuelEUProjectionYear:
    """Single year in a multi-year projection."""
    year: int
    ghg_intensity: float
    ghg_limit: float
    reduction_target_pct: float
    compliance_balance_gco2eq: float
    total_energy_mj: float
    status: str
    penalty_eur: float


# =============================================================================
# Calculator
# =============================================================================

class FuelEUCalculator:
    """FuelEU Maritime (EU 2023/1805) GHG intensity calculator."""

    def calculate_ghg_intensity(self, fuel_mt: Dict[str, float]) -> FuelEUResult:
        """
        Calculate Well-to-Wake GHG intensity for given fuel consumption.

        Args:
            fuel_mt: Dict of fuel type -> consumption in metric tons
                     e.g., {"hfo": 5000, "vlsfo": 2000}

        Returns:
            FuelEUResult with GHG intensity and per-fuel breakdown
        """
        total_energy = 0.0
        total_co2eq = 0.0
        breakdown = []

        for fuel_type, mass_mt in fuel_mt.items():
            if mass_mt <= 0:
                continue

            fuel_key = fuel_type.lower().replace(" ", "_")
            if fuel_key not in LCV:
                logger.warning("Unknown fuel type: %s, skipping", fuel_type)
                continue

            # Energy in MJ: mass (MT) * 1e6 (g/MT) * LCV (MJ/g)
            energy_mj = mass_mt * 1_000_000 * LCV[fuel_key]

            wtt = WTT_FACTORS[fuel_key]
            ttw = TTW_FACTORS[fuel_key]
            wtw = wtt + ttw

            wtt_gco2eq = energy_mj * wtt
            ttw_gco2eq = energy_mj * ttw
            wtw_gco2eq = energy_mj * wtw

            total_energy += energy_mj
            total_co2eq += wtw_gco2eq

            breakdown.append(FuelBreakdown(
                fuel_type=fuel_key,
                mass_mt=mass_mt,
                energy_mj=round(energy_mj, 2),
                wtt_gco2eq=round(wtt_gco2eq, 2),
                ttw_gco2eq=round(ttw_gco2eq, 2),
                wtw_gco2eq=round(wtw_gco2eq, 2),
                wtw_intensity=round(wtw, 4),
            ))

        if total_energy <= 0:
            return FuelEUResult(
                ghg_intensity=0.0,
                total_energy_mj=0.0,
                total_co2eq_g=0.0,
                fuel_breakdown=[],
            )

        ghg_intensity = total_co2eq / total_energy

        return FuelEUResult(
            ghg_intensity=round(ghg_intensity, 4),
            total_energy_mj=round(total_energy, 2),
            total_co2eq_g=round(total_co2eq, 2),
            fuel_breakdown=breakdown,
        )

    def calculate_compliance_balance(
        self, fuel_mt: Dict[str, float], year: int
    ) -> FuelEUComplianceResult:
        """
        Calculate surplus/deficit vs annual GHG intensity limit.

        Args:
            fuel_mt: Fuel consumption by type in MT
            year: Compliance year

        Returns:
            FuelEUComplianceResult with balance (positive=surplus)
        """
        result = self.calculate_ghg_intensity(fuel_mt)
        reduction_pct = self._get_reduction_target(year)
        limit = REFERENCE_GHG * (1 - reduction_pct / 100)

        # Balance: positive means vessel is better than limit (surplus)
        # balance = (limit - intensity) * total_energy
        balance = (limit - result.ghg_intensity) * result.total_energy_mj

        status = "compliant" if balance >= 0 else "deficit"

        return FuelEUComplianceResult(
            year=year,
            ghg_intensity=result.ghg_intensity,
            ghg_limit=round(limit, 4),
            reduction_target_pct=reduction_pct,
            compliance_balance_gco2eq=round(balance, 2),
            total_energy_mj=result.total_energy_mj,
            status=status,
        )

    def calculate_penalty(
        self,
        fuel_mt: Dict[str, float],
        year: int,
        consecutive_deficit_years: int = 0,
    ) -> FuelEUPenaltyResult:
        """
        Calculate penalty exposure for a deficit.

        Args:
            fuel_mt: Fuel consumption by type in MT
            year: Compliance year
            consecutive_deficit_years: Number of prior consecutive deficit years
                                       (for 10% escalation)

        Returns:
            FuelEUPenaltyResult with penalty EUR amount
        """
        compliance = self.calculate_compliance_balance(fuel_mt, year)

        if compliance.compliance_balance_gco2eq >= 0:
            total_fuel_mt = sum(fuel_mt.values())
            return FuelEUPenaltyResult(
                compliance_balance_gco2eq=compliance.compliance_balance_gco2eq,
                non_compliant_energy_mj=0.0,
                vlsfo_equivalent_mt=0.0,
                penalty_eur=0.0,
                penalty_per_mt_fuel=0.0,
            )

        # Non-compliant energy: |deficit| / GHG intensity
        deficit_abs = abs(compliance.compliance_balance_gco2eq)
        non_compliant_energy_mj = deficit_abs / compliance.ghg_intensity

        # Convert to VLSFO equivalent MT: energy_mj / (LCV_vlsfo * 1e6)
        vlsfo_equivalent_mt = non_compliant_energy_mj / (LCV["vlsfo"] * 1_000_000)

        # Base penalty
        penalty_eur = vlsfo_equivalent_mt * PENALTY_EUR_PER_MT_VLSFO

        # Apply escalation for consecutive deficit years
        if consecutive_deficit_years > 0:
            escalation = 1 + CONSECUTIVE_YEAR_ESCALATION * consecutive_deficit_years
            penalty_eur *= escalation

        total_fuel_mt = sum(fuel_mt.values())
        penalty_per_mt = penalty_eur / total_fuel_mt if total_fuel_mt > 0 else 0

        return FuelEUPenaltyResult(
            compliance_balance_gco2eq=compliance.compliance_balance_gco2eq,
            non_compliant_energy_mj=round(non_compliant_energy_mj, 2),
            vlsfo_equivalent_mt=round(vlsfo_equivalent_mt, 4),
            penalty_eur=round(penalty_eur, 2),
            penalty_per_mt_fuel=round(penalty_per_mt, 2),
        )

    def simulate_pooling(
        self, vessels: List[Dict], year: int
    ) -> FuelEUPoolingResult:
        """
        Simulate fleet pooling — aggregate all vessels' energy and emissions.

        Pool: sum all energy, sum all emissions → fleet-average GHG intensity.
        Compare against limit → pooled balance.

        Args:
            vessels: List of dicts with keys:
                     - name: str
                     - fuel_mt: Dict[str, float]
            year: Compliance year

        Returns:
            FuelEUPoolingResult with fleet-level and per-vessel results
        """
        reduction_pct = self._get_reduction_target(year)
        limit = REFERENCE_GHG * (1 - reduction_pct / 100)

        fleet_energy = 0.0
        fleet_co2eq = 0.0
        per_vessel = []

        for v in vessels:
            result = self.calculate_ghg_intensity(v["fuel_mt"])
            individual_balance = (limit - result.ghg_intensity) * result.total_energy_mj

            fleet_energy += result.total_energy_mj
            fleet_co2eq += result.total_co2eq_g

            per_vessel.append(FuelEUPoolingVesselResult(
                name=v["name"],
                ghg_intensity=result.ghg_intensity,
                total_energy_mj=result.total_energy_mj,
                total_co2eq_g=result.total_co2eq_g,
                individual_balance_gco2eq=round(individual_balance, 2),
                status="compliant" if individual_balance >= 0 else "deficit",
            ))

        if fleet_energy <= 0:
            return FuelEUPoolingResult(
                fleet_ghg_intensity=0.0,
                fleet_total_energy_mj=0.0,
                fleet_total_co2eq_g=0.0,
                fleet_balance_gco2eq=0.0,
                per_vessel=per_vessel,
                status="compliant",
            )

        fleet_intensity = fleet_co2eq / fleet_energy
        fleet_balance = (limit - fleet_intensity) * fleet_energy

        return FuelEUPoolingResult(
            fleet_ghg_intensity=round(fleet_intensity, 4),
            fleet_total_energy_mj=round(fleet_energy, 2),
            fleet_total_co2eq_g=round(fleet_co2eq, 2),
            fleet_balance_gco2eq=round(fleet_balance, 2),
            per_vessel=per_vessel,
            status="compliant" if fleet_balance >= 0 else "deficit",
        )

    def project_compliance(
        self,
        fuel_mt: Dict[str, float],
        start_year: int = 2025,
        end_year: int = 2050,
        annual_efficiency_improvement_pct: float = 0.0,
    ) -> List[FuelEUProjectionYear]:
        """
        Project compliance across years as limits tighten.

        Args:
            fuel_mt: Current annual fuel consumption by type
            start_year: First projection year
            end_year: Last projection year
            annual_efficiency_improvement_pct: Annual fuel reduction %

        Returns:
            List of FuelEUProjectionYear for each year
        """
        projections = []
        base_fuel = fuel_mt.copy()

        for year in range(start_year, end_year + 1):
            year_offset = year - start_year
            efficiency_factor = (1 - annual_efficiency_improvement_pct / 100) ** year_offset

            adjusted_fuel = {
                ft: amount * efficiency_factor
                for ft, amount in base_fuel.items()
            }

            compliance = self.calculate_compliance_balance(adjusted_fuel, year)
            penalty = self.calculate_penalty(adjusted_fuel, year)

            projections.append(FuelEUProjectionYear(
                year=year,
                ghg_intensity=compliance.ghg_intensity,
                ghg_limit=compliance.ghg_limit,
                reduction_target_pct=compliance.reduction_target_pct,
                compliance_balance_gco2eq=compliance.compliance_balance_gco2eq,
                total_energy_mj=compliance.total_energy_mj,
                status=compliance.status,
                penalty_eur=penalty.penalty_eur,
            ))

        return projections

    def get_limits_by_year(self) -> List[Dict]:
        """Return GHG intensity limits for all target years."""
        limits = []
        for year, pct in sorted(REDUCTION_TARGETS.items()):
            limit = REFERENCE_GHG * (1 - pct / 100)
            limits.append({
                "year": year,
                "reduction_pct": pct,
                "ghg_limit": round(limit, 2),
            })
        return limits

    @staticmethod
    def get_fuel_info() -> List[Dict]:
        """Return fuel types with their emission factor data."""
        fuels = []
        for fuel_key in LCV:
            wtt = WTT_FACTORS[fuel_key]
            ttw = TTW_FACTORS[fuel_key]
            fuels.append({
                "id": fuel_key,
                "name": fuel_key.upper().replace("_", " "),
                "lcv_mj_per_g": LCV[fuel_key],
                "wtt_gco2eq_per_mj": wtt,
                "ttw_gco2eq_per_mj": ttw,
                "wtw_gco2eq_per_mj": round(wtt + ttw, 2),
            })
        return fuels

    # ---- private helpers ----------------------------------------------------

    @staticmethod
    def _get_reduction_target(year: int) -> float:
        """Get the applicable reduction target for a given year."""
        if year < 2025:
            return 0.0

        # Exact match
        if year in REDUCTION_TARGETS:
            return REDUCTION_TARGETS[year]

        # Interpolate between defined target years
        target_years = sorted(REDUCTION_TARGETS.keys())

        if year > target_years[-1]:
            return REDUCTION_TARGETS[target_years[-1]]

        for i, ty in enumerate(target_years):
            if ty > year:
                prev_year = target_years[i - 1]
                next_year = ty
                prev_pct = REDUCTION_TARGETS[prev_year]
                next_pct = REDUCTION_TARGETS[next_year]
                ratio = (year - prev_year) / (next_year - prev_year)
                return round(prev_pct + ratio * (next_pct - prev_pct), 2)

        return REDUCTION_TARGETS[target_years[-1]]

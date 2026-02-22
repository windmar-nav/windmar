"""
Seakeeping model for WINDMAR.

Calculates ship motion response to wave conditions:
- Roll amplitude
- Pitch amplitude
- Heave/vertical acceleration
- Slamming risk
- Parametric roll risk

Uses simplified strip theory approximations suitable for
operational planning (not detailed design analysis).

References:
- Bhattacharyya (1978) "Dynamics of Marine Vehicles"
- IMO MSC.1/Circ.1228 "Revised Guidance for Avoiding Dangerous Situations"
- ISO 2631-3:1985 "Evaluation of human exposure to whole-body vibration"
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.optimization.vessel_model import VesselSpecs as _VS

logger = logging.getLogger(__name__)


class SafetyStatus(Enum):
    """Safety assessment result."""
    SAFE = "safe"
    MARGINAL = "marginal"
    DANGEROUS = "dangerous"


@dataclass
class SeakeepingSpecs:
    """
    Vessel seakeeping characteristics.

    These are typically determined by naval architect or from
    stability booklet. Default values for MR tanker.
    """
    # Metacentric height (m)
    gm_laden: float = 2.5  # GM laden (m)
    gm_ballast: float = 4.0  # GM ballast (m)

    # Natural roll period (s)
    roll_period_laden: float = 14.0  # T_roll laden
    roll_period_ballast: float = 10.0  # T_roll ballast

    # Roll damping coefficient (non-dimensional)
    roll_damping: float = 0.05  # Typical for tanker without bilge keels

    # Vertical center of gravity above keel (m)
    kg_laden: float = 8.5
    kg_ballast: float = 10.0

    # Forward perpendicular from midship (m) - for bow motion
    fp_from_midship: float = 88.0  # Half of Lpp for MR tanker

    # Bridge location from midship (m) - for crew comfort
    bridge_from_midship: float = -70.0  # Aft

    # Bow freeboard (m)
    bow_freeboard_laden: float = 6.0
    bow_freeboard_ballast: float = 12.0

    # Critical slam pressure (kPa) - hull design limit
    critical_slam_pressure: float = 100.0


@dataclass
class MotionResponse:
    """Ship motion response at a given condition."""
    # Roll motion
    roll_amplitude_deg: float  # Maximum roll angle (degrees)
    roll_period_s: float  # Roll period (s)

    # Pitch motion
    pitch_amplitude_deg: float  # Maximum pitch angle (degrees)
    pitch_period_s: float  # Pitch period (s)

    # Vertical accelerations (m/s²)
    heave_accel_ms2: float  # At center of gravity
    bow_accel_ms2: float  # At forward perpendicular
    bridge_accel_ms2: float  # At bridge

    # Risk indicators
    slamming_probability: float  # 0-1
    green_water_probability: float  # 0-1
    parametric_roll_risk: float  # 0-1

    # Wave encounter
    encounter_period_s: float
    encounter_frequency_rad: float


@dataclass
class SafetyAssessment:
    """Safety assessment for a voyage leg."""
    status: SafetyStatus
    roll_status: SafetyStatus
    pitch_status: SafetyStatus
    acceleration_status: SafetyStatus
    slamming_status: SafetyStatus

    # Detailed metrics
    motions: MotionResponse

    # Limit exceedances
    roll_limit_exceeded: bool
    pitch_limit_exceeded: bool
    accel_limit_exceeded: bool
    slam_limit_exceeded: bool

    # Recommendations
    max_safe_speed_kts: Optional[float]
    recommended_heading_change_deg: Optional[float]
    warnings: List[str]


@dataclass
class SafetyLimits:
    """
    Operational safety limits for MR Product Tanker.

    Hard avoidance limits prevent the optimizer from routing through
    conditions that exceed structural or operational envelopes.
    Motion-based limits (roll, pitch, acceleration) provide graduated
    penalties via seakeeping model assessment.

    References:
    - IMO MSC.1/Circ.1228 "Revised Guidance for Avoiding Dangerous Situations"
    - ISO 2631-3:1985 "Evaluation of human exposure to whole-body vibration"
    """
    # ── Hard avoidance limits (instant rejection, no motion calc needed) ──
    # These are absolute no-go thresholds for an MR Product Tanker (~50k DWT).
    max_wave_height_m: float = 6.0    # Hs ≥ 6 m → forbidden (BF 9+)
    max_wind_speed_kts: float = 70.0  # ≥ 70 kts → forbidden (storm force 12)

    # ── Roll limits (degrees) — motion-based graduated penalties ──
    max_roll_safe: float = 15.0       # Normal operations
    max_roll_marginal: float = 25.0   # Reduced operations
    max_roll_dangerous: float = 30.0  # Dangerous

    # ── Pitch limits (degrees) ──
    max_pitch_safe: float = 5.0
    max_pitch_marginal: float = 8.0
    max_pitch_dangerous: float = 12.0

    # ── Vertical acceleration limits (m/s²) — at bridge ──
    max_accel_safe: float = 0.2 * 9.81   # ~2 m/s² — comfortable
    max_accel_marginal: float = 0.3 * 9.81  # ~3 m/s² — tolerable
    max_accel_dangerous: float = 0.5 * 9.81  # ~5 m/s² — severe

    # ── Slamming probability limits ──
    max_slam_safe: float = 0.03   # 3% — occasional
    max_slam_marginal: float = 0.10  # 10% — frequent

    # ── Parametric roll risk ──
    max_param_roll_risk: float = 0.3


class SeakeepingModel:
    """
    Simplified seakeeping model for operational planning.

    Calculates ship motions using strip theory approximations and
    empirical correlations suitable for weather routing.
    """

    # Constants
    G = 9.81  # Gravity (m/s²)

    def __init__(
        self,
        specs: Optional[SeakeepingSpecs] = None,
        lpp: float = _VS.lpp,
        beam: float = _VS.beam,
    ):
        """
        Initialize seakeeping model.

        Args:
            specs: Seakeeping specifications
            lpp: Length between perpendiculars (m)
            beam: Beam (m)
        """
        self.specs = specs or SeakeepingSpecs()
        self.lpp = lpp
        self.beam = beam

    def calculate_motions(
        self,
        wave_height_m: float,
        wave_period_s: float,
        wave_dir_deg: float,
        heading_deg: float,
        speed_kts: float,
        is_laden: bool,
    ) -> MotionResponse:
        """
        Calculate ship motion response to wave conditions.

        Args:
            wave_height_m: Significant wave height (m)
            wave_period_s: Peak wave period (s)
            wave_dir_deg: Wave direction (degrees, from)
            heading_deg: Ship heading (degrees)
            speed_kts: Ship speed (knots)
            is_laden: Loading condition

        Returns:
            MotionResponse with all motion amplitudes
        """
        # Calculate encounter angle (0=following, 180=head seas)
        encounter_angle_deg = (wave_dir_deg - heading_deg + 180) % 360
        if encounter_angle_deg > 180:
            encounter_angle_deg = 360 - encounter_angle_deg
        encounter_angle_rad = math.radians(encounter_angle_deg)

        # Calculate wave length
        wave_length_m = self.G * wave_period_s**2 / (2 * math.pi)

        # Calculate encounter frequency
        speed_ms = speed_kts * 0.51444
        omega_wave = 2 * math.pi / wave_period_s

        # Encounter frequency (positive for head seas)
        omega_e = abs(omega_wave - omega_wave**2 * speed_ms * math.cos(encounter_angle_rad) / self.G)
        if omega_e < 0.01:
            omega_e = 0.01  # Avoid division by zero

        encounter_period_s = 2 * math.pi / omega_e

        # Get loading-dependent parameters
        gm = self.specs.gm_laden if is_laden else self.specs.gm_ballast
        roll_period = self.specs.roll_period_laden if is_laden else self.specs.roll_period_ballast
        kg = self.specs.kg_laden if is_laden else self.specs.kg_ballast
        bow_freeboard = self.specs.bow_freeboard_laden if is_laden else self.specs.bow_freeboard_ballast

        # Natural roll frequency
        omega_roll = 2 * math.pi / roll_period

        # Calculate roll response
        roll_amplitude = self._calculate_roll(
            wave_height_m, wave_period_s, wave_length_m,
            encounter_angle_rad, omega_e, omega_roll, gm
        )

        # Calculate pitch response
        pitch_amplitude, pitch_period = self._calculate_pitch(
            wave_height_m, wave_length_m, encounter_angle_rad, speed_ms
        )

        # Calculate vertical accelerations
        heave_accel = self._calculate_heave_accel(
            wave_height_m, omega_e, encounter_angle_rad
        )

        bow_accel = self._calculate_point_accel(
            heave_accel, pitch_amplitude, omega_e, self.specs.fp_from_midship
        )

        bridge_accel = self._calculate_point_accel(
            heave_accel, pitch_amplitude, omega_e, self.specs.bridge_from_midship
        )

        # Calculate slamming probability
        slam_prob = self._calculate_slamming_probability(
            wave_height_m, wave_period_s, bow_freeboard, speed_ms,
            encounter_angle_rad, pitch_amplitude
        )

        # Calculate green water probability
        green_water_prob = self._calculate_green_water_probability(
            wave_height_m, bow_freeboard, pitch_amplitude
        )

        # Calculate parametric roll risk
        param_roll_risk = self._calculate_parametric_roll_risk(
            wave_length_m, encounter_period_s, roll_period, encounter_angle_rad
        )

        return MotionResponse(
            roll_amplitude_deg=roll_amplitude,
            roll_period_s=roll_period,
            pitch_amplitude_deg=pitch_amplitude,
            pitch_period_s=pitch_period,
            heave_accel_ms2=heave_accel,
            bow_accel_ms2=bow_accel,
            bridge_accel_ms2=bridge_accel,
            slamming_probability=slam_prob,
            green_water_probability=green_water_prob,
            parametric_roll_risk=param_roll_risk,
            encounter_period_s=encounter_period_s,
            encounter_frequency_rad=omega_e,
        )

    def calculate_motions_decomposed(
        self,
        windwave_height_m: float,
        windwave_period_s: float,
        windwave_dir_deg: float,
        swell_height_m: float,
        swell_period_s: float,
        swell_dir_deg: float,
        heading_deg: float,
        speed_kts: float,
        is_laden: bool,
    ) -> MotionResponse:
        """
        Calculate motions from separate wind-wave and swell systems.

        Computes response to each system independently, then combines
        using spectral superposition (RSS for amplitudes, worst-case
        for risk indicators). This is physically more accurate than
        using a single combined sea state.

        Args:
            windwave_height_m: Wind-wave significant height (m)
            windwave_period_s: Wind-wave mean period (s)
            windwave_dir_deg: Wind-wave direction (degrees, from)
            swell_height_m: Primary swell height (m)
            swell_period_s: Primary swell period (s)
            swell_dir_deg: Primary swell direction (degrees, from)
            heading_deg: Ship heading (degrees)
            speed_kts: Ship speed (knots)
            is_laden: Loading condition

        Returns:
            MotionResponse with combined motion amplitudes
        """
        # Calculate response to each wave system
        ww_response = self.calculate_motions(
            windwave_height_m, windwave_period_s, windwave_dir_deg,
            heading_deg, speed_kts, is_laden
        )
        sw_response = self.calculate_motions(
            swell_height_m, swell_period_s, swell_dir_deg,
            heading_deg, speed_kts, is_laden
        )

        # Combine using RSS (root sum of squares) for motion amplitudes.
        # This approximates spectral superposition of independent systems.
        combined_roll = math.sqrt(ww_response.roll_amplitude_deg**2 + sw_response.roll_amplitude_deg**2)
        combined_pitch = math.sqrt(ww_response.pitch_amplitude_deg**2 + sw_response.pitch_amplitude_deg**2)
        combined_heave = math.sqrt(ww_response.heave_accel_ms2**2 + sw_response.heave_accel_ms2**2)
        combined_bow = math.sqrt(ww_response.bow_accel_ms2**2 + sw_response.bow_accel_ms2**2)
        combined_bridge = math.sqrt(ww_response.bridge_accel_ms2**2 + sw_response.bridge_accel_ms2**2)

        # For risk indicators, take worst case
        combined_slam = max(ww_response.slamming_probability, sw_response.slamming_probability)
        combined_greenwater = max(ww_response.green_water_probability, sw_response.green_water_probability)
        combined_param_roll = max(ww_response.parametric_roll_risk, sw_response.parametric_roll_risk)

        # Use the dominant system's encounter values (the one with larger roll)
        dominant = sw_response if sw_response.roll_amplitude_deg > ww_response.roll_amplitude_deg else ww_response

        return MotionResponse(
            roll_amplitude_deg=min(combined_roll, 45.0),
            roll_period_s=dominant.roll_period_s,
            pitch_amplitude_deg=min(combined_pitch, 20.0),
            pitch_period_s=dominant.pitch_period_s,
            heave_accel_ms2=combined_heave,
            bow_accel_ms2=combined_bow,
            bridge_accel_ms2=combined_bridge,
            slamming_probability=combined_slam,
            green_water_probability=combined_greenwater,
            parametric_roll_risk=combined_param_roll,
            encounter_period_s=dominant.encounter_period_s,
            encounter_frequency_rad=dominant.encounter_frequency_rad,
        )

    def _calculate_roll(
        self,
        wave_height_m: float,
        wave_period_s: float,
        wave_length_m: float,
        encounter_angle_rad: float,
        omega_e: float,
        omega_roll: float,
        gm: float,
    ) -> float:
        """
        Calculate roll amplitude using single-degree-of-freedom model.

        Roll is maximum in beam seas and at resonance.
        """
        # Wave slope (steepness)
        wave_slope = wave_height_m / wave_length_m

        # Effective wave slope (reduced for non-beam seas)
        beam_factor = abs(math.sin(encounter_angle_rad))
        effective_slope = wave_slope * beam_factor

        # Roll excitation moment coefficient
        # Simplified: proportional to wave slope and GM
        excitation = effective_slope * self.G / gm

        # Frequency ratio
        freq_ratio = omega_e / omega_roll

        # Roll response amplitude using linear response
        # RAO = 1 / sqrt((1 - r²)² + (2*zeta*r)²)
        zeta = self.specs.roll_damping

        denominator = math.sqrt(
            (1 - freq_ratio**2)**2 + (2 * zeta * freq_ratio)**2
        )
        if denominator < 0.1:
            denominator = 0.1  # Avoid extreme resonance

        rao = 1.0 / denominator

        # Roll amplitude in degrees
        # Scale by wave height and beam factor
        roll_amplitude = math.degrees(excitation * rao) * (wave_height_m / 2)

        # Cap at physical limits
        roll_amplitude = min(roll_amplitude, 45.0)

        return roll_amplitude

    def _calculate_pitch(
        self,
        wave_height_m: float,
        wave_length_m: float,
        encounter_angle_rad: float,
        speed_ms: float,
    ) -> Tuple[float, float]:
        """
        Calculate pitch amplitude and period.

        Pitch is maximum in head/following seas.
        """
        # Pitch is related to wave slope and L/lambda ratio
        l_lambda = self.lpp / wave_length_m

        # Head/following sea factor
        head_factor = abs(math.cos(encounter_angle_rad))

        # Wave slope
        wave_slope = wave_height_m / wave_length_m

        # Pitch amplitude depends on L/lambda ratio
        # Maximum response when lambda ≈ L
        if l_lambda < 0.5:
            pitch_factor = 2.0 * l_lambda
        elif l_lambda < 1.5:
            pitch_factor = 1.0 - 0.3 * abs(l_lambda - 1.0)
        else:
            pitch_factor = 0.5 / l_lambda

        # Pitch amplitude in degrees
        pitch_amplitude = math.degrees(wave_slope * head_factor * pitch_factor * 10)

        # Natural pitch period (roughly 0.5-0.6 * sqrt(Lpp))
        pitch_period = 0.55 * math.sqrt(self.lpp)

        return min(pitch_amplitude, 20.0), pitch_period

    def _calculate_heave_accel(
        self,
        wave_height_m: float,
        omega_e: float,
        encounter_angle_rad: float,
    ) -> float:
        """
        Calculate heave acceleration at CG.
        """
        # Heave amplitude ≈ wave amplitude for short ships
        heave_amplitude = wave_height_m / 2

        # Heave acceleration = amplitude * omega²
        heave_accel = heave_amplitude * omega_e**2

        # Reduce for beam seas (mostly roll)
        beam_factor = abs(math.cos(encounter_angle_rad))
        heave_accel *= (0.3 + 0.7 * beam_factor)

        return heave_accel

    def _calculate_point_accel(
        self,
        heave_accel: float,
        pitch_amplitude_deg: float,
        omega_e: float,
        distance_from_midship: float,
    ) -> float:
        """
        Calculate vertical acceleration at a point.

        Combines heave and pitch-induced acceleration.
        """
        pitch_rad = math.radians(pitch_amplitude_deg)

        # Pitch-induced vertical acceleration at distance x from midship
        pitch_accel = abs(distance_from_midship) * pitch_rad * omega_e**2

        # Combined (RSS)
        total_accel = math.sqrt(heave_accel**2 + pitch_accel**2)

        return total_accel

    def _calculate_slamming_probability(
        self,
        wave_height_m: float,
        wave_period_s: float,
        bow_freeboard: float,
        speed_ms: float,
        encounter_angle_rad: float,
        pitch_amplitude_deg: float,
    ) -> float:
        """
        Calculate probability of slamming using Ochi's criteria.

        Slamming occurs when:
        1. Bow emerges from water
        2. Re-entry velocity exceeds threshold
        """
        # Relative motion at bow (simplified)
        pitch_rad = math.radians(pitch_amplitude_deg)
        bow_vertical_motion = wave_height_m / 2 + self.specs.fp_from_midship * pitch_rad

        # Probability of emergence
        if bow_vertical_motion < 0.1:
            return 0.0

        emergence_ratio = bow_freeboard / bow_vertical_motion
        if emergence_ratio > 3.0:
            return 0.0

        prob_emergence = math.exp(-2 * emergence_ratio**2)

        # Head seas factor (slamming much worse in head seas)
        head_factor = (1 + math.cos(encounter_angle_rad)) / 2

        # Speed factor (higher speed = more slamming)
        speed_factor = min(speed_ms / 8.0, 2.0)  # Normalized to ~16 kts

        # Combined probability
        slam_prob = prob_emergence * head_factor * speed_factor

        return min(slam_prob, 1.0)

    def _calculate_green_water_probability(
        self,
        wave_height_m: float,
        bow_freeboard: float,
        pitch_amplitude_deg: float,
    ) -> float:
        """
        Calculate probability of green water on deck.
        """
        pitch_rad = math.radians(pitch_amplitude_deg)

        # Effective freeboard reduction due to pitch
        effective_freeboard = bow_freeboard - self.specs.fp_from_midship * pitch_rad

        # Relative motion amplitude
        relative_motion = wave_height_m / 2

        if effective_freeboard <= 0:
            return 1.0

        if relative_motion < 0.1:
            return 0.0

        # Probability based on freeboard exceedance
        ratio = effective_freeboard / relative_motion
        if ratio > 3.0:
            return 0.0

        prob = math.exp(-2 * ratio**2)

        return min(prob, 1.0)

    def _calculate_parametric_roll_risk(
        self,
        wave_length_m: float,
        encounter_period_s: float,
        roll_period_s: float,
        encounter_angle_rad: float,
    ) -> float:
        """
        Calculate parametric rolling risk.

        Parametric roll occurs when:
        1. Te ≈ Tr/2 (encounter period ≈ half roll period)
        2. Wave length ≈ ship length
        3. Head or following seas
        """
        # Period ratio (dangerous when Te ≈ Tr/2)
        period_ratio = encounter_period_s / (roll_period_s / 2)

        # Risk from period matching
        if abs(period_ratio - 1.0) < 0.3:
            period_risk = 1.0 - abs(period_ratio - 1.0) / 0.3
        else:
            period_risk = 0.0

        # Risk from wave length matching ship length
        l_lambda = self.lpp / wave_length_m
        if 0.8 < l_lambda < 1.2:
            length_risk = 1.0 - abs(l_lambda - 1.0) / 0.2
        else:
            length_risk = 0.0

        # Head/following seas factor
        head_follow = abs(math.cos(encounter_angle_rad))
        if head_follow > 0.7:
            heading_risk = 1.0
        else:
            heading_risk = head_follow / 0.7

        # Combined risk
        risk = period_risk * length_risk * heading_risk

        return risk


class SafetyConstraints:
    """
    Safety constraint checker for route optimization.

    Evaluates ship motions against operational limits and
    provides recommendations for safe navigation.
    """

    def __init__(
        self,
        seakeeping: Optional[SeakeepingModel] = None,
        limits: Optional[SafetyLimits] = None,
    ):
        """
        Initialize safety constraints.

        Args:
            seakeeping: Seakeeping model to use
            limits: Operational safety limits
        """
        self.seakeeping = seakeeping or SeakeepingModel()
        self.limits = limits or SafetyLimits()

    def assess_safety(
        self,
        wave_height_m: float,
        wave_period_s: float,
        wave_dir_deg: float,
        heading_deg: float,
        speed_kts: float,
        is_laden: bool,
        windwave_height_m: float = 0.0,
        windwave_period_s: float = 0.0,
        windwave_dir_deg: float = 0.0,
        swell_height_m: float = 0.0,
        swell_period_s: float = 0.0,
        swell_dir_deg: float = 0.0,
        has_decomposition: bool = False,
    ) -> SafetyAssessment:
        """
        Perform full safety assessment for a voyage leg.

        When wave decomposition is available, uses separate wind-wave
        and swell systems for more accurate motion prediction. Falls
        back to combined sea state when decomposition is not available.

        Args:
            wave_height_m: Significant wave height (m) — combined
            wave_period_s: Peak wave period (s) — combined
            wave_dir_deg: Wave direction (degrees) — combined
            heading_deg: Ship heading (degrees)
            speed_kts: Ship speed (knots)
            is_laden: Loading condition
            windwave_height_m: Wind-wave height (m) — if decomposed
            windwave_period_s: Wind-wave period (s) — if decomposed
            windwave_dir_deg: Wind-wave direction (deg) — if decomposed
            swell_height_m: Swell height (m) — if decomposed
            swell_period_s: Swell period (s) — if decomposed
            swell_dir_deg: Swell direction (deg) — if decomposed
            has_decomposition: Whether decomposed data is available

        Returns:
            SafetyAssessment with detailed evaluation
        """
        # Calculate motions — use decomposed data when available
        if has_decomposition and windwave_height_m > 0 and swell_height_m > 0:
            motions = self.seakeeping.calculate_motions_decomposed(
                windwave_height_m, windwave_period_s, windwave_dir_deg,
                swell_height_m, swell_period_s, swell_dir_deg,
                heading_deg, speed_kts, is_laden,
            )
        else:
            motions = self.seakeeping.calculate_motions(
                wave_height_m, wave_period_s, wave_dir_deg,
                heading_deg, speed_kts, is_laden,
            )

        warnings = []

        # Assess roll
        roll_status, roll_exceeded = self._assess_roll(motions.roll_amplitude_deg)
        if roll_exceeded:
            warnings.append(f"Excessive roll: {motions.roll_amplitude_deg:.1f}° (limit: {self.limits.max_roll_safe}°)")

        # Assess pitch
        pitch_status, pitch_exceeded = self._assess_pitch(motions.pitch_amplitude_deg)
        if pitch_exceeded:
            warnings.append(f"Excessive pitch: {motions.pitch_amplitude_deg:.1f}° (limit: {self.limits.max_pitch_safe}°)")

        # Assess acceleration
        accel_status, accel_exceeded = self._assess_acceleration(motions.bridge_accel_ms2)
        if accel_exceeded:
            warnings.append(f"High vertical acceleration: {motions.bridge_accel_ms2:.1f} m/s²")

        # Assess slamming
        slam_status, slam_exceeded = self._assess_slamming(motions.slamming_probability)
        if slam_exceeded:
            warnings.append(f"Slamming risk: {motions.slamming_probability*100:.0f}%")

        # Parametric roll warning
        if motions.parametric_roll_risk > self.limits.max_param_roll_risk:
            warnings.append(f"Parametric roll risk: {motions.parametric_roll_risk*100:.0f}%")

        # Green water warning
        if motions.green_water_probability > 0.1:
            warnings.append(f"Green water risk: {motions.green_water_probability*100:.0f}%")

        # Overall status (worst of all categories)
        status_values = [roll_status, pitch_status, accel_status, slam_status]
        if SafetyStatus.DANGEROUS in status_values:
            overall_status = SafetyStatus.DANGEROUS
        elif SafetyStatus.MARGINAL in status_values:
            overall_status = SafetyStatus.MARGINAL
        else:
            overall_status = SafetyStatus.SAFE

        # Calculate recommendations if not safe
        max_safe_speed = None
        heading_change = None

        if overall_status != SafetyStatus.SAFE:
            max_safe_speed = self._find_safe_speed(
                wave_height_m, wave_period_s, wave_dir_deg,
                heading_deg, is_laden
            )
            heading_change = self._suggest_heading_change(
                wave_height_m, wave_period_s, wave_dir_deg,
                heading_deg, speed_kts, is_laden
            )

        return SafetyAssessment(
            status=overall_status,
            roll_status=roll_status,
            pitch_status=pitch_status,
            acceleration_status=accel_status,
            slamming_status=slam_status,
            motions=motions,
            roll_limit_exceeded=roll_exceeded,
            pitch_limit_exceeded=pitch_exceeded,
            accel_limit_exceeded=accel_exceeded,
            slam_limit_exceeded=slam_exceeded,
            max_safe_speed_kts=max_safe_speed,
            recommended_heading_change_deg=heading_change,
            warnings=warnings,
        )

    def _assess_roll(self, roll_deg: float) -> Tuple[SafetyStatus, bool]:
        """Assess roll amplitude against limits."""
        if roll_deg >= self.limits.max_roll_dangerous:
            return SafetyStatus.DANGEROUS, True
        elif roll_deg >= self.limits.max_roll_marginal:
            return SafetyStatus.MARGINAL, True
        elif roll_deg >= self.limits.max_roll_safe:
            return SafetyStatus.MARGINAL, True
        return SafetyStatus.SAFE, False

    def _assess_pitch(self, pitch_deg: float) -> Tuple[SafetyStatus, bool]:
        """Assess pitch amplitude against limits."""
        if pitch_deg >= self.limits.max_pitch_dangerous:
            return SafetyStatus.DANGEROUS, True
        elif pitch_deg >= self.limits.max_pitch_marginal:
            return SafetyStatus.MARGINAL, True
        elif pitch_deg >= self.limits.max_pitch_safe:
            return SafetyStatus.MARGINAL, True
        return SafetyStatus.SAFE, False

    def _assess_acceleration(self, accel_ms2: float) -> Tuple[SafetyStatus, bool]:
        """Assess vertical acceleration against limits."""
        if accel_ms2 >= self.limits.max_accel_dangerous:
            return SafetyStatus.DANGEROUS, True
        elif accel_ms2 >= self.limits.max_accel_marginal:
            return SafetyStatus.MARGINAL, True
        elif accel_ms2 >= self.limits.max_accel_safe:
            return SafetyStatus.MARGINAL, True
        return SafetyStatus.SAFE, False

    def _assess_slamming(self, slam_prob: float) -> Tuple[SafetyStatus, bool]:
        """Assess slamming probability against limits."""
        if slam_prob >= self.limits.max_slam_marginal:
            return SafetyStatus.DANGEROUS, True
        elif slam_prob >= self.limits.max_slam_safe:
            return SafetyStatus.MARGINAL, True
        return SafetyStatus.SAFE, False

    def _quick_status(
        self,
        wave_height_m: float,
        wave_period_s: float,
        wave_dir_deg: float,
        heading_deg: float,
        speed_kts: float,
        is_laden: bool,
    ) -> Tuple[SafetyStatus, MotionResponse]:
        """
        Compute motions and overall status without recommendations.

        This avoids the recursion that would occur if _find_safe_speed
        or _suggest_heading_change called the full assess_safety method.
        """
        motions = self.seakeeping.calculate_motions(
            wave_height_m, wave_period_s, wave_dir_deg,
            heading_deg, speed_kts, is_laden,
        )

        roll_status, _ = self._assess_roll(motions.roll_amplitude_deg)
        pitch_status, _ = self._assess_pitch(motions.pitch_amplitude_deg)
        accel_status, _ = self._assess_acceleration(motions.bridge_accel_ms2)
        slam_status, _ = self._assess_slamming(motions.slamming_probability)

        status_values = [roll_status, pitch_status, accel_status, slam_status]
        if SafetyStatus.DANGEROUS in status_values:
            overall = SafetyStatus.DANGEROUS
        elif SafetyStatus.MARGINAL in status_values:
            overall = SafetyStatus.MARGINAL
        else:
            overall = SafetyStatus.SAFE

        return overall, motions

    def _find_safe_speed(
        self,
        wave_height_m: float,
        wave_period_s: float,
        wave_dir_deg: float,
        heading_deg: float,
        is_laden: bool,
        min_speed: float = 5.0,
    ) -> Optional[float]:
        """
        Find maximum safe speed for given conditions.

        Returns None if no safe speed exists (must change heading).
        """
        # Try decreasing speeds
        for speed in range(15, int(min_speed) - 1, -1):
            status, _ = self._quick_status(
                wave_height_m, wave_period_s, wave_dir_deg,
                heading_deg, float(speed), is_laden
            )
            if status != SafetyStatus.DANGEROUS:
                return float(speed)

        return None

    def _suggest_heading_change(
        self,
        wave_height_m: float,
        wave_period_s: float,
        wave_dir_deg: float,
        heading_deg: float,
        speed_kts: float,
        is_laden: bool,
    ) -> Optional[float]:
        """
        Suggest heading change to improve safety.

        Returns degrees to alter course (positive = starboard).
        """
        current_status, current_motions = self._quick_status(
            wave_height_m, wave_period_s, wave_dir_deg,
            heading_deg, speed_kts, is_laden
        )

        if current_status == SafetyStatus.SAFE:
            return None

        # Try alterations up to 45 degrees each side
        best_change = None
        best_roll = current_motions.roll_amplitude_deg

        for change in [10, -10, 20, -20, 30, -30, 45, -45]:
            new_heading = (heading_deg + change) % 360
            status, motions = self._quick_status(
                wave_height_m, wave_period_s, wave_dir_deg,
                new_heading, speed_kts, is_laden
            )

            if status == SafetyStatus.SAFE:
                return float(change)

            if motions.roll_amplitude_deg < best_roll:
                best_roll = motions.roll_amplitude_deg
                best_change = float(change)

        return best_change

    def get_safety_cost_factor(
        self,
        wave_height_m: float,
        wave_period_s: float,
        wave_dir_deg: float,
        heading_deg: float,
        speed_kts: float,
        is_laden: bool,
        wind_speed_kts: float = 0.0,
        skip_hard_limits: bool = False,
    ) -> float:
        """
        Get cost factor for route optimization.

        Returns a multiplier (1.0 = safe, >1.0 = penalized, inf = forbidden).

        Hard avoidance limits are checked first (wave height, wind speed)
        before computing motion-based penalties. This prevents the optimizer
        from routing through extreme conditions regardless of vessel heading.

        When *skip_hard_limits* is True, wave/wind hard limits return 10.0
        (heavy penalty) instead of inf, allowing the optimizer to route
        through extreme weather as a last resort.  Motion exceedance >1.5x
        remains inf (structural vessel limit, never skipped).
        """
        # ── Hard avoidance: instant rejection (or heavy penalty) ──
        if wave_height_m >= self.limits.max_wave_height_m:
            return 10.0 if skip_hard_limits else float('inf')
        if wind_speed_kts >= self.limits.max_wind_speed_kts:
            return 10.0 if skip_hard_limits else float('inf')

        # ── Motion-based graduated penalties ──
        assessment = self.assess_safety(
            wave_height_m, wave_period_s, wave_dir_deg,
            heading_deg, speed_kts, is_laden
        )

        if assessment.status == SafetyStatus.DANGEROUS:
            # Graduated penalty based on severity of exceedance.
            # Compute worst exceedance ratio across roll, pitch, accel.
            roll_ratio = assessment.motions.roll_amplitude_deg / self.limits.max_roll_dangerous if self.limits.max_roll_dangerous > 0 else 0
            pitch_ratio = assessment.motions.pitch_amplitude_deg / self.limits.max_pitch_dangerous if self.limits.max_pitch_dangerous > 0 else 0
            exceedance = max(roll_ratio, pitch_ratio)

            if exceedance > 1.5:
                # Extreme conditions (>1.5x dangerous threshold) — block
                return float('inf')
            elif exceedance > 1.0:
                # Scale penalty 2.0 → 5.0 as exceedance goes 1.0 → 1.5
                penalty = 2.0 + (exceedance - 1.0) / 0.5 * 3.0
                return penalty
            else:
                # Below dangerous threshold but still flagged dangerous
                return 2.0
        elif assessment.status == SafetyStatus.MARGINAL:
            # Penalty based on how marginal
            roll_penalty = max(0, assessment.motions.roll_amplitude_deg - self.limits.max_roll_safe) / 10
            pitch_penalty = max(0, assessment.motions.pitch_amplitude_deg - self.limits.max_pitch_safe) / 5
            return 1.0 + roll_penalty + pitch_penalty
        else:
            return 1.0  # Safe, no penalty


def create_default_safety_constraints(
    lpp: float = _VS.lpp,
    beam: float = _VS.beam,
) -> SafetyConstraints:
    """
    Create safety constraints with default MR tanker parameters.

    Args:
        lpp: Length between perpendiculars (m)
        beam: Beam (m)

    Returns:
        Configured SafetyConstraints instance
    """
    seakeeping_specs = SeakeepingSpecs()
    seakeeping = SeakeepingModel(specs=seakeeping_specs, lpp=lpp, beam=beam)
    limits = SafetyLimits()

    return SafetyConstraints(seakeeping=seakeeping, limits=limits)

"""
Numba-JIT compiled kernels for route optimization inner loops.

Every function here is a standalone ``@njit``-decorated scalar function
using only the ``math`` module. They are called thousands of times per
route search; JIT compilation gives 5-20x speedup on each call.

The outer search loops (heapq, datetime, Python objects) remain in pure
Python — only the hot inner math is compiled.
"""

import logging
import math

try:
    from numba import njit
except ModuleNotFoundError:
    # Fallback: functions run as plain Python (no JIT speedup)
    def njit(fn):
        return fn

    logging.getLogger(__name__).info(
        "numba not installed — kernels run as plain Python"
    )


# ===================================================================
# Geometry (from base_optimizer.py)
# ===================================================================


@njit
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in nautical miles."""
    R = 3440.065  # Earth radius in nm
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(a))


@njit
def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing from point 1 to point 2 (degrees, 0-360)."""
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(math.radians(lat2))
    y = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(
        math.radians(lat1)
    ) * math.cos(math.radians(lat2)) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


@njit
def current_effect(
    heading_deg: float, current_speed_ms: float, current_dir_deg: float
) -> float:
    """Speed adjustment from current (knots, positive = favorable)."""
    if current_speed_ms <= 0:
        return 0.0
    current_kts = current_speed_ms * 1.94384
    relative_angle = abs(((current_dir_deg - heading_deg) + 180) % 360 - 180)
    return current_kts * math.cos(math.radians(relative_angle))


@njit
def course_change_penalty(current_heading_deg: float, next_heading_deg: float) -> float:
    """Piecewise-linear penalty for course changes (0-0.20)."""
    diff = abs(((next_heading_deg - current_heading_deg) + 180) % 360 - 180)
    if diff <= 15.0:
        return 0.0
    if diff <= 45.0:
        return 0.02 * (diff - 15.0) / 30.0
    if diff <= 90.0:
        return 0.02 + 0.06 * (diff - 45.0) / 45.0
    return 0.08 + 0.12 * (min(diff, 180.0) - 90.0) / 90.0


# ===================================================================
# Seawater properties (from vessel_model.py)
# ===================================================================


@njit
def seawater_density(sst_celsius: float) -> float:
    """UNESCO 1983 simplified equation of state (salinity=35 PSU)."""
    t = sst_celsius
    rho_fw = 999.842594 + 6.793952e-2 * t - 9.095290e-3 * t**2 + 1.001685e-4 * t**3
    return rho_fw + 0.824493 * 35 - 4.0899e-3 * 35 * t


@njit
def seawater_viscosity(sst_celsius: float) -> float:
    """Kinematic viscosity of seawater (Sharqawy 2010 correlation)."""
    t = sst_celsius
    mu = 1.7910 - 6.144e-2 * t + 1.4510e-3 * t**2 - 1.6826e-5 * t**3
    rho = seawater_density(t)
    return (mu * 1e-3) / rho


# ===================================================================
# Vessel resistance (from vessel_model.py)
# ===================================================================


@njit
def holtrop_mennen_resistance(
    speed_ms: float,
    draft: float,
    displacement: float,
    cb: float,
    wetted_surface: float,
    lpp: float,
    beam: float,
    rho_sw: float,
    nu_sw: float,
) -> float:
    """Calm-water resistance via Holtrop-Mennen (simplified for tankers)."""
    # Froude number
    froude = speed_ms / math.sqrt(9.81 * lpp)

    # Reynolds number
    reynolds = speed_ms * lpp / nu_sw

    # Frictional resistance coefficient (ITTC 1957)
    cf = 0.075 / (math.log10(reynolds) - 2) ** 2

    # Hull roughness allowance
    delta_cf = 0.00025

    # Form factor (Holtrop-Mennen for tankers)
    k1 = 0.93 + 0.4871 * (beam / lpp) - 0.2156 * (beam / draft) + 0.1027 * cb
    if k1 < 0.1:
        k1 = 0.1

    # Frictional resistance
    rf = 0.5 * rho_sw * speed_ms**2 * wetted_surface * (cf + delta_cf) * (1 + k1)

    # Wave-making resistance
    rw_ratio = 4.0 * froude**2
    rw = rw_ratio * rf

    # Appendage resistance (~5% of frictional)
    rapp = 0.05 * rf

    return rf + rw + rapp


@njit
def wind_resistance(
    wind_speed_ms: float,
    wind_dir_deg: float,
    heading_deg: float,
    frontal_area: float,
    lateral_area: float,
    rho_air: float,
) -> float:
    """Wind resistance via Blendermann method (N, always >= 0)."""
    relative_angle = abs(((wind_dir_deg - heading_deg) + 180) % 360 - 180)
    relative_angle_rad = math.radians(relative_angle)

    # Longitudinal drag
    cx_drag = 0.8 * math.cos(relative_angle_rad)
    cx_clamped = cx_drag if cx_drag > 0.0 else 0.0
    direct_resistance = cx_clamped * 0.5 * rho_air * wind_speed_ms**2 * frontal_area

    # Transverse drift contribution
    cy = 0.9 * abs(math.sin(relative_angle_rad))
    drift_resistance = 0.1 * cy * 0.5 * rho_air * wind_speed_ms**2 * lateral_area

    return direct_resistance + drift_resistance


@njit
def stawave1_wave_resistance(
    sig_wave_height_m: float,
    wave_dir_deg: float,
    heading_deg: float,
    speed_ms: float,
    beam: float,
    lpp: float,
    rho_sw: float,
) -> float:
    """STAWAVE-1 added resistance in waves (ISO 15016)."""
    relative_angle = abs(((wave_dir_deg - heading_deg) + 180) % 360 - 180)
    relative_angle_rad = math.radians(relative_angle)

    directional_factor = (1 + math.cos(relative_angle_rad)) / 2

    alpha_bk = 1.0
    raw = (
        (1.0 / 16.0)
        * rho_sw
        * 9.81
        * sig_wave_height_m**2
        * beam
        * math.sqrt(beam / lpp)
        * alpha_bk
        * directional_factor
    )
    return raw


@njit
def kwon_speed_loss_pct(
    sig_wave_height_m: float,
    wave_dir_deg: float,
    heading_deg: float,
    cb: float,
    lpp: float,
) -> float:
    """Kwon's involuntary speed-loss percentage."""
    if sig_wave_height_m <= 0:
        return 0.0

    relative_angle = abs(((wave_dir_deg - heading_deg) + 180) % 360 - 180)

    cb_factor = 1.7 - 0.9 * cb
    length_factor = 180.0 / lpp
    if length_factor < 0.5:
        length_factor = 0.5
    elif length_factor > 1.5:
        length_factor = 1.5

    base_loss_pct_per_m = 3.0 * cb_factor * length_factor

    if relative_angle <= 30:
        dir_factor = 1.0
    elif relative_angle <= 60:
        dir_factor = 0.9
    elif relative_angle <= 90:
        dir_factor = 0.7
    elif relative_angle <= 150:
        dir_factor = 0.4
    else:
        dir_factor = 0.2

    delta_v_pct = base_loss_pct_per_m * sig_wave_height_m * dir_factor
    if delta_v_pct > 50.0:
        delta_v_pct = 50.0
    return delta_v_pct


@njit
def sfoc_curve(load_fraction: float, sfoc_at_mcr: float, sfoc_factor: float) -> float:
    """SFOC at given engine load (g/kWh)."""
    if load_fraction < 0.15:
        load_fraction = 0.15
    elif load_fraction > 1.0:
        load_fraction = 1.0

    if load_fraction < 0.75:
        sfoc = sfoc_at_mcr * (1.0 + 0.15 * (0.75 - load_fraction))
    else:
        sfoc = sfoc_at_mcr * (1.0 + 0.05 * (load_fraction - 0.75))

    return sfoc * sfoc_factor


# ===================================================================
# Seakeeping motions (from seakeeping.py)
# ===================================================================


@njit
def calculate_roll(
    wave_height_m: float,
    wave_length_m: float,
    encounter_angle_rad: float,
    omega_e: float,
    omega_roll: float,
    gm: float,
    roll_damping: float,
    G: float,
) -> float:
    """Roll amplitude (degrees) using single-DOF linear response model."""
    wave_slope = wave_height_m / wave_length_m
    beam_factor = abs(math.sin(encounter_angle_rad))
    effective_slope = wave_slope * beam_factor

    excitation = effective_slope * G / gm

    freq_ratio = omega_e / omega_roll

    denominator = math.sqrt(
        (1 - freq_ratio**2) ** 2 + (2 * roll_damping * freq_ratio) ** 2
    )
    if denominator < 0.1:
        denominator = 0.1

    rao = 1.0 / denominator

    roll_amplitude = math.degrees(excitation * rao) * (wave_height_m / 2)
    if roll_amplitude > 45.0:
        roll_amplitude = 45.0

    return roll_amplitude


@njit
def calculate_pitch(
    wave_height_m: float,
    wave_length_m: float,
    encounter_angle_rad: float,
    speed_ms: float,
    lpp: float,
) -> tuple:
    """Pitch amplitude (degrees) and period (s)."""
    l_lambda = lpp / wave_length_m
    head_factor = abs(math.cos(encounter_angle_rad))
    wave_slope = wave_height_m / wave_length_m

    if l_lambda < 0.5:
        pitch_factor = 2.0 * l_lambda
    elif l_lambda < 1.5:
        pitch_factor = 1.0 - 0.3 * abs(l_lambda - 1.0)
    else:
        pitch_factor = 0.5 / l_lambda

    pitch_amplitude = math.degrees(wave_slope * head_factor * pitch_factor * 10)
    if pitch_amplitude > 20.0:
        pitch_amplitude = 20.0

    pitch_period = 0.55 * math.sqrt(lpp)
    return pitch_amplitude, pitch_period


@njit
def calculate_heave_accel(
    wave_height_m: float,
    omega_e: float,
    encounter_angle_rad: float,
) -> float:
    """Heave acceleration at CG (m/s^2)."""
    heave_amplitude = wave_height_m / 2
    heave_accel = heave_amplitude * omega_e**2

    beam_factor = abs(math.cos(encounter_angle_rad))
    heave_accel *= 0.3 + 0.7 * beam_factor

    return heave_accel


@njit
def calculate_point_accel(
    heave_accel: float,
    pitch_amplitude_deg: float,
    omega_e: float,
    distance_from_midship: float,
) -> float:
    """Vertical acceleration at a point (m/s^2), combining heave + pitch."""
    pitch_rad = math.radians(pitch_amplitude_deg)
    pitch_accel = abs(distance_from_midship) * pitch_rad * omega_e**2
    return math.sqrt(heave_accel**2 + pitch_accel**2)


@njit
def calculate_slamming_probability(
    wave_height_m: float,
    wave_period_s: float,
    bow_freeboard: float,
    speed_ms: float,
    encounter_angle_rad: float,
    pitch_amplitude_deg: float,
    fp_from_midship: float,
) -> float:
    """Probability of slamming (0-1) using Ochi's criteria."""
    pitch_rad = math.radians(pitch_amplitude_deg)
    bow_vertical_motion = wave_height_m / 2 + fp_from_midship * pitch_rad

    if bow_vertical_motion < 0.1:
        return 0.0

    emergence_ratio = bow_freeboard / bow_vertical_motion
    if emergence_ratio > 3.0:
        return 0.0

    prob_emergence = math.exp(-2 * emergence_ratio**2)
    head_factor = (1 + math.cos(encounter_angle_rad)) / 2
    speed_factor = speed_ms / 8.0
    if speed_factor > 2.0:
        speed_factor = 2.0

    slam_prob = prob_emergence * head_factor * speed_factor
    if slam_prob > 1.0:
        slam_prob = 1.0
    return slam_prob


@njit
def calculate_green_water_probability(
    wave_height_m: float,
    bow_freeboard: float,
    pitch_amplitude_deg: float,
    fp_from_midship: float,
) -> float:
    """Probability of green water on deck (0-1)."""
    pitch_rad = math.radians(pitch_amplitude_deg)
    effective_freeboard = bow_freeboard - fp_from_midship * pitch_rad
    relative_motion = wave_height_m / 2

    if effective_freeboard <= 0:
        return 1.0
    if relative_motion < 0.1:
        return 0.0

    ratio = effective_freeboard / relative_motion
    if ratio > 3.0:
        return 0.0

    prob = math.exp(-2 * ratio**2)
    if prob > 1.0:
        prob = 1.0
    return prob


@njit
def calculate_parametric_roll_risk(
    wave_length_m: float,
    encounter_period_s: float,
    roll_period_s: float,
    encounter_angle_rad: float,
    lpp: float,
) -> float:
    """Parametric rolling risk indicator (0-1)."""
    period_ratio = encounter_period_s / (roll_period_s / 2)

    if abs(period_ratio - 1.0) < 0.3:
        period_risk = 1.0 - abs(period_ratio - 1.0) / 0.3
    else:
        period_risk = 0.0

    l_lambda = lpp / wave_length_m
    if 0.8 < l_lambda < 1.2:
        length_risk = 1.0 - abs(l_lambda - 1.0) / 0.2
    else:
        length_risk = 0.0

    head_follow = abs(math.cos(encounter_angle_rad))
    if head_follow > 0.7:
        heading_risk = 1.0
    else:
        heading_risk = head_follow / 0.7

    return period_risk * length_risk * heading_risk


# ===================================================================
# Warm-up: call every kernel once to trigger JIT compilation
# ===================================================================


def warm_up():
    """Pre-compile all kernels with dummy arguments.

    Call this once at application startup (e.g. FastAPI lifespan) so
    the first real route optimization isn't delayed by JIT compilation.
    """
    haversine(0.0, 0.0, 1.0, 1.0)
    bearing(0.0, 0.0, 1.0, 1.0)
    current_effect(0.0, 1.0, 0.0)
    course_change_penalty(0.0, 45.0)

    seawater_density(15.0)
    seawater_viscosity(15.0)

    holtrop_mennen_resistance(
        7.0, 11.8, 65000.0, 0.82, 7500.0, 176.0, 32.0, 1025.0, 1.19e-6
    )
    wind_resistance(15.0, 0.0, 0.0, 450.0, 2100.0, 1.225)
    stawave1_wave_resistance(3.0, 0.0, 0.0, 7.0, 32.0, 176.0, 1025.0)
    kwon_speed_loss_pct(3.0, 0.0, 0.0, 0.82, 176.0)
    sfoc_curve(0.75, 171.0, 1.0)

    calculate_roll(3.0, 100.0, 1.57, 0.5, 0.45, 2.5, 0.05, 9.81)
    calculate_pitch(3.0, 100.0, 1.57, 7.0, 176.0)
    calculate_heave_accel(3.0, 0.5, 1.57)
    calculate_point_accel(1.0, 5.0, 0.5, 88.0)
    calculate_slamming_probability(3.0, 8.0, 6.0, 7.0, 0.0, 5.0, 88.0)
    calculate_green_water_probability(3.0, 6.0, 5.0, 88.0)
    calculate_parametric_roll_risk(100.0, 7.0, 14.0, 0.0, 176.0)

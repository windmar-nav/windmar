"""
Charter Party Weather Clause Tools.

Provides commercial analysis functions for charter party disputes:
- Beaufort scale classification
- Good weather day counting along a route
- Warranted speed/consumption verification
- Off-hire event detection from engine log data

These are data-presentation and contractual-analysis tools that
reuse existing voyage, weather, and performance predictor data.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Beaufort Scale Reference Table
# =============================================================================

BEAUFORT_SCALE = [
    {
        "force": 0,
        "wind_min_kts": 0,
        "wind_max_kts": 1,
        "wave_height_m": 0.0,
        "description": "Calm",
    },
    {
        "force": 1,
        "wind_min_kts": 1,
        "wind_max_kts": 3,
        "wave_height_m": 0.1,
        "description": "Light air",
    },
    {
        "force": 2,
        "wind_min_kts": 4,
        "wind_max_kts": 6,
        "wave_height_m": 0.3,
        "description": "Light breeze",
    },
    {
        "force": 3,
        "wind_min_kts": 7,
        "wind_max_kts": 10,
        "wave_height_m": 0.6,
        "description": "Gentle breeze",
    },
    {
        "force": 4,
        "wind_min_kts": 11,
        "wind_max_kts": 16,
        "wave_height_m": 1.0,
        "description": "Moderate breeze",
    },
    {
        "force": 5,
        "wind_min_kts": 17,
        "wind_max_kts": 21,
        "wave_height_m": 2.0,
        "description": "Fresh breeze",
    },
    {
        "force": 6,
        "wind_min_kts": 22,
        "wind_max_kts": 27,
        "wave_height_m": 3.0,
        "description": "Strong breeze",
    },
    {
        "force": 7,
        "wind_min_kts": 28,
        "wind_max_kts": 33,
        "wave_height_m": 4.0,
        "description": "Near gale",
    },
    {
        "force": 8,
        "wind_min_kts": 34,
        "wind_max_kts": 40,
        "wave_height_m": 5.5,
        "description": "Gale",
    },
    {
        "force": 9,
        "wind_min_kts": 41,
        "wind_max_kts": 47,
        "wave_height_m": 7.0,
        "description": "Severe gale",
    },
    {
        "force": 10,
        "wind_min_kts": 48,
        "wind_max_kts": 55,
        "wave_height_m": 9.0,
        "description": "Storm",
    },
    {
        "force": 11,
        "wind_min_kts": 56,
        "wind_max_kts": 63,
        "wave_height_m": 11.5,
        "description": "Violent storm",
    },
    {
        "force": 12,
        "wind_min_kts": 64,
        "wind_max_kts": 999,
        "wave_height_m": 14.0,
        "description": "Hurricane",
    },
]


# =============================================================================
# Result Dataclasses
# =============================================================================


@dataclass
class GoodWeatherLeg:
    """Per-leg good weather classification."""

    leg_index: int
    wind_speed_kts: float
    wave_height_m: float
    current_speed_ms: float
    bf_force: int
    is_good_weather: bool
    time_hours: float


@dataclass
class GoodWeatherDayResult:
    """Result of good weather day counting."""

    total_days: float
    good_weather_days: float
    bad_weather_days: float
    good_weather_pct: float
    bf_threshold: int
    wave_threshold_m: Optional[float]
    current_threshold_kts: Optional[float]
    legs: List[GoodWeatherLeg] = field(default_factory=list)


@dataclass
class WarrantyLegDetail:
    """Per-leg warranty verification detail."""

    leg_index: int
    sog_kts: float
    fuel_mt: float
    time_hours: float
    distance_nm: float
    bf_force: int
    is_good_weather: bool


@dataclass
class WarrantyVerificationResult:
    """Result of warranted speed/consumption verification."""

    warranted_speed_kts: float
    achieved_speed_kts: float
    speed_margin_kts: float
    speed_compliant: bool
    warranted_consumption_mt_day: float
    achieved_consumption_mt_day: float
    consumption_margin_mt: float
    consumption_compliant: bool
    good_weather_hours: float
    total_hours: float
    legs_assessed: int
    legs_good_weather: int
    legs: List[WarrantyLegDetail] = field(default_factory=list)


@dataclass
class OffHireEvent:
    """Single off-hire event."""

    start_time: datetime
    end_time: datetime
    duration_hours: float
    reason: str
    avg_speed_kts: Optional[float] = None


@dataclass
class OffHireAnalysisResult:
    """Result of off-hire event detection."""

    total_hours: float
    on_hire_hours: float
    off_hire_hours: float
    off_hire_pct: float
    events: List[OffHireEvent] = field(default_factory=list)


# =============================================================================
# Calculator
# =============================================================================


class CharterPartyCalculator:
    """Charter party weather clause analysis calculator."""

    @staticmethod
    def classify_beaufort(wind_speed_kts: float) -> int:
        """Classify wind speed into Beaufort force 0-12."""
        if wind_speed_kts < 0:
            return 0
        for entry in BEAUFORT_SCALE:
            if wind_speed_kts <= entry["wind_max_kts"]:
                return entry["force"]
        return 12

    def count_good_weather_days(
        self,
        legs: List[Dict],
        bf_threshold: int = 4,
        wave_threshold_m: Optional[float] = None,
        current_threshold_kts: Optional[float] = None,
    ) -> GoodWeatherDayResult:
        """
        Count good weather days along a route.

        A leg is "good weather" if:
        - Beaufort force <= bf_threshold
        - Wave height <= wave_threshold_m (if provided)
        - Current speed <= current_threshold_kts (if provided, converted from m/s)

        Args:
            legs: List of dicts with keys:
                  wind_speed_kts, wave_height_m, current_speed_ms, time_hours
            bf_threshold: Maximum Beaufort force for good weather (inclusive)
            wave_threshold_m: Maximum wave height in meters (optional)
            current_threshold_kts: Maximum current speed in knots (optional)

        Returns:
            GoodWeatherDayResult with totals and per-leg breakdown
        """
        if not legs:
            return GoodWeatherDayResult(
                total_days=0.0,
                good_weather_days=0.0,
                bad_weather_days=0.0,
                good_weather_pct=0.0,
                bf_threshold=bf_threshold,
                wave_threshold_m=wave_threshold_m,
                current_threshold_kts=current_threshold_kts,
            )

        good_hours = 0.0
        bad_hours = 0.0
        leg_results = []

        for i, leg in enumerate(legs):
            wind_kts = leg.get("wind_speed_kts", 0.0)
            wave_m = leg.get("wave_height_m", 0.0)
            current_ms = leg.get("current_speed_ms", 0.0)
            time_h = leg.get("time_hours", 0.0)

            bf = self.classify_beaufort(wind_kts)

            # Current in m/s → knots for threshold comparison
            current_kts = current_ms * 1.94384

            is_good = bf <= bf_threshold
            if is_good and wave_threshold_m is not None:
                is_good = wave_m <= wave_threshold_m
            if is_good and current_threshold_kts is not None:
                is_good = current_kts <= current_threshold_kts

            if is_good:
                good_hours += time_h
            else:
                bad_hours += time_h

            leg_results.append(
                GoodWeatherLeg(
                    leg_index=i,
                    wind_speed_kts=wind_kts,
                    wave_height_m=wave_m,
                    current_speed_ms=current_ms,
                    bf_force=bf,
                    is_good_weather=is_good,
                    time_hours=time_h,
                )
            )

        total_hours = good_hours + bad_hours
        total_days = round(total_hours / 24, 4)
        good_days = round(good_hours / 24, 4)
        bad_days = round(bad_hours / 24, 4)
        good_pct = round(
            (good_hours / total_hours * 100) if total_hours > 0 else 0.0, 2
        )

        return GoodWeatherDayResult(
            total_days=total_days,
            good_weather_days=good_days,
            bad_weather_days=bad_days,
            good_weather_pct=good_pct,
            bf_threshold=bf_threshold,
            wave_threshold_m=wave_threshold_m,
            current_threshold_kts=current_threshold_kts,
            legs=leg_results,
        )

    def verify_warranty(
        self,
        legs: List[Dict],
        warranted_speed_kts: float,
        warranted_consumption_mt_day: float,
        bf_threshold: int = 4,
        speed_tolerance_pct: float = 0.0,
        consumption_tolerance_pct: float = 0.0,
    ) -> WarrantyVerificationResult:
        """
        Verify warranted speed and consumption against good-weather legs.

        Args:
            legs: List of dicts with keys:
                  wind_speed_kts, wave_height_m, current_speed_ms,
                  time_hours, distance_nm, sog_kts, fuel_mt
            warranted_speed_kts: Chartered speed in knots
            warranted_consumption_mt_day: Chartered daily fuel consumption in MT
            bf_threshold: Maximum BF force for good weather
            speed_tolerance_pct: Allowable speed deficit as % (0-20)
            consumption_tolerance_pct: Allowable consumption excess as % (0-20)

        Returns:
            WarrantyVerificationResult with compliance flags and per-leg detail
        """
        leg_details = []
        gw_distance = 0.0
        gw_time_hours = 0.0
        gw_fuel = 0.0
        total_hours = 0.0
        gw_count = 0

        for i, leg in enumerate(legs):
            wind_kts = leg.get("wind_speed_kts", 0.0)
            wave_m = leg.get("wave_height_m", 0.0)
            time_h = leg.get("time_hours", 0.0)
            dist_nm = leg.get("distance_nm", 0.0)
            sog = leg.get("sog_kts", 0.0)
            fuel = leg.get("fuel_mt", 0.0)

            bf = self.classify_beaufort(wind_kts)
            is_good = bf <= bf_threshold

            total_hours += time_h

            if is_good:
                gw_distance += dist_nm
                gw_time_hours += time_h
                gw_fuel += fuel
                gw_count += 1

            leg_details.append(
                WarrantyLegDetail(
                    leg_index=i,
                    sog_kts=sog,
                    fuel_mt=fuel,
                    time_hours=time_h,
                    distance_nm=dist_nm,
                    bf_force=bf,
                    is_good_weather=is_good,
                )
            )

        # Calculate achieved speed: distance-weighted across good-weather legs
        if gw_time_hours > 0:
            achieved_speed = gw_distance / gw_time_hours
            achieved_consumption = gw_fuel / (gw_time_hours / 24)
        else:
            achieved_speed = 0.0
            achieved_consumption = 0.0

        # Apply tolerances
        min_speed = warranted_speed_kts * (1 - speed_tolerance_pct / 100)
        max_consumption = warranted_consumption_mt_day * (
            1 + consumption_tolerance_pct / 100
        )

        speed_compliant = achieved_speed >= min_speed
        consumption_compliant = (
            achieved_consumption <= max_consumption if gw_time_hours > 0 else True
        )

        speed_margin = round(achieved_speed - warranted_speed_kts, 4)
        consumption_margin = round(
            warranted_consumption_mt_day - achieved_consumption, 4
        )

        return WarrantyVerificationResult(
            warranted_speed_kts=warranted_speed_kts,
            achieved_speed_kts=round(achieved_speed, 4),
            speed_margin_kts=speed_margin,
            speed_compliant=speed_compliant,
            warranted_consumption_mt_day=warranted_consumption_mt_day,
            achieved_consumption_mt_day=round(achieved_consumption, 4),
            consumption_margin_mt=consumption_margin,
            consumption_compliant=consumption_compliant,
            good_weather_hours=round(gw_time_hours, 4),
            total_hours=round(total_hours, 4),
            legs_assessed=len(legs),
            legs_good_weather=gw_count,
            legs=leg_details,
        )

    def detect_off_hire(
        self,
        entries: List[Dict],
        rpm_threshold: float = 10.0,
        speed_threshold: float = 1.0,
        gap_hours: float = 6.0,
    ) -> OffHireAnalysisResult:
        """
        Detect off-hire events from engine log entries.

        Off-hire conditions:
        - Zero or very low RPM (< rpm_threshold)
        - Drifting (speed < speed_threshold with some RPM)
        - At anchor or in port (event field contains 'anchor' or 'port')
        - Timestamp gaps exceeding gap_hours

        Args:
            entries: List of dicts with keys:
                     timestamp (datetime), rpm (float), speed_stw (float),
                     event (str, optional), place (str, optional)
            rpm_threshold: RPM below which engine is considered stopped
            speed_threshold: Speed (kts) below which vessel is drifting
            gap_hours: Hours gap threshold for unaccounted time

        Returns:
            OffHireAnalysisResult with total/on/off hours and event list
        """
        if not entries:
            return OffHireAnalysisResult(
                total_hours=0.0,
                on_hire_hours=0.0,
                off_hire_hours=0.0,
                off_hire_pct=0.0,
            )

        # Sort by timestamp
        sorted_entries = sorted(entries, key=lambda e: e["timestamp"])

        if len(sorted_entries) < 2:
            return OffHireAnalysisResult(
                total_hours=0.0,
                on_hire_hours=0.0,
                off_hire_hours=0.0,
                off_hire_pct=0.0,
            )

        raw_events: List[OffHireEvent] = []
        first_ts = sorted_entries[0]["timestamp"]
        last_ts = sorted_entries[-1]["timestamp"]
        total_hours = (last_ts - first_ts).total_seconds() / 3600

        for i in range(len(sorted_entries) - 1):
            curr = sorted_entries[i]
            nxt = sorted_entries[i + 1]

            ts_curr = curr["timestamp"]
            ts_next = nxt["timestamp"]
            interval_hours = (ts_next - ts_curr).total_seconds() / 3600

            if interval_hours <= 0:
                continue

            rpm = curr.get("rpm", 0.0) or 0.0
            speed = curr.get("speed_stw", 0.0) or 0.0
            event = (curr.get("event") or "").lower()
            place = (curr.get("place") or "").lower()

            reason = None

            # Check timestamp gap
            if interval_hours > gap_hours:
                reason = "Timestamp gap"
            # Check anchored/port
            elif "anchor" in event or "anchor" in place:
                reason = "At anchor"
            elif "port" in event or "port" in place:
                reason = "In port"
            # Check zero/low RPM
            elif rpm < rpm_threshold:
                reason = "Engine stopped"
            # Check drifting (some RPM but very low speed)
            elif speed < speed_threshold:
                reason = "Drifting"

            if reason:
                raw_events.append(
                    OffHireEvent(
                        start_time=ts_curr,
                        end_time=ts_next,
                        duration_hours=round(interval_hours, 4),
                        reason=reason,
                        avg_speed_kts=round(speed, 2) if speed > 0 else 0.0,
                    )
                )

        # Merge adjacent events within 1 hour with same reason
        merged = self._merge_off_hire_events(raw_events)

        off_hire_hours = round(sum(e.duration_hours for e in merged), 4)
        on_hire_hours = round(max(total_hours - off_hire_hours, 0.0), 4)
        off_hire_pct = round(
            (off_hire_hours / total_hours * 100) if total_hours > 0 else 0.0, 2
        )

        return OffHireAnalysisResult(
            total_hours=round(total_hours, 4),
            on_hire_hours=on_hire_hours,
            off_hire_hours=off_hire_hours,
            off_hire_pct=off_hire_pct,
            events=merged,
        )

    @staticmethod
    def _merge_off_hire_events(events: List[OffHireEvent]) -> List[OffHireEvent]:
        """Merge adjacent off-hire events within 1 hour with same reason."""
        if not events:
            return []

        merged = [events[0]]
        for ev in events[1:]:
            prev = merged[-1]
            gap = (ev.start_time - prev.end_time).total_seconds() / 3600

            if prev.reason == ev.reason and gap <= 1.0:
                # Merge: extend end_time and accumulate duration
                total_dur = (ev.end_time - prev.start_time).total_seconds() / 3600
                merged[-1] = OffHireEvent(
                    start_time=prev.start_time,
                    end_time=ev.end_time,
                    duration_hours=round(total_dur, 4),
                    reason=prev.reason,
                    avg_speed_kts=prev.avg_speed_kts,
                )
            else:
                merged.append(ev)

        return merged

"""
Noon report generation from persisted voyage legs.

Generates synthetic noon reports by interpolating voyage legs at 24-hour
intervals from departure, following standard maritime reporting conventions.
"""

from datetime import datetime, timedelta
from typing import List, Optional

from api.models import Voyage, VoyageLeg


def generate_noon_reports(voyage: Voyage) -> List[dict]:
    """Generate noon reports at 24h intervals from voyage legs.

    Walks through legs chronologically and at every 24h mark from departure,
    interpolates position and conditions between leg waypoints.

    Returns:
        List of noon report dicts matching NoonReportEntry schema.
    """
    legs: List[VoyageLeg] = sorted(voyage.legs, key=lambda l: l.leg_index)
    if not legs:
        return []

    departure = voyage.departure_time
    arrival = voyage.arrival_time
    total_hours = voyage.total_time_hours

    # Build cumulative timeline from legs
    timeline = _build_timeline(legs, departure)

    reports = []
    report_num = 0
    cumulative_distance = 0.0
    cumulative_fuel = 0.0
    last_report_distance = 0.0
    last_report_fuel = 0.0

    # Generate reports at 24h intervals
    interval_hours = 24.0
    report_time = departure + timedelta(hours=interval_hours)

    while report_time < arrival:
        report_num += 1
        point = _interpolate_at_time(timeline, report_time)
        if point is None:
            report_time += timedelta(hours=interval_hours)
            continue

        cumulative_distance = point["cumulative_distance_nm"]
        cumulative_fuel = point["cumulative_fuel_mt"]
        distance_since_last = cumulative_distance - last_report_distance
        fuel_since_last = cumulative_fuel - last_report_fuel

        reports.append({
            "report_number": report_num,
            "timestamp": report_time,
            "lat": round(point["lat"], 4),
            "lon": round(point["lon"], 4),
            "sog_kts": point.get("sog_kts"),
            "stw_kts": point.get("stw_kts"),
            "course_deg": point.get("bearing_deg"),
            "distance_since_last_nm": round(distance_since_last, 2),
            "fuel_since_last_mt": round(fuel_since_last, 2),
            "cumulative_distance_nm": round(cumulative_distance, 2),
            "cumulative_fuel_mt": round(cumulative_fuel, 2),
            "wind_speed_kts": point.get("wind_speed_kts"),
            "wind_dir_deg": point.get("wind_dir_deg"),
            "wave_height_m": point.get("wave_height_m"),
            "wave_dir_deg": point.get("wave_dir_deg"),
            "current_speed_ms": point.get("current_speed_ms"),
            "current_dir_deg": point.get("current_dir_deg"),
        })

        last_report_distance = cumulative_distance
        last_report_fuel = cumulative_fuel
        report_time += timedelta(hours=interval_hours)

    return reports


def _build_timeline(legs: List[VoyageLeg], departure: datetime) -> List[dict]:
    """Build a chronological timeline of leg boundaries with cumulative values."""
    timeline = []
    cum_distance = 0.0
    cum_fuel = 0.0
    cum_hours = 0.0

    for leg in legs:
        leg_dep = leg.departure_time or (departure + timedelta(hours=cum_hours))

        # Start point of leg
        timeline.append({
            "time": leg_dep,
            "lat": leg.from_lat,
            "lon": leg.from_lon,
            "cumulative_distance_nm": cum_distance,
            "cumulative_fuel_mt": cum_fuel,
            "sog_kts": leg.sog_kts,
            "stw_kts": leg.stw_kts,
            "bearing_deg": leg.bearing_deg,
            "wind_speed_kts": leg.wind_speed_kts,
            "wind_dir_deg": leg.wind_dir_deg,
            "wave_height_m": leg.wave_height_m,
            "wave_dir_deg": leg.wave_dir_deg,
            "current_speed_ms": leg.current_speed_ms,
            "current_dir_deg": leg.current_dir_deg,
        })

        cum_distance += leg.distance_nm
        cum_fuel += leg.fuel_mt
        cum_hours += leg.time_hours

        leg_arr = leg.arrival_time or (departure + timedelta(hours=cum_hours))

        # End point of leg
        timeline.append({
            "time": leg_arr,
            "lat": leg.to_lat,
            "lon": leg.to_lon,
            "cumulative_distance_nm": cum_distance,
            "cumulative_fuel_mt": cum_fuel,
            "sog_kts": leg.sog_kts,
            "stw_kts": leg.stw_kts,
            "bearing_deg": leg.bearing_deg,
            "wind_speed_kts": leg.wind_speed_kts,
            "wind_dir_deg": leg.wind_dir_deg,
            "wave_height_m": leg.wave_height_m,
            "wave_dir_deg": leg.wave_dir_deg,
            "current_speed_ms": leg.current_speed_ms,
            "current_dir_deg": leg.current_dir_deg,
        })

    return timeline


def _interpolate_at_time(timeline: List[dict], target_time: datetime) -> Optional[dict]:
    """Interpolate position and cumulative values at a given time."""
    if not timeline:
        return None

    # Find the bracketing points
    for i in range(len(timeline) - 1):
        t0 = timeline[i]["time"]
        t1 = timeline[i + 1]["time"]

        if t0 <= target_time <= t1:
            total_secs = (t1 - t0).total_seconds()
            if total_secs <= 0:
                return timeline[i].copy()

            frac = (target_time - t0).total_seconds() / total_secs

            return {
                "lat": timeline[i]["lat"] + frac * (timeline[i + 1]["lat"] - timeline[i]["lat"]),
                "lon": timeline[i]["lon"] + frac * (timeline[i + 1]["lon"] - timeline[i]["lon"]),
                "cumulative_distance_nm": (
                    timeline[i]["cumulative_distance_nm"]
                    + frac * (timeline[i + 1]["cumulative_distance_nm"] - timeline[i]["cumulative_distance_nm"])
                ),
                "cumulative_fuel_mt": (
                    timeline[i]["cumulative_fuel_mt"]
                    + frac * (timeline[i + 1]["cumulative_fuel_mt"] - timeline[i]["cumulative_fuel_mt"])
                ),
                "sog_kts": timeline[i].get("sog_kts"),
                "stw_kts": timeline[i].get("stw_kts"),
                "bearing_deg": timeline[i].get("bearing_deg"),
                "wind_speed_kts": timeline[i].get("wind_speed_kts"),
                "wind_dir_deg": timeline[i].get("wind_dir_deg"),
                "wave_height_m": timeline[i].get("wave_height_m"),
                "wave_dir_deg": timeline[i].get("wave_dir_deg"),
                "current_speed_ms": timeline[i].get("current_speed_ms"),
                "current_dir_deg": timeline[i].get("current_dir_deg"),
            }

    return None

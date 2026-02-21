"""
Departure and arrival report template builders.

Generates standard maritime report data from persisted voyage records.
"""

from typing import Dict, List, Optional

from api.models import Voyage, VoyageLeg


def build_departure_report(voyage: Voyage) -> dict:
    """Build departure report data from a voyage.

    Returns dict matching DepartureReportData schema.
    """
    legs: List[VoyageLeg] = sorted(voyage.legs, key=lambda l: l.leg_index)

    # Weather at departure — from first leg
    weather_at_departure = None
    if legs:
        first = legs[0]
        weather_at_departure = {
            "wind_speed_kts": first.wind_speed_kts,
            "wind_dir_deg": first.wind_dir_deg,
            "wave_height_m": first.wave_height_m,
            "wave_dir_deg": first.wave_dir_deg,
            "current_speed_ms": first.current_speed_ms,
            "current_dir_deg": first.current_dir_deg,
        }

    # Vessel name from snapshot
    vessel_name = None
    dwt = None
    specs = voyage.vessel_specs_snapshot
    if specs:
        vessel_name = specs.get("name")
        dwt = specs.get("deadweight") or specs.get("dwt")

    return {
        "vessel_name": vessel_name,
        "dwt": dwt,
        "departure_port": voyage.departure_port,
        "departure_time": voyage.departure_time,
        "loading_condition": "Laden" if voyage.is_laden else "Ballast",
        "destination": voyage.arrival_port,
        "eta": voyage.arrival_time,
        "planned_distance_nm": voyage.total_distance_nm,
        "planned_speed_kts": voyage.calm_speed_kts,
        "estimated_fuel_mt": voyage.total_fuel_mt,
        "weather_at_departure": weather_at_departure,
    }


def build_arrival_report(voyage: Voyage) -> dict:
    """Build arrival report data from a voyage.

    Returns dict matching ArrivalReportData schema.
    """
    legs: List[VoyageLeg] = sorted(voyage.legs, key=lambda l: l.leg_index)

    # Weather summary — min/max/avg across all legs
    weather_summary = _summarize_weather(legs)

    vessel_name = None
    specs = voyage.vessel_specs_snapshot
    if specs:
        vessel_name = specs.get("name")

    return {
        "vessel_name": vessel_name,
        "arrival_port": voyage.arrival_port,
        "arrival_time": voyage.arrival_time,
        "actual_voyage_time_hours": voyage.total_time_hours,
        "total_fuel_consumed_mt": voyage.total_fuel_mt,
        "average_speed_kts": voyage.avg_sog_kts or 0.0,
        "total_distance_nm": voyage.total_distance_nm,
        "weather_summary": weather_summary,
        "cii_estimate": voyage.cii_estimate,
    }


def _summarize_weather(legs: List[VoyageLeg]) -> Optional[Dict]:
    """Summarize weather conditions across all legs."""
    if not legs:
        return None

    wind_speeds = [l.wind_speed_kts for l in legs if l.wind_speed_kts is not None]
    wave_heights = [l.wave_height_m for l in legs if l.wave_height_m is not None]
    current_speeds = [l.current_speed_ms for l in legs if l.current_speed_ms is not None]

    summary = {}
    if wind_speeds:
        summary["wind_speed_kts"] = {
            "min": round(min(wind_speeds), 1),
            "max": round(max(wind_speeds), 1),
            "avg": round(sum(wind_speeds) / len(wind_speeds), 1),
        }
    if wave_heights:
        summary["wave_height_m"] = {
            "min": round(min(wave_heights), 1),
            "max": round(max(wave_heights), 1),
            "avg": round(sum(wave_heights) / len(wave_heights), 1),
        }
    if current_speeds:
        summary["current_speed_ms"] = {
            "min": round(min(current_speeds), 2),
            "max": round(max(current_speeds), 2),
            "avg": round(sum(current_speeds) / len(current_speeds), 2),
        }

    return summary or None

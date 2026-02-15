"""
Engine log Excel parser.

Parses engine log workbooks with multi-tier merged headers (3-4 rows).
Uses column-index-based mapping instead of header parsing, since the
merged header structure is fragile across vessels and reporting periods.

Designed for the ECOMAR engine log format (E log sheet), but validates
layout fingerprints before parsing so incompatible formats fail early.
"""

import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column index map for E log sheet
# ---------------------------------------------------------------------------
# Based on ECOMAR GARONNE format. Indices are 0-based pandas column positions
# when read with header=None.

COLUMN_MAP: Dict[str, int] = {
    # Navigation
    "date": 1,
    "time": 2,
    "datetime_combined": 3,
    "lapse_hours": 5,
    "place": 6,
    "event": 7,
    # ME Operational
    "me_revs_counter": 8,
    "revs_period": 9,
    "rpm": 10,
    "engine_distance": 11,
    "engine_speed": 12,
    "log_distance": 13,
    "slip_pct": 15,
    "speed_stw": 16,
    "me_power_kw": 17,
    "me_fuel_index_pct": 18,
    "me_load_pct": 19,
    "shaft_power": 20,
    "shaft_torque_knm": 21,
    # Technical temperatures & pressures
    "me_tc_in_c": 22,
    "me_tc_out_c": 23,
    "air_cooler_in_c": 24,
    "air_cooler_out_c": 25,
    "water_air_cooler_in_c": 26,
    "water_air_cooler_out_c": 27,
    "scav_air_press_bar": 28,
    "tc_rpm": 29,
    "fuel_temp_c": 30,
    "sw_temp_c": 31,
    "drop_after_cooler_mm": 32,
    # Cumulative running hours
    "rh_me_cum": 34,
    "rh_sg_cum": 35,
    "rh_ae1_cum": 36,
    "rh_ae2_cum": 37,
    "rh_ae3_cum": 38,
    "rh_aux_boiler_cum": 39,
    "rh_comp_boiler_cum": 40,
    # Period running hours
    "rh_me": 56,
    "rh_sg": 57,
    "rh_ae1": 58,
    "rh_ae2": 59,
    "rh_ae3": 60,
    "rh_ae_total": 61,
    "rh_aux_boiler": 62,
    "rh_comp_boiler": 63,
    # HFO Consumption (MT)
    "hfo_me_mt": 83,
    "hfo_ae_mt": 84,
    "hfo_aux_boiler_mt": 85,
    "hfo_comp_boiler_mt": 86,
    "hfo_total_mt": 87,
    "rob_vlsfo_mt": 88,
    # MGO Consumption (MT)
    "mgo_me_mt": 90,
    "mgo_ae_mt": 91,
    "mgo_aux_boiler_mt": 92,
    "mgo_comp_boiler_mt": 93,
    "mgo_igg_mt": 94,
    "mgo_incinerator_mt": 95,
    "mgo_total_mt": 96,
    "rob_mgo_mt": 97,
    # Methanol
    "methanol_me_mt": 99,
    "rob_methanol_mt": 100,
}

# Primary fields that map directly to EngineLogEntry typed columns
PRIMARY_FIELDS: Dict[str, int] = {
    "lapse_hours": 5,
    "place": 6,
    "event": 7,
    "rpm": 10,
    "engine_distance": 11,
    "speed_stw": 16,
    "me_power_kw": 17,
    "me_fuel_index_pct": 18,
    "me_load_pct": 19,
    "shaft_power": 20,
    "shaft_torque_knm": 21,
    "slip_pct": 15,
    "hfo_me_mt": 83,
    "hfo_ae_mt": 84,
    "hfo_boiler_mt": 87,  # computed specially
    "hfo_total_mt": 87,
    "mgo_me_mt": 90,
    "mgo_ae_mt": 91,
    "mgo_total_mt": 96,
    "methanol_me_mt": 99,
    "rob_vlsfo_mt": 88,
    "rob_mgo_mt": 97,
    "rob_methanol_mt": 100,
    "rh_me": 56,
    "rh_ae_total": 61,
    "tc_rpm": 29,
    "scav_air_press_bar": 28,
    "fuel_temp_c": 30,
    "sw_temp_c": 31,
}

# Data row start (0-indexed). Rows 0-5 are headers/vessel info.
DATA_START_ROW = 6

# Layout fingerprint positions: (row, col, expected_substring)
LAYOUT_FINGERPRINTS = [
    (2, 8, "main engine"),
    (3, 10, "rpm"),
    (2, 83, "hfo"),
    (2, 90, "mgo"),
]

# Event normalization lookup
EVENT_NORMALIZE: Dict[str, str] = {
    "noon": "NOON",
    "sosp": "SOSP",
    "eosp": "EOSP",
    "all fast": "ALL_FAST",
    "all clear": "ALL_CLEAR",
    "drop anchor": "DROP_ANCHOR",
    "anchor aweigh": "ANCHOR_AWEIGH",
    "drifting": "DRIFTING",
    "compl. drifting": "COMPL_DRIFTING",
    "deviation out": "DEVIATION_OUT",
    "deviation in": "DEVIATION_IN",
    "alter course": "ALTER_COURSE",
    "resume voyage": "RESUME_VOYAGE",
    "pilot on": "PILOT_ON",
    "pilot off": "PILOT_OFF",
    "1st parcel": "FIRST_PARCEL",
    "2nd parcel": "SECOND_PARCEL",
    "3rd parcel": "THIRD_PARCEL",
    "compl. disch.": "COMPL_DISCHARGE",
    "comm. c/o": "COMM_CHANGEOVER",
    "loading completed": "LOADING_COMPLETED",
    "anchorage": "ANCHORAGE",
    "bosp": "BOSP",
}


def _safe_float(value: Any) -> Optional[float]:
    """Convert value to float, returning None for NaN/Infinity/non-numeric."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() in ("nan", "n/a", "-", ""):
            return None
        try:
            val = float(value)
        except (ValueError, TypeError):
            return None
    else:
        try:
            val = float(value)
        except (ValueError, TypeError):
            return None

    if math.isnan(val) or math.isinf(val):
        return None
    return val


def _safe_str(value: Any) -> Optional[str]:
    """Convert value to stripped string, returning None for NaN/empty."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    s = str(value).strip()
    return s if s else None


def _normalize_event(raw: Optional[str]) -> Optional[str]:
    """Normalize event string to a consistent enum-like value."""
    if raw is None:
        return None
    key = raw.lower().strip()
    return EVENT_NORMALIZE.get(key, key.upper().replace(" ", "_"))


class EngineLogParser:
    """Parse engine log data from multi-tier-header Excel workbooks.

    Uses column-index-based mapping. Validates layout fingerprints
    before parsing to catch incompatible formats early.
    """

    DEFAULT_SHEET = "E log"

    def __init__(self, excel_file: Path):
        self.excel_file = Path(excel_file)
        if not self.excel_file.exists():
            raise FileNotFoundError(f"Engine log file not found: {self.excel_file}")

        self._entries: List[Dict[str, Any]] = []
        self._raw_df: Optional[pd.DataFrame] = None
        self._statistics: Optional[Dict[str, Any]] = None

    @property
    def entries(self) -> List[Dict[str, Any]]:
        return list(self._entries)

    def parse(self, sheet_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Parse engine log entries from the specified sheet."""
        sheet = sheet_name or self.DEFAULT_SHEET
        logger.info(f"Parsing engine log: {self.excel_file} sheet={sheet}")

        try:
            available_sheets = pd.ExcelFile(self.excel_file).sheet_names
        except Exception as e:
            raise ValueError(f"Cannot read Excel file: {e}") from e

        if sheet not in available_sheets:
            raise ValueError(
                f"Sheet '{sheet}' not found. Available: {available_sheets}"
            )

        self._raw_df = pd.read_excel(
            self.excel_file, sheet_name=sheet, header=None
        )
        logger.info(f"Raw shape: {self._raw_df.shape}")

        self._validate_layout()

        self._entries = []
        skipped = 0
        for row_idx in range(DATA_START_ROW, len(self._raw_df)):
            row = self._raw_df.iloc[row_idx]
            try:
                entry = self._parse_row(row, row_idx, sheet)
                if entry is not None:
                    self._entries.append(entry)
                else:
                    skipped += 1
            except Exception as e:
                logger.warning(f"Row {row_idx}: skipped ({e})")
                skipped += 1

        logger.info(
            f"Parsed {len(self._entries)} entries, skipped {skipped} rows"
        )
        self._statistics = None
        return self._entries

    def _validate_layout(self) -> None:
        """Check layout fingerprints to ensure this is a compatible format."""
        df = self._raw_df
        for row, col, expected in LAYOUT_FINGERPRINTS:
            if row >= len(df) or col >= len(df.columns):
                raise ValueError(
                    f"Layout validation failed: sheet too small "
                    f"(need row {row}, col {col})"
                )
            cell = df.iloc[row, col]
            cell_str = str(cell).lower().strip() if pd.notna(cell) else ""
            if expected.lower() not in cell_str:
                raise ValueError(
                    f"Layout validation failed at row={row}, col={col}: "
                    f"expected '{expected}', got '{cell_str}'"
                )
        logger.debug("Layout validation passed")

    def _parse_row(
        self, row: pd.Series, row_idx: int, sheet: str
    ) -> Optional[Dict[str, Any]]:
        """Parse a single data row into an entry dict."""
        raw_date = row.iloc[COLUMN_MAP["date"]] if COLUMN_MAP["date"] < len(row) else None
        if pd.isna(raw_date):
            return None

        timestamp = self._parse_timestamp(row)
        if timestamp is None:
            logger.warning(f"Row {row_idx}: cannot parse timestamp")
            return None

        entry: Dict[str, Any] = {"timestamp": timestamp}

        # String fields
        entry["place"] = _safe_str(self._cell(row, "place"))
        entry["event"] = _normalize_event(_safe_str(self._cell(row, "event")))

        # Numeric primary fields
        for field_name, col_idx in PRIMARY_FIELDS.items():
            if field_name in ("place", "event"):
                continue
            if field_name == "hfo_boiler_mt":
                aux = _safe_float(self._cell_idx(row, 85))
                comp = _safe_float(self._cell_idx(row, 86))
                if aux is not None or comp is not None:
                    entry[field_name] = (aux or 0.0) + (comp or 0.0)
                else:
                    entry[field_name] = None
                continue
            entry[field_name] = _safe_float(self._cell_idx(row, col_idx))

        # Extended data: everything in COLUMN_MAP not in PRIMARY_FIELDS
        extended: Dict[str, Any] = {}
        for field_name, col_idx in COLUMN_MAP.items():
            if field_name in PRIMARY_FIELDS or field_name in (
                "date", "time", "datetime_combined",
            ):
                continue
            val = self._cell_idx(row, col_idx)
            if pd.notna(val) if not isinstance(val, str) else bool(val):
                coerced = _safe_float(val)
                extended[field_name] = coerced if coerced is not None else _safe_str(val)

        entry["extended_data"] = extended if extended else None
        entry["source_sheet"] = sheet
        entry["source_file"] = self.excel_file.name

        return entry

    def _parse_timestamp(self, row: pd.Series) -> Optional[datetime]:
        """Parse timestamp using 3 fallback strategies."""
        # Strategy 1: combined datetime
        combined = self._cell(row, "datetime_combined")
        if pd.notna(combined):
            try:
                ts = pd.to_datetime(combined)
                if pd.notna(ts):
                    return ts.to_pydatetime()
            except (ValueError, TypeError):
                pass

        # Strategy 2: date + time
        date_val = self._cell(row, "date")
        time_val = self._cell(row, "time")
        if pd.notna(date_val):
            try:
                dt = pd.to_datetime(date_val)
                if pd.notna(time_val):
                    try:
                        from datetime import time as dt_time
                        if isinstance(time_val, dt_time):
                            dt = dt.replace(
                                hour=time_val.hour,
                                minute=time_val.minute,
                                second=time_val.second,
                            )
                        else:
                            t = pd.to_datetime(str(time_val))
                            dt = dt.replace(
                                hour=t.hour, minute=t.minute, second=t.second,
                            )
                    except (ValueError, TypeError):
                        pass
                return dt.to_pydatetime()
            except (ValueError, TypeError):
                pass

        # Strategy 3: date only
        if pd.notna(date_val):
            try:
                return pd.to_datetime(date_val).to_pydatetime()
            except (ValueError, TypeError):
                pass

        return None

    def _cell(self, row: pd.Series, field: str) -> Any:
        """Get cell value by field name from COLUMN_MAP."""
        col_idx = COLUMN_MAP.get(field)
        if col_idx is None or col_idx >= len(row):
            return None
        return row.iloc[col_idx]

    def _cell_idx(self, row: pd.Series, col_idx: int) -> Any:
        """Get cell value by column index."""
        if col_idx >= len(row):
            return None
        return row.iloc[col_idx]

    def get_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics from parsed entries."""
        if not self._entries:
            return {"total_entries": 0}

        if self._statistics is not None:
            return self._statistics

        timestamps = [e["timestamp"] for e in self._entries]
        events = [e.get("event") for e in self._entries if e.get("event")]

        event_counts: Dict[str, int] = {}
        for ev in events:
            event_counts[ev] = event_counts.get(ev, 0) + 1

        hfo_total = sum(
            e["hfo_total_mt"] for e in self._entries
            if e.get("hfo_total_mt") is not None
        )
        mgo_total = sum(
            e["mgo_total_mt"] for e in self._entries
            if e.get("mgo_total_mt") is not None
        )
        methanol_total = sum(
            e["methanol_me_mt"] for e in self._entries
            if e.get("methanol_me_mt") is not None
        )

        sea_rpms = [
            e["rpm"] for e in self._entries
            if e.get("event") == "NOON" and e.get("rpm") is not None and e["rpm"] > 0
        ]
        avg_rpm_at_sea = sum(sea_rpms) / len(sea_rpms) if sea_rpms else None

        sea_speeds = [
            e["speed_stw"] for e in self._entries
            if e.get("event") == "NOON" and e.get("speed_stw") is not None and e["speed_stw"] > 0
        ]
        avg_speed_stw = sum(sea_speeds) / len(sea_speeds) if sea_speeds else None

        self._statistics = {
            "total_entries": len(self._entries),
            "date_range": {
                "start": min(timestamps).isoformat(),
                "end": max(timestamps).isoformat(),
            },
            "events_breakdown": event_counts,
            "fuel_totals": {
                "hfo_mt": round(hfo_total, 3),
                "mgo_mt": round(mgo_total, 3),
                "methanol_mt": round(methanol_total, 3),
            },
            "avg_rpm_at_sea": round(avg_rpm_at_sea, 1) if avg_rpm_at_sea else None,
            "avg_speed_stw": round(avg_speed_stw, 2) if avg_speed_stw else None,
        }
        return self._statistics

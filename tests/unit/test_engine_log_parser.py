"""Unit tests for engine log Excel parser."""

import math
import tempfile
from datetime import datetime, time
from pathlib import Path

import pandas as pd
import pytest

from src.database.engine_log_parser import (
    COLUMN_MAP,
    DATA_START_ROW,
    EngineLogParser,
    _normalize_event,
    _safe_float,
    _safe_str,
)

# Path to real GARONNE dataset (skip tests if not present)
GARONNE_FILE = Path(
    "/home/slmar/Desktop/Spec WindMar/01. Engine Log GARONNE- 0929.xlsx"
)


def _make_elog_excel(
    rows: list[list],
    header_override: dict | None = None,
    sheet_name: str = "E log",
) -> Path:
    """Build a minimal E log Excel file with valid layout fingerprints."""
    ncols = 138
    total_rows = DATA_START_ROW + len(rows)
    data = [[None] * ncols for _ in range(total_rows)]

    # Fingerprint cells
    data[2][8] = "MAIN ENGINE PARAMETERS"
    data[3][10] = "RPM"
    data[2][83] = "HFO CONSUMPTION (MT)"
    data[2][90] = "MGO CONSUMPTION (MT)"

    if header_override:
        for (r, c), val in header_override.items():
            data[r][c] = val

    for i, row_data in enumerate(rows):
        for j, val in enumerate(row_data):
            if j < ncols:
                data[DATA_START_ROW + i][j] = val

    df = pd.DataFrame(data)
    tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
    with pd.ExcelWriter(tmp.name, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
    return Path(tmp.name)


def _make_data_row(**kwargs) -> list:
    """Build a data row list from keyword args keyed by COLUMN_MAP field names."""
    row = [None] * 138
    for field, value in kwargs.items():
        if field in COLUMN_MAP:
            row[COLUMN_MAP[field]] = value
    return row


class TestSafeFloat:
    def test_normal_float(self):
        assert _safe_float(42.5) == 42.5

    def test_zero(self):
        assert _safe_float(0) == 0.0

    def test_nan_returns_none(self):
        assert _safe_float(float("nan")) is None

    def test_inf_returns_none(self):
        assert _safe_float(float("inf")) is None
        assert _safe_float(float("-inf")) is None

    def test_none_returns_none(self):
        assert _safe_float(None) is None

    def test_string_number(self):
        assert _safe_float("3.14") == 3.14

    def test_empty_string(self):
        assert _safe_float("") is None

    def test_nan_string(self):
        assert _safe_float("nan") is None

    def test_non_numeric_string(self):
        assert _safe_float("abc") is None


class TestSafeStr:
    def test_normal_string(self):
        assert _safe_str("Noon") == "Noon"

    def test_nan_float(self):
        assert _safe_str(float("nan")) is None

    def test_none(self):
        assert _safe_str(None) is None

    def test_whitespace_stripped(self):
        assert _safe_str("  At Sea  ") == "At Sea"

    def test_empty_string(self):
        assert _safe_str("") is None


class TestEventNormalization:
    def test_noon(self):
        assert _normalize_event("Noon") == "NOON"
        assert _normalize_event("noon") == "NOON"
        assert _normalize_event("  NOON  ") == "NOON"

    def test_sosp(self):
        assert _normalize_event("SOSP") == "SOSP"
        assert _normalize_event("sosp") == "SOSP"

    def test_eosp(self):
        assert _normalize_event("EOSP") == "EOSP"

    def test_all_fast(self):
        assert _normalize_event("All Fast") == "ALL_FAST"
        assert _normalize_event("all fast") == "ALL_FAST"

    def test_all_clear(self):
        assert _normalize_event("All Clear") == "ALL_CLEAR"

    def test_drop_anchor(self):
        assert _normalize_event("Drop Anchor") == "DROP_ANCHOR"

    def test_unknown_event_uppercased(self):
        assert _normalize_event("custom event") == "CUSTOM_EVENT"

    def test_none_returns_none(self):
        assert _normalize_event(None) is None


class TestEngineLogParserInit:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            EngineLogParser(Path("/nonexistent/path.xlsx"))

    def test_valid_file(self):
        path = _make_elog_excel([])
        try:
            parser = EngineLogParser(path)
            assert parser.excel_file == path
        finally:
            path.unlink(missing_ok=True)


class TestLayoutValidation:
    def test_valid_layout_passes(self):
        path = _make_elog_excel([
            _make_data_row(date=datetime(2025, 8, 1), event="Noon", rpm=50),
        ])
        try:
            parser = EngineLogParser(path)
            entries = parser.parse()
            assert len(entries) >= 1
        finally:
            path.unlink(missing_ok=True)

    def test_wrong_format_raises_valueerror(self):
        path = _make_elog_excel([], header_override={(2, 8): "SOMETHING ELSE"})
        try:
            parser = EngineLogParser(path)
            with pytest.raises(ValueError, match="Layout validation failed"):
                parser.parse()
        finally:
            path.unlink(missing_ok=True)

    def test_wrong_sheet_name_raises(self):
        path = _make_elog_excel([])
        try:
            parser = EngineLogParser(path)
            with pytest.raises(ValueError, match="not found"):
                parser.parse(sheet_name="Nonexistent Sheet")
        finally:
            path.unlink(missing_ok=True)


class TestTimestampParsing:
    def test_combined_datetime(self):
        ts = datetime(2025, 8, 15, 12, 0, 0)
        path = _make_elog_excel([
            _make_data_row(date=ts, datetime_combined=ts, event="Noon"),
        ])
        try:
            entries = EngineLogParser(path).parse()
            assert len(entries) == 1
            assert entries[0]["timestamp"] == ts
        finally:
            path.unlink(missing_ok=True)

    def test_date_plus_time(self):
        d = datetime(2025, 8, 15)
        t = time(14, 30, 0)
        path = _make_elog_excel([_make_data_row(date=d, time=t, event="Noon")])
        try:
            entries = EngineLogParser(path).parse()
            assert len(entries) == 1
            assert entries[0]["timestamp"].hour == 14
            assert entries[0]["timestamp"].minute == 30
        finally:
            path.unlink(missing_ok=True)

    def test_date_only(self):
        d = datetime(2025, 8, 15)
        path = _make_elog_excel([_make_data_row(date=d, event="Noon")])
        try:
            entries = EngineLogParser(path).parse()
            assert len(entries) == 1
            assert entries[0]["timestamp"].date() == d.date()
        finally:
            path.unlink(missing_ok=True)


class TestDataParsing:
    def test_rpm_zero_accepted(self):
        path = _make_elog_excel([
            _make_data_row(date=datetime(2025, 8, 1), event="All Fast", rpm=0, place="Singapore"),
        ])
        try:
            entries = EngineLogParser(path).parse()
            assert len(entries) == 1
            assert entries[0]["rpm"] == 0.0
        finally:
            path.unlink(missing_ok=True)

    def test_null_becomes_none(self):
        path = _make_elog_excel([
            _make_data_row(date=datetime(2025, 8, 1), event="Noon", rpm=50),
        ])
        try:
            entries = EngineLogParser(path).parse()
            assert entries[0]["speed_stw"] is None
            assert entries[0]["me_power_kw"] is None
        finally:
            path.unlink(missing_ok=True)

    def test_fuel_totals_extracted(self):
        path = _make_elog_excel([
            _make_data_row(
                date=datetime(2025, 8, 1), event="Noon",
                hfo_me_mt=5.2, hfo_ae_mt=1.1, hfo_total_mt=7.5,
                mgo_me_mt=0.3, mgo_total_mt=0.8,
            ),
        ])
        try:
            entries = EngineLogParser(path).parse()
            assert entries[0]["hfo_me_mt"] == 5.2
            assert entries[0]["hfo_total_mt"] == 7.5
            assert entries[0]["mgo_total_mt"] == 0.8
        finally:
            path.unlink(missing_ok=True)

    def test_empty_rows_skipped(self):
        path = _make_elog_excel([
            _make_data_row(date=datetime(2025, 8, 1), event="Noon", rpm=50),
            _make_data_row(),  # empty
            _make_data_row(date=datetime(2025, 8, 2), event="Noon", rpm=55),
        ])
        try:
            entries = EngineLogParser(path).parse()
            assert len(entries) == 2
        finally:
            path.unlink(missing_ok=True)

    def test_extended_data_captured(self):
        path = _make_elog_excel([
            _make_data_row(
                date=datetime(2025, 8, 1), event="Noon", rpm=50,
                me_revs_counter=12345, engine_speed=12.5, rh_me_cum=5000,
            ),
        ])
        try:
            entries = EngineLogParser(path).parse()
            ext = entries[0]["extended_data"]
            assert ext is not None
            assert ext["me_revs_counter"] == 12345.0
            assert ext["engine_speed"] == 12.5
            assert ext["rh_me_cum"] == 5000.0
        finally:
            path.unlink(missing_ok=True)


class TestStatistics:
    def test_empty_entries(self):
        path = _make_elog_excel([])
        try:
            parser = EngineLogParser(path)
            parser.parse()
            assert parser.get_statistics()["total_entries"] == 0
        finally:
            path.unlink(missing_ok=True)

    def test_statistics_with_data(self):
        path = _make_elog_excel([
            _make_data_row(date=datetime(2025, 8, 1), event="Noon", rpm=50, speed_stw=12.0,
                           hfo_total_mt=6.0, mgo_total_mt=0.5, methanol_me_mt=1.0),
            _make_data_row(date=datetime(2025, 8, 2), event="Noon", rpm=60, speed_stw=14.0,
                           hfo_total_mt=7.0, mgo_total_mt=0.6, methanol_me_mt=1.2),
            _make_data_row(date=datetime(2025, 8, 2), event="SOSP", rpm=0,
                           hfo_total_mt=0.0, mgo_total_mt=0.1),
        ])
        try:
            parser = EngineLogParser(path)
            parser.parse()
            stats = parser.get_statistics()
            assert stats["total_entries"] == 3
            assert stats["avg_rpm_at_sea"] == 55.0
            assert stats["avg_speed_stw"] == 13.0
            assert stats["fuel_totals"]["hfo_mt"] == 13.0
            assert stats["events_breakdown"]["NOON"] == 2
            assert stats["events_breakdown"]["SOSP"] == 1
        finally:
            path.unlink(missing_ok=True)


@pytest.mark.skipif(not GARONNE_FILE.exists(), reason="GARONNE file not available")
class TestGaronneDataset:
    @pytest.fixture(scope="class")
    def parsed(self):
        parser = EngineLogParser(GARONNE_FILE)
        entries = parser.parse()
        return parser, entries

    def test_parse_real_dataset(self, parsed):
        _, entries = parsed
        assert len(entries) >= 150
        assert len(entries) <= 200

    def test_timestamps_in_range(self, parsed):
        _, entries = parsed
        for e in entries:
            assert e["timestamp"].year == 2025
            assert 7 <= e["timestamp"].month <= 9

    def test_event_types(self, parsed):
        _, entries = parsed
        events = {e["event"] for e in entries if e.get("event")}
        assert "NOON" in events
        assert "ALL_FAST" in events

    def test_rpm_values_reasonable(self, parsed):
        _, entries = parsed
        rpms = [e["rpm"] for e in entries if e.get("rpm") is not None]
        assert len(rpms) > 0
        for r in rpms:
            assert 0 <= r <= 500

    def test_fuel_totals_present(self, parsed):
        _, entries = parsed
        hfo = [e["hfo_total_mt"] for e in entries if e.get("hfo_total_mt") is not None]
        assert len(hfo) > 100

    def test_rob_tracked(self, parsed):
        _, entries = parsed
        robs = [e["rob_vlsfo_mt"] for e in entries if e.get("rob_vlsfo_mt") is not None]
        assert len(robs) > 100
        for r in robs:
            assert r >= 0

    def test_no_nan_in_output(self, parsed):
        _, entries = parsed
        for e in entries:
            for key, val in e.items():
                if isinstance(val, float):
                    assert not math.isnan(val), f"NaN in {key}"
                    assert not math.isinf(val), f"Inf in {key}"

    def test_statistics(self, parsed):
        parser, _ = parsed
        stats = parser.get_statistics()
        assert stats["total_entries"] >= 150
        assert stats["events_breakdown"]["NOON"] >= 40
        assert stats["fuel_totals"]["hfo_mt"] > 0

    def test_extended_data_populated(self, parsed):
        _, entries = parsed
        with_ext = [e for e in entries if e.get("extended_data")]
        assert len(with_ext) > 100

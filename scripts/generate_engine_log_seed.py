"""
Generate SQL seed file from anonymized engine log Excel.

Reads demo-engine-log.xlsx using the EngineLogParser, then produces
INSERT statements for the engine_log_entries table.

Usage:
    python scripts/generate_engine_log_seed.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path so we can import the parser
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database.engine_log_parser import EngineLogParser

EXCEL_PATH = PROJECT_ROOT / "data" / "MR Example.xlsx"
OUTPUT_PATH = PROJECT_ROOT / "data" / "demo-engine-log-seed.sql"

# Fixed UUIDs for demo data
DEMO_BATCH_ID = "00000000-0000-0000-0000-de0000ba1c01"
DEMO_VESSEL_ID = None  # Will be NULL — linked at runtime


def sql_value(val, col_type="float"):
    """Convert a Python value to a SQL literal."""
    if val is None:
        return "NULL"
    if col_type == "uuid":
        return f"'{val}'"
    if col_type == "timestamp":
        if isinstance(val, datetime):
            return f"'{val.strftime('%Y-%m-%d %H:%M:%S')}'"
        return f"'{val}'"
    if col_type == "text" or col_type == "string":
        # Escape single quotes
        s = str(val).replace("'", "''")
        return f"'{s}'"
    if col_type == "json":
        if val is None:
            return "NULL"
        s = json.dumps(val).replace("'", "''")
        return f"'{s}'"
    if col_type == "float":
        if val is None:
            return "NULL"
        return str(val)
    return str(val)


# Column definitions: (db_column_name, entry_key, col_type)
COLUMNS = [
    ("id", None, "uuid"),  # generated per row
    ("vessel_id", None, "uuid"),
    ("timestamp", "timestamp", "timestamp"),
    ("lapse_hours", "lapse_hours", "float"),
    ("place", "place", "string"),
    ("event", "event", "string"),
    ("rpm", "rpm", "float"),
    ("engine_distance", "engine_distance", "float"),
    ("speed_stw", "speed_stw", "float"),
    ("me_power_kw", "me_power_kw", "float"),
    ("me_load_pct", "me_load_pct", "float"),
    ("me_fuel_index_pct", "me_fuel_index_pct", "float"),
    ("shaft_power", "shaft_power", "float"),
    ("shaft_torque_knm", "shaft_torque_knm", "float"),
    ("slip_pct", "slip_pct", "float"),
    ("hfo_me_mt", "hfo_me_mt", "float"),
    ("hfo_ae_mt", "hfo_ae_mt", "float"),
    ("hfo_boiler_mt", "hfo_boiler_mt", "float"),
    ("hfo_total_mt", "hfo_total_mt", "float"),
    ("mgo_me_mt", "mgo_me_mt", "float"),
    ("mgo_ae_mt", "mgo_ae_mt", "float"),
    ("mgo_total_mt", "mgo_total_mt", "float"),
    ("methanol_me_mt", "methanol_me_mt", "float"),
    ("rob_vlsfo_mt", "rob_vlsfo_mt", "float"),
    ("rob_mgo_mt", "rob_mgo_mt", "float"),
    ("rob_methanol_mt", "rob_methanol_mt", "float"),
    ("rh_me", "rh_me", "float"),
    ("rh_ae_total", "rh_ae_total", "float"),
    ("tc_rpm", "tc_rpm", "float"),
    ("scav_air_press_bar", "scav_air_press_bar", "float"),
    ("fuel_temp_c", "fuel_temp_c", "float"),
    ("sw_temp_c", "sw_temp_c", "float"),
    ("upload_batch_id", None, "uuid"),
    ("source_sheet", "source_sheet", "string"),
    ("source_file", "source_file", "string"),
    ("created_at", None, "timestamp"),
    ("extended_data", "extended_data", "json"),
]


def generate_seed():
    """Parse the Excel file and generate SQL INSERT statements."""
    print(f"Parsing: {EXCEL_PATH}")
    parser = EngineLogParser(EXCEL_PATH)
    entries = parser.parse()
    stats = parser.get_statistics()

    print(f"Parsed {len(entries)} entries")
    print(f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
    print(f"Events: {stats['events_breakdown']}")

    if not entries:
        print("ERROR: No entries parsed. Aborting.")
        sys.exit(1)

    col_names = [c[0] for c in COLUMNS]
    col_list = ", ".join(col_names)

    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("-- =============================================================")
    lines.append("-- Windmar Demo Engine Log Seed Data")
    lines.append(f"-- Generated: {now_str}")
    lines.append(f"-- Source: {EXCEL_PATH.name}")
    lines.append(f"-- Entries: {len(entries)}")
    lines.append(f"-- Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
    lines.append(f"-- Batch ID: {DEMO_BATCH_ID}")
    lines.append("-- =============================================================")
    lines.append("")
    # Self-contained: ensure table exists before inserting
    lines.append("-- Ensure table exists (self-contained — no dependency on API startup)")
    lines.append("CREATE TABLE IF NOT EXISTS engine_log_entries (")
    lines.append("    id UUID PRIMARY KEY,")
    lines.append("    vessel_id UUID,")
    lines.append("    timestamp TIMESTAMP NOT NULL,")
    lines.append("    lapse_hours FLOAT, place VARCHAR(255), event VARCHAR(100),")
    lines.append("    rpm FLOAT, engine_distance FLOAT, speed_stw FLOAT,")
    lines.append("    me_power_kw FLOAT, me_load_pct FLOAT, me_fuel_index_pct FLOAT,")
    lines.append("    shaft_power FLOAT, shaft_torque_knm FLOAT, slip_pct FLOAT,")
    lines.append("    hfo_me_mt FLOAT, hfo_ae_mt FLOAT, hfo_boiler_mt FLOAT, hfo_total_mt FLOAT,")
    lines.append("    mgo_me_mt FLOAT, mgo_ae_mt FLOAT, mgo_total_mt FLOAT,")
    lines.append("    methanol_me_mt FLOAT,")
    lines.append("    rob_vlsfo_mt FLOAT, rob_mgo_mt FLOAT, rob_methanol_mt FLOAT,")
    lines.append("    rh_me FLOAT, rh_ae_total FLOAT,")
    lines.append("    tc_rpm FLOAT, scav_air_press_bar FLOAT, fuel_temp_c FLOAT, sw_temp_c FLOAT,")
    lines.append("    upload_batch_id UUID NOT NULL,")
    lines.append("    source_sheet VARCHAR(100), source_file VARCHAR(500),")
    lines.append("    created_at TIMESTAMP NOT NULL DEFAULT NOW(),")
    lines.append("    extended_data JSONB")
    lines.append(");")
    lines.append("CREATE INDEX IF NOT EXISTS ix_engine_log_entries_timestamp ON engine_log_entries(timestamp);")
    lines.append("CREATE INDEX IF NOT EXISTS ix_engine_log_entries_event ON engine_log_entries(event);")
    lines.append("CREATE INDEX IF NOT EXISTS ix_engine_log_entries_upload_batch_id ON engine_log_entries(upload_batch_id);")
    lines.append("CREATE INDEX IF NOT EXISTS ix_engine_log_vessel_timestamp ON engine_log_entries(vessel_id, timestamp);")
    lines.append("")
    lines.append("BEGIN;")
    lines.append("")
    lines.append(f"-- Delete existing demo batch data (idempotent)")
    lines.append(f"DELETE FROM engine_log_entries WHERE upload_batch_id = '{DEMO_BATCH_ID}';")
    lines.append("")

    for i, entry in enumerate(entries):
        # Generate a deterministic UUID for each row
        row_uuid = f"00000000-0000-0000-0000-de00{i:08x}"

        values = []
        for col_name, entry_key, col_type in COLUMNS:
            if col_name == "id":
                values.append(sql_value(row_uuid, "uuid"))
            elif col_name == "vessel_id":
                values.append("NULL")
            elif col_name == "upload_batch_id":
                values.append(sql_value(DEMO_BATCH_ID, "uuid"))
            elif col_name == "created_at":
                values.append(sql_value(now_str, "timestamp"))
            else:
                val = entry.get(entry_key)
                values.append(sql_value(val, col_type))

        values_str = ", ".join(values)
        lines.append(f"INSERT INTO engine_log_entries ({col_list})")
        lines.append(f"VALUES ({values_str});")
        lines.append("")

    lines.append("COMMIT;")
    lines.append("")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Seed SQL written to: {OUTPUT_PATH}")
    print(f"Total INSERT statements: {len(entries)}")


if __name__ == "__main__":
    generate_seed()

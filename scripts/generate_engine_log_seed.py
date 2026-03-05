"""Generate synthetic engine log data for Windmar demo.

Replaces real vessel data with realistic MR Product Tanker voyages
in the ADRS 1+2 (NW Europe) and ADRS 4 (Mediterranean) areas.

Vessel specs: 49k DWT, 183m LOA, 8840 kW MCR, 14.5 kn laden / 15.0 kn ballast
"""

import json
import random
import uuid
from datetime import datetime, timedelta, timezone

random.seed(42)

# ── Voyage definitions ────────────────────────────────────────────────
# Each voyage: (load_port, discharge_port, condition, sea_days, dist_nm)
VOYAGES = [
    ("Rotterdam", "Algeciras", "laden", 5.5, 1450),
    ("Algeciras", "Augusta", "ballast", 3.0, 830),
    ("Augusta", "Piraeus", "laden", 2.0, 520),
    ("Piraeus", "Trieste", "ballast", 3.5, 920),
    ("Trieste", "Le Havre", "laden", 7.0, 2350),
    ("Le Havre", "Gothenburg", "ballast", 2.5, 680),
    ("Gothenburg", "Milford Haven", "laden", 3.0, 810),
    ("Milford Haven", "Rotterdam", "ballast", 1.5, 420),
    ("Rotterdam", "Iskenderun", "laden", 9.0, 3100),
    ("Iskenderun", "Genoa", "ballast", 4.0, 1350),
    ("Genoa", "Barcelona", "laden", 1.5, 380),
    ("Barcelona", "Rotterdam", "ballast", 5.0, 1620),
]

# ── MR Tanker operating parameters ───────────────────────────────────
MCR_KW = 8840
SFOC_MCR = 171  # g/kWh

LADEN = {
    "service_speed": 14.5,
    "rpm_range": (58, 66),
    "power_range": (4200, 5800),
    "load_pct_range": (55, 70),
    "hfo_daily_mt": (28, 34),
    "draft": 11.8,
}

BALLAST = {
    "service_speed": 15.0,
    "rpm_range": (60, 68),
    "power_range": (3800, 5200),
    "load_pct_range": (48, 62),
    "hfo_daily_mt": (24, 30),
    "draft": 6.5,
}


def jitter(val, pct=0.05):
    """Add realistic measurement noise."""
    return val * (1 + random.uniform(-pct, pct))


def gen_noon(ts, params, rob_vlsfo, rob_mgo, dist_remaining):
    """Generate a noon report entry."""
    rpm = random.uniform(*params["rpm_range"])
    power = random.uniform(*params["power_range"])
    speed = jitter(params["service_speed"], 0.08)
    load_pct = random.uniform(*params["load_pct_range"])
    hfo_daily = random.uniform(*params["hfo_daily_mt"])
    hfo_ae = round(random.uniform(0.8, 1.5), 2)
    hfo_boiler = round(random.uniform(0.3, 0.8), 2)
    hfo_me = round(hfo_daily - hfo_ae - hfo_boiler, 2)
    mgo_ae = round(random.uniform(0.1, 0.3), 2)
    slip = round(random.uniform(1.5, 6.0), 1)
    tc_rpm = round(rpm * random.uniform(1.35, 1.50), 1)
    scav = round(random.uniform(1.2, 2.0), 2)
    fuel_temp = round(random.uniform(125, 140), 1)
    sw_temp = round(random.uniform(8, 22), 1)
    shaft_power = round(power * random.uniform(0.92, 0.97), 1)
    shaft_torque = round(shaft_power / (2 * 3.14159 * rpm / 60), 1)

    rob_vlsfo -= hfo_daily
    rob_mgo -= mgo_ae

    return {
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "lapse_hours": 24.0,
        "place": "at Sea",
        "event": "NOON",
        "rpm": round(rpm, 2),
        "engine_distance": round(speed * 24, 1),
        "speed_stw": round(speed, 1),
        "me_power_kw": round(power, 1),
        "me_load_pct": round(load_pct, 1),
        "me_fuel_index_pct": round(load_pct * random.uniform(0.95, 1.05), 1),
        "shaft_power": shaft_power,
        "shaft_torque_knm": shaft_torque,
        "slip_pct": slip,
        "hfo_me_mt": hfo_me,
        "hfo_ae_mt": hfo_ae,
        "hfo_boiler_mt": hfo_boiler,
        "hfo_total_mt": round(hfo_daily, 2),
        "mgo_me_mt": 0,
        "mgo_ae_mt": mgo_ae,
        "mgo_total_mt": mgo_ae,
        "methanol_me_mt": 0,
        "rob_vlsfo_mt": round(rob_vlsfo, 1),
        "rob_mgo_mt": round(rob_mgo, 1),
        "rob_methanol_mt": 0,
        "rh_me": round(random.uniform(8000, 25000), 0),
        "rh_ae_total": round(random.uniform(5000, 20000), 0),
        "tc_rpm": tc_rpm,
        "scav_air_press_bar": scav,
        "fuel_temp_c": fuel_temp,
        "sw_temp_c": sw_temp,
    }, rob_vlsfo, rob_mgo


def gen_port_event(ts, place, event, lapse_hours=0):
    """Generate a port event entry."""
    return {
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "lapse_hours": lapse_hours,
        "place": place,
        "event": event,
        "rpm": 0 if event in ("DROP_ANCHOR", "ALL_FAST", "EOSP") else None,
        "engine_distance": 0,
        "speed_stw": None,
        "me_power_kw": None,
        "me_load_pct": None,
        "me_fuel_index_pct": None,
        "shaft_power": None,
        "shaft_torque_knm": None,
        "slip_pct": None,
        "hfo_me_mt": 0,
        "hfo_ae_mt": round(random.uniform(0.1, 0.4), 2) if lapse_hours > 6 else 0,
        "hfo_boiler_mt": 0,
        "hfo_total_mt": 0,
        "mgo_me_mt": 0,
        "mgo_ae_mt": round(random.uniform(0.05, 0.15), 2) if lapse_hours > 6 else 0,
        "mgo_total_mt": 0,
        "methanol_me_mt": 0,
        "rob_vlsfo_mt": None,
        "rob_mgo_mt": None,
        "rob_methanol_mt": 0,
        "rh_me": None,
        "rh_ae_total": None,
        "tc_rpm": None,
        "scav_air_press_bar": None,
        "fuel_temp_c": None,
        "sw_temp_c": None,
    }


DEMO_BATCH_ID = "00000000-0000-0000-0000-de0000ba1c01"


def generate():
    entries = []
    batch_id = DEMO_BATCH_ID
    ts = datetime(2025, 7, 1, 8, 0, 0)
    rob_vlsfo = 1200.0
    rob_mgo = 180.0

    for voy_idx, (load_port, disc_port, condition, sea_days, dist_nm) in enumerate(VOYAGES):
        params = LADEN if condition == "laden" else BALLAST

        # ── Port departure sequence ──
        port_hours = random.uniform(18, 72)

        entries.append(gen_port_event(ts, load_port, "ALL_CLEAR"))
        ts += timedelta(hours=random.uniform(0.5, 1.5))
        entries.append(gen_port_event(ts, load_port, "PILOT_OFF"))
        ts += timedelta(hours=random.uniform(0.3, 0.8))
        entries.append(gen_port_event(ts, load_port, "SOSP"))
        ts += timedelta(minutes=random.randint(10, 30))
        entries.append(gen_port_event(ts, load_port, "COSP"))

        # ── Sea passage with noon reports ──
        full_days = int(sea_days)
        # Advance to next noon
        hours_to_noon = 12 - ts.hour + (0 if ts.minute == 0 else 0)
        if hours_to_noon <= 0:
            hours_to_noon += 24
        ts += timedelta(hours=hours_to_noon)

        for day in range(full_days):
            entry, rob_vlsfo, rob_mgo = gen_noon(ts, params, rob_vlsfo, rob_mgo, 0)
            entries.append(entry)
            ts += timedelta(hours=24)

        # Bunker if low
        if rob_vlsfo < 300:
            rob_vlsfo += random.uniform(600, 900)
            rob_mgo += random.uniform(80, 120)

        # ── Arrival sequence ──
        ts += timedelta(hours=random.uniform(2, 8))
        entries.append(gen_port_event(ts, disc_port, "EOSP"))
        ts += timedelta(hours=random.uniform(0.5, 2))
        entries.append(gen_port_event(ts, disc_port, "ALL_FAST", lapse_hours=round(random.uniform(1, 3), 1)))

        # Port stay
        if condition == "laden":
            ts += timedelta(hours=random.uniform(6, 18))
            entries.append(gen_port_event(ts, disc_port, "COMPL_DISCHARGE", lapse_hours=round(random.uniform(12, 36), 1)))
        else:
            ts += timedelta(hours=random.uniform(12, 36))
            entries.append(gen_port_event(ts, disc_port, "COMPL._LOADING", lapse_hours=round(random.uniform(18, 48), 1)))

        ts += timedelta(hours=random.uniform(6, 24))

    return entries, batch_id


def to_sql(entries, batch_id):
    """Generate self-contained SQL seed file."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    first_ts = entries[0]["timestamp"] if entries else "N/A"
    last_ts = entries[-1]["timestamp"] if entries else "N/A"

    lines = [
        "-- =============================================================",
        "-- Windmar Demo Engine Log Seed Data",
        f"-- Generated: {now}",
        "-- Source: Synthetic MR Tanker (NW Europe / Mediterranean)",
        f"-- Entries: {len(entries)}",
        f"-- Date range: {first_ts} to {last_ts}",
        f"-- Batch ID: {batch_id}",
        "-- =============================================================",
        "",
        "-- Ensure table exists (self-contained — no dependency on API startup)",
        "CREATE TABLE IF NOT EXISTS engine_log_entries (",
        "    id UUID PRIMARY KEY,",
        "    vessel_id UUID,",
        "    timestamp TIMESTAMP NOT NULL,",
        "    lapse_hours FLOAT, place VARCHAR(255), event VARCHAR(100),",
        "    rpm FLOAT, engine_distance FLOAT, speed_stw FLOAT,",
        "    me_power_kw FLOAT, me_load_pct FLOAT, me_fuel_index_pct FLOAT,",
        "    shaft_power FLOAT, shaft_torque_knm FLOAT, slip_pct FLOAT,",
        "    hfo_me_mt FLOAT, hfo_ae_mt FLOAT, hfo_boiler_mt FLOAT, hfo_total_mt FLOAT,",
        "    mgo_me_mt FLOAT, mgo_ae_mt FLOAT, mgo_total_mt FLOAT,",
        "    methanol_me_mt FLOAT,",
        "    rob_vlsfo_mt FLOAT, rob_mgo_mt FLOAT, rob_methanol_mt FLOAT,",
        "    rh_me FLOAT, rh_ae_total FLOAT,",
        "    tc_rpm FLOAT, scav_air_press_bar FLOAT, fuel_temp_c FLOAT, sw_temp_c FLOAT,",
        "    upload_batch_id UUID NOT NULL,",
        "    source_sheet VARCHAR(100), source_file VARCHAR(500),",
        "    created_at TIMESTAMP NOT NULL DEFAULT NOW(),",
        "    extended_data JSONB",
        ");",
        "CREATE INDEX IF NOT EXISTS ix_engine_log_entries_timestamp ON engine_log_entries(timestamp);",
        "CREATE INDEX IF NOT EXISTS ix_engine_log_entries_event ON engine_log_entries(event);",
        "CREATE INDEX IF NOT EXISTS ix_engine_log_entries_upload_batch_id ON engine_log_entries(upload_batch_id);",
        "CREATE INDEX IF NOT EXISTS ix_engine_log_vessel_timestamp ON engine_log_entries(vessel_id, timestamp);",
        "",
        "BEGIN;",
        "",
        f"-- Delete existing demo batch data (idempotent)",
        f"DELETE FROM engine_log_entries WHERE upload_batch_id = '{batch_id}';",
        "",
    ]

    cols = [
        "id", "vessel_id", "timestamp", "lapse_hours", "place", "event",
        "rpm", "engine_distance", "speed_stw", "me_power_kw",
        "me_load_pct", "me_fuel_index_pct", "shaft_power", "shaft_torque_knm",
        "slip_pct", "hfo_me_mt", "hfo_ae_mt", "hfo_boiler_mt", "hfo_total_mt",
        "mgo_me_mt", "mgo_ae_mt", "mgo_total_mt", "methanol_me_mt",
        "rob_vlsfo_mt", "rob_mgo_mt", "rob_methanol_mt",
        "rh_me", "rh_ae_total", "tc_rpm", "scav_air_press_bar",
        "fuel_temp_c", "sw_temp_c",
        "upload_batch_id", "source_sheet", "source_file", "created_at",
        "extended_data",
    ]

    def sql_val(v):
        if v is None:
            return "NULL"
        if isinstance(v, str):
            return f"'{v}'"
        return str(v)

    for idx, e in enumerate(entries):
        entry_id = f"00000000-0000-0000-0000-de00000{idx:05d}"
        vals = [
            f"'{entry_id}'",
            "NULL",
            f"'{e['timestamp']}'",
            sql_val(e.get("lapse_hours")),
            sql_val(e.get("place")),
            sql_val(e.get("event")),
            sql_val(e.get("rpm")),
            sql_val(e.get("engine_distance")),
            sql_val(e.get("speed_stw")),
            sql_val(e.get("me_power_kw")),
            sql_val(e.get("me_load_pct")),
            sql_val(e.get("me_fuel_index_pct")),
            sql_val(e.get("shaft_power")),
            sql_val(e.get("shaft_torque_knm")),
            sql_val(e.get("slip_pct")),
            sql_val(e.get("hfo_me_mt")),
            sql_val(e.get("hfo_ae_mt")),
            sql_val(e.get("hfo_boiler_mt")),
            sql_val(e.get("hfo_total_mt")),
            sql_val(e.get("mgo_me_mt")),
            sql_val(e.get("mgo_ae_mt")),
            sql_val(e.get("mgo_total_mt")),
            sql_val(e.get("methanol_me_mt")),
            sql_val(e.get("rob_vlsfo_mt")),
            sql_val(e.get("rob_mgo_mt")),
            sql_val(e.get("rob_methanol_mt")),
            sql_val(e.get("rh_me")),
            sql_val(e.get("rh_ae_total")),
            sql_val(e.get("tc_rpm")),
            sql_val(e.get("scav_air_press_bar")),
            sql_val(e.get("fuel_temp_c")),
            sql_val(e.get("sw_temp_c")),
            f"'{batch_id}'",
            "'Engine Log'",
            "'Demo Vessel.xlsx'",
            f"'{now}'",
            "NULL",
        ]

        lines.append(
            f"INSERT INTO engine_log_entries ({', '.join(cols)}) VALUES ({', '.join(vals)});"
        )

    lines.append("")
    lines.append("COMMIT;")
    return "\n".join(lines)


if __name__ == "__main__":
    entries, batch_id = generate()
    sql = to_sql(entries, batch_id)

    seed_path = "/home/slmar/projects/Windmar/data/demo-engine-log-seed.sql"
    with open(seed_path, "w") as f:
        f.write(sql)

    print(f"Generated {len(entries)} entries across {len(VOYAGES)} voyages")
    print(f"SQL written to: {seed_path}")

    # Summary
    places = set()
    events = set()
    for e in entries:
        if e.get("place"):
            places.add(e["place"])
        if e.get("event"):
            events.add(e["event"])
    print(f"Ports: {sorted(places - {'at Sea'})}")
    print(f"Events: {sorted(events)}")

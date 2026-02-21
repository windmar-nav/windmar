-- Voyage persistence tables.
-- Applied at startup via _run_voyage_migrations() with advisory lock.

CREATE TABLE IF NOT EXISTS voyages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(200),
    departure_port VARCHAR(200),
    arrival_port VARCHAR(200),
    departure_time TIMESTAMPTZ NOT NULL,
    arrival_time TIMESTAMPTZ NOT NULL,
    total_distance_nm FLOAT NOT NULL,
    total_time_hours FLOAT NOT NULL,
    total_fuel_mt FLOAT NOT NULL,
    avg_sog_kts FLOAT,
    avg_stw_kts FLOAT,
    calm_speed_kts FLOAT NOT NULL,
    is_laden BOOLEAN NOT NULL DEFAULT TRUE,
    vessel_specs_snapshot JSONB,
    cii_estimate JSONB,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS voyage_legs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    voyage_id UUID NOT NULL REFERENCES voyages(id) ON DELETE CASCADE,
    leg_index INT NOT NULL,
    from_name VARCHAR(200),
    from_lat FLOAT NOT NULL,
    from_lon FLOAT NOT NULL,
    to_name VARCHAR(200),
    to_lat FLOAT NOT NULL,
    to_lon FLOAT NOT NULL,
    distance_nm FLOAT NOT NULL,
    bearing_deg FLOAT,
    wind_speed_kts FLOAT,
    wind_dir_deg FLOAT,
    wave_height_m FLOAT,
    wave_dir_deg FLOAT,
    current_speed_ms FLOAT,
    current_dir_deg FLOAT,
    calm_speed_kts FLOAT,
    stw_kts FLOAT,
    sog_kts FLOAT,
    speed_loss_pct FLOAT,
    time_hours FLOAT NOT NULL,
    departure_time TIMESTAMPTZ,
    arrival_time TIMESTAMPTZ,
    fuel_mt FLOAT NOT NULL,
    power_kw FLOAT,
    data_source VARCHAR(50),
    UNIQUE(voyage_id, leg_index)
);

CREATE INDEX IF NOT EXISTS ix_voyages_departure ON voyages(departure_time DESC);
CREATE INDEX IF NOT EXISTS ix_voyages_name ON voyages(name);
CREATE INDEX IF NOT EXISTS ix_voyage_legs_voyage ON voyage_legs(voyage_id);

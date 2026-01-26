-- WINDMAR Database Initialization Script

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create tables
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key_hash VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    rate_limit INTEGER DEFAULT 1000,
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS vessel_specs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    length FLOAT NOT NULL,
    beam FLOAT NOT NULL,
    draft FLOAT NOT NULL,
    displacement FLOAT NOT NULL,
    deadweight FLOAT NOT NULL,
    block_coefficient FLOAT,
    midship_coefficient FLOAT,
    waterplane_coefficient FLOAT,
    lcb_fraction FLOAT,
    propeller_diameter FLOAT,
    max_speed FLOAT,
    service_speed FLOAT,
    engine_power FLOAT,
    fuel_type VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by UUID,
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS routes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vessel_id UUID REFERENCES vessel_specs(id),
    origin_lat FLOAT NOT NULL,
    origin_lon FLOAT NOT NULL,
    destination_lat FLOAT NOT NULL,
    destination_lon FLOAT NOT NULL,
    departure_time TIMESTAMP WITH TIME ZONE NOT NULL,
    route_data JSONB NOT NULL,
    total_distance FLOAT,
    total_time FLOAT,
    fuel_consumption FLOAT,
    calculation_time FLOAT,
    weather_data_source VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by UUID,
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS calibration_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vessel_id UUID REFERENCES vessel_specs(id),
    speed FLOAT NOT NULL,
    fuel_consumption FLOAT NOT NULL,
    wind_speed FLOAT,
    wind_direction FLOAT,
    wave_height FLOAT,
    current_speed FLOAT,
    current_direction FLOAT,
    recorded_at TIMESTAMP WITH TIME ZONE NOT NULL,
    data_source VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS noon_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vessel_id UUID REFERENCES vessel_specs(id),
    route_id UUID REFERENCES routes(id),
    position_lat FLOAT NOT NULL,
    position_lon FLOAT NOT NULL,
    speed_over_ground FLOAT,
    speed_through_water FLOAT,
    course FLOAT,
    fuel_consumed FLOAT,
    distance_made_good FLOAT,
    weather_conditions JSONB,
    report_time TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Create indexes
CREATE INDEX idx_api_keys_key_hash ON api_keys(key_hash);
CREATE INDEX idx_api_keys_active ON api_keys(is_active) WHERE is_active = true;
CREATE INDEX idx_vessel_specs_name ON vessel_specs(name);
CREATE INDEX idx_routes_vessel_id ON routes(vessel_id);
CREATE INDEX idx_routes_created_at ON routes(created_at);
CREATE INDEX idx_calibration_vessel_id ON calibration_data(vessel_id);
CREATE INDEX idx_calibration_recorded_at ON calibration_data(recorded_at);
CREATE INDEX idx_noon_reports_vessel_id ON noon_reports(vessel_id);
CREATE INDEX idx_noon_reports_route_id ON noon_reports(route_id);
CREATE INDEX idx_noon_reports_time ON noon_reports(report_time);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to vessel_specs
CREATE TRIGGER update_vessel_specs_updated_at BEFORE UPDATE ON vessel_specs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- PRODUCTION SECURITY NOTICE
-- ============================================================================
-- API keys must be created manually after deployment using the CLI tool:
--
--   docker-compose exec api python -m api.cli create-api-key --name "Production Key"
--
-- NEVER commit API keys to version control or seed them in init scripts.
-- ============================================================================

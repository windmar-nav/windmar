/**
 * API client for WINDMAR backend v2.
 */

import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// ============================================================================
// Types
// ============================================================================

export interface Position {
  lat: number;
  lon: number;
}

export interface WaypointData {
  id: number;
  name: string;
  lat: number;
  lon: number;
}

export interface LegData {
  from: string;
  to: string;
  distance_nm: number;
  bearing_deg: number;
}

export interface RouteData {
  name: string;
  waypoints: WaypointData[];
  total_distance_nm: number;
  legs: LegData[];
}

// Weather types
export interface WindFieldData {
  parameter: string;
  time: string;
  bbox: {
    lat_min: number;
    lat_max: number;
    lon_min: number;
    lon_max: number;
  };
  resolution: number;
  nx: number;
  ny: number;
  lats: number[];
  lons: number[];
  u: number[][];
  v: number[][];
}

export interface WaveFieldData {
  parameter: string;
  time: string;
  bbox: {
    lat_min: number;
    lat_max: number;
    lon_min: number;
    lon_max: number;
  };
  resolution: number;
  nx: number;
  ny: number;
  lats: number[];
  lons: number[];
  data: number[][];
  unit: string;
  colorscale: {
    min: number;
    max: number;
    colors: string[];
  };
}

export interface VelocityData {
  header: {
    parameterCategory: number;
    parameterNumber: number;
    lo1: number;
    la1: number;
    lo2: number;
    la2: number;
    dx: number;
    dy: number;
    nx: number;
    ny: number;
    refTime: string;
  };
  data: number[];
}

export interface PointWeather {
  position: { lat: number; lon: number };
  time: string;
  wind: {
    speed_ms: number;
    speed_kts: number;
    dir_deg: number;
  };
  waves: {
    height_m: number;
    dir_deg: number;
  };
}

// Voyage types
export interface VoyageRequest {
  waypoints: Position[];
  calm_speed_kts: number;
  is_laden: boolean;
  departure_time?: string;
  use_weather: boolean;
}

export interface LegResult {
  leg_index: number;
  from_wp: WaypointData;
  to_wp: WaypointData;
  distance_nm: number;
  bearing_deg: number;
  wind_speed_kts: number;
  wind_dir_deg: number;
  wave_height_m: number;
  wave_dir_deg: number;
  calm_speed_kts: number;
  stw_kts: number;
  sog_kts: number;
  speed_loss_pct: number;
  time_hours: number;
  departure_time: string;
  arrival_time: string;
  fuel_mt: number;
  power_kw: number;
  // Data source info
  data_source?: 'forecast' | 'blended' | 'climatology';
  forecast_weight?: number;
}

export interface DataSourceSummary {
  forecast_legs: number;
  blended_legs: number;
  climatology_legs: number;
  forecast_horizon_days: number;
  warning?: string;
}

export interface VoyageResponse {
  route_name: string;
  departure_time: string;
  arrival_time: string;
  total_distance_nm: number;
  total_time_hours: number;
  total_fuel_mt: number;
  avg_sog_kts: number;
  avg_stw_kts: number;
  legs: LegResult[];
  calm_speed_kts: number;
  is_laden: boolean;
  // Data source summary
  data_sources?: DataSourceSummary;
}

// Optimization types
export interface OptimizationRequest {
  origin: Position;
  destination: Position;
  calm_speed_kts: number;
  is_laden: boolean;
  departure_time?: string;
  optimization_target: 'fuel' | 'time';
  grid_resolution_deg: number;
}

export interface OptimizationLeg {
  from_lat: number;
  from_lon: number;
  to_lat: number;
  to_lon: number;
  distance_nm: number;
  bearing_deg: number;
  fuel_mt: number;
  time_hours: number;
  sog_kts: number;
  stw_kts: number;  // Optimized speed through water
  wind_speed_ms: number;
  wave_height_m: number;
  // Safety metrics per leg
  safety_status?: 'safe' | 'marginal' | 'dangerous';
  roll_deg?: number;
  pitch_deg?: number;
}

export interface SafetySummary {
  status: 'safe' | 'marginal' | 'dangerous';
  warnings: string[];
  max_roll_deg: number;
  max_pitch_deg: number;
  max_accel_ms2: number;
}

export interface OptimizationResponse {
  waypoints: Position[];
  total_fuel_mt: number;
  total_time_hours: number;
  total_distance_nm: number;
  direct_fuel_mt: number;
  direct_time_hours: number;
  fuel_savings_pct: number;
  time_savings_pct: number;
  legs: OptimizationLeg[];
  // Speed profile (variable speed optimization)
  speed_profile: number[];  // Optimal speed per leg (kts)
  avg_speed_kts: number;
  variable_speed_enabled: boolean;
  // Safety assessment
  safety?: SafetySummary;
  optimization_target: string;
  grid_resolution_deg: number;
  cells_explored: number;
  optimization_time_ms: number;
}

// Vessel types
export interface VesselSpecs {
  dwt: number;
  loa: number;
  beam: number;
  draft_laden: number;
  draft_ballast: number;
  mcr_kw: number;
  sfoc_at_mcr: number;
  service_speed_laden: number;
  service_speed_ballast: number;
}

// Zone types
export type ZoneType = 'eca' | 'seca' | 'hra' | 'tss' | 'vts' | 'exclusion' | 'environmental' | 'ice' | 'canal' | 'custom';
export type ZoneInteraction = 'mandatory' | 'exclusion' | 'penalty' | 'advisory';

export interface ZoneCoordinate {
  lat: number;
  lon: number;
}

export interface ZoneProperties {
  name: string;
  zone_type: ZoneType;
  interaction: ZoneInteraction;
  penalty_factor: number;
  is_builtin: boolean;
  notes?: string;
}

export interface ZoneFeature {
  type: 'Feature';
  id: string;
  properties: ZoneProperties;
  geometry: {
    type: 'Polygon';
    coordinates: number[][][]; // [lon, lat] arrays
  };
}

export interface ZoneGeoJSON {
  type: 'FeatureCollection';
  features: ZoneFeature[];
}

export interface CreateZoneRequest {
  name: string;
  zone_type: ZoneType;
  interaction: ZoneInteraction;
  coordinates: ZoneCoordinate[];
  penalty_factor?: number;
  notes?: string;
}

export interface ZoneListItem {
  id: string;
  name: string;
  zone_type: ZoneType;
  interaction: ZoneInteraction;
  penalty_factor: number;
  is_builtin: boolean;
}

// Calibration types
export interface CalibrationFactors {
  calm_water: number;
  wind: number;
  waves: number;
  sfoc_factor: number;
  calibrated_at?: string;
  num_reports_used: number;
  calibration_error: number;
  days_since_drydock: number;
}

export interface CalibrationStatus {
  calibrated: boolean;
  factors: {
    calm_water: number;
    wind: number;
    waves: number;
    sfoc_factor: number;
  };
  calibrated_at?: string;
  num_reports_used?: number;
  calibration_error_mt?: number;
  days_since_drydock?: number;
  message?: string;
}

export interface NoonReportData {
  timestamp: string;
  latitude: number;
  longitude: number;
  speed_over_ground_kts: number;
  speed_through_water_kts?: number;
  fuel_consumption_mt: number;
  period_hours: number;
  is_laden: boolean;
  heading_deg: number;
  wind_speed_kts?: number;
  wind_direction_deg?: number;
  wave_height_m?: number;
  wave_direction_deg?: number;
  engine_power_kw?: number;
}

export interface CalibrationResult {
  factors: CalibrationFactors;
  reports_used: number;
  reports_skipped: number;
  mean_error_before_mt: number;
  mean_error_after_mt: number;
  improvement_pct: number;
  residuals: Array<{
    timestamp: string;
    actual_mt: number;
    predicted_mt: number;
    error_mt: number;
    error_pct: number;
    speed_kts: number;
    is_laden: boolean;
  }>;
}

// ============================================================================
// API Functions
// ============================================================================

export const apiClient = {
  // Health check
  async healthCheck() {
    const response = await api.get('/api/health');
    return response.data;
  },

  // -------------------------------------------------------------------------
  // Weather API (Layer 1)
  // -------------------------------------------------------------------------

  async getWindField(params: {
    lat_min?: number;
    lat_max?: number;
    lon_min?: number;
    lon_max?: number;
    resolution?: number;
    time?: string;
  } = {}): Promise<WindFieldData> {
    const response = await api.get<WindFieldData>('/api/weather/wind', { params });
    return response.data;
  },

  async getWindVelocity(params: {
    lat_min?: number;
    lat_max?: number;
    lon_min?: number;
    lon_max?: number;
    resolution?: number;
    time?: string;
  } = {}): Promise<VelocityData[]> {
    const response = await api.get<VelocityData[]>('/api/weather/wind/velocity', { params });
    return response.data;
  },

  async getWaveField(params: {
    lat_min?: number;
    lat_max?: number;
    lon_min?: number;
    lon_max?: number;
    resolution?: number;
    time?: string;
  } = {}): Promise<WaveFieldData> {
    const response = await api.get<WaveFieldData>('/api/weather/waves', { params });
    return response.data;
  },

  async getWeatherAtPoint(lat: number, lon: number, time?: string): Promise<PointWeather> {
    const params: { lat: number; lon: number; time?: string } = { lat, lon };
    if (time) params.time = time;
    const response = await api.get<PointWeather>('/api/weather/point', { params });
    return response.data;
  },

  // -------------------------------------------------------------------------
  // Routes API (Layer 2)
  // -------------------------------------------------------------------------

  async parseRTZ(file: File): Promise<RouteData> {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post<RouteData>('/api/routes/parse-rtz', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  async createRouteFromWaypoints(
    waypoints: Position[],
    name: string = 'Custom Route'
  ): Promise<RouteData> {
    const response = await api.post<RouteData>(
      `/api/routes/from-waypoints?name=${encodeURIComponent(name)}`,
      waypoints
    );
    return response.data;
  },

  // -------------------------------------------------------------------------
  // Voyage API (Layer 3)
  // -------------------------------------------------------------------------

  async calculateVoyage(request: VoyageRequest): Promise<VoyageResponse> {
    const response = await api.post<VoyageResponse>('/api/voyage/calculate', request);
    return response.data;
  },

  async getWeatherAlongRoute(
    waypoints: Position[],
    time?: string
  ): Promise<{ time: string; waypoints: Array<{
    waypoint_index: number;
    position: Position;
    wind_speed_kts: number;
    wind_dir_deg: number;
    wave_height_m: number;
    wave_dir_deg: number;
  }> }> {
    const wpString = waypoints.map(wp => `${wp.lat},${wp.lon}`).join(';');
    const params: { waypoints: string; time?: string } = { waypoints: wpString };
    if (time) params.time = time;
    const response = await api.get('/api/voyage/weather-along-route', { params });
    return response.data;
  },

  // -------------------------------------------------------------------------
  // Optimization API (Layer 4)
  // -------------------------------------------------------------------------

  async optimizeRoute(request: OptimizationRequest): Promise<OptimizationResponse> {
    const response = await api.post<OptimizationResponse>('/api/optimize/route', request);
    return response.data;
  },

  async getOptimizationStatus(): Promise<{
    status: string;
    default_resolution_deg: number;
    default_max_cells: number;
    optimization_targets: string[];
  }> {
    const response = await api.get('/api/optimize/status');
    return response.data;
  },

  // -------------------------------------------------------------------------
  // Vessel API
  // -------------------------------------------------------------------------

  async getVesselSpecs(): Promise<VesselSpecs> {
    const response = await api.get<VesselSpecs>('/api/vessel/specs');
    return response.data;
  },

  async updateVesselSpecs(specs: VesselSpecs): Promise<{ status: string; message: string }> {
    const response = await api.post('/api/vessel/specs', specs);
    return response.data;
  },

  // -------------------------------------------------------------------------
  // Calibration API
  // -------------------------------------------------------------------------

  async getCalibration(): Promise<CalibrationStatus> {
    const response = await api.get<CalibrationStatus>('/api/vessel/calibration');
    return response.data;
  },

  async setCalibration(factors: Partial<CalibrationFactors>): Promise<{ status: string; message: string }> {
    const response = await api.post('/api/vessel/calibration/set', factors);
    return response.data;
  },

  async getNoonReports(): Promise<{
    count: number;
    reports: Array<{
      timestamp: string;
      latitude: number;
      longitude: number;
      speed_kts: number;
      fuel_mt: number;
      period_hours: number;
      is_laden: boolean;
    }>;
  }> {
    const response = await api.get('/api/vessel/noon-reports');
    return response.data;
  },

  async addNoonReport(report: NoonReportData): Promise<{ status: string; total_reports: number }> {
    const response = await api.post('/api/vessel/noon-reports', report);
    return response.data;
  },

  async uploadNoonReportsCSV(file: File): Promise<{ status: string; imported: number; total_reports: number }> {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post('/api/vessel/noon-reports/upload-csv', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  async clearNoonReports(): Promise<{ status: string; message: string }> {
    const response = await api.delete('/api/vessel/noon-reports');
    return response.data;
  },

  async calibrateVessel(daysSinceDrydock: number = 0): Promise<CalibrationResult> {
    const response = await api.post<CalibrationResult>(`/api/vessel/calibrate?days_since_drydock=${daysSinceDrydock}`);
    return response.data;
  },

  async estimateFouling(daysSinceDrydock: number, operatingRegions: string[] = []): Promise<{
    days_since_drydock: number;
    operating_regions: string[];
    estimated_fouling_factor: number;
    resistance_increase_pct: number;
    note: string;
  }> {
    const params = new URLSearchParams();
    params.append('days_since_drydock', String(daysSinceDrydock));
    operatingRegions.forEach(r => params.append('operating_regions', r));
    const response = await api.post(`/api/vessel/calibration/estimate-fouling?${params.toString()}`);
    return response.data;
  },

  // -------------------------------------------------------------------------
  // Zones API (Regulatory Zones)
  // -------------------------------------------------------------------------

  async getZones(): Promise<ZoneGeoJSON> {
    const response = await api.get<ZoneGeoJSON>('/api/zones');
    return response.data;
  },

  async getZonesList(): Promise<{ zones: ZoneListItem[]; count: number }> {
    const response = await api.get('/api/zones/list');
    return response.data;
  },

  async createZone(request: CreateZoneRequest): Promise<ZoneListItem> {
    const response = await api.post<ZoneListItem>('/api/zones', request);
    return response.data;
  },

  async deleteZone(zoneId: string): Promise<{ status: string; zone_id: string }> {
    const response = await api.delete(`/api/zones/${zoneId}`);
    return response.data;
  },

  async getZonesAtPoint(lat: number, lon: number): Promise<{
    position: Position;
    zones: Array<{
      id: string;
      name: string;
      zone_type: ZoneType;
      interaction: ZoneInteraction;
      penalty_factor: number;
    }>;
  }> {
    const response = await api.get('/api/zones/at-point', { params: { lat, lon } });
    return response.data;
  },
};

export default api;

/**
 * API client for WINDMAR backend v2.
 */

import axios from 'axios';
import { isDemoUser } from '@/lib/demoMode';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Attach X-API-Key to every request so the backend can resolve user tier.
api.interceptors.request.use((config) => {
  if (typeof window !== 'undefined') {
    const key = localStorage.getItem('windmar_api_key');
    if (key) {
      config.headers['X-API-Key'] = key;
    }
  }
  return config;
});

// ============================================================================
// Types
// ============================================================================

export interface Position {
  lat: number;
  lon: number;
  name?: string;
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
  ocean_mask?: boolean[][];
  ocean_mask_lats?: number[];
  ocean_mask_lons?: number[];
  ingested_at?: string;
}

export interface WaveDecomposition {
  height: number[][];
  period?: number[][];
  direction?: number[][];
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
  /** Mean wave direction grid (degrees, meteorological convention) */
  direction?: number[][];
  /** Wave decomposition */
  has_decomposition?: boolean;
  windwave?: WaveDecomposition;
  swell?: WaveDecomposition;
  ocean_mask?: boolean[][];
  ocean_mask_lats?: number[];
  ocean_mask_lons?: number[];
  colorscale: {
    min: number;
    max: number;
    data_min?: number;
    data_max?: number;
    colors: string[];
  };
  ingested_at?: string;
}

// Extended weather field types (SPEC-P1)
export interface GridFieldData {
  parameter: string;
  time: string;
  bbox: { lat_min: number; lat_max: number; lon_min: number; lon_max: number };
  resolution: number;
  nx: number;
  ny: number;
  lats: number[];
  lons: number[];
  data: number[][];
  unit: string;
  ocean_mask?: boolean[][];
  ocean_mask_lats?: number[];
  ocean_mask_lons?: number[];
  source?: string;
  colorscale?: { min: number; max: number; data_min?: number; data_max?: number; colors: string[] };
  ingested_at?: string;
}

export interface SwellFieldData extends GridFieldData {
  has_decomposition: boolean;
  total_hs: number[][];
  swell_hs: number[][] | null;
  swell_tp: number[][] | null;
  swell_dir: number[][] | null;
  windsea_hs: number[][] | null;
  windsea_tp: number[][] | null;
  windsea_dir: number[][] | null;
}

// Wave forecast types
export interface WaveForecastFrame {
  data: number[][];
  direction?: number[][];
  windwave?: WaveDecomposition;
  swell?: WaveDecomposition;
}

export interface WaveForecastFrames {
  run_time: string;
  total_hours: number;
  cached_hours: number;
  source?: string;
  lats: number[];
  lons: number[];
  ny: number;
  nx: number;
  ocean_mask?: boolean[][];
  ocean_mask_lats?: number[];
  ocean_mask_lons?: number[];
  colorscale: { min: number; max: number; data_min?: number; data_max?: number; colors: string[] };
  frames: Record<string, WaveForecastFrame>;
}

export interface CurrentForecastFrame {
  u?: number[][];
  v?: number[][];
}

export interface CurrentForecastFrames {
  run_time: string;
  total_hours: number;
  cached_hours: number;
  source?: string;
  lats: number[];
  lons: number[];
  ny: number;
  nx: number;
  frames: Record<string, CurrentForecastFrame>;
}

export interface IceForecastFrame {
  data: number[][];
}

export interface IceForecastFrames {
  run_time: string;
  total_hours: number;
  cached_hours: number;
  source?: string;
  lats: number[];
  lons: number[];
  ny: number;
  nx: number;
  ocean_mask?: boolean[][];
  ocean_mask_lats?: number[];
  ocean_mask_lons?: number[];
  frames: Record<string, IceForecastFrame>;
}

// SST forecast types
export interface SstForecastFrame {
  data: number[][];
}

export interface SstForecastFrames {
  run_time: string;
  total_hours: number;
  cached_hours: number;
  source?: string;
  lats: number[];
  lons: number[];
  ny: number;
  nx: number;
  ocean_mask?: boolean[][];
  ocean_mask_lats?: number[];
  ocean_mask_lons?: number[];
  colorscale?: { min: number; max: number; data_min?: number; data_max?: number; colors: string[] };
  frames: Record<string, SstForecastFrame>;
}

// Visibility forecast types
export interface VisForecastFrame {
  data: number[][];
}

export interface VisForecastFrames {
  run_time: string;
  total_hours: number;
  cached_hours: number;
  source?: string;
  lats: number[];
  lons: number[];
  ny: number;
  nx: number;
  ocean_mask?: boolean[][];
  ocean_mask_lats?: number[];
  ocean_mask_lons?: number[];
  colorscale?: { min: number; max: number; data_min?: number; data_max?: number; colors: string[] };
  frames: Record<string, VisForecastFrame>;
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

// Forecast types
export interface ForecastHourStatus {
  forecast_hour: number;
  valid_time: string;
  cached: boolean;
}

export interface ForecastStatus {
  run_date: string;
  run_hour: string;
  total_hours: number;
  cached_hours: number;
  complete: boolean;
  prefetch_running: boolean;
  hours: ForecastHourStatus[];
}

export interface ForecastFrames {
  run_date: string;
  run_hour: string;
  run_time: string;
  total_hours: number;
  cached_hours: number;
  source?: string;
  frames: Record<string, VelocityData[]>;
}

// Weather health/sync types
export interface WeatherSourceHealth {
  label: string;
  present: boolean;
  complete: boolean;
  fresh: boolean;
  healthy: boolean;
  frame_count: number;
  expected_frames: number;
  age_hours: number | null;
  bounds: { lat_min: number; lat_max: number; lon_min: number; lon_max: number } | null;
}

export interface WeatherHealthResponse {
  healthy: boolean;
  db_bounds: { lat_min: number; lat_max: number; lon_min: number; lon_max: number } | null;
  sources: Record<string, WeatherSourceHealth>;
}


// Voyage types
export interface VoyageRequest {
  waypoints: Position[];
  calm_speed_kts: number;
  is_laden: boolean;
  departure_time?: string;
  use_weather: boolean;
  variable_speed?: boolean;
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
  // Current data (ocean currents)
  current_speed_ms?: number;
  current_dir_deg?: number;
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
  // Variable speed optimization
  variable_speed_enabled?: boolean;
  speed_profile?: number[];
  // Data source summary
  data_sources?: DataSourceSummary;
}

// Weather along route types
export interface RouteWeatherPoint {
  distance_nm: number;
  lat: number;
  lon: number;
  wind_speed_kts: number;
  wind_dir_deg: number;
  wave_height_m: number;
  wave_dir_deg: number;
  current_speed_ms: number;
  current_dir_deg: number;
  is_waypoint: boolean;
  waypoint_index: number | null;
}

export interface WeatherAlongRouteResponse {
  time: string;
  total_distance_nm: number;
  points: RouteWeatherPoint[];
}

// Voyage History types
export interface SaveVoyageLeg {
  leg_index: number;
  from_name?: string;
  from_lat: number;
  from_lon: number;
  to_name?: string;
  to_lat: number;
  to_lon: number;
  distance_nm: number;
  bearing_deg?: number;
  wind_speed_kts?: number;
  wind_dir_deg?: number;
  wave_height_m?: number;
  wave_dir_deg?: number;
  current_speed_ms?: number;
  current_dir_deg?: number;
  calm_speed_kts?: number;
  stw_kts?: number;
  sog_kts?: number;
  speed_loss_pct?: number;
  time_hours: number;
  departure_time?: string;
  arrival_time?: string;
  fuel_mt: number;
  power_kw?: number;
  data_source?: string;
}

export interface SaveVoyageRequest {
  name?: string;
  departure_port?: string;
  arrival_port?: string;
  departure_time: string;
  arrival_time: string;
  total_distance_nm: number;
  total_time_hours: number;
  total_fuel_mt: number;
  avg_sog_kts?: number;
  avg_stw_kts?: number;
  calm_speed_kts: number;
  is_laden: boolean;
  vessel_specs_snapshot?: Record<string, unknown>;
  cii_estimate?: Record<string, unknown>;
  notes?: string;
  legs: SaveVoyageLeg[];
}

export interface VoyageSummary {
  id: string;
  name?: string;
  departure_port?: string;
  arrival_port?: string;
  departure_time: string;
  arrival_time: string;
  total_distance_nm: number;
  total_time_hours: number;
  total_fuel_mt: number;
  avg_sog_kts?: number;
  calm_speed_kts: number;
  is_laden: boolean;
  cii_estimate?: Record<string, unknown>;
  created_at: string;
}

export interface VoyageLegDetail {
  id: string;
  leg_index: number;
  from_name?: string;
  from_lat: number;
  from_lon: number;
  to_name?: string;
  to_lat: number;
  to_lon: number;
  distance_nm: number;
  bearing_deg?: number;
  wind_speed_kts?: number;
  wind_dir_deg?: number;
  wave_height_m?: number;
  wave_dir_deg?: number;
  current_speed_ms?: number;
  current_dir_deg?: number;
  calm_speed_kts?: number;
  stw_kts?: number;
  sog_kts?: number;
  speed_loss_pct?: number;
  time_hours: number;
  departure_time?: string;
  arrival_time?: string;
  fuel_mt: number;
  power_kw?: number;
  data_source?: string;
}

export interface VoyageDetail extends VoyageSummary {
  avg_stw_kts?: number;
  vessel_specs_snapshot?: Record<string, unknown>;
  notes?: string;
  updated_at: string;
  legs: VoyageLegDetail[];
}

export interface VoyageListResponse {
  voyages: VoyageSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface NoonReportEntry {
  report_number: number;
  timestamp: string;
  lat: number;
  lon: number;
  sog_kts?: number;
  stw_kts?: number;
  course_deg?: number;
  distance_since_last_nm: number;
  fuel_since_last_mt: number;
  cumulative_distance_nm: number;
  cumulative_fuel_mt: number;
  wind_speed_kts?: number;
  wind_dir_deg?: number;
  wave_height_m?: number;
  wave_dir_deg?: number;
  current_speed_ms?: number;
  current_dir_deg?: number;
}

export interface NoonReportsResponse {
  voyage_id: string;
  voyage_name?: string;
  departure_time: string;
  arrival_time: string;
  reports: NoonReportEntry[];
}

export interface DepartureReportData {
  vessel_name?: string;
  dwt?: number;
  departure_port?: string;
  departure_time: string;
  loading_condition: string;
  destination?: string;
  eta: string;
  planned_distance_nm: number;
  planned_speed_kts: number;
  estimated_fuel_mt: number;
  weather_at_departure?: Record<string, unknown>;
}

export interface ArrivalReportData {
  vessel_name?: string;
  arrival_port?: string;
  arrival_time: string;
  actual_voyage_time_hours: number;
  total_fuel_consumed_mt: number;
  average_speed_kts: number;
  total_distance_nm: number;
  weather_summary?: Record<string, unknown>;
  cii_estimate?: Record<string, unknown>;
}

export interface VoyageReportsResponse {
  voyage_id: string;
  departure_report: DepartureReportData;
  arrival_report: ArrivalReportData;
  noon_reports: NoonReportEntry[];
}

export interface VoyageListParams {
  name?: string;
  date_from?: string;
  date_to?: string;
  limit?: number;
  offset?: number;
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
  max_time_factor: number;
  // All user waypoints for multi-segment optimization
  route_waypoints?: Position[];
  // Baseline from voyage calculation (enables dual-strategy comparison)
  baseline_fuel_mt?: number;
  baseline_time_hours?: number;
  baseline_distance_nm?: number;
  // Engine selection (astar or dijkstra)
  engine?: 'astar' | 'dijkstra';
  // Safety weight: 0=fuel optimal, 1=safety priority
  safety_weight?: number;
  // Pareto front: run A* with multiple lambda values
  pareto?: boolean;
  // Variable resolution: two-tier grid (0.5° ocean + 0.1° nearshore)
  variable_resolution?: boolean;
  // Zone types to enforce during routing (mirrors map visibility)
  enforced_zone_types?: string[];
}

export interface WeatherProvenance {
  source_type: string;
  model_name: string;
  forecast_lead_hours: number;
  confidence: 'high' | 'medium' | 'low';
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
  // Weather provenance per leg
  data_source?: string;
}

export interface SafetySummary {
  status: 'safe' | 'marginal' | 'dangerous';
  warnings: string[];
  max_roll_deg: number;
  max_pitch_deg: number;
  max_accel_ms2: number;
}

export interface SpeedScenario {
  strategy: 'constant_speed' | 'match_eta';
  label: string;
  total_fuel_mt: number;
  total_time_hours: number;
  total_distance_nm: number;
  avg_speed_kts: number;
  speed_profile: number[];
  legs: OptimizationLeg[];
  fuel_savings_pct: number;
  time_savings_pct: number;
}

export interface ParetoSolution {
  lambda_value: number;
  fuel_mt: number;
  time_hours: number;
  distance_nm: number;
  speed_profile: number[];
  is_selected: boolean;
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
  variable_resolution_enabled?: boolean;
  // Safety assessment
  safety?: SafetySummary;
  // Speed strategy scenarios
  scenarios: SpeedScenario[];
  baseline_fuel_mt?: number;
  baseline_time_hours?: number;
  baseline_distance_nm?: number;
  // Pareto front (populated when pareto=true)
  pareto_front?: ParetoSolution[];
  // Weather provenance
  weather_provenance?: WeatherProvenance[];
  temporal_weather: boolean;
  optimization_target: string;
  grid_resolution_deg: number;
  cells_explored: number;
  optimization_time_ms: number;
  // Engine identifier
  engine?: string;
  // Safety fallback: true when hard limits were relaxed to find route
  safety_degraded?: boolean;
}

// Dual-engine types
export type EngineType = 'astar' | 'dijkstra';

export type OptimizedRouteKey =
  | 'astar_fuel' | 'astar_balanced' | 'astar_safety'
  | 'dijkstra_fuel' | 'dijkstra_balanced' | 'dijkstra_safety';

export type AllOptimizationResults = Record<OptimizedRouteKey, OptimizationResponse | null>;

export interface RouteVisibility {
  original: boolean;
  astar_fuel: boolean;
  astar_balanced: boolean;
  astar_safety: boolean;
  dijkstra_fuel: boolean;
  dijkstra_balanced: boolean;
  dijkstra_safety: boolean;
}

export const ROUTE_STYLES: Record<OptimizedRouteKey, { color: string; dashArray: string; label: string }> = {
  astar_fuel:     { color: '#5ab87a', dashArray: '8, 4',  label: 'A* Fuel' },
  astar_balanced: { color: '#7bc89a', dashArray: '12, 6', label: 'A* Balanced' },
  astar_safety:   { color: '#a5d4b5', dashArray: '4, 4',  label: 'A* Safety' },
  dijkstra_fuel:     { color: '#d4885a', dashArray: '8, 4',  label: 'Dijkstra Fuel' },
  dijkstra_balanced: { color: '#dda07c', dashArray: '12, 6', label: 'Dijkstra Balanced' },
  dijkstra_safety:   { color: '#e6c0a0', dashArray: '4, 4',  label: 'Dijkstra Safety' },
};

export const DEFAULT_ROUTE_VISIBILITY: RouteVisibility = {
  original: true,
  astar_fuel: true,
  astar_balanced: false,
  astar_safety: false,
  dijkstra_fuel: true,
  dijkstra_balanced: false,
  dijkstra_safety: false,
};

export const EMPTY_ALL_RESULTS: AllOptimizationResults = {
  astar_fuel: null, astar_balanced: null, astar_safety: null,
  dijkstra_fuel: null, dijkstra_balanced: null, dijkstra_safety: null,
};

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
export type ZoneType = 'eca' | 'hra' | 'tss' | 'vts' | 'exclusion' | 'environmental' | 'ice' | 'canal' | 'custom';
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

// Monte Carlo types
export interface MonteCarloRequest {
  waypoints: Position[];
  calm_speed_kts: number;
  is_laden: boolean;
  departure_time?: string;
  n_simulations?: number;
}

export interface PercentileFloat {
  p10: number;
  p50: number;
  p90: number;
}

export interface PercentileString {
  p10: string;
  p50: string;
  p90: string;
}

export interface MonteCarloResponse {
  n_simulations: number;
  eta: PercentileString;
  fuel_mt: PercentileFloat;
  total_time_hours: PercentileFloat;
  computation_time_ms: number;
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

// CII Compliance types
export interface VesselTypeInfo {
  id: string;
  name: string;
}

export interface FuelTypeInfo {
  id: string;
  name: string;
  co2_factor: number;
}

export interface CIICalculationRequest {
  fuel_consumption_mt: Record<string, number>;
  total_distance_nm: number;
  dwt: number;
  vessel_type: string;
  year: number;
  gt?: number;
}

export interface CIICalculationResponse {
  year: number;
  rating: string;
  compliance_status: string;
  attained_cii: number;
  required_cii: number;
  rating_boundaries: Record<string, number>;
  reduction_factor: number;
  total_co2_mt: number;
  total_distance_nm: number;
  capacity: number;
  vessel_type: string;
  margin_to_downgrade: number;
  margin_to_upgrade: number;
}

export interface CIIProjectionRequest {
  annual_fuel_mt: Record<string, number>;
  annual_distance_nm: number;
  dwt: number;
  vessel_type: string;
  start_year: number;
  end_year: number;
  fuel_efficiency_improvement_pct: number;
  gt?: number;
}

export interface CIIProjectionItem {
  year: number;
  rating: string;
  attained_cii: number;
  required_cii: number;
  reduction_factor: number;
  status: string;
}

export interface CIIProjectionSummary {
  current_rating: string;
  final_rating: string;
  years_until_d_rating: number | string;
  years_until_e_rating: number | string;
  recommendation: string;
}

export interface CIIProjectionResponse {
  projections: CIIProjectionItem[];
  summary: CIIProjectionSummary;
}

export interface CIIReductionRequest {
  current_fuel_mt: Record<string, number>;
  current_distance_nm: number;
  dwt: number;
  vessel_type: string;
  target_rating: string;
  target_year: number;
  gt?: number;
}

export interface CIIReductionResponse {
  reduction_needed_pct: number;
  current_cii: number;
  target_cii: number;
  current_rating: string;
  target_rating: string;
  fuel_savings_mt: number;
  message: string;
}

// CII Speed Sweep (Simulator)
export interface CIISpeedSweepRequest {
  dwt: number;
  vessel_type: string;
  distance_nm: number;
  voyages_per_year: number;
  fuel_type: string;
  year: number;
  speed_min_kts: number;
  speed_max_kts: number;
  speed_step_kts: number;
  is_laden: boolean;
}

export interface CIISpeedSweepPoint {
  speed_kts: number;
  fuel_per_voyage_mt: number;
  annual_fuel_mt: number;
  annual_co2_mt: number;
  attained_cii: number;
  required_cii: number;
  rating: string;
}

export interface CIISpeedSweepResponse {
  points: CIISpeedSweepPoint[];
  optimal_speed_kts: number;
  rating_boundaries: Record<string, number>;
}

// CII Thresholds
export interface CIIThresholdYear {
  year: number;
  required_cii: number;
  boundaries: Record<string, number>;
  reduction_factor: number;
}

export interface CIIThresholdsResponse {
  years: CIIThresholdYear[];
  vessel_type: string;
  capacity: number;
}

// CII Fleet
export interface CIIFleetVessel {
  name: string;
  dwt: number;
  vessel_type: string;
  fuel_consumption_mt: Record<string, number>;
  total_distance_nm: number;
  year: number;
  gt?: number;
}

export interface CIIFleetRequest {
  vessels: CIIFleetVessel[];
}

export interface CIIFleetResult {
  name: string;
  rating: string;
  attained_cii: number;
  required_cii: number;
  compliance_status: string;
  total_co2_mt: number;
  margin_to_downgrade: number;
  margin_to_upgrade: number;
}

export interface CIIFleetResponse {
  results: CIIFleetResult[];
  summary: Record<string, number>;
}

// FuelEU Maritime types
export interface FuelEUFuelInfo {
  id: string;
  name: string;
  lcv_mj_per_g: number;
  wtt_gco2eq_per_mj: number;
  ttw_gco2eq_per_mj: number;
  wtw_gco2eq_per_mj: number;
}

export interface FuelEUFuelBreakdown {
  fuel_type: string;
  mass_mt: number;
  energy_mj: number;
  wtt_gco2eq: number;
  ttw_gco2eq: number;
  wtw_gco2eq: number;
  wtw_intensity: number;
}

export interface FuelEUCalculateResponse {
  ghg_intensity: number;
  total_energy_mj: number;
  total_co2eq_g: number;
  fuel_breakdown: FuelEUFuelBreakdown[];
}

export interface FuelEUComplianceResponse {
  year: number;
  ghg_intensity: number;
  ghg_limit: number;
  reduction_target_pct: number;
  compliance_balance_gco2eq: number;
  total_energy_mj: number;
  status: string;
}

export interface FuelEUPenaltyResponse {
  compliance_balance_gco2eq: number;
  non_compliant_energy_mj: number;
  vlsfo_equivalent_mt: number;
  penalty_eur: number;
  penalty_per_mt_fuel: number;
}

export interface FuelEUPoolingVesselResult {
  name: string;
  ghg_intensity: number;
  total_energy_mj: number;
  total_co2eq_g: number;
  individual_balance_gco2eq: number;
  status: string;
}

export interface FuelEUPoolingResponse {
  fleet_ghg_intensity: number;
  fleet_total_energy_mj: number;
  fleet_total_co2eq_g: number;
  fleet_balance_gco2eq: number;
  per_vessel: FuelEUPoolingVesselResult[];
  status: string;
}

export interface FuelEUProjectionYear {
  year: number;
  ghg_intensity: number;
  ghg_limit: number;
  reduction_target_pct: number;
  compliance_balance_gco2eq: number;
  total_energy_mj: number;
  status: string;
  penalty_eur: number;
}

export interface FuelEUProjectResponse {
  projections: FuelEUProjectionYear[];
}

export interface FuelEULimitYear {
  year: number;
  reduction_pct: number;
  ghg_limit: number;
}

export interface FuelEULimitsResponse {
  limits: FuelEULimitYear[];
  reference_ghg: number;
}

export interface FuelEUPoolingVessel {
  name: string;
  fuel_mt: Record<string, number>;
}

// Charter Party types
export interface BeaufortEntry {
  force: number;
  wind_min_kts: number;
  wind_max_kts: number;
  wave_height_m: number;
  description: string;
}

export interface BeaufortScaleResponse {
  scale: BeaufortEntry[];
}

export interface LegWeatherInput {
  wind_speed_kts: number;
  wave_height_m?: number;
  current_speed_ms?: number;
  time_hours: number;
  distance_nm?: number;
  sog_kts?: number;
  fuel_mt?: number;
}

export interface GoodWeatherLegResponse {
  leg_index: number;
  wind_speed_kts: number;
  wave_height_m: number;
  current_speed_ms: number;
  bf_force: number;
  is_good_weather: boolean;
  time_hours: number;
}

export interface GoodWeatherResponse {
  total_days: number;
  good_weather_days: number;
  bad_weather_days: number;
  good_weather_pct: number;
  bf_threshold: number;
  wave_threshold_m: number | null;
  current_threshold_kts: number | null;
  legs: GoodWeatherLegResponse[];
}

export interface WarrantyVerificationResponse {
  warranted_speed_kts: number;
  achieved_speed_kts: number;
  speed_margin_kts: number;
  speed_compliant: boolean;
  warranted_consumption_mt_day: number;
  achieved_consumption_mt_day: number;
  consumption_margin_mt: number;
  consumption_compliant: boolean;
  good_weather_hours: number;
  total_hours: number;
  legs_assessed: number;
  legs_good_weather: number;
  legs: WarrantyLegDetailResponse[];
}

export interface WarrantyLegDetailResponse {
  leg_index: number;
  sog_kts: number;
  fuel_mt: number;
  time_hours: number;
  distance_nm: number;
  bf_force: number;
  is_good_weather: boolean;
}

export interface OffHireEventResponse {
  start_time: string;
  end_time: string;
  duration_hours: number;
  reason: string;
  avg_speed_kts: number | null;
}

export interface OffHireResponse {
  total_hours: number;
  on_hire_hours: number;
  off_hire_hours: number;
  off_hire_pct: number;
  events: OffHireEventResponse[];
}

// Fuel Analysis types
export interface FuelScenario {
  name: string;
  conditions: string;
  fuel_mt: number;
  power_kw: number;
}

// Model curves types
export interface ModelCurvesResponse {
  speed_range_kts: number[];
  resistance_theoretical_kn: number[];
  resistance_calibrated_kn: number[];
  power_kw: number[];
  sfoc_gkwh: number[];
  fuel_mt_per_day: number[];
  sfoc_curve: {
    load_pct: number[];
    sfoc_theoretical_gkwh: number[];
    sfoc_calibrated_gkwh: number[];
  };
  calibration: {
    calibrated: boolean;
    factors: {
      calm_water: number;
      wind: number;
      waves: number;
      sfoc_factor: number;
    };
    calibrated_at: string | null;
    num_reports_used: number;
    calibration_error_mt: number;
  };
}

// Performance prediction types
export interface PerformancePredictionRequest {
  is_laden: boolean;
  engine_load_pct?: number;     // Mode 1: find speed at this power
  calm_speed_kts?: number;      // Mode 2: find power for this calm-water speed
  wind_speed_kts: number;
  wind_relative_deg: number;   // 0=ahead, 90=beam, 180=astern
  wave_height_m: number;
  wave_relative_deg: number;   // 0=head seas, 90=beam, 180=following
  current_speed_kts: number;
  current_relative_deg: number; // 0=head current, 180=following
}

export interface PerformancePredictionResult {
  stw_kts: number;
  sog_kts: number;
  fuel_per_day_mt: number;
  fuel_per_nm_mt: number;
  power_kw: number;
  load_pct: number;
  sfoc_gkwh: number;
  resistance_breakdown_kn: {
    calm_water: number;
    wind: number;
    waves: number;
    total: number;
  };
  speed_loss_from_weather_pct: number;
  calm_water_speed_kts: number;
  current_effect_kts: number;
  service_speed_kts: number;
  mcr_exceeded?: boolean;
  required_power_kw?: number;
  mode?: string;
}

// Vessel model status types
export interface VesselModelStatus {
  specifications: {
    dimensions: {
      loa: number; lpp: number; beam: number;
      draft_laden: number; draft_ballast: number;
      dwt: number; displacement_laden: number; displacement_ballast: number;
    };
    hull_form: {
      cb_laden: number; cb_ballast: number;
      wetted_surface_laden: number; wetted_surface_ballast: number;
    };
    engine: {
      mcr_kw: number; sfoc_at_mcr: number;
      service_speed_laden: number; service_speed_ballast: number;
    };
    areas: {
      frontal_area_laden: number; frontal_area_ballast: number;
      lateral_area_laden: number; lateral_area_ballast: number;
    };
  };
  calibration: {
    calibrated: boolean;
    factors: { calm_water: number; wind: number; waves: number; sfoc_factor: number };
    calibrated_at: string | null;
    num_reports_used: number;
    calibration_error_mt: number;
    days_since_drydock: number;
  };
  wave_method: string;
  computed: {
    optimal_speed_laden_kts: number;
    optimal_speed_ballast_kts: number;
    daily_fuel_service_laden_mt: number;
    daily_fuel_service_ballast_mt: number;
  };
}

// Engine Log types
export interface EngineLogUploadResponse {
  status: string;
  batch_id: string;
  imported: number;
  skipped: number;
  date_range?: { start: string | null; end: string | null };
  events_summary?: Record<string, number>;
}

export interface EngineLogEntryResponse {
  id: string;
  timestamp: string;
  lapse_hours: number | null;
  place: string | null;
  event: string | null;
  rpm: number | null;
  engine_distance: number | null;
  speed_stw: number | null;
  me_power_kw: number | null;
  me_load_pct: number | null;
  me_fuel_index_pct: number | null;
  shaft_power: number | null;
  shaft_torque_knm: number | null;
  slip_pct: number | null;
  hfo_me_mt: number | null;
  hfo_ae_mt: number | null;
  hfo_boiler_mt: number | null;
  hfo_total_mt: number | null;
  mgo_me_mt: number | null;
  mgo_ae_mt: number | null;
  mgo_total_mt: number | null;
  methanol_me_mt: number | null;
  rob_vlsfo_mt: number | null;
  rob_mgo_mt: number | null;
  rob_methanol_mt: number | null;
  rh_me: number | null;
  rh_ae_total: number | null;
  tc_rpm: number | null;
  scav_air_press_bar: number | null;
  fuel_temp_c: number | null;
  sw_temp_c: number | null;
  upload_batch_id: string;
  source_sheet: string | null;
  source_file: string | null;
  extended_data: Record<string, unknown> | null;
}

export interface EngineLogBatch {
  batch_id: string;
  count: number;
  date_start: string | null;
  date_end: string | null;
  source_file: string | null;
}

export interface EngineLogSummaryResponse {
  total_entries: number;
  date_range?: { start: string | null; end: string | null };
  events_breakdown?: Record<string, number>;
  fuel_summary?: { hfo_mt: number; mgo_mt: number; methanol_mt: number };
  avg_rpm_at_sea: number | null;
  avg_speed_stw: number | null;
  batches?: EngineLogBatch[];
}

export interface EngineLogCalibrateResponse {
  status: string;
  factors: {
    calm_water: number;
    wind: number;
    waves: number;
    sfoc_factor: number;
    calibrated_at: string | null;
    num_reports_used: number;
    calibration_error: number;
    days_since_drydock: number;
  };
  entries_used: number;
  entries_skipped: number;
  mean_error_before_mt: number;
  mean_error_after_mt: number;
  improvement_pct: number;
}

export interface EngineLogEntriesParams {
  event?: string;
  date_from?: string;
  date_to?: string;
  min_rpm?: number;
  batch_id?: string;
  limit?: number;
  offset?: number;
}

// Live Sensor types
export interface SensorStatus {
  connected: boolean;
  streaming: boolean;
  connection_type: string;
  message_count: number;
  parse_errors: number;
  last_message_time: string | null;
}

export interface LiveData {
  timestamp: string;
  position: {
    latitude: number;
    longitude: number;
  };
  velocity: {
    sog_kts: number;
    cog_deg: number;
  };
  attitude: {
    heading_deg: number;
    roll_deg: number;
    pitch_deg: number;
  };
  motion: {
    heave_m: number;
    surge_m: number;
  };
  status: {
    satellites: number;
    gnss_fix: number;
    hdop: number;
  };
}

export interface SensorConfig {
  connection_type: string;
  port?: string;
  baudrate?: number;
}

export function createLiveWebSocket(
  onData: (data: LiveData) => void,
  onError: (error: Event) => void,
  onClose: () => void,
): WebSocket {
  const wsUrl = API_BASE_URL.replace(/^http/, 'ws') + '/api/live/ws';
  const ws = new WebSocket(wsUrl);

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data) as LiveData;
      onData(data);
    } catch (e) {
      console.error('Failed to parse live data:', e);
    }
  };

  ws.onerror = onError;
  ws.onclose = onClose;

  return ws;
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
    forecast_hour?: number;
  } = {}): Promise<VelocityData[]> {
    const response = await api.get<VelocityData[]>('/api/weather/wind/velocity', { params });
    return response.data;
  },

  // Forecast timeline API
  async getForecastStatus(params: {
    lat_min?: number;
    lat_max?: number;
    lon_min?: number;
    lon_max?: number;
  } = {}): Promise<ForecastStatus> {
    const response = await api.get<ForecastStatus>('/api/weather/forecast/status', { params });
    return response.data;
  },

  async triggerForecastPrefetch(params: {
    lat_min?: number;
    lat_max?: number;
    lon_min?: number;
    lon_max?: number;
  } = {}): Promise<{ status: string; message: string }> {
    const response = await api.post('/api/weather/forecast/prefetch', null, { params });
    return response.data;
  },

  async getForecastFrames(params: {
    lat_min?: number;
    lat_max?: number;
    lon_min?: number;
    lon_max?: number;
  } = {}): Promise<ForecastFrames> {
    const response = await api.get<ForecastFrames>('/api/weather/forecast/frames', { params });
    return response.data;
  },

  // Wave forecast timeline API
  async getWaveForecastStatus(params: {
    lat_min?: number;
    lat_max?: number;
    lon_min?: number;
    lon_max?: number;
  } = {}): Promise<ForecastStatus> {
    const response = await api.get<ForecastStatus>('/api/weather/forecast/wave/status', { params });
    return response.data;
  },

  async triggerWaveForecastPrefetch(params: {
    lat_min?: number;
    lat_max?: number;
    lon_min?: number;
    lon_max?: number;
  } = {}): Promise<{ status: string; message: string }> {
    const response = await api.post('/api/weather/forecast/wave/prefetch', null, { params });
    return response.data;
  },

  async getWaveForecastFrames(params: {
    lat_min?: number;
    lat_max?: number;
    lon_min?: number;
    lon_max?: number;
  } = {}): Promise<WaveForecastFrames> {
    const response = await api.get<WaveForecastFrames>('/api/weather/forecast/wave/frames', { params });
    return response.data;
  },

  // Current forecast
  async getCurrentForecastStatus(params: {
    lat_min?: number;
    lat_max?: number;
    lon_min?: number;
    lon_max?: number;
  } = {}): Promise<ForecastStatus> {
    const response = await api.get<ForecastStatus>('/api/weather/forecast/current/status', { params });
    return response.data;
  },

  async triggerCurrentForecastPrefetch(params: {
    lat_min?: number;
    lat_max?: number;
    lon_min?: number;
    lon_max?: number;
  } = {}): Promise<{ status: string; message: string }> {
    const response = await api.post('/api/weather/forecast/current/prefetch', null, { params });
    return response.data;
  },

  async getCurrentForecastFrames(params: {
    lat_min?: number;
    lat_max?: number;
    lon_min?: number;
    lon_max?: number;
  } = {}): Promise<CurrentForecastFrames> {
    const response = await api.get<CurrentForecastFrames>('/api/weather/forecast/current/frames', { params });
    return response.data;
  },

  // Ice forecast
  async getIceForecastStatus(params: {
    lat_min?: number;
    lat_max?: number;
    lon_min?: number;
    lon_max?: number;
  } = {}): Promise<ForecastStatus> {
    const response = await api.get<ForecastStatus>('/api/weather/forecast/ice/status', { params });
    return response.data;
  },

  async triggerIceForecastPrefetch(params: {
    lat_min?: number;
    lat_max?: number;
    lon_min?: number;
    lon_max?: number;
  } = {}): Promise<{ status: string; message: string }> {
    const response = await api.post('/api/weather/forecast/ice/prefetch', null, { params });
    return response.data;
  },

  async getIceForecastFrames(params: {
    lat_min?: number;
    lat_max?: number;
    lon_min?: number;
    lon_max?: number;
  } = {}): Promise<IceForecastFrames> {
    const response = await api.get<IceForecastFrames>('/api/weather/forecast/ice/frames', { params });
    return response.data;
  },

  // SST forecast
  async getSstForecastStatus(params: {
    lat_min?: number;
    lat_max?: number;
    lon_min?: number;
    lon_max?: number;
  } = {}): Promise<ForecastStatus> {
    const response = await api.get<ForecastStatus>('/api/weather/forecast/sst/status', { params });
    return response.data;
  },

  async triggerSstForecastPrefetch(params: {
    lat_min?: number;
    lat_max?: number;
    lon_min?: number;
    lon_max?: number;
  } = {}): Promise<{ status: string; message: string }> {
    const response = await api.post('/api/weather/forecast/sst/prefetch', null, { params });
    return response.data;
  },

  async getSstForecastFrames(params: {
    lat_min?: number;
    lat_max?: number;
    lon_min?: number;
    lon_max?: number;
  } = {}): Promise<SstForecastFrames> {
    const response = await api.get<SstForecastFrames>('/api/weather/forecast/sst/frames', { params });
    return response.data;
  },

  // Visibility forecast
  async getVisForecastStatus(params: {
    lat_min?: number;
    lat_max?: number;
    lon_min?: number;
    lon_max?: number;
  } = {}): Promise<ForecastStatus> {
    const response = await api.get<ForecastStatus>('/api/weather/forecast/visibility/status', { params });
    return response.data;
  },

  async triggerVisForecastPrefetch(params: {
    lat_min?: number;
    lat_max?: number;
    lon_min?: number;
    lon_max?: number;
  } = {}): Promise<{ status: string; message: string }> {
    const response = await api.post('/api/weather/forecast/visibility/prefetch', null, { params });
    return response.data;
  },

  async getVisForecastFrames(params: {
    lat_min?: number;
    lat_max?: number;
    lon_min?: number;
    lon_max?: number;
  } = {}): Promise<VisForecastFrames> {
    const response = await api.get<VisForecastFrames>('/api/weather/forecast/visibility/frames', { params });
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

  async getCurrentVelocity(params: {
    lat_min?: number;
    lat_max?: number;
    lon_min?: number;
    lon_max?: number;
    resolution?: number;
    time?: string;
  } = {}): Promise<VelocityData[]> {
    const response = await api.get<VelocityData[]>('/api/weather/currents/velocity', { params });
    return response.data;
  },

  async getWeatherAtPoint(lat: number, lon: number, time?: string): Promise<PointWeather> {
    const params: { lat: number; lon: number; time?: string } = { lat, lon };
    if (time) params.time = time;
    const response = await api.get<PointWeather>('/api/weather/point', { params });
    return response.data;
  },

  // Extended weather fields (SPEC-P1)
  async getSstField(params: { lat_min?: number; lat_max?: number; lon_min?: number; lon_max?: number; resolution?: number } = {}): Promise<GridFieldData> {
    const response = await api.get<GridFieldData>('/api/weather/sst', { params });
    return response.data;
  },

  async getVisibilityField(params: { lat_min?: number; lat_max?: number; lon_min?: number; lon_max?: number; resolution?: number } = {}): Promise<GridFieldData> {
    const response = await api.get<GridFieldData>('/api/weather/visibility', { params });
    return response.data;
  },

  async getCurrentField(params: { lat_min?: number; lat_max?: number; lon_min?: number; lon_max?: number; resolution?: number } = {}): Promise<GridFieldData> {
    const response = await api.get<GridFieldData>('/api/weather/currents', { params });
    return response.data;
  },

  async getIceField(params: { lat_min?: number; lat_max?: number; lon_min?: number; lon_max?: number; resolution?: number } = {}): Promise<GridFieldData> {
    const response = await api.get<GridFieldData>('/api/weather/ice', { params });
    return response.data;
  },

  async getWeatherFreshness(): Promise<{
    status: string;
    age_hours: number | null;
    color: string;
    message?: string;
  }> {
    if (isDemoUser()) {
      return { status: 'demo', age_hours: 0, color: 'green', message: 'Demo snapshot' };
    }
    const response = await api.get('/api/weather/freshness');
    return response.data;
  },

  async getSwellField(params: { lat_min?: number; lat_max?: number; lon_min?: number; lon_max?: number; resolution?: number } = {}): Promise<SwellFieldData> {
    const response = await api.get<SwellFieldData>('/api/weather/swell', { params });
    return response.data;
  },

  async getWeatherHealth(): Promise<WeatherHealthResponse> {
    const response = await api.get<WeatherHealthResponse>('/api/weather/health');
    return response.data;
  },

  async resyncWeatherLayer(
    layer: string,
    bbox?: { lat_min: number; lat_max: number; lon_min: number; lon_max: number },
  ): Promise<{ status: string; ingested_at: string }> {
    const params = bbox ? { params: bbox } : {};
    const response = await api.post(`/api/weather/${layer}/resync`, null, { timeout: 300000, ...params });
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
    time?: string,
    interpolation_points: number = 5,
  ): Promise<WeatherAlongRouteResponse> {
    const wpString = waypoints.map(wp => `${wp.lat},${wp.lon}`).join(';');
    const params: { waypoints: string; time?: string; interpolation_points: number } = {
      waypoints: wpString,
      interpolation_points,
    };
    if (time) params.time = time;
    const response = await api.get('/api/voyage/weather-along-route', { params });
    return response.data;
  },

  // -------------------------------------------------------------------------
  // Optimization API (Layer 4)
  // -------------------------------------------------------------------------

  async optimizeRoute(request: OptimizationRequest): Promise<OptimizationResponse> {
    const response = await api.post<OptimizationResponse>('/api/optimize/route', request, {
      timeout: 180000, // 3 min timeout per engine (fine grids on long routes)
    });
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

  async benchmarkEngines(request: {
    origin: Position;
    destination: Position;
    calm_speed_kts: number;
    is_laden: boolean;
    grid_resolution_deg: number;
    safety_weight: number;
    variable_resolution: boolean;
    engines: string[];
  }): Promise<{
    results: Array<{
      engine: string;
      total_fuel_mt: number;
      total_time_hours: number;
      total_distance_nm: number;
      cells_explored: number;
      optimization_time_ms: number;
      waypoint_count: number;
      error?: string | null;
    }>;
    grid_resolution_deg: number;
    optimization_target: string;
  }> {
    const response = await api.post('/api/optimize/benchmark', request, {
      timeout: 300000, // 5 min for both engines
    });
    return response.data;
  },

  // -------------------------------------------------------------------------
  // Ocean / land check
  // -------------------------------------------------------------------------

  async checkOcean(lat: number, lon: number): Promise<boolean> {
    const response = await api.get<{ ocean: boolean }>('/api/check-ocean', {
      params: { lat, lon },
    });
    return response.data.ocean;
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

  async uploadNoonReportsExcel(file: File): Promise<{ status: string; imported: number; total_reports: number }> {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post('/api/vessel/noon-reports/upload-excel', formData, {
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

  // -------------------------------------------------------------------------
  // Monte Carlo API
  // -------------------------------------------------------------------------

  async runMonteCarlo(request: MonteCarloRequest): Promise<MonteCarloResponse> {
    const response = await api.post<MonteCarloResponse>('/api/voyage/monte-carlo', request, {
      timeout: 120000, // 2 min timeout for CPU-bound simulation
    });
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

  // -------------------------------------------------------------------------
  // CII Compliance API
  // -------------------------------------------------------------------------

  async getVesselTypes(): Promise<{ vessel_types: VesselTypeInfo[] }> {
    const response = await api.get('/api/cii/vessel-types');
    return response.data;
  },

  async getFuelTypes(): Promise<{ fuel_types: FuelTypeInfo[] }> {
    const response = await api.get('/api/cii/fuel-types');
    return response.data;
  },

  async calculateCII(request: CIICalculationRequest): Promise<CIICalculationResponse> {
    const response = await api.post<CIICalculationResponse>('/api/cii/calculate', request);
    return response.data;
  },

  async projectCII(request: CIIProjectionRequest): Promise<CIIProjectionResponse> {
    const response = await api.post<CIIProjectionResponse>('/api/cii/project', request);
    return response.data;
  },

  async calculateCIIReduction(request: CIIReductionRequest): Promise<CIIReductionResponse> {
    const response = await api.post<CIIReductionResponse>('/api/cii/reduction', request);
    return response.data;
  },

  async simulateCIISpeed(request: CIISpeedSweepRequest): Promise<CIISpeedSweepResponse> {
    const response = await api.post<CIISpeedSweepResponse>('/api/cii/speed-sweep', request);
    return response.data;
  },

  async getCIIThresholds(params: { dwt: number; vessel_type?: string; gt?: number }): Promise<CIIThresholdsResponse> {
    const response = await api.get<CIIThresholdsResponse>('/api/cii/thresholds', { params });
    return response.data;
  },

  async calculateFleetCII(request: CIIFleetRequest): Promise<CIIFleetResponse> {
    const response = await api.post<CIIFleetResponse>('/api/cii/fleet', request);
    return response.data;
  },

  // -------------------------------------------------------------------------
  // FuelEU Maritime API
  // -------------------------------------------------------------------------

  async getFuelEUFuelTypes(): Promise<{ fuel_types: FuelEUFuelInfo[] }> {
    const response = await api.get('/api/fueleu/fuel-types');
    return response.data;
  },

  async getFuelEULimits(): Promise<FuelEULimitsResponse> {
    const response = await api.get<FuelEULimitsResponse>('/api/fueleu/limits');
    return response.data;
  },

  async calculateFuelEU(request: { fuel_consumption_mt: Record<string, number>; year: number }): Promise<FuelEUCalculateResponse> {
    const response = await api.post<FuelEUCalculateResponse>('/api/fueleu/calculate', request);
    return response.data;
  },

  async calculateFuelEUCompliance(request: { fuel_consumption_mt: Record<string, number>; year: number }): Promise<FuelEUComplianceResponse> {
    const response = await api.post<FuelEUComplianceResponse>('/api/fueleu/compliance', request);
    return response.data;
  },

  async calculateFuelEUPenalty(request: { fuel_consumption_mt: Record<string, number>; year: number; consecutive_deficit_years?: number }): Promise<FuelEUPenaltyResponse> {
    const response = await api.post<FuelEUPenaltyResponse>('/api/fueleu/penalty', request);
    return response.data;
  },

  async simulateFuelEUPooling(request: { vessels: FuelEUPoolingVessel[]; year: number }): Promise<FuelEUPoolingResponse> {
    const response = await api.post<FuelEUPoolingResponse>('/api/fueleu/pooling', request);
    return response.data;
  },

  async projectFuelEU(request: { fuel_consumption_mt: Record<string, number>; start_year: number; end_year: number; annual_efficiency_improvement_pct?: number }): Promise<FuelEUProjectResponse> {
    const response = await api.post<FuelEUProjectResponse>('/api/fueleu/project', request);
    return response.data;
  },

  // -------------------------------------------------------------------------
  // Fuel Analysis API
  // -------------------------------------------------------------------------

  async getFuelScenarios(): Promise<{ scenarios: FuelScenario[] }> {
    const response = await api.get<{ scenarios: FuelScenario[] }>('/api/vessel/fuel-scenarios');
    return response.data;
  },

  async getModelCurves(): Promise<ModelCurvesResponse> {
    const response = await api.get<ModelCurvesResponse>('/api/vessel/model-curves');
    return response.data;
  },

  async getVesselModelStatus(): Promise<VesselModelStatus> {
    const response = await api.get<VesselModelStatus>('/api/vessel/model-status');
    return response.data;
  },

  async predictPerformance(req: PerformancePredictionRequest): Promise<PerformancePredictionResult> {
    const response = await api.post<PerformancePredictionResult>('/api/vessel/predict', req);
    return response.data;
  },

  // -------------------------------------------------------------------------
  // Live Sensor API
  // -------------------------------------------------------------------------

  async getSensorStatus(): Promise<SensorStatus> {
    const response = await api.get<SensorStatus>('/api/live/status');
    return response.data;
  },

  async connectSensor(config: SensorConfig): Promise<{ status: string }> {
    const response = await api.post('/api/live/connect', config);
    return response.data;
  },

  async disconnectSensor(): Promise<{ status: string }> {
    const response = await api.post('/api/live/disconnect');
    return response.data;
  },

  // -------------------------------------------------------------------------
  // Voyage History API
  // -------------------------------------------------------------------------

  async saveVoyage(request: SaveVoyageRequest): Promise<VoyageSummary> {
    const response = await api.post<VoyageSummary>('/api/voyages', request);
    return response.data;
  },

  async listVoyages(params: VoyageListParams = {}): Promise<VoyageListResponse> {
    const response = await api.get<VoyageListResponse>('/api/voyages', { params });
    return response.data;
  },

  async getVoyage(id: string): Promise<VoyageDetail> {
    const response = await api.get<VoyageDetail>(`/api/voyages/${id}`);
    return response.data;
  },

  async deleteVoyage(id: string): Promise<{ status: string; voyage_id: string }> {
    const response = await api.delete(`/api/voyages/${id}`);
    return response.data;
  },

  async getVoyageNoonReports(id: string): Promise<NoonReportsResponse> {
    const response = await api.get<NoonReportsResponse>(`/api/voyages/${id}/noon-reports`);
    return response.data;
  },

  async getVoyageReports(id: string): Promise<VoyageReportsResponse> {
    const response = await api.get<VoyageReportsResponse>(`/api/voyages/${id}/reports`);
    return response.data;
  },

  async downloadVoyagePDF(id: string): Promise<Blob> {
    const response = await api.get(`/api/voyages/${id}/pdf`, {
      responseType: 'blob',
    });
    return response.data;
  },

  // -------------------------------------------------------------------------
  // Engine Log API
  // -------------------------------------------------------------------------

  async uploadEngineLog(file: File): Promise<EngineLogUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post<EngineLogUploadResponse>('/api/engine-log/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 120000,
    });
    return response.data;
  },

  async getEngineLogEntries(params: EngineLogEntriesParams = {}): Promise<EngineLogEntryResponse[]> {
    const response = await api.get<EngineLogEntryResponse[]>('/api/engine-log/entries', { params });
    return response.data;
  },

  async getEngineLogSummary(batchId?: string): Promise<EngineLogSummaryResponse> {
    const params = batchId ? { batch_id: batchId } : {};
    const response = await api.get<EngineLogSummaryResponse>('/api/engine-log/summary', { params });
    return response.data;
  },

  async deleteEngineLogBatch(batchId: string): Promise<{ status: string; batch_id: string; deleted_count: number }> {
    const response = await api.delete(`/api/engine-log/batch/${batchId}`);
    return response.data;
  },

  async calibrateFromEngineLog(batchId?: string, daysSinceDrydock?: number): Promise<EngineLogCalibrateResponse> {
    const params: Record<string, string | number> = {};
    if (batchId) params.batch_id = batchId;
    if (daysSinceDrydock !== undefined) params.days_since_drydock = daysSinceDrydock;
    const response = await api.post<EngineLogCalibrateResponse>('/api/engine-log/calibrate', null, { params });
    return response.data;
  },

  // -------------------------------------------------------------------------
  // Charter Party Tools API
  // -------------------------------------------------------------------------

  async getBeaufortScale(): Promise<BeaufortScaleResponse> {
    const response = await api.get<BeaufortScaleResponse>('/api/charter-party/beaufort-scale');
    return response.data;
  },

  async analyzeGoodWeather(request: { legs: LegWeatherInput[]; bf_threshold?: number; wave_threshold_m?: number; current_threshold_kts?: number }): Promise<GoodWeatherResponse> {
    const response = await api.post<GoodWeatherResponse>('/api/charter-party/good-weather/from-legs', request);
    return response.data;
  },

  async verifyWarranty(request: { legs: LegWeatherInput[]; warranted_speed_kts: number; warranted_consumption_mt_day: number; bf_threshold?: number; speed_tolerance_pct?: number; consumption_tolerance_pct?: number }): Promise<WarrantyVerificationResponse> {
    const response = await api.post<WarrantyVerificationResponse>('/api/charter-party/verify-warranty/from-legs', request);
    return response.data;
  },

  async detectOffHire(request: { date_from?: string; date_to?: string; rpm_threshold?: number; speed_threshold?: number; gap_hours?: number }): Promise<OffHireResponse> {
    const response = await api.post<OffHireResponse>('/api/charter-party/off-hire', request);
    return response.data;
  },
};

export default api;

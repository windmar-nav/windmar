'use client';

import { useState, useEffect, useCallback } from 'react';
import dynamic from 'next/dynamic';
import Header from '@/components/Header';
import Card from '@/components/Card';
import { WaypointList } from '@/components/WaypointEditor';
import RouteImport, { SampleRTZButton } from '@/components/RouteImport';
import VoyageResults, { VoyageProfile } from '@/components/VoyageResults';
import SavedRoutes from '@/components/SavedRoutes';
import CalibrationPanel from '@/components/CalibrationPanel';
import {
  Navigation,
  Ship,
  Play,
  Loader2,
  Wind,
  Waves,
  Settings,
  Upload,
  MousePointer,
  Eye,
  EyeOff,
  RefreshCw,
  Compass,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  ShieldAlert,
  Shield,
  PenTool,
} from 'lucide-react';
import { apiClient, Position, WindFieldData, VoyageResponse, OptimizationResponse, CreateZoneRequest } from '@/lib/api';

// Dynamic imports for map components (client-side only)
const MapContainer = dynamic(
  () => import('react-leaflet').then((mod) => mod.MapContainer),
  { ssr: false }
);
const TileLayer = dynamic(
  () => import('react-leaflet').then((mod) => mod.TileLayer),
  { ssr: false }
);
const WaypointEditor = dynamic(() => import('@/components/WaypointEditor'), {
  ssr: false,
});
const WindLayer = dynamic(
  () => import('@/components/WindLayer'),
  { ssr: false }
);
const ZoneLayer = dynamic(
  () => import('@/components/ZoneLayer'),
  { ssr: false }
);
const ZoneEditor = dynamic(
  () => import('@/components/ZoneEditor'),
  { ssr: false }
);

// Map bounds for Europe/Mediterranean
const DEFAULT_CENTER: [number, number] = [45, 10];
const DEFAULT_ZOOM = 5;

type ViewMode = 'edit' | 'results';
type WeatherLayer = 'wind' | 'waves' | 'none';

export default function HomePage() {
  // Route state
  const [waypoints, setWaypoints] = useState<Position[]>([]);
  const [isEditing, setIsEditing] = useState(true);
  const [routeName, setRouteName] = useState('Custom Route');

  // Voyage parameters
  const [calmSpeed, setCalmSpeed] = useState(14.5);
  const [isLaden, setIsLaden] = useState(true);
  const [useWeather, setUseWeather] = useState(true);

  // Results
  const [voyageResult, setVoyageResult] = useState<VoyageResponse | null>(null);
  const [isCalculating, setIsCalculating] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>('edit');

  // Optimization
  const [optimizationResult, setOptimizationResult] = useState<OptimizationResponse | null>(null);
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [showOptimizedRoute, setShowOptimizedRoute] = useState(false);

  // Weather visualization
  const [weatherLayer, setWeatherLayer] = useState<WeatherLayer>('wind');
  const [windData, setWindData] = useState<WindFieldData | null>(null);
  const [isLoadingWeather, setIsLoadingWeather] = useState(false);

  // Zone state
  const [showZones, setShowZones] = useState(true);
  const [isDrawingZone, setIsDrawingZone] = useState(false);
  const [zoneKey, setZoneKey] = useState(0); // Force re-render of zones

  // Load weather data
  const loadWeatherData = useCallback(async () => {
    setIsLoadingWeather(true);
    try {
      const data = await apiClient.getWindField({
        lat_min: 30,
        lat_max: 60,
        lon_min: -15,
        lon_max: 40,
        resolution: 1.0,
      });
      setWindData(data);
    } catch (error) {
      console.error('Failed to load weather:', error);
    } finally {
      setIsLoadingWeather(false);
    }
  }, []);

  // Load weather on mount
  useEffect(() => {
    loadWeatherData();
  }, [loadWeatherData]);

  // Handle RTZ import
  const handleRouteImport = (importedWaypoints: Position[], name: string) => {
    setWaypoints(importedWaypoints);
    setRouteName(name);
    setVoyageResult(null);
    setViewMode('edit');
  };

  // Handle loading saved route
  const handleLoadRoute = (loadedWaypoints: Position[]) => {
    setWaypoints(loadedWaypoints);
    setVoyageResult(null);
    setViewMode('edit');
    setIsEditing(true);
  };

  // Clear route
  const handleClearRoute = () => {
    setWaypoints([]);
    setRouteName('Custom Route');
    setVoyageResult(null);
    setViewMode('edit');
  };

  // Calculate voyage
  const handleCalculate = async () => {
    if (waypoints.length < 2) {
      alert('Please add at least 2 waypoints');
      return;
    }

    setIsCalculating(true);
    try {
      const result = await apiClient.calculateVoyage({
        waypoints,
        calm_speed_kts: calmSpeed,
        is_laden: isLaden,
        use_weather: useWeather,
      });
      setVoyageResult(result);
      setViewMode('results');
      setIsEditing(false);
    } catch (error) {
      console.error('Voyage calculation failed:', error);
      alert('Voyage calculation failed. Please check the backend is running.');
    } finally {
      setIsCalculating(false);
    }
  };

  // Optimize route using A*
  const handleOptimize = async () => {
    if (waypoints.length < 2) {
      alert('Please add at least 2 waypoints (origin and destination)');
      return;
    }

    setIsOptimizing(true);
    setOptimizationResult(null);

    try {
      const result = await apiClient.optimizeRoute({
        origin: waypoints[0],
        destination: waypoints[waypoints.length - 1],
        calm_speed_kts: calmSpeed,
        is_laden: isLaden,
        optimization_target: 'fuel',
        grid_resolution_deg: 0.5,
      });

      setOptimizationResult(result);
      setShowOptimizedRoute(true);

      // Show a summary alert
      const savings = result.fuel_savings_pct > 0
        ? `Fuel savings: ${result.fuel_savings_pct.toFixed(1)}%`
        : 'No savings found (direct route is optimal)';

      alert(`Route optimized!\n\n${savings}\nCells explored: ${result.cells_explored}\nTime: ${result.optimization_time_ms.toFixed(0)}ms`);

    } catch (error) {
      console.error('Route optimization failed:', error);
      alert('Route optimization failed. Please check the backend is running.');
    } finally {
      setIsOptimizing(false);
    }
  };

  // Apply optimized route as waypoints
  const applyOptimizedRoute = () => {
    if (optimizationResult) {
      setWaypoints(optimizationResult.waypoints);
      setOptimizationResult(null);
      setShowOptimizedRoute(false);
      setVoyageResult(null);
    }
  };

  // Save new zone
  const handleSaveZone = async (request: CreateZoneRequest) => {
    await apiClient.createZone(request);
    setZoneKey(prev => prev + 1); // Force zone layer to reload
    setIsDrawingZone(false);
  };

  // Calculate total distance
  const totalDistance = waypoints.reduce((sum, wp, i) => {
    if (i === 0) return 0;
    const prev = waypoints[i - 1];
    const R = 3440.065;
    const lat1 = (prev.lat * Math.PI) / 180;
    const lat2 = (wp.lat * Math.PI) / 180;
    const dlat = ((wp.lat - prev.lat) * Math.PI) / 180;
    const dlon = ((wp.lon - prev.lon) * Math.PI) / 180;
    const a =
      Math.sin(dlat / 2) ** 2 +
      Math.cos(lat1) * Math.cos(lat2) * Math.sin(dlon / 2) ** 2;
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return sum + R * c;
  }, 0);

  return (
    <div className="min-h-screen bg-gradient-maritime">
      <Header />

      <main className="container mx-auto px-4 pt-20 pb-8">
        {/* Title */}
        <div className="mb-4">
          <h1 className="text-3xl font-bold text-white">WINDMAR</h1>
          <p className="text-gray-400">Maritime Weather Routing System</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
          {/* Left Panel - Controls */}
          <div className="lg:col-span-1 space-y-4">
            {/* Route Input Card */}
            <Card title="Route" icon={<Navigation className="w-5 h-5" />}>
              {/* Mode Tabs */}
              <div className="flex space-x-2 mb-4">
                <button
                  onClick={() => setIsEditing(true)}
                  className={`flex-1 flex items-center justify-center space-x-1 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                    isEditing
                      ? 'bg-primary-500 text-white'
                      : 'bg-maritime-dark text-gray-400 hover:text-white'
                  }`}
                >
                  <MousePointer className="w-4 h-4" />
                  <span>Draw</span>
                </button>
                <button
                  onClick={() => setIsEditing(false)}
                  className={`flex-1 flex items-center justify-center space-x-1 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                    !isEditing
                      ? 'bg-primary-500 text-white'
                      : 'bg-maritime-dark text-gray-400 hover:text-white'
                  }`}
                >
                  <Upload className="w-4 h-4" />
                  <span>Import</span>
                </button>
              </div>

              {/* Import Section */}
              {!isEditing && (
                <div className="mb-4">
                  <RouteImport onImport={handleRouteImport} />
                  <div className="mt-2 text-center">
                    <SampleRTZButton />
                  </div>
                </div>
              )}

              {/* Waypoint List */}
              <WaypointList
                waypoints={waypoints}
                onWaypointsChange={setWaypoints}
                onClear={handleClearRoute}
                totalDistance={totalDistance}
              />

              {isEditing && waypoints.length === 0 && (
                <p className="text-xs text-gray-500 mt-2">
                  Click on the map to add waypoints
                </p>
              )}

              {/* Saved Routes */}
              <div className="mt-4 pt-4 border-t border-white/10">
                <SavedRoutes
                  currentWaypoints={waypoints}
                  onLoadRoute={handleLoadRoute}
                />
              </div>
            </Card>

            {/* Voyage Parameters Card */}
            <Card title="Voyage Parameters" icon={<Ship className="w-5 h-5" />}>
              <div className="space-y-4">
                {/* Calm Speed */}
                <div>
                  <label className="block text-sm text-gray-300 mb-2">
                    Calm Water Speed
                  </label>
                  <div className="flex items-center space-x-2">
                    <input
                      type="range"
                      min="8"
                      max="18"
                      step="0.5"
                      value={calmSpeed}
                      onChange={(e) => setCalmSpeed(parseFloat(e.target.value))}
                      className="flex-1"
                    />
                    <span className="w-16 text-right text-white font-semibold">
                      {calmSpeed} kts
                    </span>
                  </div>
                </div>

                {/* Loading Condition */}
                <div>
                  <label className="block text-sm text-gray-300 mb-2">
                    Loading Condition
                  </label>
                  <div className="flex space-x-2">
                    <button
                      onClick={() => setIsLaden(true)}
                      className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors ${
                        isLaden
                          ? 'bg-primary-500 text-white'
                          : 'bg-maritime-dark text-gray-400 hover:text-white'
                      }`}
                    >
                      Laden
                    </button>
                    <button
                      onClick={() => setIsLaden(false)}
                      className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors ${
                        !isLaden
                          ? 'bg-primary-500 text-white'
                          : 'bg-maritime-dark text-gray-400 hover:text-white'
                      }`}
                    >
                      Ballast
                    </button>
                  </div>
                </div>

                {/* Weather Toggle */}
                <div className="flex items-center justify-between p-3 bg-maritime-dark rounded-lg">
                  <div className="flex items-center space-x-2">
                    <Wind className="w-4 h-4 text-primary-400" />
                    <span className="text-sm text-white">Use Weather</span>
                  </div>
                  <button
                    onClick={() => setUseWeather(!useWeather)}
                    className={`relative w-10 h-6 rounded-full transition-colors ${
                      useWeather ? 'bg-primary-500' : 'bg-gray-600'
                    }`}
                  >
                    <span
                      className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-transform ${
                        useWeather ? 'left-5' : 'left-1'
                      }`}
                    />
                  </button>
                </div>

                {/* Calculate Button */}
                <button
                  onClick={handleCalculate}
                  disabled={isCalculating || waypoints.length < 2}
                  className="w-full flex items-center justify-center space-x-2 py-3 bg-gradient-to-r from-primary-500 to-ocean-500 text-white font-semibold rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isCalculating ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      <span>Calculating...</span>
                    </>
                  ) : (
                    <>
                      <Play className="w-5 h-5" />
                      <span>Calculate Voyage</span>
                    </>
                  )}
                </button>

                {/* Optimize Button */}
                <button
                  onClick={handleOptimize}
                  disabled={isOptimizing || waypoints.length < 2}
                  className="w-full flex items-center justify-center space-x-2 py-3 bg-gradient-to-r from-green-600 to-emerald-500 text-white font-semibold rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isOptimizing ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      <span>Optimizing...</span>
                    </>
                  ) : (
                    <>
                      <Compass className="w-5 h-5" />
                      <span>Optimize Route (A*)</span>
                    </>
                  )}
                </button>

                {/* Optimization Results */}
                {optimizationResult && (
                  <div className="mt-3 p-3 bg-green-500/10 border border-green-500/30 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-green-400">Route Optimized</span>
                      <TrendingDown className="w-4 h-4 text-green-400" />
                    </div>
                    <div className="text-xs text-gray-300 space-y-1">
                      <div className="flex justify-between">
                        <span>Fuel savings:</span>
                        <span className="text-green-400 font-semibold">
                          {optimizationResult.fuel_savings_pct.toFixed(1)}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Distance:</span>
                        <span>{optimizationResult.total_distance_nm.toFixed(0)} nm</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Waypoints:</span>
                        <span>{optimizationResult.waypoints.length}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Avg speed:</span>
                        <span>{optimizationResult.avg_speed_kts.toFixed(1)} kts</span>
                      </div>
                    </div>

                    {/* Speed Profile */}
                    {optimizationResult.variable_speed_enabled && optimizationResult.speed_profile.length > 1 && (
                      <div className="mt-2 pt-2 border-t border-white/10">
                        <div className="text-xs text-gray-400 mb-1">Speed Profile (per leg):</div>
                        <div className="flex items-end h-8 gap-px">
                          {optimizationResult.speed_profile.map((speed, i) => {
                            const minSpeed = Math.min(...optimizationResult.speed_profile);
                            const maxSpeed = Math.max(...optimizationResult.speed_profile);
                            const range = maxSpeed - minSpeed || 1;
                            const height = ((speed - minSpeed) / range) * 100;
                            return (
                              <div
                                key={i}
                                className="flex-1 bg-primary-500 rounded-t"
                                style={{ height: `${Math.max(20, height)}%` }}
                                title={`Leg ${i + 1}: ${speed.toFixed(1)} kts`}
                              />
                            );
                          })}
                        </div>
                        <div className="flex justify-between text-xs text-gray-500 mt-1">
                          <span>{Math.min(...optimizationResult.speed_profile).toFixed(0)} kts</span>
                          <span>{Math.max(...optimizationResult.speed_profile).toFixed(0)} kts</span>
                        </div>
                      </div>
                    )}

                    {/* Safety Assessment */}
                    {optimizationResult.safety && (
                      <div className={`mt-2 p-2 rounded ${
                        optimizationResult.safety.status === 'safe'
                          ? 'bg-green-500/20 border border-green-500/30'
                          : optimizationResult.safety.status === 'marginal'
                          ? 'bg-yellow-500/20 border border-yellow-500/30'
                          : 'bg-red-500/20 border border-red-500/30'
                      }`}>
                        <div className="flex items-center space-x-2 mb-1">
                          {optimizationResult.safety.status === 'safe' ? (
                            <CheckCircle className="w-4 h-4 text-green-400" />
                          ) : optimizationResult.safety.status === 'marginal' ? (
                            <AlertTriangle className="w-4 h-4 text-yellow-400" />
                          ) : (
                            <ShieldAlert className="w-4 h-4 text-red-400" />
                          )}
                          <span className={`text-xs font-medium ${
                            optimizationResult.safety.status === 'safe'
                              ? 'text-green-400'
                              : optimizationResult.safety.status === 'marginal'
                              ? 'text-yellow-400'
                              : 'text-red-400'
                          }`}>
                            {optimizationResult.safety.status === 'safe'
                              ? 'Safe Passage'
                              : optimizationResult.safety.status === 'marginal'
                              ? 'Marginal Conditions'
                              : 'Dangerous Conditions'}
                          </span>
                        </div>
                        <div className="text-xs text-gray-400 space-y-0.5">
                          <div>Max Roll: {optimizationResult.safety.max_roll_deg.toFixed(1)}°</div>
                          <div>Max Pitch: {optimizationResult.safety.max_pitch_deg.toFixed(1)}°</div>
                        </div>
                        {optimizationResult.safety.warnings.length > 0 && (
                          <div className="mt-1 text-xs text-yellow-400">
                            {optimizationResult.safety.warnings.slice(0, 2).map((w, i) => (
                              <div key={i} className="truncate">• {w}</div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}

                    <button
                      onClick={applyOptimizedRoute}
                      className="mt-2 w-full py-2 text-sm bg-green-600 text-white rounded hover:bg-green-500 transition-colors"
                    >
                      Apply Optimized Route
                    </button>
                  </div>
                )}
              </div>
            </Card>

            {/* Weather Layer Controls */}
            <Card title="Weather Display" icon={<Wind className="w-5 h-5" />}>
              <div className="space-y-2">
                <WeatherLayerButton
                  icon={<Wind className="w-4 h-4" />}
                  label="Wind"
                  active={weatherLayer === 'wind'}
                  onClick={() => setWeatherLayer(weatherLayer === 'wind' ? 'none' : 'wind')}
                />
                <WeatherLayerButton
                  icon={<Waves className="w-4 h-4" />}
                  label="Waves"
                  active={weatherLayer === 'waves'}
                  onClick={() => setWeatherLayer(weatherLayer === 'waves' ? 'none' : 'waves')}
                  disabled
                />
                <button
                  onClick={loadWeatherData}
                  disabled={isLoadingWeather}
                  className="w-full flex items-center justify-center space-x-2 py-2 text-sm text-gray-400 hover:text-white transition-colors"
                >
                  <RefreshCw className={`w-4 h-4 ${isLoadingWeather ? 'animate-spin' : ''}`} />
                  <span>Refresh Weather</span>
                </button>
              </div>
            </Card>

            {/* Vessel Calibration */}
            <CalibrationPanel />

            {/* Regulatory Zones */}
            <Card title="Regulatory Zones" icon={<Shield className="w-5 h-5" />}>
              <div className="space-y-2">
                <WeatherLayerButton
                  icon={<Shield className="w-4 h-4" />}
                  label="Show Zones"
                  active={showZones}
                  onClick={() => setShowZones(!showZones)}
                />
                <button
                  onClick={() => setIsDrawingZone(!isDrawingZone)}
                  className={`w-full flex items-center justify-between px-3 py-2 rounded-lg transition-colors ${
                    isDrawingZone
                      ? 'bg-amber-500/20 border border-amber-500/50 text-amber-400'
                      : 'bg-maritime-dark text-gray-400 hover:text-white'
                  }`}
                >
                  <div className="flex items-center space-x-2">
                    <PenTool className="w-4 h-4" />
                    <span className="text-sm">Draw Custom Zone</span>
                  </div>
                </button>
                {isDrawingZone && (
                  <p className="text-xs text-amber-400 px-2">
                    Click on the map to draw a polygon, then fill in zone properties.
                  </p>
                )}
              </div>
            </Card>
          </div>

          {/* Center - Map */}
          <div className="lg:col-span-2">
            <Card className="h-[calc(100vh-180px)] min-h-[500px]">
              <MapComponent
                waypoints={waypoints}
                onWaypointsChange={setWaypoints}
                isEditing={isEditing}
                windData={weatherLayer === 'wind' ? windData : null}
                showZones={showZones}
                zoneKey={zoneKey}
                isDrawingZone={isDrawingZone}
                onSaveZone={handleSaveZone}
                onCancelZone={() => setIsDrawingZone(false)}
              />
            </Card>
          </div>

          {/* Right Panel - Results */}
          <div className="lg:col-span-1">
            {voyageResult ? (
              <div className="space-y-4">
                <VoyageResults voyage={voyageResult} />
                <VoyageProfile voyage={voyageResult} />
              </div>
            ) : (
              <Card>
                <div className="text-center py-12">
                  <Ship className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                  <h3 className="text-lg font-semibold text-gray-400 mb-2">
                    No Voyage Calculated
                  </h3>
                  <p className="text-sm text-gray-500">
                    Add waypoints and click "Calculate Voyage" to see results with
                    per-leg SOG, ETA, and weather conditions.
                  </p>
                </div>
              </Card>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

// Weather layer toggle button
function WeatherLayerButton({
  icon,
  label,
  active,
  onClick,
  disabled,
}: {
  icon: React.ReactNode;
  label: string;
  active: boolean;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`w-full flex items-center justify-between px-3 py-2 rounded-lg transition-colors ${
        active
          ? 'bg-primary-500/20 border border-primary-500/50 text-primary-400'
          : 'bg-maritime-dark text-gray-400 hover:text-white'
      } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
    >
      <div className="flex items-center space-x-2">
        {icon}
        <span className="text-sm">{label}</span>
      </div>
      {active ? (
        <Eye className="w-4 h-4" />
      ) : (
        <EyeOff className="w-4 h-4" />
      )}
    </button>
  );
}

// Map component wrapper
function MapComponent({
  waypoints,
  onWaypointsChange,
  isEditing,
  windData,
  showZones = true,
  zoneKey = 0,
  isDrawingZone = false,
  onSaveZone,
  onCancelZone,
}: {
  waypoints: Position[];
  onWaypointsChange: (wps: Position[]) => void;
  isEditing: boolean;
  windData: WindFieldData | null;
  showZones?: boolean;
  zoneKey?: number;
  isDrawingZone?: boolean;
  onSaveZone?: (request: CreateZoneRequest) => Promise<void>;
  onCancelZone?: () => void;
}) {
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  if (!isMounted) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-maritime-dark rounded-lg">
        <Loader2 className="w-8 h-8 animate-spin text-primary-400" />
      </div>
    );
  }

  return (
    <MapContainer
      center={DEFAULT_CENTER}
      zoom={DEFAULT_ZOOM}
      style={{ height: '100%', width: '100%' }}
      className="rounded-lg"
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
      />

      {/* Zone Layer */}
      {showZones && <ZoneLayer key={zoneKey} visible={showZones} />}

      {/* Zone Editor (drawing) */}
      {isDrawingZone && onSaveZone && onCancelZone && (
        <ZoneEditor
          isDrawing={isDrawingZone}
          onSaveZone={onSaveZone}
          onCancel={onCancelZone}
        />
      )}

      {/* Wind Layer */}
      {windData && <WindLayer windData={windData} showArrows />}

      {/* Waypoint Editor */}
      <WaypointEditor
        waypoints={waypoints}
        onWaypointsChange={onWaypointsChange}
        isEditing={isEditing}
      />
    </MapContainer>
  );
}

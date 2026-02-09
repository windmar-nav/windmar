'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import dynamic from 'next/dynamic';
import Header from '@/components/Header';
import Card from '@/components/Card';
import TabPanel, { ActiveTab } from '@/components/TabPanel';
import RouteTab from '@/components/RouteTab';
import AnalysisTab from '@/components/AnalysisTab';
import MapOverlayControls from '@/components/MapOverlayControls';
import { apiClient, Position, WindFieldData, WaveFieldData, VelocityData, VoyageResponse, OptimizationResponse, CreateZoneRequest, WaveForecastFrames } from '@/lib/api';
import { getAnalyses, saveAnalysis, deleteAnalysis, updateAnalysisMonteCarlo, AnalysisEntry } from '@/lib/analysisStorage';

// Dynamic import for MapComponent (client-side only)
const MapComponent = dynamic(() => import('@/components/MapComponent'), { ssr: false });

type WeatherLayer = 'wind' | 'waves' | 'currents' | 'none';

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
  const [isCalculating, setIsCalculating] = useState(false);

  // Optimization
  const [optimizationResult, setOptimizationResult] = useState<OptimizationResponse | null>(null);
  const [isOptimizing, setIsOptimizing] = useState(false);

  // Weather visualization
  const [weatherLayer, setWeatherLayer] = useState<WeatherLayer>('wind');
  const [windData, setWindData] = useState<WindFieldData | null>(null);
  const [waveData, setWaveData] = useState<WaveFieldData | null>(null);
  const [windVelocityData, setWindVelocityData] = useState<VelocityData[] | null>(null);
  const [currentVelocityData, setCurrentVelocityData] = useState<VelocityData[] | null>(null);
  const [isLoadingWeather, setIsLoadingWeather] = useState(false);

  // Viewport state
  const [viewport, setViewport] = useState<{
    bounds: { lat_min: number; lat_max: number; lon_min: number; lon_max: number };
    zoom: number;
  } | null>(null);

  // Forecast timeline state
  const [forecastEnabled, setForecastEnabled] = useState(false);
  const [forecastHour, setForecastHour] = useState(0);

  // Zone state
  const [showZones, setShowZones] = useState(true);
  const [isDrawingZone, setIsDrawingZone] = useState(false);
  const [zoneKey, setZoneKey] = useState(0);

  // Tab state
  const [activeTab, setActiveTab] = useState<ActiveTab>('route');

  // Analysis state
  const [analyses, setAnalyses] = useState<AnalysisEntry[]>([]);
  const [displayedAnalysisId, setDisplayedAnalysisId] = useState<string | null>(null);
  const [simulatingId, setSimulatingId] = useState<string | null>(null);

  // Load analyses from localStorage on mount
  useEffect(() => {
    setAnalyses(getAnalyses());
  }, []);

  // Derive weather resolution from zoom level
  const getResolutionForZoom = (zoom: number): number => {
    if (zoom <= 4) return 2.0;
    if (zoom <= 6) return 1.0;
    return 0.5;
  };

  // Keep a stable ref to the current viewport so callbacks don't need it as a dep
  const viewportRef = useRef(viewport);
  useEffect(() => { viewportRef.current = viewport; }, [viewport]);

  // Load weather data for current viewport (stable — no object deps)
  const loadWeatherData = useCallback(async (vp?: typeof viewport) => {
    const v = vp || viewportRef.current;
    if (!v) return;

    setIsLoadingWeather(true);
    try {
      const params = {
        lat_min: v.bounds.lat_min,
        lat_max: v.bounds.lat_max,
        lon_min: v.bounds.lon_min,
        lon_max: v.bounds.lon_max,
        resolution: getResolutionForZoom(v.zoom),
      };
      const [wind, waves, windVel, currentVel] = await Promise.all([
        apiClient.getWindField(params),
        apiClient.getWaveField(params),
        apiClient.getWindVelocity(params),
        apiClient.getCurrentVelocity(params).catch(() => null),
      ]);
      setWindData(wind);
      setWaveData(waves);
      setWindVelocityData(windVel);
      setCurrentVelocityData(currentVel);
    } catch (error) {
      console.error('Failed to load weather:', error);
    } finally {
      setIsLoadingWeather(false);
    }
  }, []); // stable — reads viewport from ref

  // Reload weather when viewport changes
  useEffect(() => {
    if (viewport) {
      loadWeatherData(viewport);
    }
  }, [viewport]); // eslint-disable-line react-hooks/exhaustive-deps

  // Handle wind forecast hour change from timeline (stable)
  const handleForecastHourChange = useCallback((hour: number, data: VelocityData[] | null) => {
    setForecastHour(hour);
    if (data) {
      setWindVelocityData(data);
    } else if (hour === 0) {
      loadWeatherData();
    }
  }, [loadWeatherData]);

  // Handle wave forecast hour change — construct WaveFieldData from frame
  const handleWaveForecastHourChange = useCallback((hour: number, allFrames: WaveForecastFrames | null) => {
    setForecastHour(hour);
    if (!allFrames || hour === 0) {
      if (hour === 0) loadWeatherData();
      return;
    }
    const frame = allFrames.frames[String(hour)];
    if (!frame) return;

    const synth: WaveFieldData = {
      parameter: 'wave_height',
      time: allFrames.run_time,
      bbox: {
        lat_min: allFrames.lats[0],
        lat_max: allFrames.lats[allFrames.lats.length - 1],
        lon_min: allFrames.lons[0],
        lon_max: allFrames.lons[allFrames.lons.length - 1],
      },
      resolution: allFrames.lats.length > 1 ? Math.abs(allFrames.lats[1] - allFrames.lats[0]) : 1,
      nx: allFrames.nx,
      ny: allFrames.ny,
      lats: allFrames.lats,
      lons: allFrames.lons,
      data: frame.data,
      unit: 'm',
      direction: frame.direction,
      has_decomposition: !!frame.windwave && !!frame.swell,
      windwave: frame.windwave,
      swell: frame.swell,
      ocean_mask: allFrames.ocean_mask,
      ocean_mask_lats: allFrames.ocean_mask_lats,
      ocean_mask_lons: allFrames.ocean_mask_lons,
      colorscale: allFrames.colorscale,
    };
    setWaveData(synth);
  }, [loadWeatherData]);

  // Handle RTZ import
  const handleRouteImport = (importedWaypoints: Position[], name: string) => {
    setWaypoints(importedWaypoints);
    setRouteName(name);
    setDisplayedAnalysisId(null);
  };

  // Handle loading saved route
  const handleLoadRoute = (loadedWaypoints: Position[]) => {
    setWaypoints(loadedWaypoints);
    setIsEditing(true);
    setDisplayedAnalysisId(null);
  };

  // Clear route
  const handleClearRoute = () => {
    setWaypoints([]);
    setRouteName('Custom Route');
    setDisplayedAnalysisId(null);
  };

  // Calculate voyage → create analysis entry
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

      // Save analysis entry
      const entry = saveAnalysis(
        routeName,
        waypoints,
        { calmSpeed, isLaden, useWeather },
        result,
      );

      // Refresh analyses list and switch to analysis tab
      setAnalyses(getAnalyses());
      setDisplayedAnalysisId(entry.id);
      setActiveTab('analysis');
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

      const savings = result.fuel_savings_pct > 0
        ? `Fuel savings: ${result.fuel_savings_pct.toFixed(1)}%`
        : 'No savings found (direct route is optimal)';

      // Weather provenance info
      let wxInfo = '';
      if (result.temporal_weather && result.weather_provenance?.length) {
        const sources = result.weather_provenance.map(
          (p) => `${p.model_name}: ${p.confidence} confidence (${p.forecast_lead_hours}h lead)`
        ).join('\n  ');
        wxInfo = `\n\nWeather: Time-varying (temporal)\n  ${sources}`;
      } else {
        wxInfo = '\n\nWeather: Single snapshot';
      }

      alert(`Route optimized!\n\n${savings}\nCells explored: ${result.cells_explored}\nTime: ${result.optimization_time_ms.toFixed(0)}ms${wxInfo}`);

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
      setDisplayedAnalysisId(null);
    }
  };

  // Save new zone
  const handleSaveZone = async (request: CreateZoneRequest) => {
    await apiClient.createZone(request);
    setZoneKey(prev => prev + 1);
    setIsDrawingZone(false);
  };

  // Analysis actions
  const handleShowOnMap = (id: string) => {
    if (displayedAnalysisId === id) {
      setDisplayedAnalysisId(null);
    } else {
      setDisplayedAnalysisId(id);
      // Load the analysis waypoints onto the map
      const analysis = analyses.find(a => a.id === id);
      if (analysis) {
        setWaypoints(analysis.waypoints);
        setIsEditing(false);
      }
    }
  };

  const handleDeleteAnalysis = (id: string) => {
    deleteAnalysis(id);
    setAnalyses(getAnalyses());
    if (displayedAnalysisId === id) {
      setDisplayedAnalysisId(null);
    }
  };

  const handleRunSimulation = async (id: string) => {
    const analysis = analyses.find(a => a.id === id);
    if (!analysis) return;

    setSimulatingId(id);
    try {
      const mcResult = await apiClient.runMonteCarlo({
        waypoints: analysis.waypoints,
        calm_speed_kts: analysis.parameters.calmSpeed,
        is_laden: analysis.parameters.isLaden,
        departure_time: analysis.parameters.departureTime,
        n_simulations: 100,
      });

      updateAnalysisMonteCarlo(id, mcResult);
      setAnalyses(getAnalyses());
    } catch (error) {
      console.error('Monte Carlo simulation failed:', error);
      alert('Monte Carlo simulation failed. Please check the backend is running.');
    } finally {
      setSimulatingId(null);
    }
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

  // Determine if map should show editable waypoints
  const mapEditing = activeTab === 'route' ? isEditing : false;

  return (
    <div className="min-h-screen bg-gradient-maritime">
      <Header />

      <main className="px-4 pt-20 pb-4 h-screen flex flex-col">
        <div className="flex-1 grid grid-cols-1 lg:grid-cols-[360px_1fr] gap-3 min-h-0">
          {/* Left — Tab Panel */}
          <div className="min-h-0 bg-maritime-medium/50 backdrop-blur-sm rounded-lg border border-white/5 overflow-hidden">
            <TabPanel activeTab={activeTab} onTabChange={setActiveTab}>
              {activeTab === 'route' ? (
                <RouteTab
                  waypoints={waypoints}
                  onWaypointsChange={setWaypoints}
                  isEditing={isEditing}
                  onIsEditingChange={setIsEditing}
                  routeName={routeName}
                  onRouteImport={handleRouteImport}
                  onLoadRoute={handleLoadRoute}
                  onClearRoute={handleClearRoute}
                  totalDistance={totalDistance}
                  calmSpeed={calmSpeed}
                  onCalmSpeedChange={setCalmSpeed}
                  isLaden={isLaden}
                  onIsLadenChange={setIsLaden}
                  useWeather={useWeather}
                  onUseWeatherChange={setUseWeather}
                  isCalculating={isCalculating}
                  onCalculate={handleCalculate}
                  isOptimizing={isOptimizing}
                  onOptimize={handleOptimize}
                  optimizationResult={optimizationResult}
                  onApplyOptimizedRoute={applyOptimizedRoute}
                  showZones={showZones}
                  onShowZonesChange={setShowZones}
                  isDrawingZone={isDrawingZone}
                  onIsDrawingZoneChange={setIsDrawingZone}
                />
              ) : (
                <AnalysisTab
                  analyses={analyses}
                  displayedAnalysisId={displayedAnalysisId}
                  onShowOnMap={handleShowOnMap}
                  onDelete={handleDeleteAnalysis}
                  onRunSimulation={handleRunSimulation}
                  simulatingId={simulatingId}
                />
              )}
            </TabPanel>
          </div>

          {/* Right — Map */}
          <div className="min-h-0">
            <Card className="h-full min-h-[500px]">
              <MapComponent
                waypoints={waypoints}
                onWaypointsChange={setWaypoints}
                isEditing={mapEditing}
                weatherLayer={weatherLayer}
                windData={windData}
                waveData={waveData}
                windVelocityData={windVelocityData}
                currentVelocityData={currentVelocityData}
                showZones={showZones}
                zoneKey={zoneKey}
                isDrawingZone={isDrawingZone}
                onSaveZone={handleSaveZone}
                onCancelZone={() => setIsDrawingZone(false)}
                forecastEnabled={forecastEnabled}
                onForecastClose={() => setForecastEnabled(false)}
                onForecastHourChange={handleForecastHourChange}
                onWaveForecastHourChange={handleWaveForecastHourChange}
                onViewportChange={setViewport}
                viewportBounds={viewport?.bounds ?? null}
              >
                <MapOverlayControls
                  weatherLayer={weatherLayer}
                  onWeatherLayerChange={setWeatherLayer}
                  forecastEnabled={forecastEnabled}
                  onForecastToggle={() => setForecastEnabled(!forecastEnabled)}
                  isLoadingWeather={isLoadingWeather}
                  onRefresh={loadWeatherData}
                />
              </MapComponent>
            </Card>
          </div>
        </div>
      </main>
    </div>
  );
}

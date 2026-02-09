'use client';

import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import dynamic from 'next/dynamic';
import Header from '@/components/Header';
import MapOverlayControls from '@/components/MapOverlayControls';
import RouteIndicatorPanel from '@/components/RouteIndicatorPanel';
import AnalysisSlidePanel from '@/components/AnalysisSlidePanel';
import { useVoyage } from '@/components/VoyageContext';
import { apiClient, Position, WindFieldData, WaveFieldData, VelocityData, VoyageResponse, OptimizationResponse, CreateZoneRequest, WaveForecastFrames } from '@/lib/api';
import { getAnalyses, saveAnalysis, deleteAnalysis, updateAnalysisMonteCarlo, AnalysisEntry } from '@/lib/analysisStorage';
import { debugLog } from '@/lib/debugLog';
import DebugConsole from '@/components/DebugConsole';

const MapComponent = dynamic(() => import('@/components/MapComponent'), { ssr: false });

type WeatherLayer = 'wind' | 'waves' | 'currents' | 'none';

export default function HomePage() {
  // Voyage context (shared with header dropdowns)
  const { calmSpeed, isLaden, useWeather, zoneVisibility, isDrawingZone, setIsDrawingZone } = useVoyage();

  // Route state
  const [waypoints, setWaypoints] = useState<Position[]>([]);
  const [isEditing, setIsEditing] = useState(true);
  const [routeName, setRouteName] = useState('Custom Route');

  // Results
  const [isCalculating, setIsCalculating] = useState(false);

  // Optimization
  const [optimizationResult, setOptimizationResult] = useState<OptimizationResponse | null>(null);
  const [isOptimizing, setIsOptimizing] = useState(false);

  // Weather visualization
  const [weatherLayer, setWeatherLayer] = useState<WeatherLayer>('none');
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
  const [zoneKey, setZoneKey] = useState(0);

  // Analysis state
  const [analysisOpen, setAnalysisOpen] = useState(false);
  const [analyses, setAnalyses] = useState<AnalysisEntry[]>([]);
  const [displayedAnalysisId, setDisplayedAnalysisId] = useState<string | null>(null);
  const [simulatingId, setSimulatingId] = useState<string | null>(null);

  // Load analyses from localStorage on mount
  useEffect(() => {
    setAnalyses(getAnalyses());
  }, []);

  // Compute visible zone types from context
  const visibleZoneTypes = useMemo(() => {
    return Object.entries(zoneVisibility)
      .filter(([, visible]) => visible)
      .map(([type]) => type);
  }, [zoneVisibility]);

  // Derive weather resolution from zoom level
  const getResolutionForZoom = (zoom: number): number => {
    if (zoom <= 4) return 2.0;
    if (zoom <= 6) return 1.0;
    return 0.5;
  };

  // Keep stable refs so callbacks don't need these as deps
  const viewportRef = useRef(viewport);
  useEffect(() => { viewportRef.current = viewport; }, [viewport]);

  const weatherLayerRef = useRef(weatherLayer);
  useEffect(() => { weatherLayerRef.current = weatherLayer; }, [weatherLayer]);

  const waypointsRef = useRef(waypoints);
  useEffect(() => { waypointsRef.current = waypoints; }, [waypoints]);

  // Route-aware bounds helpers
  function getRouteBounds(wps: Position[], margin: number = 3) {
    if (wps.length === 0) return null;
    let latMin = Infinity, latMax = -Infinity, lonMin = Infinity, lonMax = -Infinity;
    for (const wp of wps) {
      latMin = Math.min(latMin, wp.lat);
      latMax = Math.max(latMax, wp.lat);
      lonMin = Math.min(lonMin, wp.lon);
      lonMax = Math.max(lonMax, wp.lon);
    }
    return {
      lat_min: Math.max(-85, latMin - margin),
      lat_max: Math.min(85, latMax + margin),
      lon_min: lonMin - margin,
      lon_max: lonMax + margin,
    };
  }

  function unionBounds(
    a: { lat_min: number; lat_max: number; lon_min: number; lon_max: number },
    b: { lat_min: number; lat_max: number; lon_min: number; lon_max: number } | null,
  ) {
    if (!b) return a;
    return {
      lat_min: Math.min(a.lat_min, b.lat_min),
      lat_max: Math.max(a.lat_max, b.lat_max),
      lon_min: Math.min(a.lon_min, b.lon_min),
      lon_max: Math.max(a.lon_max, b.lon_max),
    };
  }

  // Load weather data
  const loadWeatherData = useCallback(async (vp?: typeof viewport, layer?: WeatherLayer) => {
    const v = vp || viewportRef.current;
    const activeLayer = layer ?? weatherLayerRef.current;
    if (!v || activeLayer === 'none') return;

    const routeBounds = getRouteBounds(waypointsRef.current);
    const effectiveBounds = unionBounds(v.bounds, routeBounds);

    const params = {
      lat_min: effectiveBounds.lat_min,
      lat_max: effectiveBounds.lat_max,
      lon_min: effectiveBounds.lon_min,
      lon_max: effectiveBounds.lon_max,
      resolution: getResolutionForZoom(v.zoom),
    };

    setIsLoadingWeather(true);
    const t0 = performance.now();
    debugLog('info', 'API', `Loading ${activeLayer} weather: zoom=${v.zoom}, bbox=[${params.lat_min.toFixed(1)},${params.lat_max.toFixed(1)},${params.lon_min.toFixed(1)},${params.lon_max.toFixed(1)}]`);
    try {
      if (activeLayer === 'wind') {
        const [wind, windVel] = await Promise.all([
          apiClient.getWindField(params),
          apiClient.getWindVelocity(params),
        ]);
        const dt = (performance.now() - t0).toFixed(0);
        debugLog('info', 'API', `Wind loaded in ${dt}ms: grid=${wind?.ny}x${wind?.nx}`);
        setWindData(wind);
        setWindVelocityData(windVel);
      } else if (activeLayer === 'waves') {
        const waves = await apiClient.getWaveField(params);
        const dt = (performance.now() - t0).toFixed(0);
        debugLog('info', 'API', `Waves loaded in ${dt}ms: grid=${waves?.ny}x${waves?.nx}`);
        setWaveData(waves);
      } else if (activeLayer === 'currents') {
        const currentVel = await apiClient.getCurrentVelocity(params).catch(() => null);
        const dt = (performance.now() - t0).toFixed(0);
        debugLog('info', 'API', `Currents loaded in ${dt}ms: ${currentVel ? 'yes' : 'no data'}`);
        setCurrentVelocityData(currentVel);
      }
    } catch (error) {
      debugLog('error', 'API', `Weather load failed: ${error}`);
    } finally {
      setIsLoadingWeather(false);
    }
  }, []);

  // Reload weather when viewport or active layer changes
  useEffect(() => {
    if (viewport && weatherLayer !== 'none') {
      loadWeatherData(viewport, weatherLayer);
    }
  }, [viewport, weatherLayer]); // eslint-disable-line react-hooks/exhaustive-deps

  // Handle wind forecast hour change from timeline
  const handleForecastHourChange = useCallback((hour: number, data: VelocityData[] | null) => {
    setForecastHour(hour);
    if (data) {
      setWindVelocityData(data);
    } else if (hour === 0) {
      loadWeatherData();
    }
  }, [loadWeatherData]);

  // Handle wave forecast hour change
  const handleWaveForecastHourChange = useCallback((hour: number, allFrames: WaveForecastFrames | null) => {
    setForecastHour(hour);
    if (!allFrames) {
      debugLog('warn', 'WAVE', `Hour ${hour}: no frame data available`);
      if (hour === 0) loadWeatherData();
      return;
    }
    const frame = allFrames.frames[String(hour)];
    if (!frame) {
      debugLog('warn', 'WAVE', `Hour ${hour}: frame not found in ${Object.keys(allFrames.frames).length} frames`);
      return;
    }

    const midRow = Math.floor((frame.data?.length || 0) / 2);
    const sample = frame.data?.[midRow]?.[0]?.toFixed(2) ?? 'null';
    debugLog('info', 'WAVE', `Frame T+${hour}h: sample=${sample}, grid=${allFrames.ny}x${allFrames.nx}`);

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

  // Handle current forecast hour change
  const handleCurrentForecastHourChange = useCallback((hour: number, allFrames: any | null) => {
    setForecastHour(hour);
    if (!allFrames) {
      debugLog('warn', 'CURRENT', `Hour ${hour}: no frame data available`);
      if (hour === 0) loadWeatherData();
      return;
    }
    const frame = allFrames.frames?.[String(hour)];
    if (!frame || !frame.u || !frame.v) {
      debugLog('warn', 'CURRENT', `Hour ${hour}: frame not found or missing u/v`);
      return;
    }
    debugLog('info', 'CURRENT', `Frame T+${hour}h: u_sample=${frame.u?.[Math.floor(frame.u.length/2)]?.[0]?.toFixed(3)}`);

    const lats = allFrames.lats as number[];
    const lons = allFrames.lons as number[];
    const ny = lats.length;
    const nx = lons.length;
    const uFlat: number[] = [];
    const vFlat: number[] = [];
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        uFlat.push(frame.u[j]?.[i] ?? 0);
        vFlat.push(frame.v[j]?.[i] ?? 0);
      }
    }
    const header = {
      parameterCategory: 2,
      parameterNumber: 2,
      lo1: lons[0],
      la1: lats[ny - 1],
      lo2: lons[nx - 1],
      la2: lats[0],
      dx: lons.length > 1 ? Math.abs(lons[1] - lons[0]) : 1,
      dy: lats.length > 1 ? Math.abs(lats[1] - lats[0]) : 1,
      nx,
      ny,
      refTime: allFrames.run_time || '',
    };
    setCurrentVelocityData([
      { header: { ...header, parameterNumber: 2 }, data: uFlat },
      { header: { ...header, parameterNumber: 3 }, data: vFlat },
    ]);
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

  // Calculate voyage
  const handleCalculate = async () => {
    if (waypoints.length < 2) {
      alert('Please add at least 2 waypoints');
      return;
    }

    setIsCalculating(true);
    const t0 = performance.now();
    debugLog('info', 'VOYAGE', `Start Calculation: ${waypoints.length} waypoints, speed=${calmSpeed}kts, weather=${useWeather}`);
    try {
      const result = await apiClient.calculateVoyage({
        waypoints,
        calm_speed_kts: calmSpeed,
        is_laden: isLaden,
        use_weather: useWeather,
      });
      const dt = ((performance.now() - t0) / 1000).toFixed(1);
      debugLog('info', 'VOYAGE', `Calculation completed in ${dt}s: ${result.total_distance_nm}nm, ${result.total_time_hours.toFixed(1)}h, ${result.total_fuel_mt.toFixed(1)}mt fuel`);

      const entry = saveAnalysis(
        routeName,
        waypoints,
        { calmSpeed, isLaden, useWeather },
        result,
      );

      setAnalyses(getAnalyses());
      setDisplayedAnalysisId(entry.id);
      setAnalysisOpen(true);
    } catch (error) {
      const dt = ((performance.now() - t0) / 1000).toFixed(1);
      debugLog('error', 'VOYAGE', `Calculation failed after ${dt}s: ${error}`);
      alert('Voyage calculation failed. Please check the backend is running.');
    } finally {
      setIsCalculating(false);
    }
  };

  // Optimize route
  const handleOptimize = async () => {
    if (waypoints.length < 2) {
      alert('Please add at least 2 waypoints (origin and destination)');
      return;
    }

    setIsOptimizing(true);
    setOptimizationResult(null);

    const t0 = performance.now();
    debugLog('info', 'ROUTE', `Optimizing: ${waypoints[0].lat.toFixed(2)},${waypoints[0].lon.toFixed(2)} → ${waypoints[waypoints.length-1].lat.toFixed(2)},${waypoints[waypoints.length-1].lon.toFixed(2)}, speed=${calmSpeed}kts`);

    try {
      const result = await apiClient.optimizeRoute({
        origin: waypoints[0],
        destination: waypoints[waypoints.length - 1],
        calm_speed_kts: calmSpeed,
        is_laden: isLaden,
        optimization_target: 'fuel',
        grid_resolution_deg: 0.5,
        max_time_factor: 1.15,
      });

      const dt = ((performance.now() - t0) / 1000).toFixed(1);
      debugLog('info', 'ROUTE', `Optimized in ${dt}s: ${result.waypoints.length} waypoints, ${result.cells_explored} cells, temporal=${result.temporal_weather}, savings=${result.fuel_savings_pct?.toFixed(1)}%`);
      setOptimizationResult(result);

      const savings = result.fuel_savings_pct > 0
        ? `Fuel savings: ${result.fuel_savings_pct.toFixed(1)}%`
        : 'No savings found (direct route is optimal)';

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
      debugLog('error', 'ROUTE', `Optimization failed: ${error}`);
      alert('Route optimization failed. Please check the backend is running.');
    } finally {
      setIsOptimizing(false);
    }
  };

  // Apply optimized route
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

  // Get displayed analysis for route indicator
  const displayedAnalysis = displayedAnalysisId
    ? analyses.find(a => a.id === displayedAnalysisId)
    : null;

  return (
    <div className="min-h-screen bg-gradient-maritime">
      <Header />
      <DebugConsole />

      <main className="pt-16 h-screen">
        <div className="h-full">
          <MapComponent
            waypoints={waypoints}
            onWaypointsChange={setWaypoints}
            isEditing={isEditing}
            weatherLayer={weatherLayer}
            windData={windData}
            waveData={waveData}
            windVelocityData={windVelocityData}
            currentVelocityData={currentVelocityData}
            showZones={visibleZoneTypes.length > 0}
            visibleZoneTypes={visibleZoneTypes}
            zoneKey={zoneKey}
            isDrawingZone={isDrawingZone}
            onSaveZone={handleSaveZone}
            onCancelZone={() => setIsDrawingZone(false)}
            forecastEnabled={forecastEnabled}
            onForecastClose={() => setForecastEnabled(false)}
            onForecastHourChange={handleForecastHourChange}
            onWaveForecastHourChange={handleWaveForecastHourChange}
            onCurrentForecastHourChange={handleCurrentForecastHourChange}
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
              analysisOpen={analysisOpen}
              onAnalysisToggle={() => setAnalysisOpen(!analysisOpen)}
            />

            <RouteIndicatorPanel
              waypoints={waypoints}
              onWaypointsChange={setWaypoints}
              routeName={routeName}
              totalDistance={totalDistance}
              isEditing={isEditing}
              onIsEditingChange={setIsEditing}
              isCalculating={isCalculating}
              onCalculate={handleCalculate}
              isOptimizing={isOptimizing}
              onOptimize={handleOptimize}
              optimizationResult={optimizationResult}
              onApplyOptimizedRoute={applyOptimizedRoute}
              onRouteImport={handleRouteImport}
              onLoadRoute={handleLoadRoute}
              onClearRoute={handleClearRoute}
              analysisFuel={displayedAnalysis?.result.total_fuel_mt}
              analysisTime={displayedAnalysis?.result.total_time_hours}
              analysisAvgSpeed={displayedAnalysis?.result.avg_sog_kts}
            />

            <AnalysisSlidePanel
              open={analysisOpen}
              onClose={() => setAnalysisOpen(false)}
              analyses={analyses}
              displayedAnalysisId={displayedAnalysisId}
              onShowOnMap={handleShowOnMap}
              onDelete={handleDeleteAnalysis}
              onRunSimulation={handleRunSimulation}
              simulatingId={simulatingId}
            />
          </MapComponent>
        </div>
      </main>
    </div>
  );
}

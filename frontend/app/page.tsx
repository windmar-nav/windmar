'use client';

import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import dynamic from 'next/dynamic';
import Header from '@/components/Header';
import MapOverlayControls from '@/components/MapOverlayControls';
import AnalysisPanel from '@/components/AnalysisPanel';
import { useVoyage } from '@/components/VoyageContext';
import { apiClient, Position, OptimizationResponse, ParetoSolution, CreateZoneRequest, OptimizedRouteKey, AllOptimizationResults, EMPTY_ALL_RESULTS } from '@/lib/api';
import { getAnalyses, saveAnalysis, deleteAnalysis, updateAnalysisMonteCarlo, updateAnalysisOptimizations, AnalysisEntry } from '@/lib/analysisStorage';
import { debugLog } from '@/lib/debugLog';
import DebugConsole from '@/components/DebugConsole';
import { useToast } from '@/components/Toast';
import { useWeatherDisplay } from '@/hooks/useWeatherDisplay';

const MapComponent = dynamic(() => import('@/components/MapComponent'), { ssr: false });

export default function HomePage() {
  // Voyage context (shared with header dropdowns, persisted across navigation)
  const {
    viewMode, departureTime,
    calmSpeed, isLaden, useWeather,
    zoneVisibility, isDrawingZone, setIsDrawingZone,
    waypoints, setWaypoints,
    routeName, setRouteName,
    allResults, setAllResults,
    routeVisibility, setRouteVisibility,
    weatherLayer, setWeatherLayer,
    lastViewport, setLastViewport,
    gridResolution, variableResolution, paretoEnabled,
    variableSpeed,
  } = useVoyage();

  // Toast notifications
  const toast = useToast();

  // Ephemeral state (local to this page)
  const [isEditing, setIsEditing] = useState(true);
  const [isCalculating, setIsCalculating] = useState(false);
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [paretoFront, setParetoFront] = useState<ParetoSolution[] | null>(null);
  const [isRunningPareto, setIsRunningPareto] = useState(false);

  // Viewport state (init from context for cross-navigation persistence)
  const [viewport, setViewportLocal] = useState<{
    bounds: { lat_min: number; lat_max: number; lon_min: number; lon_max: number };
    zoom: number;
  } | null>(lastViewport);
  const setViewport = useCallback((vp: typeof viewport) => {
    setViewportLocal(vp);
    if (vp) setLastViewport(vp);
  }, [setLastViewport]);

  // Weather display (all state + handlers encapsulated in hook)
  const weather = useWeatherDisplay(weatherLayer, setWeatherLayer, viewport, waypoints.length);

  // Zone state
  const [zoneKey, setZoneKey] = useState(0);

  // Fit-to-route state
  const [fitBounds, setFitBounds] = useState<[[number, number], [number, number]] | null>(null);
  const [fitKey, setFitKey] = useState(0);

  // Analysis state
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

  // Fit map to route bounds
  const handleFitRoute = useCallback(() => {
    if (waypoints.length < 2) return;
    let latMin = Infinity, latMax = -Infinity, lonMin = Infinity, lonMax = -Infinity;
    for (const wp of waypoints) {
      latMin = Math.min(latMin, wp.lat);
      latMax = Math.max(latMax, wp.lat);
      lonMin = Math.min(lonMin, wp.lon);
      lonMax = Math.max(lonMax, wp.lon);
    }
    setFitBounds([[latMin, lonMin], [latMax, lonMax]]);
    setFitKey(prev => prev + 1);
  }, [waypoints]);

  // Clear route
  const handleClearRoute = () => {
    setWaypoints([]);
    setRouteName('Custom Route');
    setDisplayedAnalysisId(null);
    setAllResults(EMPTY_ALL_RESULTS);
  };

  // Get displayed analysis for route indicator and optimization baseline
  const displayedAnalysis = displayedAnalysisId
    ? analyses.find(a => a.id === displayedAnalysisId) ?? null
    : null;

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
        departure_time: departureTime || undefined,
        variable_speed: variableSpeed,
      });
      const dt = ((performance.now() - t0) / 1000).toFixed(1);
      debugLog('info', 'VOYAGE', `Calculation completed in ${dt}s: ${result.total_distance_nm}nm, ${result.total_time_hours.toFixed(1)}h, ${result.total_fuel_mt.toFixed(1)}mt fuel`);

      const entry = saveAnalysis(
        routeName,
        waypoints,
        { calmSpeed, isLaden, useWeather, departureTime: departureTime || undefined },
        result,
      );

      setAnalyses(getAnalyses());
      setDisplayedAnalysisId(entry.id);
    } catch (error) {
      const dt = ((performance.now() - t0) / 1000).toFixed(1);
      debugLog('error', 'VOYAGE', `Calculation failed after ${dt}s: ${error}`);
      alert('Voyage calculation failed. Please check the backend is running.');
    } finally {
      setIsCalculating(false);
    }
  };

  // Optimize route — always fire all 6 requests (2 engines x 3 weights)
  const handleOptimize = async () => {
    if (waypoints.length < 2) {
      alert('Please add at least 2 waypoints (origin and destination)');
      return;
    }

    setIsOptimizing(true);
    setAllResults(EMPTY_ALL_RESULTS);

    const t0 = performance.now();
    debugLog('info', 'ROUTE', `All-routes optimize: ${waypoints[0].lat.toFixed(2)},${waypoints[0].lon.toFixed(2)} → ${waypoints[waypoints.length-1].lat.toFixed(2)},${waypoints[waypoints.length-1].lon.toFixed(2)}, speed=${calmSpeed}kts`);

    const baseRequest = {
      origin: waypoints[0],
      destination: waypoints[waypoints.length - 1],
      calm_speed_kts: calmSpeed,
      is_laden: isLaden,
      departure_time: departureTime || undefined,
      optimization_target: 'fuel' as const,
      grid_resolution_deg: gridResolution,
      max_time_factor: 1.15,
      route_waypoints: waypoints.length > 2 ? waypoints : undefined,
      baseline_fuel_mt: displayedAnalysis?.result.total_fuel_mt,
      baseline_time_hours: displayedAnalysis?.result.total_time_hours,
      baseline_distance_nm: displayedAnalysis?.result.total_distance_nm,
      variable_resolution: variableResolution,
    };

    const combos: { engine: 'astar' | 'dijkstra'; weight: number; key: OptimizedRouteKey }[] = [
      { engine: 'astar', weight: 0.0, key: 'astar_fuel' },
      { engine: 'astar', weight: 0.5, key: 'astar_balanced' },
      { engine: 'astar', weight: 1.0, key: 'astar_safety' },
      { engine: 'dijkstra', weight: 0.0, key: 'dijkstra_fuel' },
      { engine: 'dijkstra', weight: 0.5, key: 'dijkstra_balanced' },
      { engine: 'dijkstra', weight: 1.0, key: 'dijkstra_safety' },
    ];

    try {
      // Run sequentially per engine to avoid saturating the weather
      // data connection pool (6 parallel requests overwhelm the backend).
      // Results are pushed to the UI progressively so the user sees
      // routes appear as they complete.
      const results = { ...EMPTY_ALL_RESULTS };

      for (const { engine, weight, key } of combos) {
        debugLog('info', 'ROUTE', `Firing ${engine} w=${weight}...`);
        try {
          const r = await apiClient.optimizeRoute({ ...baseRequest, engine, safety_weight: weight });
          results[key] = r as OptimizationResponse | null;
        } catch (err) {
          debugLog('warn', 'ROUTE', `${engine} w=${weight} failed: ${err}`);
          results[key] = null;
        }
        // Progressive update — show each result as it arrives
        setAllResults({ ...results });
      }

      const dt = ((performance.now() - t0) / 1000).toFixed(1);
      const ok = Object.values(results).filter(Boolean).length;
      debugLog('info', 'ROUTE', `All-routes done in ${dt}s: ${ok}/6 succeeded`);

      // Notify user if Dijkstra failed but A* succeeded
      const astarOk = [results.astar_fuel, results.astar_balanced, results.astar_safety].some(Boolean);
      const dijkstraOk = [results.dijkstra_fuel, results.dijkstra_balanced, results.dijkstra_safety].some(Boolean);
      if (astarOk && !dijkstraOk) {
        toast.warning('Dijkstra routes unavailable', 'Dijkstra engine could not find routes for this voyage. A* routes are shown.');
      }

      if (displayedAnalysisId && ok > 0) {
        updateAnalysisOptimizations(displayedAnalysisId, results);
        setAnalyses(getAnalyses());
      }
    } catch (error) {
      debugLog('error', 'ROUTE', `All-routes optimization failed: ${error}`);
    } finally {
      setIsOptimizing(false);
    }

    // Auto-trigger Pareto analysis if enabled in settings
    if (paretoEnabled) {
      handlePareto();
    }
  };

  // Run Pareto analysis (A* engine only)
  const handlePareto = async () => {
    if (waypoints.length < 2) return;
    setIsRunningPareto(true);
    setParetoFront(null);
    const t0 = performance.now();
    debugLog('info', 'ROUTE', 'Running Pareto analysis...');
    try {
      const r = await apiClient.optimizeRoute({
        origin: waypoints[0],
        destination: waypoints[waypoints.length - 1],
        calm_speed_kts: calmSpeed,
        is_laden: isLaden,
        departure_time: departureTime || undefined,
        optimization_target: 'fuel',
        grid_resolution_deg: gridResolution,
        max_time_factor: 1.15,
        route_waypoints: waypoints.length > 2 ? waypoints : undefined,
        baseline_fuel_mt: displayedAnalysis?.result.total_fuel_mt,
        baseline_time_hours: displayedAnalysis?.result.total_time_hours,
        baseline_distance_nm: displayedAnalysis?.result.total_distance_nm,
        variable_resolution: variableResolution,
        engine: 'astar',
        pareto: true,
      });
      const resp = r as OptimizationResponse;
      if (resp.pareto_front && resp.pareto_front.length > 0) {
        setParetoFront(resp.pareto_front);
        debugLog('info', 'ROUTE', `Pareto done in ${((performance.now() - t0) / 1000).toFixed(1)}s: ${resp.pareto_front.length} solutions`);
      } else {
        debugLog('warn', 'ROUTE', 'Pareto returned no solutions');
        toast.warning('No Pareto solutions found', 'The fuel-time trade-off space may be too narrow for current conditions.');
      }
    } catch (error) {
      debugLog('error', 'ROUTE', `Pareto analysis failed: ${error}`);
      toast.error('Pareto analysis failed', 'Check that the backend is running and try again.');
    } finally {
      setIsRunningPareto(false);
    }
  };

  // Apply optimized route from a specific key
  const applyOptimizedRoute = (key: OptimizedRouteKey) => {
    const result = allResults[key];
    if (result) {
      setWaypoints(result.waypoints);
      setAllResults(EMPTY_ALL_RESULTS);
      setDisplayedAnalysisId(null);
    }
  };

  // Dismiss optimized routes (keep original)
  const dismissOptimizedRoute = () => {
    setAllResults(EMPTY_ALL_RESULTS);
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
      setIsEditing(true);
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

  // Route coverage warning: warn when waypoints extend beyond forecast viewport
  const lastWarnedRouteRef = useRef<string>('');
  useEffect(() => {
    if (!weather.forecastEnabled || !viewport || waypoints.length < 2) return;
    const routeHash = waypoints.map(w => `${w.lat.toFixed(3)},${w.lon.toFixed(3)}`).join(';');
    if (routeHash === lastWarnedRouteRef.current) return;

    const b = viewport.bounds;
    const latSpan = b.lat_max - b.lat_min;
    const lonSpan = b.lon_max - b.lon_min;
    const margin = 0.1; // 10% margin

    const outOfBounds = waypoints.some(wp =>
      wp.lat < b.lat_min - latSpan * margin ||
      wp.lat > b.lat_max + latSpan * margin ||
      wp.lon < b.lon_min - lonSpan * margin ||
      wp.lon > b.lon_max + lonSpan * margin
    );

    if (outOfBounds) {
      toast.warning(
        'Route extends beyond forecast coverage',
        'Pan the map to include all waypoints for full forecast data.'
      );
      lastWarnedRouteRef.current = routeHash;
    }
  }, [waypoints, viewport, weather.forecastEnabled]); // eslint-disable-line react-hooks/exhaustive-deps

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
      <Header onFitRoute={handleFitRoute} />
      <DebugConsole />

      <main className="pt-16 h-screen">
        <div className="h-full">
          <MapComponent
            waypoints={waypoints}
            onWaypointsChange={setWaypoints}
            isEditing={isEditing}
            allResults={allResults}
            routeVisibility={routeVisibility}
            weatherLayer={weatherLayer}
            windData={weather.windData}
            waveData={weather.waveData}
            windVelocityData={weather.windVelocityData}
            currentVelocityData={weather.currentVelocityData}
            showZones={visibleZoneTypes.length > 0}
            visibleZoneTypes={visibleZoneTypes}
            zoneKey={zoneKey}
            isDrawingZone={isDrawingZone}
            onSaveZone={handleSaveZone}
            onCancelZone={() => setIsDrawingZone(false)}
            forecastEnabled={weather.forecastEnabled}
            dataTimestamp={weather.layerIngestedAt}
            onForecastClose={() => weather.setForecastEnabled(false)}
            onForecastHourChange={weather.handleForecastHourChange}
            onWaveForecastHourChange={weather.handleWaveForecastHourChange}
            onCurrentForecastHourChange={weather.handleCurrentForecastHourChange}
            onIceForecastHourChange={weather.handleIceForecastHourChange}
            onSwellForecastHourChange={weather.handleSwellForecastHourChange}
            onSstForecastHourChange={weather.handleSstForecastHourChange}
            onVisForecastHourChange={weather.handleVisForecastHourChange}
            onViewportChange={setViewport}
            viewportBounds={viewport?.bounds ?? null}
            weatherModelLabel={weather.weatherModelLabel}
            extendedWeatherData={weather.extendedWeatherData}
            fitBounds={fitBounds}
            fitKey={fitKey}
          >
            {/* Weather mode: overlay controls */}
            <MapOverlayControls
              weatherLayer={weatherLayer}
              onWeatherLayerChange={setWeatherLayer}
              forecastEnabled={weather.forecastEnabled}
              onForecastToggle={() => weather.setForecastEnabled(!weather.forecastEnabled)}
              isLoadingWeather={weather.isLoadingWeather}
              layerIngestedAt={weather.layerIngestedAt}
              resyncRunning={weather.resyncRunning}
              onResync={weather.handleResync}
            />

            {/* Analysis mode: left panel */}
            {viewMode === 'analysis' && (
              <AnalysisPanel
                waypoints={waypoints}
                routeName={routeName}
                onRouteNameChange={setRouteName}
                totalDistance={totalDistance}
                onRouteImport={handleRouteImport}
                onClearRoute={handleClearRoute}
                isEditing={isEditing}
                onIsEditingChange={setIsEditing}
                isCalculating={isCalculating}
                onCalculate={handleCalculate}
                isOptimizing={isOptimizing}
                onOptimize={handleOptimize}
                allResults={allResults}
                onApplyRoute={applyOptimizedRoute}
                onDismissRoutes={dismissOptimizedRoute}
                routeVisibility={routeVisibility}
                onRouteVisibilityChange={setRouteVisibility}
                isSimulating={simulatingId !== null}
                onRunSimulations={() => {
                  if (displayedAnalysisId) handleRunSimulation(displayedAnalysisId);
                }}
                displayedAnalysis={displayedAnalysis}
                paretoFront={paretoFront}
                isRunningPareto={isRunningPareto}
                onRunPareto={handlePareto}
              />
            )}
          </MapComponent>
        </div>
      </main>
    </div>
  );
}

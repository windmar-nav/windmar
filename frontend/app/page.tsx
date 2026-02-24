'use client';

import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import dynamic from 'next/dynamic';
import Header from '@/components/Header';
import MapOverlayControls from '@/components/MapOverlayControls';
import AnalysisPanel from '@/components/AnalysisPanel';
import { useVoyage } from '@/components/VoyageContext';
import { apiClient, Position, WindFieldData, WaveFieldData, VelocityData, OptimizationResponse, ParetoSolution, CreateZoneRequest, WaveForecastFrames, IceForecastFrames, SstForecastFrames, VisForecastFrames, OptimizedRouteKey, AllOptimizationResults, EMPTY_ALL_RESULTS } from '@/lib/api';
import { getAnalyses, saveAnalysis, deleteAnalysis, updateAnalysisMonteCarlo, updateAnalysisOptimizations, AnalysisEntry } from '@/lib/analysisStorage';
import { debugLog } from '@/lib/debugLog';
import DebugConsole from '@/components/DebugConsole';
import { useToast } from '@/components/Toast';

const MapComponent = dynamic(() => import('@/components/MapComponent'), { ssr: false });

type WeatherLayer = 'wind' | 'waves' | 'currents' | 'ice' | 'visibility' | 'sst' | 'swell' | 'none';

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
  } = useVoyage();

  // Toast notifications
  const toast = useToast();

  // Ephemeral state (local to this page)
  const [isEditing, setIsEditing] = useState(true);
  const [isCalculating, setIsCalculating] = useState(false);
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [variableResolution, setVariableResolution] = useState(true);
  const [paretoFront, setParetoFront] = useState<ParetoSolution[] | null>(null);
  const [isRunningPareto, setIsRunningPareto] = useState(false);

  // Per-layer staleness indicator (ISO timestamp from backend)
  const [layerIngestedAt, setLayerIngestedAt] = useState<string | null>(null);
  const [resyncRunning, setResyncRunning] = useState(false);
  const resyncRunningRef = useRef(false);

  // Weather visualization (layer selection persisted in VoyageContext)
  const [windData, setWindData] = useState<WindFieldData | null>(null);
  const windDataBaseRef = useRef<WindFieldData | null>(null); // preserve ocean mask for forecast
  const windFieldCacheRef = useRef<Record<string, WindFieldData>>({}); // per-frame cache
  const windFieldCacheVersionRef = useRef<string>(''); // invalidated on new GFS run
  const [waveData, setWaveData] = useState<WaveFieldData | null>(null);
  const [windVelocityData, setWindVelocityData] = useState<VelocityData[] | null>(null);
  const [currentVelocityData, setCurrentVelocityData] = useState<VelocityData[] | null>(null);
  const [extendedWeatherData, setExtendedWeatherData] = useState<any>(null);
  const [isLoadingWeather, setIsLoadingWeather] = useState(false);

  // Viewport state (init from context for cross-navigation persistence)
  const [viewport, setViewportLocal] = useState<{
    bounds: { lat_min: number; lat_max: number; lon_min: number; lon_max: number };
    zoom: number;
  } | null>(lastViewport);
  const setViewport = useCallback((vp: typeof viewport) => {
    setViewportLocal(vp);
    if (vp) setLastViewport(vp);
  }, [setLastViewport]);

  // Forecast timeline state
  const [forecastEnabled, setForecastEnabled] = useState(false);
  const [forecastHour, setForecastHour] = useState(0);

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

  // Load weather data
  const loadWeatherData = useCallback(async (vp?: typeof viewport, layer?: WeatherLayer) => {
    const v = vp || viewportRef.current;
    const activeLayer = layer ?? weatherLayerRef.current;
    if (!v || activeLayer === 'none') return;

    // Weather viewer uses viewport only — route analysis endpoints
    // fetch their own weather server-side with route-corridor bounds.
    const params = {
      lat_min: v.bounds.lat_min,
      lat_max: v.bounds.lat_max,
      lon_min: v.bounds.lon_min,
      lon_max: v.bounds.lon_max,
      resolution: getResolutionForZoom(v.zoom),
    };

    // NOTE: Do NOT clear extended data here — keep old data visible
    // while the new viewport's data loads (avoids 2-6s blank flash).

    setIsLoadingWeather(true);
    const t0 = performance.now();
    debugLog('info', 'API', `Loading ${activeLayer} weather: zoom=${v.zoom}, bbox=[${params.lat_min.toFixed(1)},${params.lat_max.toFixed(1)},${params.lon_min.toFixed(1)},${params.lon_max.toFixed(1)}]`);
    // Helper: treat empty 204 responses as null
    const orNull = <T,>(v: T): T | null => (v && typeof v === 'object' ? v : null);
    try {
      if (activeLayer === 'wind') {
        const [wind, windVel] = await Promise.all([
          apiClient.getWindField(params).then(orNull),
          apiClient.getWindVelocity(params).then(orNull),
        ]);
        const dt = (performance.now() - t0).toFixed(0);
        debugLog('info', 'API', `Wind loaded in ${dt}ms: grid=${wind?.ny}x${wind?.nx}`);
        if (wind) { setWindData(wind); windDataBaseRef.current = wind; }
        if (windVel) setWindVelocityData(windVel);
        if (wind?.ingested_at && !resyncRunningRef.current) setLayerIngestedAt(wind.ingested_at);
        // Background prefetch waves for instant layer switching
        apiClient.getWaveField(params).then(orNull).then(w => {
          if (w) setWaveData(w);
        }).catch(() => {});
      } else if (activeLayer === 'waves') {
        const waves = await apiClient.getWaveField(params).then(orNull);
        const dt = (performance.now() - t0).toFixed(0);
        debugLog('info', 'API', `Waves loaded in ${dt}ms: grid=${waves?.ny}x${waves?.nx}`);
        if (waves) setWaveData(waves);
        if (waves?.ingested_at && !resyncRunningRef.current) setLayerIngestedAt(waves.ingested_at);
      } else if (activeLayer === 'currents') {
        const currentVel = await apiClient.getCurrentVelocity(params).then(orNull).catch(() => null);
        const dt = (performance.now() - t0).toFixed(0);
        debugLog('info', 'API', `Currents loaded in ${dt}ms: ${currentVel ? 'yes' : 'no data'}`);
        setCurrentVelocityData(currentVel);
      } else if (activeLayer === 'ice') {
        const data = await apiClient.getIceField(params).then(orNull);
        const dt = (performance.now() - t0).toFixed(0);
        debugLog('info', 'API', `Ice loaded in ${dt}ms: grid=${data?.ny}x${data?.nx}`);
        if (data) setExtendedWeatherData(data);
        if (data?.ingested_at && !resyncRunningRef.current) setLayerIngestedAt(data.ingested_at);
      } else if (activeLayer === 'visibility') {
        const data = await apiClient.getVisibilityField(params).then(orNull);
        const dt = (performance.now() - t0).toFixed(0);
        debugLog('info', 'API', `Visibility loaded in ${dt}ms: grid=${data?.ny}x${data?.nx}`);
        if (data) setExtendedWeatherData(data);
        if (data?.ingested_at && !resyncRunningRef.current) setLayerIngestedAt(data.ingested_at);
      } else if (activeLayer === 'sst') {
        const data = await apiClient.getSstField(params).then(orNull);
        const dt = (performance.now() - t0).toFixed(0);
        debugLog('info', 'API', `SST loaded in ${dt}ms: grid=${data?.ny}x${data?.nx}`);
        if (data) setExtendedWeatherData(data);
        if (data?.ingested_at && !resyncRunningRef.current) setLayerIngestedAt(data.ingested_at);
      } else if (activeLayer === 'swell') {
        const data = await apiClient.getSwellField(params);
        const dt = (performance.now() - t0).toFixed(0);
        debugLog('info', 'API', `Swell loaded in ${dt}ms: grid=${data?.ny}x${data?.nx}`);
        setExtendedWeatherData(data);
        if (data?.ingested_at && !resyncRunningRef.current) setLayerIngestedAt(data.ingested_at);
      }
    } catch (error) {
      debugLog('error', 'API', `Weather load failed: ${error}`);
    } finally {
      setIsLoadingWeather(false);
    }
  }, []);

  // Reload weather when layer changes
  useEffect(() => {
    if (viewport && weatherLayer !== 'none') {
      loadWeatherData(viewport, weatherLayer);
    }
    if (weatherLayer === 'none') {
      setLayerIngestedAt(null);
    }
  }, [weatherLayer]); // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-load wind when route has 2+ waypoints and no layer is active
  useEffect(() => {
    if (waypoints.length >= 2 && weatherLayer === 'none') {
      setWeatherLayer('wind');
    }
  }, [waypoints.length]); // eslint-disable-line react-hooks/exhaustive-deps

  // Handle wind forecast hour change from timeline
  const handleForecastHourChange = useCallback((hour: number, data: VelocityData[] | null) => {
    setForecastHour(hour);
    if (data && data.length >= 2) {
      setWindVelocityData(data);

      // Lazy-cache: build WindFieldData once per (run, hour), reuse on loops
      const hdr = data[0].header;
      const version = hdr.refTime || '';
      if (version !== windFieldCacheVersionRef.current) {
        windFieldCacheRef.current = {};
        windFieldCacheVersionRef.current = version;
      }
      const key = String(hour);
      let field = windFieldCacheRef.current[key];
      if (!field) {
        const nx = hdr.nx;
        const ny = hdr.ny;
        const flatU = data[0].data;
        const flatV = data[1].data;
        // Guard: skip if grid dimensions don't match data length (stale frame during pan)
        if (!nx || !ny || flatU.length < nx * ny || flatV.length < nx * ny) return;
        const lats = Array.from({ length: ny }, (_, j) => hdr.la1 - j * hdr.dy);
        const lons = Array.from({ length: nx }, (_, i) => hdr.lo1 + i * hdr.dx);
        const u2d = Array.from({ length: ny }, (_, j) => {
          const off = j * nx;
          return Array.from({ length: nx }, (_, i) => flatU[off + i] ?? 0);
        });
        const v2d = Array.from({ length: ny }, (_, j) => {
          const off = j * nx;
          return Array.from({ length: nx }, (_, i) => flatV[off + i] ?? 0);
        });
        const base = windDataBaseRef.current;
        field = {
          parameter: 'wind',
          time: version,
          bbox: {
            lat_min: Math.min(hdr.la1, hdr.la2),
            lat_max: Math.max(hdr.la1, hdr.la2),
            lon_min: hdr.lo1,
            lon_max: hdr.lo2,
          },
          resolution: hdr.dx,
          nx, ny, lats, lons,
          u: u2d, v: v2d,
          ocean_mask: base?.ocean_mask,
          ocean_mask_lats: base?.ocean_mask_lats,
          ocean_mask_lons: base?.ocean_mask_lons,
        };
        windFieldCacheRef.current[key] = field;
      }
      setWindData(field);
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

  // Handle ice forecast hour change
  const handleIceForecastHourChange = useCallback((hour: number, allFrames: IceForecastFrames | null) => {
    setForecastHour(hour);
    if (!allFrames) {
      debugLog('warn', 'ICE', `Hour ${hour}: no frame data available`);
      if (hour === 0) loadWeatherData();
      return;
    }
    const frame = allFrames.frames?.[String(hour)];
    if (!frame || !frame.data) {
      debugLog('warn', 'ICE', `Hour ${hour}: frame not found in ${Object.keys(allFrames.frames).length} frames`);
      return;
    }
    debugLog('info', 'ICE', `Frame Day ${hour / 24}: grid=${allFrames.ny}x${allFrames.nx}`);

    setExtendedWeatherData({
      parameter: 'ice_concentration',
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
      unit: 'fraction',
      ocean_mask: allFrames.ocean_mask,
      ocean_mask_lats: allFrames.ocean_mask_lats,
      ocean_mask_lons: allFrames.ocean_mask_lons,
    });
  }, [loadWeatherData]);

  // Handle swell forecast hour change (extracts swell decomposition from wave frames)
  const handleSwellForecastHourChange = useCallback((hour: number, allFrames: WaveForecastFrames | null) => {
    setForecastHour(hour);
    if (!allFrames) {
      debugLog('warn', 'SWELL', `Hour ${hour}: no frame data available`);
      if (hour === 0) loadWeatherData();
      return;
    }
    const frame = allFrames.frames[String(hour)];
    if (!frame) {
      debugLog('warn', 'SWELL', `Hour ${hour}: frame not found in ${Object.keys(allFrames.frames).length} frames`);
      return;
    }

    // Use swell decomposition height if available, fall back to total Hs
    const swellData = frame.swell?.height ?? frame.data;
    debugLog('info', 'SWELL', `Frame T+${hour}h: hasSwell=${!!frame.swell}, grid=${allFrames.ny}x${allFrames.nx}`);

    setExtendedWeatherData({
      parameter: 'swell',
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
      data: swellData,
      unit: 'm',
      has_decomposition: !!frame.swell,
      total_hs: frame.data,
      swell_hs: frame.swell?.height ?? null,
      swell_tp: frame.swell?.period ?? null,
      swell_dir: frame.swell?.direction ?? null,
      windsea_hs: frame.windwave?.height ?? null,
      windsea_tp: frame.windwave?.period ?? null,
      windsea_dir: frame.windwave?.direction ?? null,
      ocean_mask: allFrames.ocean_mask,
      ocean_mask_lats: allFrames.ocean_mask_lats,
      ocean_mask_lons: allFrames.ocean_mask_lons,
      colorscale: allFrames.colorscale,
    });
  }, [loadWeatherData]);

  // Handle SST forecast hour change
  const handleSstForecastHourChange = useCallback((hour: number, allFrames: SstForecastFrames | null) => {
    setForecastHour(hour);
    if (!allFrames) {
      debugLog('warn', 'SST', `Hour ${hour}: no frame data available`);
      if (hour === 0) loadWeatherData();
      return;
    }
    const frame = allFrames.frames?.[String(hour)];
    if (!frame || !frame.data) {
      debugLog('warn', 'SST', `Hour ${hour}: frame not found in ${Object.keys(allFrames.frames).length} frames`);
      return;
    }
    debugLog('info', 'SST', `Frame T+${hour}h: grid=${allFrames.ny}x${allFrames.nx}`);

    setExtendedWeatherData({
      parameter: 'sst',
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
      unit: '°C',
      ocean_mask: allFrames.ocean_mask,
      ocean_mask_lats: allFrames.ocean_mask_lats,
      ocean_mask_lons: allFrames.ocean_mask_lons,
      colorscale: allFrames.colorscale,
    });
  }, [loadWeatherData]);

  // Handle visibility forecast hour change
  const handleVisForecastHourChange = useCallback((hour: number, allFrames: VisForecastFrames | null) => {
    setForecastHour(hour);
    if (!allFrames) {
      debugLog('warn', 'VIS', `Hour ${hour}: no frame data available`);
      if (hour === 0) loadWeatherData();
      return;
    }
    const frame = allFrames.frames?.[String(hour)];
    if (!frame || !frame.data) {
      debugLog('warn', 'VIS', `Hour ${hour}: frame not found in ${Object.keys(allFrames.frames).length} frames`);
      return;
    }
    debugLog('info', 'VIS', `Frame T+${hour}h: grid=${allFrames.ny}x${allFrames.nx}`);

    setExtendedWeatherData({
      parameter: 'visibility',
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
      unit: 'km',
      ocean_mask: allFrames.ocean_mask,
      ocean_mask_lats: allFrames.ocean_mask_lats,
      ocean_mask_lons: allFrames.ocean_mask_lons,
      colorscale: allFrames.colorscale,
    });
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
      grid_resolution_deg: 0.2,
      max_time_factor: 1.15,
      route_waypoints: waypoints.length > 2 ? waypoints : undefined,
      baseline_fuel_mt: displayedAnalysis?.result.total_fuel_mt,
      baseline_time_hours: displayedAnalysis?.result.total_time_hours,
      baseline_distance_nm: displayedAnalysis?.result.total_distance_nm,
      variable_resolution: variableResolution,
    };

    const combos: { engine: 'astar' | 'visir'; weight: number; key: OptimizedRouteKey }[] = [
      { engine: 'astar', weight: 0.0, key: 'astar_fuel' },
      { engine: 'astar', weight: 0.5, key: 'astar_balanced' },
      { engine: 'astar', weight: 1.0, key: 'astar_safety' },
      { engine: 'visir', weight: 0.0, key: 'visir_fuel' },
      { engine: 'visir', weight: 0.5, key: 'visir_balanced' },
      { engine: 'visir', weight: 1.0, key: 'visir_safety' },
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
        } catch {
          results[key] = null;
        }
        // Progressive update — show each result as it arrives
        setAllResults({ ...results });
      }

      const dt = ((performance.now() - t0) / 1000).toFixed(1);
      const ok = Object.values(results).filter(Boolean).length;
      debugLog('info', 'ROUTE', `All-routes done in ${dt}s: ${ok}/6 succeeded`);

      if (displayedAnalysisId && ok > 0) {
        updateAnalysisOptimizations(displayedAnalysisId, results);
        setAnalyses(getAnalyses());
      }
    } catch (error) {
      debugLog('error', 'ROUTE', `All-routes optimization failed: ${error}`);
    } finally {
      setIsOptimizing(false);
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
        grid_resolution_deg: 0.2,
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
    if (!forecastEnabled || !viewport || waypoints.length < 2) return;
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
  }, [waypoints, viewport, forecastEnabled]); // eslint-disable-line react-hooks/exhaustive-deps

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

  // Derive weather model label for map watermark
  const weatherModelLabel = useMemo(() => {
    if (weatherLayer === 'none') return undefined;
    if (weatherLayer === 'wind') return 'NOAA GFS 0.25\u00B0';
    if (weatherLayer === 'waves') return 'CMEMS WAV 1/12\u00B0';
    if (weatherLayer === 'currents') return 'CMEMS PHY 1/12\u00B0';
    if (weatherLayer === 'ice') return 'CMEMS ICE';
    if (weatherLayer === 'visibility') return 'NOAA GFS';
    if (weatherLayer === 'sst') return 'CMEMS PHY';
    if (weatherLayer === 'swell') return 'CMEMS WAV';
    return undefined;
  }, [weatherLayer]);

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
            dataTimestamp={layerIngestedAt}
            onForecastClose={() => setForecastEnabled(false)}
            onForecastHourChange={handleForecastHourChange}
            onWaveForecastHourChange={handleWaveForecastHourChange}
            onCurrentForecastHourChange={handleCurrentForecastHourChange}
            onIceForecastHourChange={handleIceForecastHourChange}
            onSwellForecastHourChange={handleSwellForecastHourChange}
            onSstForecastHourChange={handleSstForecastHourChange}
            onVisForecastHourChange={handleVisForecastHourChange}
            onViewportChange={setViewport}
            viewportBounds={viewport?.bounds ?? null}
            weatherModelLabel={weatherModelLabel}
            extendedWeatherData={extendedWeatherData}
            fitBounds={fitBounds}
            fitKey={fitKey}
          >
            {/* Weather mode: overlay controls */}
            <MapOverlayControls
              weatherLayer={weatherLayer}
              onWeatherLayerChange={setWeatherLayer}
              forecastEnabled={forecastEnabled}
              onForecastToggle={() => setForecastEnabled(!forecastEnabled)}
              isLoadingWeather={isLoadingWeather}
              layerIngestedAt={layerIngestedAt}
              resyncRunning={resyncRunning}
              onResync={async () => {
                if (weatherLayer === 'none') return;
                debugLog('info', 'WEATHER', `Resync: re-fetching ${weatherLayer}...`);
                setResyncRunning(true);
                resyncRunningRef.current = true;
                try {
                  await apiClient.resyncWeatherLayer(weatherLayer, viewportRef.current?.bounds);
                  // Force a unique timestamp so dataTimestamp always changes,
                  // even when the same GFS model run is re-ingested.
                  setLayerIngestedAt(new Date().toISOString());
                  // Reload overlay grid (skip ingested_at overwrite via ref guard)
                  await loadWeatherData(viewportRef.current ?? undefined, weatherLayer);
                } catch (error) {
                  debugLog('error', 'WEATHER', `Resync failed: ${error}`);
                } finally {
                  resyncRunningRef.current = false;
                  setResyncRunning(false);
                }
              }}
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
                variableResolution={variableResolution}
                onVariableResolutionChange={setVariableResolution}
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

'use client';

import { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import {
  apiClient,
  WindFieldData,
  WaveFieldData,
  VelocityData,
  GridFieldData,
  SwellFieldData,
  WaveForecastFrames,
  IceForecastFrames,
  SstForecastFrames,
  VisForecastFrames,
} from '@/lib/api';
import { debugLog } from '@/lib/debugLog';

type WeatherLayer = 'wind' | 'waves' | 'currents' | 'ice' | 'visibility' | 'sst' | 'swell' | 'none';

/**
 * Convert a WindFieldData grid (u[][], v[][]) to leaflet-velocity format.
 * Eliminates the separate /api/weather/wind/velocity fetch — ~0ms overhead.
 */
function windFieldToVelocity(wind: WindFieldData): VelocityData[] {
  const { ny, nx } = wind;
  const flatU: number[] = new Array(ny * nx);
  const flatV: number[] = new Array(ny * nx);
  // leaflet-velocity expects la1=top (descending lat order).
  // WindFieldData.lats may be ascending — flip rows if so.
  const ascending = wind.lats.length > 1 && wind.lats[0] < wind.lats[wind.lats.length - 1];
  for (let j = 0; j < ny; j++) {
    const srcRow = ascending ? ny - 1 - j : j;
    for (let i = 0; i < nx; i++) {
      flatU[j * nx + i] = wind.u[srcRow]?.[i] ?? 0;
      flatV[j * nx + i] = wind.v[srcRow]?.[i] ?? 0;
    }
  }
  const header = {
    parameterCategory: 2,
    parameterNumber: 2,
    la1: wind.bbox.lat_max,
    la2: wind.bbox.lat_min,
    lo1: wind.bbox.lon_min,
    lo2: wind.bbox.lon_max,
    dx: wind.resolution,
    dy: wind.resolution,
    nx,
    ny,
    refTime: wind.time || '',
  };
  return [
    { header: { ...header, parameterNumber: 2 }, data: flatU },
    { header: { ...header, parameterNumber: 3 }, data: flatV },
  ];
}

interface ViewportState {
  bounds: { lat_min: number; lat_max: number; lon_min: number; lon_max: number };
  zoom: number;
}

interface WeatherFetchParams {
  lat_min: number;
  lat_max: number;
  lon_min: number;
  lon_max: number;
  resolution: number;
}

// Current forecast frames — not in @/lib/api, only used here
interface CurrentForecastFrame {
  u: number[][];
  v: number[][];
}
interface CurrentForecastFrames {
  run_time: string;
  lats: number[];
  lons: number[];
  frames: Record<string, CurrentForecastFrame>;
}

// ---------------------------------------------------------------------------
// Shared helper: build GridFieldData from ice / sst / vis forecast frames.
// All three are structurally compatible via GridFrameSource.
// Swell uses its own builder (different output shape: SwellFieldData).
// ---------------------------------------------------------------------------
interface GridFrameSource {
  run_time: string;
  ny: number;
  nx: number;
  lats: number[];
  lons: number[];
  frames: Record<string, { data: number[][] }>;
  ocean_mask?: boolean[][];
  ocean_mask_lats?: number[];
  ocean_mask_lons?: number[];
  colorscale?: { min: number; max: number; data_min?: number; data_max?: number; colors: string[] };
}

function buildBbox(lats: number[], lons: number[]) {
  return {
    lat_min: lats[0],
    lat_max: lats[lats.length - 1],
    lon_min: lons[0],
    lon_max: lons[lons.length - 1],
  };
}

function resolFromLats(lats: number[]): number {
  return lats.length > 1 ? Math.abs(lats[1] - lats[0]) : 1;
}

function buildGridFrameData(
  parameter: string,
  unit: string,
  source: GridFrameSource,
  hour: number,
): GridFieldData | null {
  const frame = source.frames?.[String(hour)];
  if (!frame?.data) return null;
  return {
    parameter,
    time: source.run_time,
    bbox: buildBbox(source.lats, source.lons),
    resolution: resolFromLats(source.lats),
    nx: source.nx,
    ny: source.ny,
    lats: source.lats,
    lons: source.lons,
    data: frame.data,
    unit,
    ocean_mask: source.ocean_mask,
    ocean_mask_lats: source.ocean_mask_lats,
    ocean_mask_lons: source.ocean_mask_lons,
    colorscale: source.colorscale,
  };
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------
export function useWeatherDisplay(
  weatherLayer: WeatherLayer,
  setWeatherLayer: (layer: WeatherLayer) => void,
  viewport: ViewportState | null,
  waypointCount: number,
) {
  // ---- display data ----
  const [windData, setWindData] = useState<WindFieldData | null>(null);
  const [windVelocityData, setWindVelocityData] = useState<VelocityData[] | null>(null);
  const windDataBaseRef = useRef<WindFieldData | null>(null);
  const windFieldCacheRef = useRef<Record<string, WindFieldData>>({});
  const windFieldCacheVersionRef = useRef<string>('');
  const [waveData, setWaveData] = useState<WaveFieldData | null>(null);
  const [currentVelocityData, setCurrentVelocityData] = useState<VelocityData[] | null>(null);
  const [extendedWeatherData, setExtendedWeatherData] = useState<GridFieldData | null>(null);

  // ---- UI state ----
  const [isLoadingWeather, setIsLoadingWeather] = useState(false);
  const [layerIngestedAt, setLayerIngestedAt] = useState<string | null>(null);
  const [resyncRunning, setResyncRunning] = useState(false);
  const [forecastEnabled, setForecastEnabled] = useState(false);
  const [currentForecastHour, setCurrentForecastHour] = useState(0);

  // ---- request tracking: monotonic counter guards stale responses ----
  const loadRequestRef = useRef(0);

  // ---- data coverage tracking: skip viewport refetches when data already covers view ----
  const dataExtentRef = useRef<{
    lat_min: number; lat_max: number; lon_min: number; lon_max: number;
    layer: WeatherLayer;
  } | null>(null);
  const storeDataExtent = (bbox: { lat_min: number; lat_max: number; lon_min: number; lon_max: number }, layer: WeatherLayer) => {
    dataExtentRef.current = {
      lat_min: Math.min(bbox.lat_min, bbox.lat_max),
      lat_max: Math.max(bbox.lat_min, bbox.lat_max),
      lon_min: Math.min(bbox.lon_min, bbox.lon_max),
      lon_max: Math.max(bbox.lon_min, bbox.lon_max),
      layer,
    };
  };

  // ---- stable refs (avoid re-creating callbacks on every render) ----
  const viewportRef = useRef(viewport);
  useEffect(() => { viewportRef.current = viewport; }, [viewport]);
  const weatherLayerRef = useRef(weatherLayer);
  useEffect(() => { weatherLayerRef.current = weatherLayer; }, [weatherLayer]);

  // ---- resolution helper ----
  const getResolutionForZoom = (zoom: number): number => {
    if (zoom <= 4) return 2.0;
    if (zoom <= 6) return 1.0;
    return 0.5;
  };

  // ------------------------------------------------------------------
  // Load weather data for the current viewport + active layer.
  //
  // RACE-CONDITION GUARD: each call increments loadRequestRef.
  // After every await, isStale() checks whether a newer request has
  // started. If so, this request discards its results — only the
  // latest request may touch state.
  //
  // options.skipIngestedAt: caller manages layerIngestedAt externally.
  // Used by handleResync to force a cache-busting timestamp after load.
  // ------------------------------------------------------------------
  const loadWeatherData = useCallback(async (
    vp?: ViewportState,
    layer?: WeatherLayer,
    options?: { skipIngestedAt?: boolean },
  ) => {
    const v = vp || viewportRef.current;
    const activeLayer = layer ?? weatherLayerRef.current;
    if (!v || activeLayer === 'none') return;

    const requestId = ++loadRequestRef.current;
    const isStale = () => loadRequestRef.current !== requestId;

    const params: WeatherFetchParams = {
      lat_min: v.bounds.lat_min,
      lat_max: v.bounds.lat_max,
      lon_min: v.bounds.lon_min,
      lon_max: v.bounds.lon_max,
      resolution: getResolutionForZoom(v.zoom),
    };

    setIsLoadingWeather(true);
    const t0 = performance.now();
    debugLog('info', 'API', `Loading ${activeLayer} weather: zoom=${v.zoom}, bbox=[${params.lat_min.toFixed(1)},${params.lat_max.toFixed(1)},${params.lon_min.toFixed(1)},${params.lon_max.toFixed(1)}]`);
    const orNull = <T,>(v: T): T | null => (v && typeof v === 'object' ? v : null);

    try {
      if (activeLayer === 'wind') {
        const wind = await apiClient.getWindField(params).then(orNull);
        if (isStale()) return;
        const dt = (performance.now() - t0).toFixed(0);
        debugLog('info', 'API', `Wind loaded in ${dt}ms: grid=${wind?.ny}x${wind?.nx}`);
        if (wind) {
          setWindData(wind);
          windDataBaseRef.current = wind;
          setWindVelocityData(windFieldToVelocity(wind));
          if (wind.bbox) storeDataExtent(wind.bbox, 'wind');
        }
        if (wind?.ingested_at && !options?.skipIngestedAt) setLayerIngestedAt(wind.ingested_at);
        // Background prefetch waves for instant layer switching
        apiClient.getWaveField(params).then(orNull).then(w => {
          if (!isStale() && w) setWaveData(w);
        }).catch(() => {});

      } else if (activeLayer === 'waves') {
        const waves = await apiClient.getWaveField(params).then(orNull);
        if (isStale()) return;
        const dt = (performance.now() - t0).toFixed(0);
        debugLog('info', 'API', `Waves loaded in ${dt}ms: grid=${waves?.ny}x${waves?.nx}`);
        if (waves) {
          setWaveData(waves);
          if (waves.bbox) storeDataExtent(waves.bbox, 'waves');
        }
        if (waves?.ingested_at && !options?.skipIngestedAt) setLayerIngestedAt(waves.ingested_at);

      } else if (activeLayer === 'currents') {
        const [currentVel, currentGrid] = await Promise.all([
          apiClient.getCurrentVelocity(params).then(orNull).catch(() => null),
          apiClient.getCurrentField(params).then(orNull).catch(() => null),
        ]);
        if (isStale()) return;
        const dt = (performance.now() - t0).toFixed(0);
        debugLog('info', 'API', `Currents loaded in ${dt}ms: ${currentVel ? 'yes' : 'no data'}`);
        setCurrentVelocityData(currentVel);
        if (currentGrid) {
          setExtendedWeatherData(currentGrid as GridFieldData);
          const g = currentGrid as GridFieldData;
          if (g.bbox) storeDataExtent(g.bbox, 'currents');
        }
        if ((currentGrid as GridFieldData | null)?.ingested_at && !options?.skipIngestedAt) {
          setLayerIngestedAt((currentGrid as GridFieldData).ingested_at!);
        }

      } else {
        // Generic handler for ice / visibility / sst / swell
        const fetchers: Partial<Record<WeatherLayer, (p: WeatherFetchParams) => Promise<GridFieldData>>> = {
          ice: apiClient.getIceField.bind(apiClient),
          visibility: apiClient.getVisibilityField.bind(apiClient),
          sst: apiClient.getSstField.bind(apiClient),
          swell: apiClient.getSwellField.bind(apiClient),
        };
        const fetcher = fetchers[activeLayer];
        if (fetcher) {
          const data = await fetcher(params).then(orNull);
          if (isStale()) return;
          const dt = (performance.now() - t0).toFixed(0);
          debugLog('info', 'API', `${activeLayer} loaded in ${dt}ms: grid=${data?.ny}x${data?.nx}`);
          if (data) {
            setExtendedWeatherData(data);
            if (data.bbox) storeDataExtent(data.bbox, activeLayer);
          }
          if (data?.ingested_at && !options?.skipIngestedAt) setLayerIngestedAt(data.ingested_at);
        }
      }
    } catch (error) {
      if (isStale()) return;
      debugLog('error', 'API', `Weather load failed: ${error}`);
    } finally {
      if (!isStale()) {
        setIsLoadingWeather(false);
      }
    }
  }, []);

  // ---- Auto-reload on layer change ----
  // Clear previous layer data immediately to free memory before loading new layer.
  // Skip single-frame load when forecast timeline is active — the timeline
  // manages extendedWeatherData directly and would be overwritten by the
  // single-frame response arriving after the timeline restores its frames.
  useEffect(() => {
    setExtendedWeatherData(null);
    dataExtentRef.current = null; // Force refetch for new layer
    if (viewport && weatherLayer !== 'none' && !forecastEnabled) {
      loadWeatherData(viewport, weatherLayer);
    }
    if (weatherLayer === 'none') {
      setLayerIngestedAt(null);
    }
  }, [weatherLayer, forecastEnabled]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---- Debounced viewport refresh for arrow/grid overlays ----
  // Skip refetch when existing data already covers the visible viewport.
  // The server returns global-bbox data (covering cache), so small
  // pan/zoom changes rarely need new data.
  const viewportRefreshTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  useEffect(() => {
    if (!viewport || weatherLayer === 'none' || forecastEnabled) return;
    const ext = dataExtentRef.current;
    if (ext && ext.layer === weatherLayer) {
      const b = viewport.bounds;
      if (ext.lat_min <= b.lat_min && ext.lat_max >= b.lat_max &&
          ext.lon_min <= b.lon_min && ext.lon_max >= b.lon_max) {
        return; // data covers viewport — no refetch needed
      }
    }

    clearTimeout(viewportRefreshTimer.current);
    viewportRefreshTimer.current = setTimeout(() => {
      loadWeatherData(viewport, weatherLayer);
    }, 1200);
    return () => clearTimeout(viewportRefreshTimer.current);
  }, [viewport]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---- Auto-load wind when route has 2+ waypoints ----
  useEffect(() => {
    if (waypointCount >= 2 && weatherLayer === 'none') {
      setWeatherLayer('wind');
    }
  }, [waypointCount]); // eslint-disable-line react-hooks/exhaustive-deps

  // ------------------------------------------------------------------
  // Forecast timeline handlers
  // ------------------------------------------------------------------

  // Wind: VelocityData[] → WindFieldData (with per-frame cache)
  const handleForecastHourChange = useCallback((hour: number, data: VelocityData[] | null) => {
    setCurrentForecastHour(hour);
    if (data && data.length >= 2) {
      const hdr = data[0].header;
      const version = hdr.refTime || '';
      if (version !== windFieldCacheVersionRef.current) {
        windFieldCacheRef.current = {};
        windFieldCacheVersionRef.current = version;
      }
      const key = String(hour);
      let field = windFieldCacheRef.current[key];
      if (!field) {
        const { nx, ny } = hdr;
        const flatU = data[0].data;
        const flatV = data[1].data;
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
      setWindVelocityData(data);
    } else if (hour === 0) {
      loadWeatherData();
    }
  }, [loadWeatherData]);

  // Waves: WaveForecastFrames → WaveFieldData
  const handleWaveForecastHourChange = useCallback((hour: number, allFrames: WaveForecastFrames | null) => {
    setCurrentForecastHour(hour);
    if (!allFrames) { if (hour === 0) loadWeatherData(); return; }
    const frame = allFrames.frames[String(hour)];
    if (!frame) return;
    setWaveData({
      parameter: 'wave_height',
      time: allFrames.run_time,
      bbox: buildBbox(allFrames.lats, allFrames.lons),
      resolution: resolFromLats(allFrames.lats),
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
    });
  }, [loadWeatherData]);

  // Ice: IceForecastFrames → GridFieldData
  const handleIceForecastHourChange = useCallback((hour: number, allFrames: IceForecastFrames | null) => {
    setCurrentForecastHour(hour);
    if (!allFrames) { if (hour === 0) loadWeatherData(); return; }
    const result = buildGridFrameData('ice_concentration', 'fraction', allFrames, hour);
    if (result) setExtendedWeatherData(result);
  }, [loadWeatherData]);

  // SST: SstForecastFrames → GridFieldData
  const handleSstForecastHourChange = useCallback((hour: number, allFrames: SstForecastFrames | null) => {
    setCurrentForecastHour(hour);
    if (!allFrames) { if (hour === 0) loadWeatherData(); return; }
    const result = buildGridFrameData('sst', '\u00B0C', allFrames, hour);
    if (result) setExtendedWeatherData(result);
  }, [loadWeatherData]);

  // Visibility: VisForecastFrames → GridFieldData
  const handleVisForecastHourChange = useCallback((hour: number, allFrames: VisForecastFrames | null) => {
    setCurrentForecastHour(hour);
    if (!allFrames) { if (hour === 0) loadWeatherData(); return; }
    const result = buildGridFrameData('visibility', 'km', allFrames, hour);
    if (result) setExtendedWeatherData(result);
  }, [loadWeatherData]);

  // Swell: WaveForecastFrames → SwellFieldData (built directly, not via helper)
  const handleSwellForecastHourChange = useCallback((hour: number, allFrames: WaveForecastFrames | null) => {
    setCurrentForecastHour(hour);
    if (!allFrames) { if (hour === 0) loadWeatherData(); return; }
    const frame = allFrames.frames[String(hour)];
    if (!frame) return;
    const swell: SwellFieldData = {
      parameter: 'swell',
      time: allFrames.run_time,
      bbox: buildBbox(allFrames.lats, allFrames.lons),
      resolution: resolFromLats(allFrames.lats),
      nx: allFrames.nx,
      ny: allFrames.ny,
      lats: allFrames.lats,
      lons: allFrames.lons,
      data: frame.swell?.height ?? frame.data,
      unit: 'm',
      ocean_mask: allFrames.ocean_mask,
      ocean_mask_lats: allFrames.ocean_mask_lats,
      ocean_mask_lons: allFrames.ocean_mask_lons,
      colorscale: allFrames.colorscale,
      has_decomposition: !!frame.swell,
      total_hs: frame.data,
      swell_hs: frame.swell?.height ?? null,
      swell_tp: frame.swell?.period ?? null,
      swell_dir: frame.swell?.direction ?? null,
      windsea_hs: frame.windwave?.height ?? null,
      windsea_tp: frame.windwave?.period ?? null,
      windsea_dir: frame.windwave?.direction ?? null,
    };
    setExtendedWeatherData(swell);
  }, [loadWeatherData]);

  // Currents: 2D u/v arrays → VelocityData[] (leaflet-velocity format)
  const handleCurrentForecastHourChange = useCallback((hour: number, allFrames: CurrentForecastFrames | null) => {
    setCurrentForecastHour(hour);
    if (!allFrames) { if (hour === 0) loadWeatherData(); return; }
    const frame = allFrames.frames?.[String(hour)];
    if (!frame?.u || !frame?.v) return;

    const { lats, lons } = allFrames;
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
      nx, ny,
      refTime: allFrames.run_time || '',
    };
    setCurrentVelocityData([
      { header: { ...header, parameterNumber: 2 }, data: uFlat },
      { header: { ...header, parameterNumber: 3 }, data: vFlat },
    ]);
  }, [loadWeatherData]);

  // ------------------------------------------------------------------
  // Resync handler
  //
  // Tells the backend to re-ingest the active layer, then reloads.
  // skipIngestedAt prevents loadWeatherData from setting the timestamp —
  // we force our own AFTER load to guarantee ForecastTimeline cache
  // invalidation even when the same model run is re-ingested.
  // ------------------------------------------------------------------
  const handleResync = useCallback(async () => {
    const layer = weatherLayerRef.current;
    if (layer === 'none') return;
    debugLog('info', 'WEATHER', `Resync: re-fetching ${layer}...`);
    setResyncRunning(true);
    try {
      await apiClient.resyncWeatherLayer(layer, viewportRef.current?.bounds);
      await loadWeatherData(viewportRef.current ?? undefined, layer, { skipIngestedAt: true });
      setLayerIngestedAt(new Date().toISOString());
    } catch (error) {
      debugLog('error', 'WEATHER', `Resync failed: ${error}`);
    } finally {
      setResyncRunning(false);
    }
  }, [loadWeatherData]);

  // ------------------------------------------------------------------
  // Weather model label (map watermark)
  // ------------------------------------------------------------------
  const weatherModelLabel = useMemo(() => {
    if (weatherLayer === 'none') return undefined;
    const labels: Record<string, string> = {
      wind: 'NOAA GFS 0.25\u00B0',
      waves: 'CMEMS WAV 1/12\u00B0',
      currents: 'CMEMS PHY 1/12\u00B0',
      ice: 'CMEMS ICE',
      visibility: 'NOAA GFS',
      sst: 'CMEMS PHY',
      swell: 'CMEMS WAV',
    };
    return labels[weatherLayer];
  }, [weatherLayer]);

  return {
    // Display data
    windData,
    windVelocityData,
    waveData,
    currentVelocityData,
    extendedWeatherData,
    // UI state
    isLoadingWeather,
    layerIngestedAt,
    resyncRunning,
    forecastEnabled,
    setForecastEnabled,
    currentForecastHour,
    weatherModelLabel,
    // Actions
    loadWeatherData,
    handleResync,
    // Forecast timeline callbacks
    handleForecastHourChange,
    handleWaveForecastHourChange,
    handleCurrentForecastHourChange,
    handleIceForecastHourChange,
    handleSwellForecastHourChange,
    handleSstForecastHourChange,
    handleVisForecastHourChange,
  };
}

'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { Play, Pause, X, Clock, Loader2, Info } from 'lucide-react';
import { apiClient, VelocityData, ForecastFrames, WaveForecastFrames, WaveForecastFrame, CurrentForecastFrames, IceForecastFrames, SstForecastFrames, VisForecastFrames } from '@/lib/api';
import { debugLog } from '@/lib/debugLog';

type LayerType = 'wind' | 'waves' | 'currents' | 'ice' | 'swell' | 'sst' | 'visibility';

export interface ViewportBounds {
  lat_min: number;
  lat_max: number;
  lon_min: number;
  lon_max: number;
}

interface ForecastTimelineProps {
  visible: boolean;
  onClose: () => void;
  onForecastHourChange: (hour: number, data: VelocityData[] | null) => void;
  /** Callback for wave forecast frame changes */
  onWaveForecastHourChange?: (hour: number, frame: WaveForecastFrames | null) => void;
  /** Callback for current forecast frame changes */
  onCurrentForecastHourChange?: (hour: number, frame: CurrentForecastFrames | null) => void;
  /** Callback for ice forecast frame changes */
  onIceForecastHourChange?: (hour: number, frame: IceForecastFrames | null) => void;
  /** Callback for swell forecast frame changes (reuses wave frames) */
  onSwellForecastHourChange?: (hour: number, frame: WaveForecastFrames | null) => void;
  /** Callback for SST forecast frame changes */
  onSstForecastHourChange?: (hour: number, frame: SstForecastFrames | null) => void;
  /** Callback for visibility forecast frame changes */
  onVisForecastHourChange?: (hour: number, frame: VisForecastFrames | null) => void;
  layerType?: LayerType;
  /** Display name override — shows the actual weather layer name in the title */
  displayLayerName?: string;
  viewportBounds?: ViewportBounds | null;
  dataTimestamp?: string | null;
}

// Default forecast hours (used until actual frame data arrives from DB)
const DEFAULT_FORECAST_HOURS = Array.from({ length: 41 }, (_, i) => i * 3); // 0,3,6,...,120
const DEFAULT_ICE_FORECAST_HOURS = Array.from({ length: 10 }, (_, i) => i * 24); // 0,24,48,...,216
const SPEED_OPTIONS = [1, 2, 4];
const SPEED_INTERVAL: Record<number, number> = { 1: 2000, 2: 1000, 4: 500 };

/** Extract sorted numeric hours from frame keys returned by the backend. */
function deriveHoursFromFrames(frames: Record<string, unknown>): number[] {
  const hours = Object.keys(frames).map(Number).filter(n => !isNaN(n)).sort((a, b) => a - b);
  return hours.length > 0 ? hours : [];
}

/** Find the forecast hour closest to "now" given a run time and available frame keys.
 *  Returns the nearest available hour that is <= now, or hour 0 as fallback. */
function nearestHourToNow(runTimeStr: string, frames: Record<string, unknown>): number {
  try {
    const hours = deriveHoursFromFrames(frames);
    if (hours.length === 0) return 0;
    // Parse run time — formats: "20260217 00Z" or ISO "2026-02-17T00:00:00Z"
    let runMs: number;
    if (runTimeStr.includes('T') || runTimeStr.includes('-')) {
      runMs = new Date(runTimeStr).getTime();
    } else {
      const parts = runTimeStr.split(' ');
      const d = parts[0];
      const h = parseInt(parts[1]?.replace('Z', '') || '0');
      runMs = Date.UTC(parseInt(d.slice(0, 4)), parseInt(d.slice(4, 6)) - 1, parseInt(d.slice(6, 8)), h);
    }
    if (isNaN(runMs)) { debugLog('info', 'TIMELINE', `nearestHourToNow: invalid runMs for "${runTimeStr}"`); return 0; }
    const elapsedHours = (Date.now() - runMs) / 3_600_000;
    // Find largest available hour <= elapsedHours (i.e., most recent past frame)
    let best = hours[0];
    for (const h of hours) {
      if (h <= elapsedHours) best = h;
      else break;
    }
    debugLog('info', 'TIMELINE', `nearestHourToNow: run="${runTimeStr}" elapsed=${elapsedHours.toFixed(1)}h → T+${best}h`);
    return best;
  } catch {
    return 0;
  }
}

export default function ForecastTimeline({
  visible,
  onClose,
  onForecastHourChange,
  onWaveForecastHourChange,
  onCurrentForecastHourChange,
  onIceForecastHourChange,
  onSwellForecastHourChange,
  onSstForecastHourChange,
  onVisForecastHourChange,
  layerType = 'wind',
  displayLayerName,
  viewportBounds,
  dataTimestamp,
}: ForecastTimelineProps) {
  const isWindMode = layerType === 'wind';
  const isWaveMode = layerType === 'waves';
  const isCurrentMode = layerType === 'currents';
  const isIceMode = layerType === 'ice';
  const isSwellMode = layerType === 'swell';
  const isSstMode = layerType === 'sst';
  const isVisMode = layerType === 'visibility';
  const hasForecast = isWindMode || isWaveMode || isCurrentMode || isIceMode || isSwellMode || isSstMode || isVisMode;

  const [currentHour, setCurrentHour] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [loadProgress, setLoadProgress] = useState({ cached: 0, total: 0 });
  const [runTime, setRunTime] = useState<string | null>(null);
  const [prefetchComplete, setPrefetchComplete] = useState(false);

  // Dynamic hours derived from actual DB/API response (replaces hardcoded constants)
  const [availableHours, setAvailableHours] = useState<number[]>([]);
  const availableHoursRef = useRef<number[]>([]);
  useEffect(() => { availableHoursRef.current = availableHours; }, [availableHours]);

  // Reset available hours when layer changes so stale data from a previous layer doesn't persist
  useEffect(() => { setAvailableHours([]); setCurrentHour(0); }, [layerType]);

  // Invalidate all cached frames when data timestamp changes (e.g., after resync)
  const prevTimestampRef = useRef(dataTimestamp);
  useEffect(() => {
    if (dataTimestamp && dataTimestamp !== prevTimestampRef.current) {
      debugLog('info', 'TIMELINE', `Data timestamp changed — clearing frame caches for re-fetch`);
      setWindFrames({});
      windFramesRef.current = {};
      setWaveFrameData(null);
      waveFrameDataRef.current = null;
      setCurrentFrameData(null);
      currentFrameDataRef.current = null;
      setIceFrameData(null);
      iceFrameDataRef.current = null;
      setSstFrameData(null);
      sstFrameDataRef.current = null;
      setVisFrameData(null);
      visFrameDataRef.current = null;
      setPrefetchComplete(false);
      setAvailableHours([]);
      setCurrentHour(0);
    }
    prevTimestampRef.current = dataTimestamp;
  }, [dataTimestamp]);

  // Effective hours: use DB-derived if available, else defaults
  const defaultHours = isIceMode ? DEFAULT_ICE_FORECAST_HOURS : DEFAULT_FORECAST_HOURS;
  const activeHours = availableHours.length > 0 ? availableHours : defaultHours;
  const sliderMax = activeHours.length > 0 ? activeHours[activeHours.length - 1] : 0;
  const sliderStep = activeHours.length >= 2 ? activeHours[1] - activeHours[0] : (isIceMode ? 24 : 3);

  // Wind frames
  const [windFrames, setWindFrames] = useState<Record<string, VelocityData[]>>({});
  const windFramesRef = useRef<Record<string, VelocityData[]>>({});

  // Wave frames (full response with shared metadata)
  const [waveFrameData, setWaveFrameData] = useState<WaveForecastFrames | null>(null);
  const waveFrameDataRef = useRef<WaveForecastFrames | null>(null);

  // Current frames
  const [currentFrameData, setCurrentFrameData] = useState<CurrentForecastFrames | null>(null);
  const currentFrameDataRef = useRef<CurrentForecastFrames | null>(null);

  // Ice frames
  const [iceFrameData, setIceFrameData] = useState<IceForecastFrames | null>(null);
  const iceFrameDataRef = useRef<IceForecastFrames | null>(null);

  // SST frames
  const [sstFrameData, setSstFrameData] = useState<SstForecastFrames | null>(null);
  const sstFrameDataRef = useRef<SstForecastFrames | null>(null);

  // Visibility frames
  const [visFrameData, setVisFrameData] = useState<VisForecastFrames | null>(null);
  const visFrameDataRef = useRef<VisForecastFrames | null>(null);

  const playRafRef = useRef<number | null>(null);
  const playLastTickRef = useRef(0);
  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const sliderRafRef = useRef<number | null>(null);

  // Track viewport bounds — updates freely on pan/zoom but does NOT trigger re-fetch
  // (boundsKey cache-clearing effect removed). Only dataTimestamp (resync) triggers re-fetch.
  const boundsRef = useRef<ViewportBounds | null>(viewportBounds ?? null);
  const hasBounds = boundsRef.current !== null;
  useEffect(() => {
    if (viewportBounds) boundsRef.current = viewportBounds;
  }, [viewportBounds]);

  // Bounds key for cache invalidation: coarse 10°-rounded viewport bounds.
  // Only triggers re-fetch when the user pans to a truly different ocean region,
  // NOT on every small pan (which would miss the backend's per-bounds frame cache).
  const BOUNDS_GRID = 10; // degrees

  // Pad viewport bounds OUT to grid cell edges so fetched data always covers the
  // full grid cell.  This prevents truncated overlays when panning within a cell.
  // Cap to MAX_SPAN degrees per axis to avoid multi-GB responses when zoomed out.
  const MAX_SPAN = 120; // degrees — keeps response ~290 MB at 0.25° grid
  const paddedBounds = () => {
    const b = boundsRef.current;
    if (!b) return {};
    let lat_min = Math.floor(b.lat_min / BOUNDS_GRID) * BOUNDS_GRID;
    let lat_max = Math.ceil(b.lat_max / BOUNDS_GRID) * BOUNDS_GRID;
    let lon_min = Math.floor(b.lon_min / BOUNDS_GRID) * BOUNDS_GRID;
    let lon_max = Math.ceil(b.lon_max / BOUNDS_GRID) * BOUNDS_GRID;
    // Clamp oversized spans around viewport center
    if (lat_max - lat_min > MAX_SPAN) {
      const mid = (b.lat_min + b.lat_max) / 2;
      lat_min = Math.floor((mid - MAX_SPAN / 2) / BOUNDS_GRID) * BOUNDS_GRID;
      lat_max = lat_min + MAX_SPAN;
    }
    if (lon_max - lon_min > MAX_SPAN) {
      const mid = (b.lon_min + b.lon_max) / 2;
      lon_min = Math.floor((mid - MAX_SPAN / 2) / BOUNDS_GRID) * BOUNDS_GRID;
      lon_max = lon_min + MAX_SPAN;
    }
    return { lat_min, lat_max, lon_min, lon_max };
  };
  // NOTE: No auto-refetch on pan/zoom. Data loads once when timeline opens.
  // Manual resync (dataTimestamp change) handles region changes.

  // Keep refs in sync
  useEffect(() => { windFramesRef.current = windFrames; }, [windFrames]);
  useEffect(() => { waveFrameDataRef.current = waveFrameData; }, [waveFrameData]);
  useEffect(() => { currentFrameDataRef.current = currentFrameData; }, [currentFrameData]);
  useEffect(() => { iceFrameDataRef.current = iceFrameData; }, [iceFrameData]);
  useEffect(() => { sstFrameDataRef.current = sstFrameData; }, [sstFrameData]);
  useEffect(() => { visFrameDataRef.current = visFrameData; }, [visFrameData]);

  // ------------------------------------------------------------------
  // Wind forecast: load all frames
  // ------------------------------------------------------------------
  const loadWindFrames = useCallback(async (bounds?: ViewportBounds | null) => {
    try {
      // Use the provided bounds (from prefetch) to avoid mismatch if the user panned
      const bp = bounds ?? boundsRef.current ?? {};
      const data: ForecastFrames = await apiClient.getForecastFrames(bp);
      setWindFrames(data.frames);
      setAvailableHours(deriveHoursFromFrames(data.frames));
      const rt = `${data.run_date} ${data.run_hour}Z`;
      setRunTime(rt);
      setPrefetchComplete(true);
      setIsLoading(false);
      const initHour = nearestHourToNow(rt, data.frames);
      const initKey = String(initHour);
      if (data.frames[initKey]) {
        setCurrentHour(initHour);
        onForecastHourChange(initHour, data.frames[initKey]);
      } else if (data.frames['0']) {
        onForecastHourChange(0, data.frames['0']);
      }
    } catch (e) {
      console.error('Failed to load wind forecast frames:', e);
      setIsLoading(false);
    }
  }, [onForecastHourChange]);

  // ------------------------------------------------------------------
  // Wave forecast: load all frames
  // ------------------------------------------------------------------
  const loadWaveFrames = useCallback(async () => {
    try {
      debugLog('info', 'WAVE', 'Loading wave forecast frames from API...');
      const t0 = performance.now();
      const bp = paddedBounds();
      const data: WaveForecastFrames = await apiClient.getWaveForecastFrames(bp);
      const dt = ((performance.now() - t0) / 1000).toFixed(1);
      const frameKeys = Object.keys(data.frames);
      debugLog('info', 'WAVE', `Loaded ${frameKeys.length} frames in ${dt}s, grid=${data.ny}x${data.nx}`);
      // Verify frames are different
      if (frameKeys.length >= 2) {
        const f0 = data.frames[frameKeys[0]]?.data;
        const f1 = data.frames[frameKeys[frameKeys.length - 1]]?.data;
        if (f0 && f1) {
          const s0 = f0[Math.floor(f0.length/2)]?.[0]?.toFixed(2);
          const s1 = f1[Math.floor(f1.length/2)]?.[0]?.toFixed(2);
          debugLog('info', 'WAVE', `Frame verification: first=${s0}, last=${s1}, different=${s0 !== s1}`);
        }
      }
      setWaveFrameData(data);
      setAvailableHours(deriveHoursFromFrames(data.frames));
      const rt = data.run_time;
      let rtFormatted = rt || '';
      if (rt) {
        try {
          const d = new Date(rt);
          rtFormatted = `${d.getUTCFullYear()}${String(d.getUTCMonth() + 1).padStart(2, '0')}${String(d.getUTCDate()).padStart(2, '0')} ${String(d.getUTCHours()).padStart(2, '0')}Z`;
          setRunTime(rtFormatted);
        } catch {
          setRunTime(rt);
        }
      }
      setPrefetchComplete(true);
      setIsLoading(false);
      if (onWaveForecastHourChange) {
        const initHour = nearestHourToNow(rtFormatted || rt || '', data.frames);
        debugLog('info', 'WAVE', `Setting initial frame T+${initHour}h`);
        setCurrentHour(initHour);
        onWaveForecastHourChange(initHour, data);
      }
    } catch (e) {
      debugLog('error', 'WAVE', `Failed to load wave forecast frames: ${e}`);
      setIsLoading(false);
    }
  }, [onWaveForecastHourChange]);

  // ------------------------------------------------------------------
  // Wind prefetch effect — direct frame load (matches wave/current/ice pattern)
  // ------------------------------------------------------------------
  useEffect(() => {
    if (!visible || !isWindMode || !boundsRef.current) return;

    // Client-side cache hit — restore UI state instantly, skip network
    if (Object.keys(windFramesRef.current).length > 0) {
      setAvailableHours(deriveHoursFromFrames(windFramesRef.current));
      setPrefetchComplete(true);
      setIsLoading(false);
      const initHour = nearestHourToNow(runTime ?? '', windFramesRef.current);
      const initKey = String(initHour);
      if (windFramesRef.current[initKey]) {
        setCurrentHour(initHour);
        onForecastHourChange(initHour, windFramesRef.current[initKey]);
      } else if (windFramesRef.current['0']) {
        onForecastHourChange(0, windFramesRef.current['0']);
      }
      return;
    }

    const bp = paddedBounds();
    setIsLoading(true);
    setPrefetchComplete(false);
    loadWindFrames(bp);

    // Fire-and-forget: warm the GRIB file cache for future requests
    apiClient.triggerForecastPrefetch(bp).catch(() => {});
  }, [visible, isWindMode, hasBounds, dataTimestamp, loadWindFrames]);

  // ------------------------------------------------------------------
  // Wave prefetch effect — loads frames directly (backend extracts on-the-fly)
  // ------------------------------------------------------------------
  useEffect(() => {
    if (!visible || !isWaveMode || !boundsRef.current) return;
    if (waveFrameDataRef.current) {
      const data = waveFrameDataRef.current;
      setAvailableHours(deriveHoursFromFrames(data.frames));
      setPrefetchComplete(true);
      setIsLoading(false);
      if (onWaveForecastHourChange) {
        const initHour = nearestHourToNow(runTime ?? data.run_time ?? '', data.frames);
        setCurrentHour(initHour);
        onWaveForecastHourChange(initHour, data);
      }
      return;
    }
    setIsLoading(true);
    setPrefetchComplete(false);
    loadWaveFrames();
  }, [visible, isWaveMode, hasBounds, dataTimestamp, loadWaveFrames]);

  // ------------------------------------------------------------------
  // Current forecast: load all frames
  // ------------------------------------------------------------------
  const loadCurrentFrames = useCallback(async () => {
    try {
      debugLog('info', 'CURRENT', 'Loading current forecast frames from API...');
      const t0 = performance.now();
      const bp = paddedBounds();
      const data: CurrentForecastFrames = await apiClient.getCurrentForecastFrames(bp);
      const dt = ((performance.now() - t0) / 1000).toFixed(1);
      const frameKeys = Object.keys(data.frames);
      debugLog('info', 'CURRENT', `Loaded ${frameKeys.length} frames in ${dt}s, grid=${data.ny}x${data.nx}`);
      setCurrentFrameData(data);
      setAvailableHours(deriveHoursFromFrames(data.frames));
      let rtFormatted = data.run_time || '';
      if (data.run_time) {
        try {
          const d = new Date(data.run_time);
          rtFormatted = `${d.getUTCFullYear()}${String(d.getUTCMonth() + 1).padStart(2, '0')}${String(d.getUTCDate()).padStart(2, '0')} ${String(d.getUTCHours()).padStart(2, '0')}Z`;
          setRunTime(rtFormatted);
        } catch {
          setRunTime(data.run_time);
        }
      }
      setPrefetchComplete(true);
      setIsLoading(false);
      if (onCurrentForecastHourChange) {
        const initHour = nearestHourToNow(rtFormatted, data.frames);
        debugLog('info', 'CURRENT', `Setting initial current frame T+${initHour}h`);
        setCurrentHour(initHour);
        onCurrentForecastHourChange(initHour, data);
      }
    } catch (e) {
      debugLog('error', 'CURRENT', `Failed to load current forecast frames: ${e}`);
      setIsLoading(false);
    }
  }, [onCurrentForecastHourChange]);

  // ------------------------------------------------------------------
  // Current prefetch effect
  // ------------------------------------------------------------------
  useEffect(() => {
    if (!visible || !isCurrentMode || !boundsRef.current) return;

    if (currentFrameDataRef.current) {
      const data = currentFrameDataRef.current;
      setAvailableHours(deriveHoursFromFrames(data.frames));
      setPrefetchComplete(true);
      setIsLoading(false);
      if (onCurrentForecastHourChange) {
        const initHour = nearestHourToNow(runTime ?? data.run_time ?? '', data.frames);
        setCurrentHour(initHour);
        onCurrentForecastHourChange(initHour, data);
      }
      return;
    }
    setIsLoading(true);
    setPrefetchComplete(false);
    loadCurrentFrames();
  }, [visible, isCurrentMode, hasBounds, dataTimestamp, loadCurrentFrames]);

  // ------------------------------------------------------------------
  // Ice forecast: load all frames
  // ------------------------------------------------------------------
  const loadIceFrames = useCallback(async () => {
    try {
      debugLog('info', 'ICE', 'Loading ice forecast frames from API...');
      const t0 = performance.now();
      const bp = paddedBounds();
      const data: IceForecastFrames = await apiClient.getIceForecastFrames(bp);
      const dt = ((performance.now() - t0) / 1000).toFixed(1);
      const frameKeys = Object.keys(data.frames);
      debugLog('info', 'ICE', `Loaded ${frameKeys.length} frames in ${dt}s, grid=${data.ny}x${data.nx}`);
      setIceFrameData(data);
      setAvailableHours(deriveHoursFromFrames(data.frames));
      let rtFormatted = data.run_time || '';
      if (data.run_time) {
        try {
          const d = new Date(data.run_time);
          rtFormatted = `${d.getUTCFullYear()}${String(d.getUTCMonth() + 1).padStart(2, '0')}${String(d.getUTCDate()).padStart(2, '0')} ${String(d.getUTCHours()).padStart(2, '0')}Z`;
          setRunTime(rtFormatted);
        } catch {
          setRunTime(data.run_time);
        }
      }
      setPrefetchComplete(true);
      setIsLoading(false);
      if (onIceForecastHourChange) {
        const initHour = nearestHourToNow(rtFormatted, data.frames);
        debugLog('info', 'ICE', `Setting initial ice frame T+${initHour}h`);
        setCurrentHour(initHour);
        onIceForecastHourChange(initHour, data);
      }
    } catch (e) {
      debugLog('error', 'ICE', `Failed to load ice forecast frames: ${e}`);
      setIsLoading(false);
    }
  }, [onIceForecastHourChange]);

  // ------------------------------------------------------------------
  // Ice prefetch effect
  // ------------------------------------------------------------------
  useEffect(() => {
    if (!visible || !isIceMode || !boundsRef.current) return;

    if (iceFrameDataRef.current) {
      const data = iceFrameDataRef.current;
      setAvailableHours(deriveHoursFromFrames(data.frames));
      setPrefetchComplete(true);
      setIsLoading(false);
      if (onIceForecastHourChange) {
        const initHour = nearestHourToNow(runTime ?? data.run_time ?? '', data.frames);
        setCurrentHour(initHour);
        onIceForecastHourChange(initHour, data);
      }
      return;
    }
    setIsLoading(true);
    setPrefetchComplete(false);
    loadIceFrames();
  }, [visible, isIceMode, hasBounds, dataTimestamp, loadIceFrames]);

  // ------------------------------------------------------------------
  // Swell prefetch effect (reuses wave data — swell is embedded in wave frames)
  // Loads wave frames into cache but fires onSwellForecastHourChange (NOT wave callback)
  // ------------------------------------------------------------------
  const loadSwellFrames = useCallback(async () => {
    try {
      debugLog('info', 'SWELL', 'Loading swell (wave) forecast frames from API...');
      const t0 = performance.now();
      const bp = paddedBounds();
      const data: WaveForecastFrames = await apiClient.getWaveForecastFrames(bp);
      const dt = ((performance.now() - t0) / 1000).toFixed(1);
      const frameKeys = Object.keys(data.frames);
      debugLog('info', 'SWELL', `Loaded ${frameKeys.length} frames in ${dt}s, grid=${data.ny}x${data.nx}`);
      setWaveFrameData(data);
      setAvailableHours(deriveHoursFromFrames(data.frames));
      setPrefetchComplete(true);
      setIsLoading(false);
      if (onSwellForecastHourChange) {
        const rtStr = data.run_time ?? '';
        const initHour = nearestHourToNow(rtStr, data.frames);
        debugLog('info', 'SWELL', `Setting initial swell frame T+${initHour}h`);
        setCurrentHour(initHour);
        onSwellForecastHourChange(initHour, data);
      }
    } catch (e) {
      debugLog('error', 'SWELL', `Failed to load swell forecast frames: ${e}`);
      setIsLoading(false);
    }
  }, [onSwellForecastHourChange]);

  useEffect(() => {
    if (!visible || !isSwellMode || !boundsRef.current) return;

    if (waveFrameDataRef.current) {
      const data = waveFrameDataRef.current;
      setAvailableHours(deriveHoursFromFrames(data.frames));
      setPrefetchComplete(true);
      setIsLoading(false);
      if (onSwellForecastHourChange) {
        const initHour = nearestHourToNow(runTime ?? data.run_time ?? '', data.frames);
        setCurrentHour(initHour);
        onSwellForecastHourChange(initHour, data);
      }
      return;
    }
    setIsLoading(true);
    setPrefetchComplete(false);
    loadSwellFrames();
  }, [visible, isSwellMode, hasBounds, dataTimestamp, loadSwellFrames]);

  // ------------------------------------------------------------------
  // SST forecast: load all frames
  // ------------------------------------------------------------------
  const loadSstFrames = useCallback(async () => {
    try {
      debugLog('info', 'SST', 'Loading SST forecast frames from API...');
      const t0 = performance.now();
      const bp = paddedBounds();
      const data: SstForecastFrames = await apiClient.getSstForecastFrames(bp);
      const dt = ((performance.now() - t0) / 1000).toFixed(1);
      const frameKeys = Object.keys(data.frames);
      debugLog('info', 'SST', `Loaded ${frameKeys.length} frames in ${dt}s, grid=${data.ny}x${data.nx}`);
      setSstFrameData(data);
      setAvailableHours(deriveHoursFromFrames(data.frames));
      let rtFormatted = data.run_time || '';
      if (data.run_time) {
        try {
          const d = new Date(data.run_time);
          rtFormatted = `${d.getUTCFullYear()}${String(d.getUTCMonth() + 1).padStart(2, '0')}${String(d.getUTCDate()).padStart(2, '0')} ${String(d.getUTCHours()).padStart(2, '0')}Z`;
          setRunTime(rtFormatted);
        } catch {
          setRunTime(data.run_time);
        }
      }
      setPrefetchComplete(true);
      setIsLoading(false);
      if (onSstForecastHourChange) {
        const initHour = nearestHourToNow(rtFormatted, data.frames);
        debugLog('info', 'SST', `Setting initial SST frame T+${initHour}h`);
        setCurrentHour(initHour);
        onSstForecastHourChange(initHour, data);
      }
    } catch (e) {
      debugLog('error', 'SST', `Failed to load SST forecast frames: ${e}`);
      setIsLoading(false);
    }
  }, [onSstForecastHourChange]);

  // ------------------------------------------------------------------
  // SST prefetch effect — direct load from DB (matches wave/current/ice pattern)
  // ------------------------------------------------------------------
  useEffect(() => {
    if (!visible || !isSstMode || !boundsRef.current) return;

    if (sstFrameDataRef.current) {
      const data = sstFrameDataRef.current;
      setAvailableHours(deriveHoursFromFrames(data.frames));
      setPrefetchComplete(true);
      setIsLoading(false);
      if (onSstForecastHourChange) {
        const initHour = nearestHourToNow(runTime ?? data.run_time ?? '', data.frames);
        setCurrentHour(initHour);
        onSstForecastHourChange(initHour, data);
      }
      return;
    }
    setIsLoading(true);
    setPrefetchComplete(false);
    loadSstFrames();
  }, [visible, isSstMode, hasBounds, dataTimestamp, loadSstFrames]);

  // ------------------------------------------------------------------
  // Visibility forecast: load all frames
  // ------------------------------------------------------------------
  const loadVisFrames = useCallback(async () => {
    try {
      debugLog('info', 'VIS', 'Loading visibility forecast frames from API...');
      const t0 = performance.now();
      const bp = paddedBounds();
      const data: VisForecastFrames = await apiClient.getVisForecastFrames(bp);
      const dt = ((performance.now() - t0) / 1000).toFixed(1);
      const frameKeys = Object.keys(data.frames);
      debugLog('info', 'VIS', `Loaded ${frameKeys.length} frames in ${dt}s, grid=${data.ny}x${data.nx}`);
      setVisFrameData(data);
      setAvailableHours(deriveHoursFromFrames(data.frames));
      let rtFormatted = data.run_time || '';
      if (data.run_time) {
        try {
          const d = new Date(data.run_time);
          rtFormatted = `${d.getUTCFullYear()}${String(d.getUTCMonth() + 1).padStart(2, '0')}${String(d.getUTCDate()).padStart(2, '0')} ${String(d.getUTCHours()).padStart(2, '0')}Z`;
          setRunTime(rtFormatted);
        } catch {
          setRunTime(data.run_time);
        }
      }
      setPrefetchComplete(true);
      setIsLoading(false);
      if (onVisForecastHourChange) {
        const initHour = nearestHourToNow(rtFormatted, data.frames);
        debugLog('info', 'VIS', `Setting initial visibility frame T+${initHour}h`);
        setCurrentHour(initHour);
        onVisForecastHourChange(initHour, data);
      }
    } catch (e) {
      debugLog('error', 'VIS', `Failed to load visibility forecast frames: ${e}`);
      setIsLoading(false);
    }
  }, [onVisForecastHourChange]);

  // ------------------------------------------------------------------
  // Visibility prefetch effect — direct load from DB (matches wave/current/ice pattern)
  // ------------------------------------------------------------------
  useEffect(() => {
    if (!visible || !isVisMode || !boundsRef.current) return;

    if (visFrameDataRef.current) {
      const data = visFrameDataRef.current;
      setAvailableHours(deriveHoursFromFrames(data.frames));
      setPrefetchComplete(true);
      setIsLoading(false);
      if (onVisForecastHourChange) {
        const initHour = nearestHourToNow(runTime ?? data.run_time ?? '', data.frames);
        setCurrentHour(initHour);
        onVisForecastHourChange(initHour, data);
      }
      return;
    }
    setIsLoading(true);
    setPrefetchComplete(false);
    loadVisFrames();
  }, [visible, isVisMode, hasBounds, dataTimestamp, loadVisFrames]);

  // ------------------------------------------------------------------
  // Play/pause
  // ------------------------------------------------------------------
  // rAF-based playback — aligned with browser paint cycle so frame
  // advances never fire while the previous render is still in progress.
  useEffect(() => {
    if (isPlaying && prefetchComplete && hasForecast) {
      const interval = SPEED_INTERVAL[speed];
      playLastTickRef.current = performance.now();

      const tick = () => {
        const now = performance.now();
        if (now - playLastTickRef.current >= interval) {
          playLastTickRef.current = now;
          setCurrentHour((prev) => {
            const fallback = isIceMode ? DEFAULT_ICE_FORECAST_HOURS : DEFAULT_FORECAST_HOURS;
            const hrs = availableHoursRef.current.length > 0 ? availableHoursRef.current : fallback;
            const idx = hrs.indexOf(prev);
            const nextIdx = idx >= 0 ? (idx + 1) % hrs.length : 0;
            const nextHour = hrs[nextIdx];

            if (isWindMode) {
              onForecastHourChange(nextHour, windFramesRef.current[String(nextHour)] || null);
            } else if (isWaveMode && onWaveForecastHourChange && waveFrameDataRef.current) {
              onWaveForecastHourChange(nextHour, waveFrameDataRef.current);
            } else if (isCurrentMode && onCurrentForecastHourChange && currentFrameDataRef.current) {
              onCurrentForecastHourChange(nextHour, currentFrameDataRef.current);
            } else if (isIceMode && onIceForecastHourChange && iceFrameDataRef.current) {
              onIceForecastHourChange(nextHour, iceFrameDataRef.current);
            } else if (isSwellMode && onSwellForecastHourChange && waveFrameDataRef.current) {
              onSwellForecastHourChange(nextHour, waveFrameDataRef.current);
            } else if (isSstMode && onSstForecastHourChange && sstFrameDataRef.current) {
              onSstForecastHourChange(nextHour, sstFrameDataRef.current);
            } else if (isVisMode && onVisForecastHourChange && visFrameDataRef.current) {
              onVisForecastHourChange(nextHour, visFrameDataRef.current);
            }
            return nextHour;
          });
        }
        playRafRef.current = requestAnimationFrame(tick);
      };

      playRafRef.current = requestAnimationFrame(tick);
    }
    return () => { if (playRafRef.current !== null) { cancelAnimationFrame(playRafRef.current); playRafRef.current = null; } };
  }, [isPlaying, speed, prefetchComplete, hasForecast, isWindMode, isWaveMode, isCurrentMode, isIceMode, isSwellMode, isSstMode, isVisMode, onForecastHourChange, onWaveForecastHourChange, onCurrentForecastHourChange, onIceForecastHourChange, onSwellForecastHourChange, onSstForecastHourChange, onVisForecastHourChange]);

  // Slider change — rAF-throttled so rapid scrubbing only renders once
  // per browser frame.  The slider UI updates immediately (setCurrentHour)
  // while heavy layer updates are deferred to the next paint.
  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const hour = parseInt(e.target.value);
    setCurrentHour(hour);
    if (sliderRafRef.current !== null) cancelAnimationFrame(sliderRafRef.current);
    sliderRafRef.current = requestAnimationFrame(() => {
      sliderRafRef.current = null;
      if (isWindMode) {
        onForecastHourChange(hour, windFrames[String(hour)] || null);
      } else if (isWaveMode && onWaveForecastHourChange && waveFrameData) {
        onWaveForecastHourChange(hour, waveFrameData);
      } else if (isCurrentMode && onCurrentForecastHourChange && currentFrameData) {
        onCurrentForecastHourChange(hour, currentFrameData);
      } else if (isIceMode && onIceForecastHourChange && iceFrameData) {
        onIceForecastHourChange(hour, iceFrameData);
      } else if (isSwellMode && onSwellForecastHourChange && waveFrameData) {
        onSwellForecastHourChange(hour, waveFrameData);
      } else if (isSstMode && onSstForecastHourChange && sstFrameData) {
        onSstForecastHourChange(hour, sstFrameData);
      } else if (isVisMode && onVisForecastHourChange && visFrameData) {
        onVisForecastHourChange(hour, visFrameData);
      }
    });
  };

  // Close
  const handleClose = () => {
    setIsPlaying(false);
    if (sliderRafRef.current !== null) { cancelAnimationFrame(sliderRafRef.current); sliderRafRef.current = null; }
    setCurrentHour(0);
    onForecastHourChange(0, null);
    if (onWaveForecastHourChange) onWaveForecastHourChange(0, null);
    if (onCurrentForecastHourChange) onCurrentForecastHourChange(0, null);
    if (onIceForecastHourChange) onIceForecastHourChange(0, null);
    if (onSwellForecastHourChange) onSwellForecastHourChange(0, null);
    if (onSstForecastHourChange) onSstForecastHourChange(0, null);
    if (onVisForecastHourChange) onVisForecastHourChange(0, null);
    onClose();
  };

  // Parse runTime into a Date (shared by formatValidTime + slider labels)
  const parseRunTime = (): Date | null => {
    if (!runTime) return null;
    try {
      const parts = runTime.split(' ');
      const dateStr = parts[0];
      const hourStr = parts[1]?.replace('Z', '') || '0';
      return new Date(
        Date.UTC(
          parseInt(dateStr.slice(0, 4)),
          parseInt(dateStr.slice(4, 6)) - 1,
          parseInt(dateStr.slice(6, 8)),
          parseInt(hourStr)
        )
      );
    } catch { return null; }
  };

  const DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

  // Format valid time — shown in the main time display
  const formatValidTime = (fh: number): string => {
    const base = parseRunTime();
    if (!base) return `T+${fh}h`;
    try {
      const valid = new Date(base.getTime() + fh * 3600_000);
      return `${DAYS[valid.getUTCDay()]} ${valid.getUTCDate()} ${MONTHS[valid.getUTCMonth()]} ${String(valid.getUTCHours()).padStart(2, '0')}:00 UTC`;
    } catch {
      return `T+${fh}h`;
    }
  };

  // Compact label for slider ticks
  const formatSliderTick = (fh: number): string => {
    const base = parseRunTime();
    if (!base) return isIceMode ? `Day ${fh / 24}` : `${fh}h`;
    try {
      const valid = new Date(base.getTime() + fh * 3600_000);
      if (isIceMode) {
        return `${DAYS[valid.getUTCDay()]} ${valid.getUTCDate()} ${MONTHS[valid.getUTCMonth()]}`;
      }
      return `${DAYS[valid.getUTCDay()]} ${valid.getUTCDate()}\n${String(valid.getUTCHours()).padStart(2, '0')}:00`;
    } catch {
      return `${fh}h`;
    }
  };

  const formatDataTimestamp = (iso: string): string => {
    try {
      const d = new Date(iso);
      return `${DAYS[d.getUTCDay()]} ${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${String(d.getUTCHours()).padStart(2, '0')}:${String(d.getUTCMinutes()).padStart(2, '0')} UTC`;
    } catch {
      return iso;
    }
  };

  if (!visible) return null;

  // Currents only: static info bar (no forecast model for currents)
  if (!hasForecast) {
    return (
      <div className="absolute bottom-0 left-0 right-0 z-[1001] bg-maritime-dark/95 backdrop-blur-sm border-t border-white/10">
        <div className="px-4 py-3 flex items-center gap-4">
          <div className="flex-shrink-0 w-9 h-9 flex items-center justify-center rounded-full bg-gray-700 text-gray-400">
            <Info className="w-4 h-4" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-1.5 text-white text-sm font-medium">
              <Clock className="w-3.5 h-3.5 text-primary-400" />
              <span>Currents — Live data</span>
              {dataTimestamp && (
                <>
                  <span className="text-gray-400 text-xs">|</span>
                  <span className="text-gray-300 text-xs">Updated {formatDataTimestamp(dataTimestamp)}</span>
                </>
              )}
            </div>
            <div className="text-xs text-gray-500 mt-0.5">
              Forecast timeline available for Wind, Waves, Currents and Ice layers
            </div>
          </div>
          <button onClick={handleClose} className="flex-shrink-0 w-7 h-7 flex items-center justify-center rounded text-gray-400 hover:text-white hover:bg-gray-700 transition-colors">
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>
    );
  }

  // Wind / Waves / Currents / Ice: full scrubber
  const defaultLabel = isWindMode ? 'Wind Speed' : isWaveMode ? 'Waves' : isCurrentMode ? 'Currents' : isIceMode ? 'Ice' : isSwellMode ? 'Swell' : isSstMode ? 'Sea Surface Temp' : 'Visibility';
  const layerLabel = displayLayerName || defaultLabel;

  // Color-code: green when forecast data loaded, gray when not
  const sourceColor = (() => {
    if (isWindMode && Object.keys(windFrames).length > 0) return 'text-green-400';
    if (isWaveMode && waveFrameData) return 'text-green-400';
    if (isCurrentMode && currentFrameData) return 'text-green-400';
    if (isIceMode && iceFrameData) return 'text-green-400';
    if (isSwellMode && waveFrameData) return 'text-green-400';
    if (isSstMode && sstFrameData) return 'text-green-400';
    if (isVisMode && visFrameData) return 'text-green-400';
    return 'text-gray-500';
  })();
  const hasData = sourceColor === 'text-green-400';

  return (
    <div className="absolute bottom-0 left-0 right-0 z-[1001] bg-maritime-dark/95 backdrop-blur-sm border-t border-white/10">
      {isLoading && (
        <div className="h-1 bg-gray-700">
          <div
            className="h-full bg-primary-500 transition-all duration-300"
            style={{ width: `${loadProgress.total > 0 ? (loadProgress.cached / loadProgress.total) * 100 : 0}%` }}
          />
        </div>
      )}

      <div className="px-4 py-3 flex items-center gap-4">
        {/* Play/Pause */}
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          disabled={!prefetchComplete}
          className="flex-shrink-0 w-9 h-9 flex items-center justify-center rounded-full bg-primary-500 text-white disabled:opacity-40 disabled:cursor-not-allowed hover:bg-primary-400 transition-colors"
        >
          {isLoading ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : isPlaying ? (
            <Pause className="w-4 h-4" />
          ) : (
            <Play className="w-4 h-4 ml-0.5" />
          )}
        </button>

        {/* Time display */}
        <div className="flex-shrink-0 min-w-[170px]">
          <div className="flex items-center gap-1.5 text-white text-sm font-medium">
            <Clock className="w-3.5 h-3.5 text-primary-400" />
            <span>{layerLabel} {isIceMode ? `Day ${currentHour / 24}` : `T+${currentHour}h`}</span>
            <span className="text-gray-400 text-xs">|</span>
            <span className="text-gray-300 text-xs">{formatValidTime(currentHour)}</span>
          </div>
          {isLoading && (
            <div className="text-xs text-gray-500 mt-0.5">
              {loadProgress.total > 0
                ? `Loading ${loadProgress.cached}/${loadProgress.total} frames...`
                : 'Loading frames...'}
            </div>
          )}
        </div>

        {/* Slider */}
        <div className="flex-1 min-w-0">
          <input
            type="range"
            min={0}
            max={sliderMax}
            step={sliderStep}
            value={currentHour}
            onChange={handleSliderChange}
            disabled={!prefetchComplete}
            className={`w-full h-2 rounded-full appearance-none cursor-pointer ${hasData ? 'bg-green-900/50' : 'bg-gray-700'} disabled:opacity-40 disabled:cursor-not-allowed
              [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-primary-400
              [&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-primary-400 [&::-moz-range-thumb]:border-0`}
          />
          <div className="flex justify-between mt-1 text-[10px] leading-tight">
            {(() => {
              // Generate ~6-10 evenly-spaced labels from the active hours
              const maxLabels = isIceMode ? activeHours.length : 7;
              const labelStep = Math.max(1, Math.floor((activeHours.length - 1) / (maxLabels - 1)));
              const indices: number[] = [];
              for (let i = 0; i < activeHours.length; i += labelStep) indices.push(i);
              if (indices[indices.length - 1] !== activeHours.length - 1) indices.push(activeHours.length - 1);
              return indices.map(i => {
                const h = activeHours[i];
                const label = formatSliderTick(h);
                const lines = label.split('\n');
                return (
                  <span key={h} className={`${sourceColor} text-center whitespace-pre`}>
                    {lines.length > 1 ? <>{lines[0]}<br/>{lines[1]}</> : lines[0]}
                  </span>
                );
              });
            })()}
          </div>
        </div>

        {/* Speed */}
        <div className="flex-shrink-0 flex items-center gap-1">
          {SPEED_OPTIONS.map((s) => (
            <button
              key={s}
              onClick={() => setSpeed(s)}
              className={`px-2 py-1 rounded text-xs font-medium transition-colors ${
                speed === s ? 'bg-primary-500 text-white' : 'bg-gray-700 text-gray-400 hover:text-white'
              }`}
            >
              {s}x
            </button>
          ))}
        </div>

        {/* Close */}
        <button onClick={handleClose} className="flex-shrink-0 w-7 h-7 flex items-center justify-center rounded text-gray-400 hover:text-white hover:bg-gray-700 transition-colors">
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}

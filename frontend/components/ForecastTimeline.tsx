'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { Play, Pause, X, Clock, Loader2, Info } from 'lucide-react';
import { apiClient, VelocityData, ForecastFrames, WaveForecastFrames, WaveForecastFrame, CurrentForecastFrames } from '@/lib/api';
import { debugLog } from '@/lib/debugLog';

type LayerType = 'wind' | 'waves' | 'currents';

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
  layerType?: LayerType;
  viewportBounds?: ViewportBounds | null;
  dataTimestamp?: string | null;
}

const FORECAST_HOURS = Array.from({ length: 41 }, (_, i) => i * 3); // 0,3,6,...,120
const SPEED_OPTIONS = [1, 2, 4];
const SPEED_INTERVAL: Record<number, number> = { 1: 2000, 2: 1000, 4: 500 };

export default function ForecastTimeline({
  visible,
  onClose,
  onForecastHourChange,
  onWaveForecastHourChange,
  onCurrentForecastHourChange,
  layerType = 'wind',
  viewportBounds,
  dataTimestamp,
}: ForecastTimelineProps) {
  const isWindMode = layerType === 'wind';
  const isWaveMode = layerType === 'waves';
  const isCurrentMode = layerType === 'currents';
  const hasForecast = isWindMode || isWaveMode || isCurrentMode;

  const [currentHour, setCurrentHour] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [loadProgress, setLoadProgress] = useState({ cached: 0, total: 41 });
  const [runTime, setRunTime] = useState<string | null>(null);
  const [prefetchComplete, setPrefetchComplete] = useState(false);

  // Wind frames
  const [windFrames, setWindFrames] = useState<Record<string, VelocityData[]>>({});
  const windFramesRef = useRef<Record<string, VelocityData[]>>({});

  // Wave frames (full response with shared metadata)
  const [waveFrameData, setWaveFrameData] = useState<WaveForecastFrames | null>(null);
  const waveFrameDataRef = useRef<WaveForecastFrames | null>(null);

  // Current frames
  const [currentFrameData, setCurrentFrameData] = useState<CurrentForecastFrames | null>(null);
  const currentFrameDataRef = useRef<CurrentForecastFrames | null>(null);

  const playIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Snapshot viewport bounds into a ref — never let it revert to null once set
  const boundsRef = useRef<ViewportBounds | null>(viewportBounds ?? null);
  const hasBounds = boundsRef.current !== null;
  useEffect(() => {
    if (viewportBounds) boundsRef.current = viewportBounds;
  }, [viewportBounds]);

  // Keep refs in sync
  useEffect(() => { windFramesRef.current = windFrames; }, [windFrames]);
  useEffect(() => { waveFrameDataRef.current = waveFrameData; }, [waveFrameData]);
  useEffect(() => { currentFrameDataRef.current = currentFrameData; }, [currentFrameData]);

  // ------------------------------------------------------------------
  // Wind forecast: load all frames
  // ------------------------------------------------------------------
  const loadWindFrames = useCallback(async () => {
    try {
      const bp = boundsRef.current ?? {};
      const data: ForecastFrames = await apiClient.getForecastFrames(bp);
      setWindFrames(data.frames);
      setRunTime(`${data.run_date} ${data.run_hour}Z`);
      setPrefetchComplete(true);
      setIsLoading(false);
      if (data.frames['0']) onForecastHourChange(0, data.frames['0']);
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
      const bp = boundsRef.current ?? {};
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
      const rt = data.run_time;
      if (rt) {
        try {
          const d = new Date(rt);
          setRunTime(
            `${d.getUTCFullYear()}${String(d.getUTCMonth() + 1).padStart(2, '0')}${String(d.getUTCDate()).padStart(2, '0')} ${String(d.getUTCHours()).padStart(2, '0')}Z`
          );
        } catch {
          setRunTime(rt);
        }
      }
      setPrefetchComplete(true);
      setIsLoading(false);
      if (data.frames['0'] && onWaveForecastHourChange) {
        debugLog('info', 'WAVE', 'Setting initial frame T+0h');
        onWaveForecastHourChange(0, data);
      }
    } catch (e) {
      debugLog('error', 'WAVE', `Failed to load wave forecast frames: ${e}`);
      setIsLoading(false);
    }
  }, [onWaveForecastHourChange]);

  // ------------------------------------------------------------------
  // Wind prefetch effect
  // ------------------------------------------------------------------
  useEffect(() => {
    if (!visible || !isWindMode || !boundsRef.current) return;

    let cancelled = false;
    const bp = boundsRef.current;

    const start = async () => {
      setIsLoading(true);
      setPrefetchComplete(false);
      try {
        await apiClient.triggerForecastPrefetch(bp);
        const poll = async () => {
          if (cancelled) return;
          try {
            const st = await apiClient.getForecastStatus(bp);
            setLoadProgress({ cached: st.cached_hours, total: st.total_hours });
            setRunTime(`${st.run_date} ${st.run_hour}Z`);
            if (st.complete || st.cached_hours === st.total_hours) {
              if (pollIntervalRef.current) { clearInterval(pollIntervalRef.current); pollIntervalRef.current = null; }
              await loadWindFrames();
            }
          } catch (e) { console.error('Wind forecast poll failed:', e); }
        };
        await poll();
        pollIntervalRef.current = setInterval(poll, 3000);
      } catch (e) {
        console.error('Wind forecast prefetch trigger failed:', e);
        setIsLoading(false);
      }
    };

    start();
    return () => { cancelled = true; if (pollIntervalRef.current) { clearInterval(pollIntervalRef.current); pollIntervalRef.current = null; } };
  }, [visible, isWindMode, hasBounds, loadWindFrames]);

  // ------------------------------------------------------------------
  // Wave prefetch effect
  // ------------------------------------------------------------------
  useEffect(() => {
    if (!visible || !isWaveMode || !boundsRef.current) return;

    let cancelled = false;
    const bp = boundsRef.current;

    const start = async () => {
      setIsLoading(true);
      setPrefetchComplete(false);
      try {
        await apiClient.triggerWaveForecastPrefetch(bp);
        const poll = async () => {
          if (cancelled) return;
          try {
            const st = await apiClient.getWaveForecastStatus(bp);
            setLoadProgress({ cached: st.cached_hours, total: st.total_hours });
            if (st.complete || st.cached_hours === st.total_hours) {
              if (pollIntervalRef.current) { clearInterval(pollIntervalRef.current); pollIntervalRef.current = null; }
              await loadWaveFrames();
            }
          } catch (e) { console.error('Wave forecast poll failed:', e); }
        };
        await poll();
        pollIntervalRef.current = setInterval(poll, 5000); // wave download is slower, poll less often
      } catch (e) {
        console.error('Wave forecast prefetch trigger failed:', e);
        setIsLoading(false);
      }
    };

    start();
    return () => { cancelled = true; if (pollIntervalRef.current) { clearInterval(pollIntervalRef.current); pollIntervalRef.current = null; } };
  }, [visible, isWaveMode, hasBounds, loadWaveFrames]);

  // ------------------------------------------------------------------
  // Current forecast: load all frames
  // ------------------------------------------------------------------
  const loadCurrentFrames = useCallback(async () => {
    try {
      debugLog('info', 'CURRENT', 'Loading current forecast frames from API...');
      const t0 = performance.now();
      const bp = boundsRef.current ?? {};
      const data: CurrentForecastFrames = await apiClient.getCurrentForecastFrames(bp);
      const dt = ((performance.now() - t0) / 1000).toFixed(1);
      const frameKeys = Object.keys(data.frames);
      debugLog('info', 'CURRENT', `Loaded ${frameKeys.length} frames in ${dt}s, grid=${data.ny}x${data.nx}`);
      setCurrentFrameData(data);
      if (data.run_time) {
        try {
          const d = new Date(data.run_time);
          setRunTime(
            `${d.getUTCFullYear()}${String(d.getUTCMonth() + 1).padStart(2, '0')}${String(d.getUTCDate()).padStart(2, '0')} ${String(d.getUTCHours()).padStart(2, '0')}Z`
          );
        } catch {
          setRunTime(data.run_time);
        }
      }
      setPrefetchComplete(true);
      setIsLoading(false);
      if (data.frames['0'] && onCurrentForecastHourChange) {
        debugLog('info', 'CURRENT', 'Setting initial current frame T+0h');
        onCurrentForecastHourChange(0, data);
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

    let cancelled = false;
    const bp = boundsRef.current;

    const start = async () => {
      setIsLoading(true);
      setPrefetchComplete(false);
      debugLog('info', 'CURRENT', 'Triggering current forecast prefetch...');
      try {
        await apiClient.triggerCurrentForecastPrefetch(bp);
        const poll = async () => {
          if (cancelled) return;
          try {
            const st = await apiClient.getCurrentForecastStatus(bp);
            setLoadProgress({ cached: st.cached_hours, total: st.total_hours });
            if (st.complete || st.cached_hours === st.total_hours) {
              if (pollIntervalRef.current) { clearInterval(pollIntervalRef.current); pollIntervalRef.current = null; }
              await loadCurrentFrames();
            }
          } catch (e) { debugLog('error', 'CURRENT', `Current forecast poll failed: ${e}`); }
        };
        await poll();
        pollIntervalRef.current = setInterval(poll, 5000);
      } catch (e) {
        debugLog('error', 'CURRENT', `Current forecast prefetch trigger failed: ${e}`);
        setIsLoading(false);
      }
    };

    start();
    return () => { cancelled = true; if (pollIntervalRef.current) { clearInterval(pollIntervalRef.current); pollIntervalRef.current = null; } };
  }, [visible, isCurrentMode, hasBounds, loadCurrentFrames]);

  // ------------------------------------------------------------------
  // Play/pause
  // ------------------------------------------------------------------
  useEffect(() => {
    if (isPlaying && prefetchComplete && hasForecast) {
      playIntervalRef.current = setInterval(() => {
        setCurrentHour((prev) => {
          const idx = FORECAST_HOURS.indexOf(prev);
          const nextIdx = (idx + 1) % FORECAST_HOURS.length;
          const nextHour = FORECAST_HOURS[nextIdx];

          if (isWindMode) {
            const fd = windFramesRef.current[String(nextHour)] || null;
            onForecastHourChange(nextHour, fd);
          } else if (isWaveMode && onWaveForecastHourChange && waveFrameDataRef.current) {
            onWaveForecastHourChange(nextHour, waveFrameDataRef.current);
          } else if (isCurrentMode && onCurrentForecastHourChange && currentFrameDataRef.current) {
            onCurrentForecastHourChange(nextHour, currentFrameDataRef.current);
          }
          return nextHour;
        });
      }, SPEED_INTERVAL[speed]);
    }
    return () => { if (playIntervalRef.current) { clearInterval(playIntervalRef.current); playIntervalRef.current = null; } };
  }, [isPlaying, speed, prefetchComplete, hasForecast, isWindMode, isWaveMode, isCurrentMode, onForecastHourChange, onWaveForecastHourChange, onCurrentForecastHourChange]);

  // Slider change
  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const hour = parseInt(e.target.value);
    setCurrentHour(hour);
    if (isWindMode) {
      onForecastHourChange(hour, windFrames[String(hour)] || null);
    } else if (isWaveMode && onWaveForecastHourChange && waveFrameData) {
      onWaveForecastHourChange(hour, waveFrameData);
    } else if (isCurrentMode && onCurrentForecastHourChange && currentFrameData) {
      onCurrentForecastHourChange(hour, currentFrameData);
    }
  };

  // Close
  const handleClose = () => {
    setIsPlaying(false);
    setCurrentHour(0);
    onForecastHourChange(0, null);
    if (onWaveForecastHourChange) onWaveForecastHourChange(0, null);
    if (onCurrentForecastHourChange) onCurrentForecastHourChange(0, null);
    onClose();
  };

  // Format valid time
  const formatValidTime = (fh: number): string => {
    if (!runTime) return `T+${fh}h`;
    try {
      const parts = runTime.split(' ');
      const dateStr = parts[0];
      const hourStr = parts[1]?.replace('Z', '') || '0';
      const base = new Date(
        parseInt(dateStr.slice(0, 4)),
        parseInt(dateStr.slice(4, 6)) - 1,
        parseInt(dateStr.slice(6, 8)),
        parseInt(hourStr)
      );
      base.setHours(base.getHours() + fh);
      const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
      return `${days[base.getDay()]} ${String(base.getUTCHours()).padStart(2, '0')}:00 UTC`;
    } catch {
      return `T+${fh}h`;
    }
  };

  const formatDataTimestamp = (iso: string): string => {
    try {
      const d = new Date(iso);
      const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
      return `${days[d.getUTCDay()]} ${String(d.getUTCHours()).padStart(2, '0')}:${String(d.getUTCMinutes()).padStart(2, '0')} UTC`;
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
              Forecast timeline available for Wind and Waves layers
            </div>
          </div>
          <button onClick={handleClose} className="flex-shrink-0 w-7 h-7 flex items-center justify-center rounded text-gray-400 hover:text-white hover:bg-gray-700 transition-colors">
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>
    );
  }

  // Wind / Waves / Currents: full scrubber
  const layerLabel = isWindMode ? 'Wind' : isWaveMode ? 'Waves' : 'Currents';

  return (
    <div className="absolute bottom-0 left-0 right-0 z-[1001] bg-maritime-dark/95 backdrop-blur-sm border-t border-white/10">
      {isLoading && (
        <div className="h-1 bg-gray-700">
          <div
            className="h-full bg-primary-500 transition-all duration-300"
            style={{ width: `${(loadProgress.cached / loadProgress.total) * 100}%` }}
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
            <span>{layerLabel} T+{currentHour}h</span>
            <span className="text-gray-400 text-xs">|</span>
            <span className="text-gray-300 text-xs">{formatValidTime(currentHour)}</span>
          </div>
          {isLoading && (
            <div className="text-xs text-gray-500 mt-0.5">
              Loading {loadProgress.cached}/{loadProgress.total} frames...
            </div>
          )}
        </div>

        {/* Slider */}
        <div className="flex-1 min-w-0">
          <input
            type="range"
            min={0}
            max={120}
            step={3}
            value={currentHour}
            onChange={handleSliderChange}
            disabled={!prefetchComplete}
            className="w-full h-2 rounded-full appearance-none cursor-pointer bg-gray-700 disabled:opacity-40 disabled:cursor-not-allowed
              [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-primary-400
              [&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-primary-400 [&::-moz-range-thumb]:border-0"
          />
          <div className="flex justify-between mt-1 text-[10px] text-gray-500">
            <span>0h</span><span>24h</span><span>48h</span><span>72h</span><span>96h</span><span>120h</span>
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

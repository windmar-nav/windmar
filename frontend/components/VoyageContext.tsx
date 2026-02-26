'use client';

import { createContext, useContext, useState, useEffect, useCallback, useRef, ReactNode } from 'react';
import type { Position, AllOptimizationResults, RouteVisibility } from '@/lib/api';
import { EMPTY_ALL_RESULTS, DEFAULT_ROUTE_VISIBILITY, apiClient } from '@/lib/api';

const ZONE_TYPES = ['eca', 'hra', 'tss', 'vts', 'ice', 'canal', 'environmental', 'exclusion'] as const;

// ── SessionStorage-backed state ──
// Survives full page reloads and hard navigations, clears when tab closes.
function useSessionState<T>(key: string, fallback: T): [T, (v: T | ((prev: T) => T)) => void] {
  const [value, setValue] = useState<T>(() => {
    if (typeof window === 'undefined') return fallback;
    try {
      const stored = sessionStorage.getItem(key);
      return stored ? JSON.parse(stored) : fallback;
    } catch { return fallback; }
  });
  const isFirstRender = useRef(true);
  useEffect(() => {
    if (isFirstRender.current) { isFirstRender.current = false; return; }
    try { sessionStorage.setItem(key, JSON.stringify(value)); }
    catch { /* quota exceeded — non-critical */ }
  }, [key, value]);
  return [value, setValue];
}

type WeatherLayerType = 'wind' | 'waves' | 'currents' | 'ice' | 'visibility' | 'sst' | 'swell' | 'none';

interface ViewportState {
  bounds: { lat_min: number; lat_max: number; lon_min: number; lon_max: number };
  zoom: number;
}

interface VoyageContextValue {
  // View mode
  viewMode: 'weather' | 'analysis';
  setViewMode: (v: 'weather' | 'analysis') => void;

  // Departure time (ISO datetime-local string)
  departureTime: string;
  setDepartureTime: (v: string) => void;

  // Voyage params
  calmSpeed: number;
  setCalmSpeed: (v: number) => void;
  isLaden: boolean;
  setIsLaden: (v: boolean) => void;
  useWeather: boolean;
  setUseWeather: (v: boolean) => void;

  // Route state (persisted across navigation)
  waypoints: Position[];
  setWaypoints: (v: Position[]) => void;
  routeName: string;
  setRouteName: (v: string) => void;
  allResults: AllOptimizationResults;
  setAllResults: (v: AllOptimizationResults) => void;
  routeVisibility: RouteVisibility;
  setRouteVisibility: (v: RouteVisibility) => void;

  // Zone visibility per type — all false by default
  zoneVisibility: Record<string, boolean>;
  setZoneTypeVisible: (type: string, visible: boolean) => void;
  isDrawingZone: boolean;
  setIsDrawingZone: (v: boolean) => void;

  // Weather layer persistence
  weatherLayer: WeatherLayerType;
  setWeatherLayer: (v: WeatherLayerType) => void;

  // Viewport persistence
  lastViewport: ViewportState | null;
  setLastViewport: (v: ViewportState) => void;

  // Optimization settings (persisted across navigation)
  gridResolution: number;
  setGridResolution: (v: number) => void;
  variableResolution: boolean;
  setVariableResolution: (v: boolean) => void;
  paretoEnabled: boolean;
  setParetoEnabled: (v: boolean) => void;

  // Variable speed (voyage calculation)
  variableSpeed: boolean;
  setVariableSpeed: (v: boolean) => void;

  // Displayed analysis (persisted across navigation)
  displayedAnalysisId: string | null;
  setDisplayedAnalysisId: (v: string | null) => void;

  // Sync speed from backend vessel specs
  refreshSpecs: () => Promise<void>;
}

const VoyageContext = createContext<VoyageContextValue | null>(null);

export function VoyageProvider({ children }: { children: ReactNode }) {
  // ── Session-persisted state (survives full page reloads) ──
  const [viewMode, setViewMode] = useSessionState<'weather' | 'analysis'>('wm:viewMode', 'weather');
  const [departureTime, setDepartureTime] = useSessionState('wm:departureTime', new Date().toISOString().slice(0, 16));
  const [calmSpeed, setCalmSpeed] = useSessionState('wm:calmSpeed', 14.5);
  const [isLaden, setIsLaden] = useSessionState('wm:isLaden', true);
  const [useWeather, setUseWeather] = useSessionState('wm:useWeather', true);
  const [weatherLayer, setWeatherLayer] = useSessionState<WeatherLayerType>('wm:weatherLayer', 'none');
  const [lastViewport, setLastViewport] = useSessionState<ViewportState | null>('wm:lastViewport', null);

  // Optimization settings
  const [gridResolution, setGridResolution] = useSessionState('wm:gridRes', 0.2);
  const [variableResolution, setVariableResolution] = useSessionState('wm:varRes', true);
  const [paretoEnabled, setParetoEnabled] = useSessionState('wm:pareto', false);
  const [variableSpeed, setVariableSpeed] = useSessionState('wm:varSpeed', false);
  const [displayedAnalysisId, setDisplayedAnalysisId] = useSessionState<string | null>('wm:analysisId', null);

  // Route state (session-persisted)
  const [waypoints, setWaypoints] = useSessionState<Position[]>('wm:waypoints', []);
  const [routeName, setRouteName] = useSessionState('wm:routeName', 'Custom Route');
  const [allResults, setAllResults] = useSessionState<AllOptimizationResults>('wm:allResults', EMPTY_ALL_RESULTS);
  const [routeVisibility, setRouteVisibility] = useSessionState<RouteVisibility>('wm:routeVis', DEFAULT_ROUTE_VISIBILITY);

  // ── Ephemeral state (not worth persisting) ──
  const [isDrawingZone, setIsDrawingZone] = useState(false);

  // Cache backend vessel speeds so laden/ballast toggle can pick the right one
  const [vesselSpeeds, setVesselSpeeds] = useState<{ laden: number; ballast: number } | null>(null);

  const refreshSpecs = useCallback(async () => {
    try {
      const specs = await apiClient.getVesselSpecs();
      setVesselSpeeds({ laden: specs.service_speed_laden, ballast: specs.service_speed_ballast });
    } catch {
      // Keep default if API unreachable
    }
  }, []);

  // Load vessel specs from backend on mount
  useEffect(() => { refreshSpecs(); }, [refreshSpecs]);

  // Sync calmSpeed when vessel speeds are loaded or laden/ballast toggles
  useEffect(() => {
    if (vesselSpeeds) {
      setCalmSpeed(isLaden ? vesselSpeeds.laden : vesselSpeeds.ballast);
    }
  }, [vesselSpeeds, isLaden]);

  const [zoneVisibility, setZoneVisibility] = useState<Record<string, boolean>>(() => {
    const init: Record<string, boolean> = {};
    for (const t of ZONE_TYPES) init[t] = false;
    // TSS on by default — routes must comply with traffic separation schemes
    init['tss'] = true;
    return init;
  });

  const setZoneTypeVisible = (type: string, visible: boolean) => {
    setZoneVisibility((prev) => ({ ...prev, [type]: visible }));
  };

  return (
    <VoyageContext.Provider
      value={{
        viewMode, setViewMode,
        departureTime, setDepartureTime,
        calmSpeed, setCalmSpeed,
        isLaden, setIsLaden,
        useWeather, setUseWeather,
        waypoints, setWaypoints,
        routeName, setRouteName,
        allResults, setAllResults,
        routeVisibility, setRouteVisibility,
        zoneVisibility, setZoneTypeVisible,
        isDrawingZone, setIsDrawingZone,
        weatherLayer, setWeatherLayer,
        lastViewport, setLastViewport,
        gridResolution, setGridResolution,
        variableResolution, setVariableResolution,
        paretoEnabled, setParetoEnabled,
        variableSpeed, setVariableSpeed,
        displayedAnalysisId, setDisplayedAnalysisId,
        refreshSpecs,
      }}
    >
      {children}
    </VoyageContext.Provider>
  );
}

export function useVoyage(): VoyageContextValue {
  const ctx = useContext(VoyageContext);
  if (!ctx) throw new Error('useVoyage must be used within VoyageProvider');
  return ctx;
}

export { ZONE_TYPES };

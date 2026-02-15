'use client';

import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import type { Position, AllOptimizationResults, RouteVisibility } from '@/lib/api';
import { EMPTY_ALL_RESULTS, DEFAULT_ROUTE_VISIBILITY, apiClient } from '@/lib/api';

const ZONE_TYPES = ['eca', 'seca', 'hra', 'tss', 'vts', 'ice', 'canal', 'environmental', 'exclusion'] as const;

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

  // Sync speed from backend vessel specs
  refreshSpecs: () => Promise<void>;
}

const VoyageContext = createContext<VoyageContextValue | null>(null);

export function VoyageProvider({ children }: { children: ReactNode }) {
  const [viewMode, setViewMode] = useState<'weather' | 'analysis'>('weather');
  const [departureTime, setDepartureTime] = useState(
    () => new Date().toISOString().slice(0, 16),
  );
  const [calmSpeed, setCalmSpeed] = useState(13);
  const [isLaden, setIsLaden] = useState(true);
  const [useWeather, setUseWeather] = useState(true);
  const [isDrawingZone, setIsDrawingZone] = useState(false);

  // Route state (persisted)
  const [waypoints, setWaypoints] = useState<Position[]>([]);
  const [routeName, setRouteName] = useState('Custom Route');
  const [allResults, setAllResults] = useState<AllOptimizationResults>(EMPTY_ALL_RESULTS);
  const [routeVisibility, setRouteVisibility] = useState<RouteVisibility>(DEFAULT_ROUTE_VISIBILITY);

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

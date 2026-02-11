'use client';

import { useState, useRef } from 'react';
import {
  PenLine, Upload, Navigation, Compass, Loader2, ChevronDown, ChevronUp,
  Trash2, MapPin, Play, Eye, EyeOff,
} from 'lucide-react';
import { Position, OptimizationResponse, SpeedScenario, EngineType, DualOptimizationResults, RouteVisibility } from '@/lib/api';
import SavedRoutes from '@/components/SavedRoutes';
import RouteImport from '@/components/RouteImport';

interface RouteIndicatorPanelProps {
  waypoints: Position[];
  onWaypointsChange: (wps: Position[]) => void;
  routeName: string;
  totalDistance: number;
  isEditing: boolean;
  onIsEditingChange: (editing: boolean) => void;
  isCalculating: boolean;
  onCalculate: () => void;
  isOptimizing: boolean;
  onOptimize: () => void;
  optimizationResults: DualOptimizationResults;
  onApplyOptimizedRoute: (engine: EngineType) => void;
  onDismissOptimizedRoute: () => void;
  routeVisibility: RouteVisibility;
  onRouteVisibilityChange: (v: RouteVisibility) => void;
  onRouteImport: (waypoints: Position[], name: string) => void;
  onLoadRoute: (waypoints: Position[]) => void;
  onClearRoute: () => void;
  // Whether a voyage baseline has been calculated (gates Optimize button)
  hasBaseline?: boolean;
  // Analysis result summary (if displayed)
  analysisFuel?: number;
  analysisTime?: number;
  analysisAvgSpeed?: number;
}

export default function RouteIndicatorPanel({
  waypoints,
  onWaypointsChange,
  routeName,
  totalDistance,
  isEditing,
  onIsEditingChange,
  isCalculating,
  onCalculate,
  isOptimizing,
  onOptimize,
  optimizationResults,
  onApplyOptimizedRoute,
  onDismissOptimizedRoute,
  onRouteImport,
  routeVisibility,
  onRouteVisibilityChange,
  onLoadRoute,
  onClearRoute,
  hasBaseline,
  analysisFuel,
  analysisTime,
  analysisAvgSpeed,
}: RouteIndicatorPanelProps) {
  const [savedRoutesExpanded, setSavedRoutesExpanded] = useState(false);
  const [showImport, setShowImport] = useState(false);
  const [selectedStrategy, setSelectedStrategy] = useState<'constant_speed' | 'match_eta'>('match_eta');
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Calculate bearing from first to last WP
  const bearing = waypoints.length >= 2
    ? calculateBearing(waypoints[0], waypoints[waypoints.length - 1])
    : 0;

  const bearingLabel = bearingToCardinal(bearing);

  // No route state
  if (waypoints.length === 0) {
    return (
      <div className="absolute bottom-4 left-4 z-[1000] w-72">
        <div className="bg-maritime-dark/90 backdrop-blur-md border border-white/10 rounded-xl shadow-2xl p-4">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-medium text-gray-400">No Route Selected</span>
            <div className="flex items-center gap-1">
              <button
                onClick={() => onIsEditingChange(true)}
                className={`p-1.5 rounded transition-colors ${
                  isEditing ? 'bg-primary-500/20 text-primary-400' : 'text-gray-500 hover:text-white'
                }`}
                title="Draw route on map"
              >
                <PenLine className="w-4 h-4" />
              </button>
              <button
                onClick={() => setShowImport(!showImport)}
                className={`p-1.5 rounded transition-colors ${
                  showImport ? 'bg-primary-500/20 text-primary-400' : 'text-gray-500 hover:text-white'
                }`}
                title="Import RTZ"
              >
                <Upload className="w-4 h-4" />
              </button>
            </div>
          </div>
          <p className="text-xs text-gray-500">Draw on map or import RTZ</p>

          {showImport && (
            <div className="mt-3 pt-3 border-t border-white/10">
              <RouteImport onImport={onRouteImport} />
            </div>
          )}

          {/* Saved Routes */}
          <div className="mt-3 pt-3 border-t border-white/10">
            <button
              onClick={() => setSavedRoutesExpanded(!savedRoutesExpanded)}
              className="flex items-center justify-between w-full text-sm text-gray-400 hover:text-white transition-colors"
            >
              <span>Saved Routes</span>
              {savedRoutesExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>
            {savedRoutesExpanded && (
              <div className="mt-2">
                <SavedRoutes currentWaypoints={waypoints} onLoadRoute={onLoadRoute} />
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Route loaded
  const firstWp = waypoints[0];
  const lastWp = waypoints[waypoints.length - 1];

  return (
    <div className="absolute bottom-4 left-4 z-[1000] w-96">
      <div className="bg-maritime-dark/90 backdrop-blur-md border border-white/10 rounded-xl shadow-2xl overflow-hidden">
        {/* Route header */}
        <div className="p-3 flex items-center justify-between">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-white truncate">{routeName}</span>
            </div>
            <div className="text-xs text-gray-500 mt-0.5 truncate">
              {firstWp.lat.toFixed(2)}°N, {firstWp.lon.toFixed(2)}°W → {lastWp.lat.toFixed(2)}°N, {lastWp.lon.toFixed(2)}°W
            </div>
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={() => onIsEditingChange(!isEditing)}
              className={`p-1.5 rounded transition-colors ${
                isEditing ? 'bg-primary-500/20 text-primary-400' : 'text-gray-500 hover:text-white'
              }`}
              title={isEditing ? 'Stop editing' : 'Edit route'}
            >
              <PenLine className="w-4 h-4" />
            </button>
            <button
              onClick={() => setShowImport(!showImport)}
              className={`p-1.5 rounded transition-colors ${
                showImport ? 'bg-primary-500/20 text-primary-400' : 'text-gray-500 hover:text-white'
              }`}
              title="Import RTZ"
            >
              <Upload className="w-4 h-4" />
            </button>
            <button
              onClick={onClearRoute}
              className="p-1.5 rounded text-gray-500 hover:text-red-400 transition-colors"
              title="Clear route"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Import section */}
        {showImport && (
          <div className="px-3 pb-3 border-t border-white/10 pt-3">
            <RouteImport onImport={onRouteImport} />
          </div>
        )}

        {/* Stats row */}
        <div className="px-3 pb-3 grid grid-cols-2 gap-x-4 gap-y-1 text-xs border-t border-white/10 pt-2">
          <div className="flex justify-between">
            <span className="text-gray-500">Distance:</span>
            <span className="text-white font-medium">{totalDistance.toFixed(0)} nm</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">WPs:</span>
            <span className="text-white font-medium">{waypoints.length}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">Bearing:</span>
            <span className="text-white font-medium">{bearing.toFixed(0)}° {bearingLabel}</span>
          </div>
        </div>

        {/* Analysis results if displayed */}
        {analysisFuel !== undefined && (
          <div className="px-3 pb-3 grid grid-cols-3 gap-2 text-xs border-t border-white/10 pt-2">
            <div>
              <span className="text-gray-500">Fuel:</span>
              <span className="text-white font-medium ml-1">{analysisFuel.toFixed(1)} MT</span>
            </div>
            <div>
              <span className="text-gray-500">ETA:</span>
              <span className="text-white font-medium ml-1">{analysisTime?.toFixed(1)}h</span>
            </div>
            <div>
              <span className="text-gray-500">Avg:</span>
              <span className="text-white font-medium ml-1">{analysisAvgSpeed?.toFixed(1)} kts</span>
            </div>
          </div>
        )}

        {/* Action buttons */}
        <div className="px-3 pb-3 flex gap-2">
          <button
            onClick={onCalculate}
            disabled={isCalculating || waypoints.length < 2}
            className="flex-1 flex items-center justify-center gap-1.5 py-2 bg-gradient-to-r from-primary-500 to-ocean-500 text-white text-sm font-medium rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isCalculating ? (
              <><Loader2 className="w-4 h-4 animate-spin" />Calculating...</>
            ) : (
              <><Play className="w-4 h-4" />Calculate Voyage</>
            )}
          </button>
          <button
            onClick={onOptimize}
            disabled={isOptimizing || waypoints.length < 2 || !hasBaseline}
            className="flex-1 flex items-center justify-center gap-1.5 py-2 bg-gradient-to-r from-green-600 to-emerald-500 text-white text-sm font-medium rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed"
            title={!hasBaseline ? 'Calculate Voyage first to establish baseline' : 'Compare routes using A* and VISIR engines'}
          >
            {isOptimizing ? (
              <><Loader2 className="w-4 h-4 animate-spin" />Optimizing (A* + VISIR)...</>
            ) : (
              <><Compass className="w-4 h-4" />Compare Routes</>
            )}
          </button>
        </div>

        {/* Dual-engine route comparison */}
        {(optimizationResults.astar || optimizationResults.visir) && (() => {
          const astar = optimizationResults.astar;
          const visir = optimizationResults.visir;
          // Use baseline from whichever result is available
          const ref = astar || visir;
          const baselineFuel = ref!.baseline_fuel_mt ?? ref!.direct_fuel_mt;
          const baselineTime = ref!.baseline_time_hours ?? ref!.direct_time_hours;
          const baselineDist = ref!.baseline_distance_nm ?? totalDistance;

          // Helper to get best scenario values from a result
          const getScenarioVal = (r: OptimizationResponse | null, field: keyof SpeedScenario) => {
            if (!r) return null;
            const sc = r.scenarios?.find(s => s.strategy === selectedStrategy) || r.scenarios?.[0];
            return sc ? (sc as any)[field] : (r as any)[field];
          };

          const astarFuel = getScenarioVal(astar, 'total_fuel_mt') ?? astar?.total_fuel_mt;
          const visirFuel = getScenarioVal(visir, 'total_fuel_mt') ?? visir?.total_fuel_mt;
          // Fuel change vs Original: negative = reduction (good), positive = increase (bad)
          const astarFuelChange = baselineFuel > 0 && astarFuel != null ? (((astarFuel as number) - baselineFuel) / baselineFuel) * 100 : 0;
          const visirFuelChange = baselineFuel > 0 && visirFuel != null ? (((visirFuel as number) - baselineFuel) / baselineFuel) * 100 : 0;

          return (
            <div className="px-3 pb-3">
              {/* Visibility toggles */}
              <div className="flex items-center gap-2 mb-2">
                <button
                  onClick={() => onRouteVisibilityChange({ ...routeVisibility, original: !routeVisibility.original })}
                  className="flex items-center gap-1 text-xs text-gray-400 hover:text-white transition-colors"
                  title="Toggle original route"
                >
                  {routeVisibility.original ? <Eye className="w-3.5 h-3.5 text-blue-400" /> : <EyeOff className="w-3.5 h-3.5" />}
                  <span className="text-blue-400">Original</span>
                </button>
                {astar && (
                  <button
                    onClick={() => onRouteVisibilityChange({ ...routeVisibility, astar: !routeVisibility.astar })}
                    className="flex items-center gap-1 text-xs text-gray-400 hover:text-white transition-colors"
                    title="Toggle A* route"
                  >
                    {routeVisibility.astar ? <Eye className="w-3.5 h-3.5 text-green-400" /> : <EyeOff className="w-3.5 h-3.5" />}
                    <span className="text-green-400">A*</span>
                  </button>
                )}
                {visir && (
                  <button
                    onClick={() => onRouteVisibilityChange({ ...routeVisibility, visir: !routeVisibility.visir })}
                    className="flex items-center gap-1 text-xs text-gray-400 hover:text-white transition-colors"
                    title="Toggle VISIR route"
                  >
                    {routeVisibility.visir ? <Eye className="w-3.5 h-3.5 text-orange-400" /> : <EyeOff className="w-3.5 h-3.5" />}
                    <span className="text-orange-400">VISIR</span>
                  </button>
                )}
              </div>

              {/* Comparison table */}
              <div className="p-2 bg-white/5 border border-white/10 rounded-lg mb-2">
                <div className="text-xs font-medium text-gray-300 mb-2">Route Comparison</div>
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-gray-500">
                      <th className="text-left font-normal pb-1"></th>
                      <th className="text-right font-normal pb-1 text-blue-400">Original</th>
                      {astar && <th className="text-right font-normal pb-1 text-green-400">A*</th>}
                      {visir && <th className="text-right font-normal pb-1 text-orange-400">VISIR</th>}
                    </tr>
                  </thead>
                  <tbody className="text-gray-300">
                    <tr>
                      <td>Distance</td>
                      <td className="text-right">{baselineDist.toFixed(0)} nm</td>
                      {astar && <td className="text-right">{(getScenarioVal(astar, 'total_distance_nm') ?? astar.total_distance_nm).toFixed(0)} nm</td>}
                      {visir && <td className="text-right">{(getScenarioVal(visir, 'total_distance_nm') ?? visir.total_distance_nm).toFixed(0)} nm</td>}
                    </tr>
                    <tr>
                      <td>Fuel</td>
                      <td className="text-right">{baselineFuel.toFixed(1)} mt</td>
                      {astar && <td className="text-right">{(astarFuel as number).toFixed(1)} mt</td>}
                      {visir && <td className="text-right">{(visirFuel as number).toFixed(1)} mt</td>}
                    </tr>
                    <tr className="font-medium">
                      <td>Fuel &Delta;</td>
                      <td className="text-right">-</td>
                      {astar && (
                        <td className={`text-right ${(astarFuelChange as number) < 0 ? 'text-green-400' : 'text-amber-400'}`}>
                          {(astarFuelChange as number) > 0 ? '+' : ''}{(astarFuelChange as number).toFixed(1)}%
                        </td>
                      )}
                      {visir && (
                        <td className={`text-right ${(visirFuelChange as number) < 0 ? 'text-orange-400' : 'text-amber-400'}`}>
                          {(visirFuelChange as number) > 0 ? '+' : ''}{(visirFuelChange as number).toFixed(1)}%
                        </td>
                      )}
                    </tr>
                    <tr>
                      <td>Time</td>
                      <td className="text-right">{baselineTime.toFixed(1)} h</td>
                      {astar && <td className="text-right">{(getScenarioVal(astar, 'total_time_hours') ?? astar.total_time_hours).toFixed(1)} h</td>}
                      {visir && <td className="text-right">{(getScenarioVal(visir, 'total_time_hours') ?? visir.total_time_hours).toFixed(1)} h</td>}
                    </tr>
                    <tr>
                      <td>Avg Speed</td>
                      <td className="text-right">{baselineTime > 0 ? (baselineDist / baselineTime).toFixed(1) : '-'} kts</td>
                      {astar && <td className="text-right">{(getScenarioVal(astar, 'avg_speed_kts') ?? astar.avg_speed_kts).toFixed(1)} kts</td>}
                      {visir && <td className="text-right">{(getScenarioVal(visir, 'avg_speed_kts') ?? visir.avg_speed_kts).toFixed(1)} kts</td>}
                    </tr>
                    <tr>
                      <td>Waypoints</td>
                      <td className="text-right">{waypoints.length}</td>
                      {astar && <td className="text-right">{astar.waypoints.length}</td>}
                      {visir && <td className="text-right">{visir.waypoints.length}</td>}
                    </tr>
                  </tbody>
                </table>
              </div>

              {/* Action buttons */}
              <div className="flex gap-2">
                <button
                  onClick={onDismissOptimizedRoute}
                  className="py-2 px-3 text-xs border border-white/20 text-gray-300 rounded-lg hover:bg-white/5 transition-colors"
                >
                  Dismiss
                </button>
                {astar && (
                  <button
                    onClick={() => onApplyOptimizedRoute('astar')}
                    className="flex-1 py-2 text-xs bg-green-600 text-white rounded-lg hover:bg-green-500 transition-colors"
                  >
                    Apply A*
                  </button>
                )}
                {visir && (
                  <button
                    onClick={() => onApplyOptimizedRoute('visir')}
                    className="flex-1 py-2 text-xs bg-orange-600 text-white rounded-lg hover:bg-orange-500 transition-colors"
                  >
                    Apply VISIR
                  </button>
                )}
              </div>
            </div>
          );
        })()}

        {/* Saved Routes */}
        <div className="px-3 pb-3 border-t border-white/10 pt-2">
          <button
            onClick={() => setSavedRoutesExpanded(!savedRoutesExpanded)}
            className="flex items-center justify-between w-full text-sm text-gray-400 hover:text-white transition-colors"
          >
            <span>Saved Routes</span>
            {savedRoutesExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
          {savedRoutesExpanded && (
            <div className="mt-2">
              <SavedRoutes currentWaypoints={waypoints} onLoadRoute={onLoadRoute} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function calculateBearing(from: Position, to: Position): number {
  const lat1 = (from.lat * Math.PI) / 180;
  const lat2 = (to.lat * Math.PI) / 180;
  const dLon = ((to.lon - from.lon) * Math.PI) / 180;

  const y = Math.sin(dLon) * Math.cos(lat2);
  const x = Math.cos(lat1) * Math.sin(lat2) - Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLon);
  const brng = (Math.atan2(y, x) * 180) / Math.PI;
  return (brng + 360) % 360;
}

function bearingToCardinal(deg: number): string {
  const dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];
  return dirs[Math.round(deg / 22.5) % 16];
}

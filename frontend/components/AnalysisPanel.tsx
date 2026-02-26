'use client';

import { useState, useRef, useEffect } from 'react';
import {
  Navigation, Clock, Fuel, Ship, Loader2, Trash2,
  Upload, Play, Zap, Dice5, ExternalLink, Eye, EyeOff,
  PenLine, MapPin, Download, FolderOpen, Check, X, CheckCircle, TrendingUp, Settings,
} from 'lucide-react';
import Link from 'next/link';
import RouteImport, { SampleRTZButton } from '@/components/RouteImport';
import { useVoyage } from '@/components/VoyageContext';
import { DEMO_MODE } from '@/lib/demoMode';
import {
  Position, AllOptimizationResults, OptimizedRouteKey,
  RouteVisibility, ROUTE_STYLES, EMPTY_ALL_RESULTS,
  CalibrationStatus, ParetoSolution, apiClient,
} from '@/lib/api';
import { AnalysisEntry } from '@/lib/analysisStorage';
import ParetoChart from '@/components/ParetoChart';

interface AnalysisPanelProps {
  waypoints: Position[];
  routeName: string;
  onRouteNameChange: (name: string) => void;
  totalDistance: number;
  onRouteImport: (wps: Position[], name: string) => void;
  onClearRoute: () => void;
  isEditing: boolean;
  onIsEditingChange: (v: boolean) => void;
  isCalculating: boolean;
  onCalculate: () => void;
  isOptimizing: boolean;
  onOptimize: () => void;
  allResults: AllOptimizationResults;
  onApplyRoute: (key: OptimizedRouteKey) => void;
  onDismissRoutes: () => void;
  routeVisibility: RouteVisibility;
  onRouteVisibilityChange: (v: RouteVisibility) => void;
  isSimulating: boolean;
  onRunSimulations: () => void;
  displayedAnalysis: AnalysisEntry | null;
  paretoFront: ParetoSolution[] | null;
  isRunningPareto: boolean;
  onRunPareto: () => void;
}

const WEIGHT_LABELS: Record<string, string> = { fuel: 'Fuel', balanced: 'Balanced', safety: 'Safety' };

export default function AnalysisPanel({
  waypoints,
  routeName,
  onRouteNameChange,
  totalDistance,
  onRouteImport,
  onClearRoute,
  isEditing,
  onIsEditingChange,
  isCalculating,
  onCalculate,
  isOptimizing,
  onOptimize,
  allResults,
  onApplyRoute,
  onDismissRoutes,
  routeVisibility,
  onRouteVisibilityChange,
  isSimulating,
  onRunSimulations,
  displayedAnalysis,
  paretoFront,
  isRunningPareto,
  onRunPareto,
}: AnalysisPanelProps) {
  const { departureTime, setDepartureTime, calmSpeed, isLaden, variableSpeed, setVariableSpeed } = useVoyage();
  const [showImport, setShowImport] = useState(false);
  const [isRenaming, setIsRenaming] = useState(false);
  const [renameValue, setRenameValue] = useState('');
  const loadFileRef = useRef<HTMLInputElement>(null);
  const [calibration, setCalibration] = useState<CalibrationStatus | null>(null);

  useEffect(() => {
    apiClient.getCalibration().then(setCalibration).catch(() => {});
  }, []);

  const hasRoute = waypoints.length >= 2;
  const hasBaseline = !!displayedAnalysis;
  const hasOptimized = Object.values(allResults).some(Boolean);

  const formatDuration = (hours: number): string => {
    const days = Math.floor(hours / 24);
    const h = Math.floor(hours % 24);
    const m = Math.round((hours % 1) * 60);
    if (days > 0) return `${days}d ${h}h ${m}m`;
    return `${h}h ${m}m`;
  };

  return (
    <div className="absolute left-4 top-3 bottom-4 w-80 z-[1000] flex flex-col bg-maritime-dark/95 backdrop-blur-md border border-white/10 rounded-xl shadow-2xl overflow-hidden">
      {/* Hidden file input for loading routes */}
      <input
        ref={loadFileRef}
        type="file"
        accept=".json"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (!file) return;
          loadRouteFromFile(file, onRouteImport);
          if (loadFileRef.current) loadFileRef.current.value = '';
        }}
      />
      {/* Header */}
      <div className="px-4 py-3 border-b border-white/10">
        <h2 className="text-sm font-semibold text-white">Route Analysis</h2>
      </div>

      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* ── Route Section ── */}
        {!hasRoute ? (
          <div className="space-y-3">
            <div className="text-center py-6">
              <Navigation className="w-10 h-10 text-gray-600 mx-auto mb-3" />
              <p className="text-sm text-gray-400 mb-3">No route loaded</p>
              <div className="flex flex-col gap-2 items-center">
                <button
                  onClick={() => setShowImport(true)}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-primary-500/20 text-primary-400 hover:bg-primary-500/30 transition-colors"
                >
                  <Upload className="w-4 h-4" />
                  Import RTZ
                </button>
                <button
                  onClick={() => loadSampleRoute(onRouteImport)}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-white/5 text-gray-300 hover:bg-white/10 transition-colors"
                >
                  <MapPin className="w-4 h-4" />
                  Load Sample Route
                </button>
                <button
                  onClick={() => loadFileRef.current?.click()}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-white/5 text-gray-300 hover:bg-white/10 transition-colors"
                >
                  <FolderOpen className="w-4 h-4" />
                  Load from File
                </button>
                <button
                  onClick={() => onIsEditingChange(true)}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-white/5 text-gray-300 hover:bg-white/10 transition-colors"
                >
                  <PenLine className="w-4 h-4" />
                  Draw on Map
                </button>
              </div>
            </div>
            {showImport && (
              <RouteImport onImport={(wps, name) => { onRouteImport(wps, name); setShowImport(false); }} />
            )}
          </div>
        ) : (
          <>
            {/* Route summary */}
            <div className="p-3 rounded-lg bg-white/5 border border-white/10">
              <div className="flex items-center justify-between mb-2">
                {isRenaming ? (
                  <div className="flex items-center gap-1 flex-1 pr-2">
                    <input
                      autoFocus
                      value={renameValue}
                      onChange={(e) => setRenameValue(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && renameValue.trim()) { onRouteNameChange(renameValue.trim()); setIsRenaming(false); }
                        if (e.key === 'Escape') setIsRenaming(false);
                      }}
                      className="flex-1 px-2 py-0.5 rounded text-sm bg-white/10 border border-white/20 text-white focus:border-primary-500/50 focus:outline-none"
                    />
                    <button onClick={() => { if (renameValue.trim()) { onRouteNameChange(renameValue.trim()); } setIsRenaming(false); }} className="p-0.5 text-green-400 hover:text-green-300"><Check className="w-3.5 h-3.5" /></button>
                    <button onClick={() => setIsRenaming(false)} className="p-0.5 text-gray-400 hover:text-white"><X className="w-3.5 h-3.5" /></button>
                  </div>
                ) : (
                  <button
                    onClick={() => { setRenameValue(routeName); setIsRenaming(true); }}
                    className="text-sm font-medium text-white truncate pr-2 hover:text-primary-300 transition-colors text-left"
                    title="Click to rename"
                  >
                    {routeName}
                  </button>
                )}
                <div className="flex items-center gap-1 flex-shrink-0">
                  <button
                    onClick={() => onIsEditingChange(!isEditing)}
                    className={`p-1 rounded transition-colors ${
                      isEditing ? 'text-primary-400 bg-primary-500/10' : 'text-gray-400 hover:text-white'
                    }`}
                    title={isEditing ? 'Stop editing' : 'Edit route on map'}
                  >
                    <PenLine className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={() => saveRouteToFile(routeName, waypoints)}
                    className="p-1 rounded text-gray-400 hover:text-white transition-colors"
                    title="Save route to file"
                  >
                    <Download className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={onClearRoute}
                    className="p-1 rounded text-gray-400 hover:text-red-400 transition-colors"
                    title="Clear route"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
              <div className="flex items-center gap-3 text-xs text-gray-400">
                <span>{waypoints.length} waypoints</span>
                <span>{totalDistance.toFixed(1)} nm</span>
              </div>
            </div>

            {/* ── Departure & Parameters ── */}
            <div className="space-y-2">
              <label className="block">
                <span className="text-xs text-gray-400">Departure Time</span>
                <input
                  type="datetime-local"
                  value={departureTime}
                  onChange={(e) => setDepartureTime(e.target.value)}
                  className="mt-1 w-full px-3 py-1.5 rounded-lg text-sm bg-white/5 border border-white/10 text-white focus:border-primary-500/50 focus:outline-none"
                />
              </label>
              <div className="flex items-center gap-3 text-xs text-gray-400">
                <span>{calmSpeed} kts</span>
                <span>{isLaden ? 'Laden' : 'Ballast'}</span>
              </div>
            </div>

            {/* ── Variable Speed toggle ── */}
            <label className="flex items-center gap-2 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={variableSpeed}
                onChange={(e) => setVariableSpeed(e.target.checked)}
                className="accent-ocean-500 w-3.5 h-3.5"
              />
              <div>
                <span className="text-xs text-gray-300">Variable speed</span>
                <span className="text-[10px] text-gray-500 ml-1.5">optimize speed per leg</span>
              </div>
            </label>

            {/* ── Calculate Voyage ── */}
            <button
              onClick={onCalculate}
              disabled={isCalculating}
              className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium bg-primary-500/20 text-primary-400 hover:bg-primary-500/30 transition-colors disabled:opacity-50"
            >
              {isCalculating ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Play className="w-4 h-4" />
              )}
              {isCalculating ? 'Calculating...' : 'Calculate Voyage'}
            </button>

            {/* ── Baseline Summary ── */}
            {hasBaseline && (
              <div className="p-3 rounded-lg bg-primary-500/5 border border-primary-500/20 space-y-2">
                <div className="text-xs font-medium text-primary-400 mb-1">Voyage Summary</div>
                {/* Calibration indicator */}
                {calibration && (
                  <div className={`flex items-center gap-1.5 text-[10px] mb-1 ${
                    calibration.calibrated ? 'text-green-400' : 'text-gray-500'
                  }`}>
                    {calibration.calibrated ? (
                      <>
                        <CheckCircle className="w-3 h-3" />
                        <span>
                          Calibrated model ({calibration.num_reports_used} reports
                          {calibration.calibrated_at && `, ${new Date(calibration.calibrated_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}`})
                        </span>
                      </>
                    ) : (
                      <span>Theoretical model (uncalibrated)</span>
                    )}
                  </div>
                )}
                <div className="grid grid-cols-2 gap-2">
                  <MetricItem
                    icon={<Clock className="w-3 h-3" />}
                    label="ETA"
                    value={new Date(displayedAnalysis.result.arrival_time).toLocaleString(undefined, {
                      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
                    })}
                  />
                  <MetricItem
                    icon={<Fuel className="w-3 h-3" />}
                    label="Fuel"
                    value={`${displayedAnalysis.result.total_fuel_mt.toFixed(1)} MT`}
                  />
                  <MetricItem
                    icon={<Ship className="w-3 h-3" />}
                    label="Avg SOG"
                    value={`${displayedAnalysis.result.avg_sog_kts.toFixed(1)} kts`}
                  />
                  <MetricItem
                    icon={<Navigation className="w-3 h-3" />}
                    label="Duration"
                    value={formatDuration(displayedAnalysis.result.total_time_hours)}
                  />
                </div>

                {/* Speed profile summary */}
                {displayedAnalysis.result.variable_speed_enabled && displayedAnalysis.result.speed_profile && (
                  <div className="mt-1 p-2 rounded bg-ocean-500/10 border border-ocean-500/20">
                    <div className="text-[10px] text-ocean-400 font-medium mb-1">Variable Speed Profile</div>
                    <div className="flex items-end gap-px h-6">
                      {displayedAnalysis.result.speed_profile.map((spd, i) => {
                        const min = Math.min(...displayedAnalysis.result.speed_profile!);
                        const max = Math.max(...displayedAnalysis.result.speed_profile!);
                        const range = max - min || 1;
                        const h = 20 + ((spd - min) / range) * 80; // 20-100%
                        return (
                          <div
                            key={i}
                            className="flex-1 bg-ocean-400/60 rounded-t-sm"
                            style={{ height: `${h}%` }}
                            title={`Leg ${i + 1}: ${spd} kts`}
                          />
                        );
                      })}
                    </div>
                    <div className="flex justify-between text-[9px] text-gray-500 mt-0.5">
                      <span>{Math.min(...displayedAnalysis.result.speed_profile).toFixed(1)} kts</span>
                      <span>{Math.max(...displayedAnalysis.result.speed_profile).toFixed(1)} kts</span>
                    </div>
                  </div>
                )}

                {/* View Full Analysis link */}
                <Link
                  href={`/analysis?id=${displayedAnalysis.id}`}
                  className="flex items-center gap-1.5 mt-2 text-xs text-primary-400 hover:text-primary-300 transition-colors"
                >
                  <ExternalLink className="w-3 h-3" />
                  View Full Analysis
                </Link>
              </div>
            )}

            {/* ── Optimization Settings link ── */}
            <Link
              href="/settings"
              className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs text-gray-400 hover:text-gray-200 hover:bg-white/5 border border-white/10 transition-colors"
            >
              <Settings className="w-3.5 h-3.5" />
              <span>Settings</span>
              <span className="ml-auto text-gray-600">&rarr;</span>
            </Link>

            {/* ── Optimize Route ── */}
            <div className="space-y-1">
              <button
                onClick={onOptimize}
                disabled={isOptimizing || !hasRoute || !hasBaseline}
                className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium bg-ocean-500/20 text-ocean-400 hover:bg-ocean-500/30 transition-colors disabled:opacity-50"
                title={!hasBaseline ? 'Calculate voyage first to establish a baseline' : undefined}
              >
                {isOptimizing ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Zap className="w-4 h-4" />
                )}
                {isOptimizing ? 'Optimizing...' : 'Optimize Route'}
              </button>
              {hasRoute && !hasBaseline && !isOptimizing && (
                <p className="text-[10px] text-gray-500 text-center">Calculate voyage first to establish a baseline</p>
              )}
            </div>

            {/* ── Optimization Results ── */}
            {hasOptimized && (() => {
              const baselineFuel = hasBaseline ? displayedAnalysis!.result.total_fuel_mt : null;

              // Count how many routes actually save fuel
              const fuelSavingCount = baselineFuel
                ? Object.values(allResults).filter(r => r && r.total_fuel_mt < baselineFuel).length
                : 0;

              return (
                <div className="space-y-2">
                  <div className="text-xs font-medium text-gray-300">Optimized Routes</div>
                  {(['astar', 'dijkstra'] as const).map(engine => {
                    const keys = [`${engine}_fuel`, `${engine}_balanced`, `${engine}_safety`] as OptimizedRouteKey[];
                    const hasAny = keys.some(k => allResults[k]);
                    if (!hasAny) return null;

                    const visibleKeys = keys.filter(key => !!allResults[key]);

                    if (visibleKeys.length === 0) return null;

                    return (
                      <div key={engine} className="space-y-1">
                        <div className="text-[10px] text-gray-500 uppercase tracking-wider">
                          {engine === 'astar' ? 'A*' : 'Dijkstra'}
                        </div>
                        {visibleKeys.map(key => {
                          const r = allResults[key]!;
                          const weight = key.split('_')[1];
                          const style = ROUTE_STYLES[key];
                          const vis = routeVisibility[key];
                          const fuelDeltaPct = baselineFuel
                            ? (r.total_fuel_mt - baselineFuel) / baselineFuel * 100
                            : null;
                          const savesFuel = fuelDeltaPct !== null && fuelDeltaPct < 0;
                          const isSafetyRoute = weight === 'balanced' || weight === 'safety';
                          const routeLabel = savesFuel
                            ? WEIGHT_LABELS[weight]
                            : isSafetyRoute
                              ? `${WEIGHT_LABELS[weight]} (safer)`
                              : WEIGHT_LABELS[weight];

                          return (
                            <div key={key} className="flex items-center gap-2 px-2 py-1.5 rounded bg-white/5 text-xs">
                              <button
                                onClick={() => onRouteVisibilityChange({ ...routeVisibility, [key]: !vis })}
                                className="text-gray-400 hover:text-white"
                              >
                                {vis ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
                              </button>
                              <div className="w-3 h-0.5 rounded" style={{ backgroundColor: style.color }} />
                              <span className="text-gray-300 flex-1">{routeLabel}</span>
                              <span className="text-gray-400">{r.total_fuel_mt.toFixed(1)} MT</span>
                              {fuelDeltaPct !== null && (
                                <span className={`text-[10px] ${fuelDeltaPct < 0 ? 'text-green-400' : 'text-amber-400'}`}>
                                  {fuelDeltaPct > 0 ? '+' : ''}{fuelDeltaPct.toFixed(1)}%
                                </span>
                              )}
                              <button
                                onClick={() => onApplyRoute(key)}
                                className="text-primary-400 hover:text-primary-300"
                                title="Apply this route"
                              >
                                <Play className="w-3 h-3" />
                              </button>
                            </div>
                          );
                        })}
                      </div>
                    );
                  })}

                  {/* No fuel-saving route found message */}
                  {baselineFuel && fuelSavingCount === 0 && (
                    <div className="px-3 py-2 rounded bg-white/5 border border-white/10 text-xs text-gray-400 text-center">
                      No fuel-saving route found — base route is near-optimal for current conditions
                    </div>
                  )}

                  <button
                    onClick={onDismissRoutes}
                    className="w-full text-xs text-gray-500 hover:text-gray-300 py-1 transition-colors"
                  >
                    Dismiss optimized routes
                  </button>
                </div>
              );
            })()}

            {/* ── Pareto Analysis ── */}
            {hasRoute && (
              <button
                onClick={onRunPareto}
                disabled={isRunningPareto || isOptimizing}
                className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-white/5 text-gray-300 hover:bg-white/10 transition-colors disabled:opacity-50"
              >
                {isRunningPareto ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <TrendingUp className="w-4 h-4" />
                )}
                {isRunningPareto ? 'Running Pareto...' : 'Pareto Analysis'}
              </button>
            )}

            {/* ── Pareto Chart ── */}
            {paretoFront && paretoFront.length > 0 && (
              <ParetoChart solutions={paretoFront} />
            )}

            {/* ── Run Simulations (Monte Carlo) ── */}
            {hasBaseline && (
              <>
                <button
                  onClick={onRunSimulations}
                  disabled={isSimulating}
                  className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-white/5 text-gray-300 hover:bg-white/10 transition-colors disabled:opacity-50"
                >
                  {isSimulating ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Dice5 className="w-4 h-4" />
                  )}
                  {isSimulating ? 'Running Simulations...' : 'Run Simulations'}
                </button>

                {/* Monte Carlo results */}
                {displayedAnalysis.monteCarlo && (
                  <div className="p-3 rounded-lg bg-white/5 border border-white/10 space-y-2">
                    <div className="text-xs font-medium text-gray-300">Monte Carlo ({displayedAnalysis.monteCarlo.n_simulations} sims)</div>
                    <div className="grid grid-cols-3 gap-1 text-[10px]">
                      <div className="text-gray-500" />
                      <div className="text-center text-gray-500">P10</div>
                      <div className="text-center text-gray-500">P50</div>
                    </div>
                    <div className="grid grid-cols-3 gap-1 text-xs">
                      <div className="text-gray-400">Fuel</div>
                      <div className="text-center text-gray-300">{displayedAnalysis.monteCarlo.fuel_mt.p10.toFixed(1)}</div>
                      <div className="text-center text-gray-300">{displayedAnalysis.monteCarlo.fuel_mt.p50.toFixed(1)}</div>
                    </div>
                    <div className="grid grid-cols-3 gap-1 text-xs">
                      <div className="text-gray-400">Duration</div>
                      <div className="text-center text-gray-300">{formatDuration(displayedAnalysis.monteCarlo.total_time_hours.p10)}</div>
                      <div className="text-center text-gray-300">{formatDuration(displayedAnalysis.monteCarlo.total_time_hours.p50)}</div>
                    </div>
                  </div>
                )}
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}

const SAMPLE_WAYPOINTS: Position[] = [
  { lat: 51.9225, lon: 4.4792, name: 'Rotterdam' },
  { lat: 51.0500, lon: 1.5000, name: 'Dover Strait' },
  { lat: 48.4500, lon: -5.1000, name: 'Ushant' },
  { lat: 42.8800, lon: -9.2700, name: 'Finisterre' },
  { lat: 36.1408, lon: -5.3536, name: 'Gibraltar' },
  { lat: 38.8000, lon: 8.4000, name: 'Sardinia South' },
  { lat: 37.2333, lon: 15.2167, name: 'Augusta' },
];

function loadSampleRoute(onImport: (wps: Position[], name: string) => void) {
  onImport(SAMPLE_WAYPOINTS, 'Rotterdam to Augusta');
}

function saveRouteToFile(name: string, waypoints: Position[]) {
  const data = JSON.stringify({ name, waypoints }, null, 2);
  const blob = new Blob([data], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${name.replace(/[^a-zA-Z0-9_-]/g, '_')}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

function loadRouteFromFile(file: File, onImport: (wps: Position[], name: string) => void) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const data = JSON.parse(reader.result as string);
      if (!Array.isArray(data.waypoints) || data.waypoints.length < 2) return;
      const wps: Position[] = data.waypoints.map((wp: any) => ({
        lat: Number(wp.lat),
        lon: Number(wp.lon),
        name: wp.name || undefined,
      }));
      onImport(wps, data.name || file.name.replace(/\.json$/, ''));
    } catch {
      // Invalid file — silently ignore
    }
  };
  reader.readAsText(file);
}

function MetricItem({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="space-y-0.5">
      <div className="flex items-center gap-1 text-[10px] text-gray-500">
        {icon}
        {label}
      </div>
      <div className="text-xs text-gray-200">{value}</div>
    </div>
  );
}

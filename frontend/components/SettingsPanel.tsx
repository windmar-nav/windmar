'use client';

import { useState } from 'react';
import { Settings, Grid3X3, Gauge, Shield, BarChart3, Loader2 } from 'lucide-react';
import { apiClient, Position } from '@/lib/api';

export interface OptimizationSettings {
  gridResolution: number;
  safetyWeight: number;
  variableResolution: boolean;
  pareto: boolean;
}

interface BenchmarkResult {
  engine: string;
  total_fuel_mt: number;
  total_time_hours: number;
  total_distance_nm: number;
  cells_explored: number;
  optimization_time_ms: number;
  waypoint_count: number;
  error?: string | null;
}

interface SettingsPanelProps {
  settings: OptimizationSettings;
  onSettingsChange: (settings: OptimizationSettings) => void;
  waypoints: Position[];
  calmSpeed: number;
  isLaden: boolean;
}

export default function SettingsPanel({
  settings,
  onSettingsChange,
  waypoints,
  calmSpeed,
  isLaden,
}: SettingsPanelProps) {
  const [expanded, setExpanded] = useState(false);
  const [benchmarkResults, setBenchmarkResults] = useState<BenchmarkResult[] | null>(null);
  const [isBenchmarking, setIsBenchmarking] = useState(false);

  const hasRoute = waypoints.length >= 2;

  const runBenchmark = async () => {
    if (!hasRoute) return;
    setIsBenchmarking(true);
    setBenchmarkResults(null);
    try {
      const resp = await apiClient.benchmarkEngines({
        origin: waypoints[0],
        destination: waypoints[waypoints.length - 1],
        calm_speed_kts: calmSpeed,
        is_laden: isLaden,
        grid_resolution_deg: settings.gridResolution,
        safety_weight: settings.safetyWeight,
        variable_resolution: settings.variableResolution,
        engines: ['astar', 'visir'],
      });
      setBenchmarkResults(resp.results);
    } catch (err) {
      console.error('Benchmark failed:', err);
    } finally {
      setIsBenchmarking(false);
    }
  };

  return (
    <div className="border border-white/10 rounded-lg overflow-hidden">
      {/* Toggle header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2 px-3 py-2 text-xs font-medium text-gray-400 hover:text-gray-200 hover:bg-white/5 transition-colors"
      >
        <Settings className="w-3.5 h-3.5" />
        <span>Optimization Settings</span>
        <span className="ml-auto text-[9px] text-gray-600">{expanded ? '▲' : '▼'}</span>
      </button>

      {expanded && (
        <div className="px-3 pb-3 space-y-3 border-t border-white/5">
          {/* Grid Resolution */}
          <div className="pt-2">
            <label className="flex items-center gap-1.5 text-[10px] text-gray-500 mb-1">
              <Grid3X3 className="w-3 h-3" />
              Grid Resolution: {settings.gridResolution.toFixed(2)}°
            </label>
            <input
              type="range"
              min="0.05"
              max="1.0"
              step="0.05"
              value={settings.gridResolution}
              onChange={(e) => onSettingsChange({ ...settings, gridResolution: parseFloat(e.target.value) })}
              className="w-full h-1 accent-ocean-500 cursor-pointer"
            />
            <div className="flex justify-between text-[9px] text-gray-600">
              <span>0.05° (fine)</span>
              <span>1.0° (coarse)</span>
            </div>
          </div>

          {/* Safety Weight */}
          <div>
            <label className="flex items-center gap-1.5 text-[10px] text-gray-500 mb-1">
              <Shield className="w-3 h-3" />
              Safety Weight: {settings.safetyWeight.toFixed(1)}
            </label>
            <input
              type="range"
              min="0"
              max="1.0"
              step="0.1"
              value={settings.safetyWeight}
              onChange={(e) => onSettingsChange({ ...settings, safetyWeight: parseFloat(e.target.value) })}
              className="w-full h-1 accent-ocean-500 cursor-pointer"
            />
            <div className="flex justify-between text-[9px] text-gray-600">
              <span>0.0 (fuel)</span>
              <span>1.0 (safety)</span>
            </div>
          </div>

          {/* Toggles */}
          <div className="space-y-1.5">
            <label className="flex items-center gap-2 text-[10px] text-gray-400 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={settings.variableResolution}
                onChange={(e) => onSettingsChange({ ...settings, variableResolution: e.target.checked })}
                className="accent-ocean-500"
              />
              <Grid3X3 className="w-3 h-3" />
              Variable resolution grid
              <span className="ml-auto text-[9px] text-gray-600">0.05°/0.5°</span>
            </label>

            <label className="flex items-center gap-2 text-[10px] text-gray-400 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={settings.pareto}
                onChange={(e) => onSettingsChange({ ...settings, pareto: e.target.checked })}
                className="accent-ocean-500"
              />
              <Gauge className="w-3 h-3" />
              Pareto analysis (fuel/time trade-offs)
            </label>
          </div>

          {/* Benchmark */}
          <div className="pt-1 border-t border-white/5">
            <button
              onClick={runBenchmark}
              disabled={!hasRoute || isBenchmarking}
              className="w-full flex items-center justify-center gap-1.5 px-3 py-1.5 rounded text-[10px] font-medium bg-white/5 text-gray-400 hover:bg-white/10 hover:text-gray-200 transition-colors disabled:opacity-40"
            >
              {isBenchmarking ? (
                <Loader2 className="w-3 h-3 animate-spin" />
              ) : (
                <BarChart3 className="w-3 h-3" />
              )}
              {isBenchmarking ? 'Running...' : 'Benchmark A* vs VISIR'}
            </button>

            {benchmarkResults && (
              <div className="mt-2 space-y-1">
                <div className="text-[9px] text-gray-500 uppercase tracking-wider">Results</div>
                <table className="w-full text-[10px] text-gray-400">
                  <thead>
                    <tr className="text-gray-500">
                      <th className="text-left py-0.5">Engine</th>
                      <th className="text-right py-0.5">Fuel (MT)</th>
                      <th className="text-right py-0.5">Time</th>
                      <th className="text-right py-0.5">ms</th>
                    </tr>
                  </thead>
                  <tbody>
                    {benchmarkResults.map((r) => (
                      <tr key={r.engine} className={r.error ? 'text-red-400' : ''}>
                        <td className="py-0.5 font-medium">{r.engine === 'astar' ? 'A*' : 'VISIR'}</td>
                        <td className="text-right py-0.5">{r.error ? '—' : r.total_fuel_mt.toFixed(1)}</td>
                        <td className="text-right py-0.5">{r.error ? '—' : `${r.total_time_hours.toFixed(1)}h`}</td>
                        <td className="text-right py-0.5">{r.error ? 'ERR' : r.optimization_time_ms.toFixed(0)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

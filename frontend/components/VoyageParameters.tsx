'use client';

import {
  Ship,
  Play,
  Loader2,
  Wind,
  Compass,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  ShieldAlert,
} from 'lucide-react';
import Card from '@/components/Card';
import { OptimizationResponse } from '@/lib/api';

interface VoyageParametersProps {
  calmSpeed: number;
  onCalmSpeedChange: (speed: number) => void;
  isLaden: boolean;
  onIsLadenChange: (laden: boolean) => void;
  useWeather: boolean;
  onUseWeatherChange: (use: boolean) => void;
  isCalculating: boolean;
  onCalculate: () => void;
  isOptimizing: boolean;
  onOptimize: () => void;
  waypointCount: number;
  optimizationResult: OptimizationResponse | null;
  onApplyOptimizedRoute: () => void;
}

export default function VoyageParameters({
  calmSpeed,
  onCalmSpeedChange,
  isLaden,
  onIsLadenChange,
  useWeather,
  onUseWeatherChange,
  isCalculating,
  onCalculate,
  isOptimizing,
  onOptimize,
  waypointCount,
  optimizationResult,
  onApplyOptimizedRoute,
}: VoyageParametersProps) {
  return (
    <Card title="Voyage Parameters" icon={<Ship className="w-5 h-5" />}>
      <div className="space-y-4">
        {/* Calm Speed */}
        <div>
          <label className="block text-sm text-gray-300 mb-2">
            Calm Water Speed
          </label>
          <div className="flex items-center space-x-2">
            <input
              type="range"
              min="8"
              max="18"
              step="0.5"
              value={calmSpeed}
              onChange={(e) => onCalmSpeedChange(parseFloat(e.target.value))}
              className="flex-1"
            />
            <span className="w-16 text-right text-white font-semibold">
              {calmSpeed} kts
            </span>
          </div>
        </div>

        {/* Loading Condition */}
        <div>
          <label className="block text-sm text-gray-300 mb-2">
            Loading Condition
          </label>
          <div className="flex space-x-2">
            <button
              onClick={() => onIsLadenChange(true)}
              className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors ${
                isLaden
                  ? 'bg-primary-500 text-white'
                  : 'bg-maritime-dark text-gray-400 hover:text-white'
              }`}
            >
              Laden
            </button>
            <button
              onClick={() => onIsLadenChange(false)}
              className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors ${
                !isLaden
                  ? 'bg-primary-500 text-white'
                  : 'bg-maritime-dark text-gray-400 hover:text-white'
              }`}
            >
              Ballast
            </button>
          </div>
        </div>

        {/* Weather Toggle */}
        <div className="flex items-center justify-between p-3 bg-maritime-dark rounded-lg">
          <div className="flex items-center space-x-2">
            <Wind className="w-4 h-4 text-primary-400" />
            <span className="text-sm text-white">Use Weather</span>
          </div>
          <button
            onClick={() => onUseWeatherChange(!useWeather)}
            className={`relative w-10 h-6 rounded-full transition-colors ${
              useWeather ? 'bg-primary-500' : 'bg-gray-600'
            }`}
          >
            <span
              className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-transform ${
                useWeather ? 'left-5' : 'left-1'
              }`}
            />
          </button>
        </div>

        {/* Calculate Button */}
        <button
          onClick={onCalculate}
          disabled={isCalculating || waypointCount < 2}
          className="w-full flex items-center justify-center space-x-2 py-3 bg-gradient-to-r from-primary-500 to-ocean-500 text-white font-semibold rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isCalculating ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Calculating...</span>
            </>
          ) : (
            <>
              <Play className="w-5 h-5" />
              <span>Calculate Voyage</span>
            </>
          )}
        </button>

        {/* Optimize Button */}
        <button
          onClick={onOptimize}
          disabled={isOptimizing || waypointCount < 2}
          className="w-full flex items-center justify-center space-x-2 py-3 bg-gradient-to-r from-green-600 to-emerald-500 text-white font-semibold rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isOptimizing ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Optimizing...</span>
            </>
          ) : (
            <>
              <Compass className="w-5 h-5" />
              <span>Optimize Route (A*)</span>
            </>
          )}
        </button>

        {/* Optimization Results */}
        {optimizationResult && (
          <OptimizationResultPanel
            result={optimizationResult}
            onApply={onApplyOptimizedRoute}
          />
        )}
      </div>
    </Card>
  );
}

function OptimizationResultPanel({
  result,
  onApply,
}: {
  result: OptimizationResponse;
  onApply: () => void;
}) {
  return (
    <div className="mt-3 p-3 bg-green-500/10 border border-green-500/30 rounded-lg">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-green-400">Route Optimized</span>
        <TrendingDown className="w-4 h-4 text-green-400" />
      </div>
      <div className="text-xs text-gray-300 space-y-1">
        <div className="flex justify-between">
          <span>Fuel savings:</span>
          <span className="text-green-400 font-semibold">
            {result.fuel_savings_pct.toFixed(1)}%
          </span>
        </div>
        <div className="flex justify-between">
          <span>Distance:</span>
          <span>{result.total_distance_nm.toFixed(0)} nm</span>
        </div>
        <div className="flex justify-between">
          <span>Waypoints:</span>
          <span>{result.waypoints.length}</span>
        </div>
        <div className="flex justify-between">
          <span>Avg speed:</span>
          <span>{result.avg_speed_kts.toFixed(1)} kts</span>
        </div>
      </div>

      {/* Speed Profile */}
      {result.variable_speed_enabled && result.speed_profile.length > 1 && (
        <div className="mt-2 pt-2 border-t border-white/10">
          <div className="text-xs text-gray-400 mb-1">Speed Profile (per leg):</div>
          <div className="flex items-end h-8 gap-px">
            {result.speed_profile.map((speed, i) => {
              const minSpeed = Math.min(...result.speed_profile);
              const maxSpeed = Math.max(...result.speed_profile);
              const range = maxSpeed - minSpeed || 1;
              const height = ((speed - minSpeed) / range) * 100;
              return (
                <div
                  key={i}
                  className="flex-1 bg-primary-500 rounded-t"
                  style={{ height: `${Math.max(20, height)}%` }}
                  title={`Leg ${i + 1}: ${speed.toFixed(1)} kts`}
                />
              );
            })}
          </div>
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>{Math.min(...result.speed_profile).toFixed(0)} kts</span>
            <span>{Math.max(...result.speed_profile).toFixed(0)} kts</span>
          </div>
        </div>
      )}

      {/* Safety Assessment */}
      {result.safety && (
        <div className={`mt-2 p-2 rounded ${
          result.safety.status === 'safe'
            ? 'bg-green-500/20 border border-green-500/30'
            : result.safety.status === 'marginal'
            ? 'bg-yellow-500/20 border border-yellow-500/30'
            : 'bg-red-500/20 border border-red-500/30'
        }`}>
          <div className="flex items-center space-x-2 mb-1">
            {result.safety.status === 'safe' ? (
              <CheckCircle className="w-4 h-4 text-green-400" />
            ) : result.safety.status === 'marginal' ? (
              <AlertTriangle className="w-4 h-4 text-yellow-400" />
            ) : (
              <ShieldAlert className="w-4 h-4 text-red-400" />
            )}
            <span className={`text-xs font-medium ${
              result.safety.status === 'safe'
                ? 'text-green-400'
                : result.safety.status === 'marginal'
                ? 'text-yellow-400'
                : 'text-red-400'
            }`}>
              {result.safety.status === 'safe'
                ? 'Safe Passage'
                : result.safety.status === 'marginal'
                ? 'Marginal Conditions'
                : 'Dangerous Conditions'}
            </span>
          </div>
          <div className="text-xs text-gray-400 space-y-0.5">
            <div>Max Roll: {result.safety.max_roll_deg.toFixed(1)}°</div>
            <div>Max Pitch: {result.safety.max_pitch_deg.toFixed(1)}°</div>
          </div>
          {result.safety.warnings.length > 0 && (
            <div className="mt-1 text-xs text-yellow-400">
              {result.safety.warnings.slice(0, 2).map((w, i) => (
                <div key={i} className="truncate">• {w}</div>
              ))}
            </div>
          )}
        </div>
      )}

      <button
        onClick={onApply}
        className="mt-2 w-full py-2 text-sm bg-green-600 text-white rounded hover:bg-green-500 transition-colors"
      >
        Apply Optimized Route
      </button>
    </div>
  );
}

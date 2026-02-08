'use client';

import { format } from 'date-fns';
import { MonteCarloResult } from '@/lib/analysisStorage';

interface MonteCarloPanelProps {
  result: MonteCarloResult;
}

export default function MonteCarloPanel({ result }: MonteCarloPanelProps) {
  const formatDate = (iso: string) => {
    try {
      return format(new Date(iso), 'MMM d, HH:mm');
    } catch {
      return iso;
    }
  };

  const formatDuration = (hours: number): string => {
    const days = Math.floor(hours / 24);
    const remainingHours = Math.floor(hours % 24);
    if (days > 0) return `${days}d ${remainingHours}h`;
    const minutes = Math.round((hours % 1) * 60);
    return `${remainingHours}h ${minutes}m`;
  };

  return (
    <div className="bg-maritime-dark rounded-lg p-3 border border-purple-500/20">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-xs font-medium text-purple-400">
          Monte Carlo Simulation
        </h4>
        <span className="text-[10px] text-gray-500">
          {result.n_simulations} runs · {(result.computation_time_ms / 1000).toFixed(1)}s
        </span>
      </div>

      {/* ETA range */}
      <div className="mb-3">
        <div className="text-xs text-gray-400 mb-1.5">ETA Range (80% confidence)</div>
        <RangeBar
          p10Label={formatDate(result.eta.p10)}
          p50Label={formatDate(result.eta.p50)}
          p90Label={formatDate(result.eta.p90)}
          p10Value={result.total_time_hours.p10}
          p50Value={result.total_time_hours.p50}
          p90Value={result.total_time_hours.p90}
          color="blue"
        />
        <div className="text-[10px] text-gray-500 mt-1">
          {formatDuration(result.total_time_hours.p10)} — {formatDuration(result.total_time_hours.p90)}
        </div>
      </div>

      {/* Fuel range */}
      <div>
        <div className="text-xs text-gray-400 mb-1.5">Fuel Range (80% confidence)</div>
        <RangeBar
          p10Label={`${result.fuel_mt.p10.toFixed(1)} MT`}
          p50Label={`${result.fuel_mt.p50.toFixed(1)} MT`}
          p90Label={`${result.fuel_mt.p90.toFixed(1)} MT`}
          p10Value={result.fuel_mt.p10}
          p50Value={result.fuel_mt.p50}
          p90Value={result.fuel_mt.p90}
          color="green"
        />
      </div>
    </div>
  );
}

function RangeBar({
  p10Label,
  p50Label,
  p90Label,
  p10Value,
  p50Value,
  p90Value,
  color,
}: {
  p10Label: string;
  p50Label: string;
  p90Label: string;
  p10Value: number;
  p50Value: number;
  p90Value: number;
  color: 'blue' | 'green';
}) {
  const range = p90Value - p10Value;
  const p50Pct = range > 0 ? ((p50Value - p10Value) / range) * 100 : 50;

  const barBg = color === 'blue' ? 'bg-blue-500/30' : 'bg-green-500/30';
  const markerBg = color === 'blue' ? 'bg-blue-400' : 'bg-green-400';

  return (
    <div>
      {/* Labels */}
      <div className="flex justify-between text-[10px] text-gray-500 mb-0.5">
        <span>P10 (optimistic)</span>
        <span>P90 (conservative)</span>
      </div>

      {/* Bar */}
      <div className={`relative h-5 ${barBg} rounded-full overflow-hidden`}>
        {/* P50 marker */}
        <div
          className={`absolute top-0 h-full w-0.5 ${markerBg}`}
          style={{ left: `${p50Pct}%` }}
        />
        {/* P50 diamond */}
        <div
          className={`absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-2.5 h-2.5 ${markerBg} rotate-45 rounded-sm`}
          style={{ left: `${p50Pct}%` }}
        />
      </div>

      {/* Value labels */}
      <div className="flex justify-between mt-0.5">
        <span className="text-[10px] text-gray-400">{p10Label}</span>
        <span className={`text-[10px] font-medium ${color === 'blue' ? 'text-blue-400' : 'text-green-400'}`}>
          {p50Label}
        </span>
        <span className="text-[10px] text-gray-400">{p90Label}</span>
      </div>
    </div>
  );
}

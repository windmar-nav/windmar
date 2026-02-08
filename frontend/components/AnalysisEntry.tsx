'use client';

import { useState } from 'react';
import {
  Navigation,
  Clock,
  Fuel,
  Ship,
  ChevronDown,
  ChevronUp,
  MapPin,
  Trash2,
  Loader2,
  Dice5,
} from 'lucide-react';
import { format } from 'date-fns';
import VoyageResults, { VoyageProfile } from '@/components/VoyageResults';
import MonteCarloPanel from '@/components/MonteCarloPanel';
import { AnalysisEntry as AnalysisEntryType } from '@/lib/analysisStorage';

interface AnalysisEntryCardProps {
  analysis: AnalysisEntryType;
  isDisplayed: boolean;
  onShowOnMap: () => void;
  onDelete: () => void;
  onRunSimulation: () => void;
  isSimulating: boolean;
}

export default function AnalysisEntryCard({
  analysis,
  isDisplayed,
  onShowOnMap,
  onDelete,
  onRunSimulation,
  isSimulating,
}: AnalysisEntryCardProps) {
  const [expanded, setExpanded] = useState(false);
  const result = analysis.result;

  const formatDuration = (hours: number): string => {
    const days = Math.floor(hours / 24);
    const remainingHours = Math.floor(hours % 24);
    if (days > 0) return `${days}d ${remainingHours}h`;
    const minutes = Math.round((hours % 1) * 60);
    return `${remainingHours}h ${minutes}m`;
  };

  return (
    <div className={`rounded-lg border transition-colors ${
      isDisplayed
        ? 'border-primary-500/50 bg-primary-500/5'
        : 'border-white/10 bg-maritime-dark/50'
    }`}>
      {/* Collapsed header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full p-3 text-left"
      >
        <div className="flex items-start justify-between">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-white truncate">
                {analysis.routeName}
              </span>
              {isDisplayed && (
                <span className="shrink-0 px-1.5 py-0.5 text-[10px] rounded bg-primary-500/20 text-primary-400">
                  On Map
                </span>
              )}
            </div>
            <div className="text-xs text-gray-500 mt-0.5">
              {format(new Date(analysis.timestamp), 'MMM d, HH:mm')}
              {' · '}
              {analysis.parameters.calmSpeed} kts
              {' · '}
              {analysis.parameters.isLaden ? 'Laden' : 'Ballast'}
            </div>
          </div>
          {expanded ? (
            <ChevronUp className="w-4 h-4 text-gray-400 shrink-0 mt-1" />
          ) : (
            <ChevronDown className="w-4 h-4 text-gray-400 shrink-0 mt-1" />
          )}
        </div>

        {/* Summary metrics */}
        <div className="grid grid-cols-4 gap-2 mt-2">
          <MetricChip
            icon={<Navigation className="w-3 h-3" />}
            value={`${result.total_distance_nm.toFixed(0)} nm`}
          />
          <MetricChip
            icon={<Clock className="w-3 h-3" />}
            value={formatDuration(result.total_time_hours)}
          />
          <MetricChip
            icon={<Fuel className="w-3 h-3" />}
            value={`${result.total_fuel_mt.toFixed(1)} MT`}
          />
          <MetricChip
            icon={<Ship className="w-3 h-3" />}
            value={`${result.avg_sog_kts.toFixed(1)} kts`}
          />
        </div>
      </button>

      {/* Expanded content */}
      {expanded && (
        <div className="border-t border-white/10 p-3 space-y-3">
          {/* Action buttons */}
          <div className="flex gap-2">
            <button
              onClick={onShowOnMap}
              className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded text-xs font-medium transition-colors ${
                isDisplayed
                  ? 'bg-primary-500/20 text-primary-400 border border-primary-500/30'
                  : 'bg-maritime-dark text-gray-300 hover:text-white'
              }`}
            >
              <MapPin className="w-3.5 h-3.5" />
              Show on Map
            </button>
            <button
              onClick={onRunSimulation}
              disabled={isSimulating}
              className="flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded text-xs font-medium bg-maritime-dark text-gray-300 hover:text-white transition-colors disabled:opacity-50"
            >
              {isSimulating ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Dice5 className="w-3.5 h-3.5" />
              )}
              {isSimulating ? 'Simulating...' : 'Run Simulation'}
            </button>
            <button
              onClick={onDelete}
              className="px-2 py-1.5 rounded text-xs text-gray-400 hover:text-red-400 bg-maritime-dark transition-colors"
              title="Delete analysis"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Monte Carlo results */}
          {analysis.monteCarlo && (
            <MonteCarloPanel result={analysis.monteCarlo} />
          )}

          {/* Voyage results */}
          <VoyageResults voyage={result} />
          <VoyageProfile voyage={result} />
        </div>
      )}
    </div>
  );
}

function MetricChip({ icon, value }: { icon: React.ReactNode; value: string }) {
  return (
    <div className="flex items-center gap-1 text-xs text-gray-400">
      <span className="text-primary-400">{icon}</span>
      <span className="truncate">{value}</span>
    </div>
  );
}

'use client';

import { BarChart3, Ship } from 'lucide-react';
import { AnalysisEntry as AnalysisEntryType } from '@/lib/analysisStorage';
import AnalysisEntryCard from '@/components/AnalysisEntry';

interface AnalysisTabProps {
  analyses: AnalysisEntryType[];
  displayedAnalysisId: string | null;
  onShowOnMap: (id: string) => void;
  onDelete: (id: string) => void;
  onRunSimulation: (id: string) => void;
  simulatingId: string | null;
}

export default function AnalysisTab({
  analyses,
  displayedAnalysisId,
  onShowOnMap,
  onDelete,
  onRunSimulation,
  simulatingId,
}: AnalysisTabProps) {
  if (analyses.length === 0) {
    return (
      <div className="text-center py-12">
        <BarChart3 className="w-16 h-16 text-gray-600 mx-auto mb-4" />
        <h3 className="text-lg font-semibold text-gray-400 mb-2">
          No Analyses Yet
        </h3>
        <p className="text-sm text-gray-500">
          Go to the Route tab, set waypoints and parameters, then click
          "Calculate Voyage" to create an analysis.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-gray-300">
          {analyses.length} {analyses.length === 1 ? 'analysis' : 'analyses'}
        </h3>
      </div>

      {analyses.map((analysis) => (
        <AnalysisEntryCard
          key={analysis.id}
          analysis={analysis}
          isDisplayed={displayedAnalysisId === analysis.id}
          onShowOnMap={() => onShowOnMap(analysis.id)}
          onDelete={() => onDelete(analysis.id)}
          onRunSimulation={() => onRunSimulation(analysis.id)}
          isSimulating={simulatingId === analysis.id}
        />
      ))}
    </div>
  );
}

'use client';

import { X } from 'lucide-react';
import AnalysisTab from '@/components/AnalysisTab';
import { AnalysisEntry } from '@/lib/analysisStorage';

interface AnalysisSlidePanelProps {
  open: boolean;
  onClose: () => void;
  analyses: AnalysisEntry[];
  displayedAnalysisId: string | null;
  onShowOnMap: (id: string) => void;
  onDelete: (id: string) => void;
  onRunSimulation: (id: string) => void;
  simulatingId: string | null;
}

export default function AnalysisSlidePanel({
  open,
  onClose,
  analyses,
  displayedAnalysisId,
  onShowOnMap,
  onDelete,
  onRunSimulation,
  simulatingId,
}: AnalysisSlidePanelProps) {
  return (
    <div
      className={`absolute top-0 right-0 z-[1000] h-full w-96 bg-maritime-dark/95 backdrop-blur-md border-l border-white/10 shadow-2xl transition-transform duration-300 ${
        open ? 'translate-x-0' : 'translate-x-full'
      }`}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-white/10">
        <h2 className="text-sm font-semibold text-white">Analyses</h2>
        <button
          onClick={onClose}
          className="p-1 rounded text-gray-400 hover:text-white hover:bg-white/10 transition-colors"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      {/* Content */}
      <div className="overflow-y-auto h-[calc(100%-56px)] p-3">
        <AnalysisTab
          analyses={analyses}
          displayedAnalysisId={displayedAnalysisId}
          onShowOnMap={onShowOnMap}
          onDelete={onDelete}
          onRunSimulation={onRunSimulation}
          simulatingId={simulatingId}
        />
      </div>
    </div>
  );
}

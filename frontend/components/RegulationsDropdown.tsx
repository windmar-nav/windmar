'use client';

import { PenTool } from 'lucide-react';
import { useVoyage, ZONE_TYPES } from '@/components/VoyageContext';

const ZONE_LABELS: Record<string, string> = {
  eca: 'ECA',
  hra: 'HRA',
  tss: 'TSS',
  vts: 'VTS',
  ice: 'Ice',
  canal: 'Canal',
  environmental: 'Environmental',
  exclusion: 'Exclusion',
};

// Only TSS is active for now — other zone types are shown but disabled
const ACTIVE_ZONE_TYPES = new Set(['tss']);

export default function RegulationsDropdown() {
  const { zoneVisibility, setZoneTypeVisible, isDrawingZone, setIsDrawingZone } = useVoyage();

  return (
    <div className="absolute top-full right-0 mt-2 w-64 bg-maritime-dark/95 backdrop-blur-md border border-white/10 rounded-xl shadow-2xl p-4 space-y-3 z-50">
      <div className="text-xs text-gray-400 uppercase tracking-wider mb-1">Zone Types</div>
      <div className="space-y-1">
        {ZONE_TYPES.map((type) => {
          const isActive = ACTIVE_ZONE_TYPES.has(type);
          return (
            <label
              key={type}
              className={`flex items-center justify-between px-2 py-1.5 rounded transition-colors ${
                isActive ? 'hover:bg-white/5 cursor-pointer' : 'opacity-40 cursor-not-allowed'
              }`}
            >
              <span className={`text-sm ${isActive ? 'text-gray-300' : 'text-gray-500'}`}>
                {ZONE_LABELS[type] || type}
                {!isActive && <span className="text-[10px] ml-1.5 text-gray-600">soon</span>}
              </span>
              <input
                type="checkbox"
                checked={zoneVisibility[type] || false}
                onChange={(e) => setZoneTypeVisible(type, e.target.checked)}
                disabled={!isActive}
                className="w-4 h-4 rounded border-white/20 bg-maritime-medium text-primary-500 focus:ring-primary-500 focus:ring-offset-0 disabled:opacity-30 disabled:cursor-not-allowed"
              />
            </label>
          );
        })}
      </div>

      <div className="border-t border-white/10 pt-3 space-y-2">
        <button
          onClick={() => setIsDrawingZone(!isDrawingZone)}
          className={`w-full flex items-center justify-center space-x-2 px-3 py-2 rounded-lg text-sm transition-colors ${
            isDrawingZone
              ? 'bg-amber-500/20 border border-amber-500/50 text-amber-400'
              : 'bg-maritime-medium text-gray-400 hover:text-white'
          }`}
        >
          <PenTool className="w-4 h-4" />
          <span>Draw Custom Zone</span>
        </button>
      </div>
    </div>
  );
}

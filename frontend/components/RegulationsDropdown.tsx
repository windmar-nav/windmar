'use client';

import { PenTool, Gauge, Fuel } from 'lucide-react';
import Link from 'next/link';
import { useVoyage, ZONE_TYPES } from '@/components/VoyageContext';

const ZONE_LABELS: Record<string, string> = {
  eca: 'ECA',
  seca: 'SECA',
  hra: 'HRA',
  tss: 'TSS',
  vts: 'VTS',
  ice: 'Ice',
  canal: 'Canal',
  environmental: 'Environmental',
  exclusion: 'Exclusion',
};

export default function RegulationsDropdown() {
  const { zoneVisibility, setZoneTypeVisible, isDrawingZone, setIsDrawingZone } = useVoyage();

  return (
    <div className="absolute top-full right-0 mt-2 w-64 bg-maritime-dark/95 backdrop-blur-md border border-white/10 rounded-xl shadow-2xl p-4 space-y-3 z-50">
      <div className="text-xs text-gray-400 uppercase tracking-wider mb-1">Zone Types</div>
      <div className="space-y-1">
        {ZONE_TYPES.map((type) => (
          <label
            key={type}
            className="flex items-center justify-between px-2 py-1.5 rounded hover:bg-white/5 cursor-pointer transition-colors"
          >
            <span className="text-sm text-gray-300">{ZONE_LABELS[type] || type}</span>
            <input
              type="checkbox"
              checked={zoneVisibility[type] || false}
              onChange={(e) => setZoneTypeVisible(type, e.target.checked)}
              className="w-4 h-4 rounded border-white/20 bg-maritime-medium text-primary-500 focus:ring-primary-500 focus:ring-offset-0"
            />
          </label>
        ))}
      </div>

      <div className="border-t border-white/10 pt-3 space-y-2">
        <Link
          href="/cii-compliance"
          className="w-full flex items-center justify-center space-x-2 px-3 py-2 rounded-lg text-sm bg-maritime-medium text-gray-400 hover:text-white transition-colors"
        >
          <Gauge className="w-4 h-4" />
          <span>CII Compliance</span>
        </Link>
        <Link
          href="/fueleu-compliance"
          className="w-full flex items-center justify-center space-x-2 px-3 py-2 rounded-lg text-sm bg-maritime-medium text-gray-400 hover:text-white transition-colors"
        >
          <Fuel className="w-4 h-4" />
          <span>FuelEU Maritime</span>
        </Link>
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

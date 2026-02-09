'use client';

import { Wind } from 'lucide-react';
import { useVoyage } from '@/components/VoyageContext';

export default function VoyageDropdown() {
  const { calmSpeed, setCalmSpeed, isLaden, setIsLaden, useWeather, setUseWeather } = useVoyage();

  return (
    <div className="absolute top-full right-0 mt-2 w-72 bg-maritime-dark/95 backdrop-blur-md border border-white/10 rounded-xl shadow-2xl p-4 space-y-4 z-50">
      {/* Calm Speed */}
      <div>
        <label className="block text-sm text-gray-300 mb-2">Calm Water Speed</label>
        <div className="flex items-center space-x-2">
          <input
            type="range"
            min="8"
            max="18"
            step="0.5"
            value={calmSpeed}
            onChange={(e) => setCalmSpeed(parseFloat(e.target.value))}
            className="flex-1"
          />
          <span className="w-16 text-right text-white font-semibold text-sm">
            {calmSpeed} kts
          </span>
        </div>
      </div>

      {/* Loading Condition */}
      <div>
        <label className="block text-sm text-gray-300 mb-2">Loading Condition</label>
        <div className="flex space-x-2">
          <button
            onClick={() => setIsLaden(true)}
            className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors ${
              isLaden
                ? 'bg-primary-500 text-white'
                : 'bg-maritime-medium text-gray-400 hover:text-white'
            }`}
          >
            Laden
          </button>
          <button
            onClick={() => setIsLaden(false)}
            className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors ${
              !isLaden
                ? 'bg-primary-500 text-white'
                : 'bg-maritime-medium text-gray-400 hover:text-white'
            }`}
          >
            Ballast
          </button>
        </div>
      </div>

      {/* Weather Toggle */}
      <div className="flex items-center justify-between p-3 bg-maritime-medium rounded-lg">
        <div className="flex items-center space-x-2">
          <Wind className="w-4 h-4 text-primary-400" />
          <span className="text-sm text-white">Use Weather</span>
        </div>
        <button
          onClick={() => setUseWeather(!useWeather)}
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
    </div>
  );
}

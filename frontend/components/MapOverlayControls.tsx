'use client';

import { Wind, Waves, Droplets, Clock, RefreshCw, Snowflake, CloudFog, AudioWaveform, Thermometer, Database } from 'lucide-react';
import { WeatherLayer } from '@/components/MapComponent';

interface MapOverlayControlsProps {
  weatherLayer: WeatherLayer;
  onWeatherLayerChange: (layer: WeatherLayer) => void;
  forecastEnabled: boolean;
  onForecastToggle: () => void;
  isLoadingWeather: boolean;
  onResync: () => void;
  layerIngestedAt: string | null;
  resyncRunning: boolean;
}

function computeStaleness(ingestedAt: string | null): { label: string; color: string } | null {
  if (!ingestedAt) return null;
  const ageMs = Date.now() - new Date(ingestedAt).getTime();
  if (isNaN(ageMs)) return null;
  const ageHours = ageMs / (1000 * 60 * 60);

  if (ageHours < 1) {
    const ageMin = Math.max(1, Math.round(ageHours * 60));
    return { label: `${ageMin}m ago`, color: 'text-green-400' };
  }
  if (ageHours < 12) {
    return { label: `${Math.round(ageHours)}h ago`, color: ageHours < 4 ? 'text-green-400' : 'text-yellow-400' };
  }
  const ageDays = Math.round(ageHours / 24);
  return { label: ageDays >= 1 ? `${ageDays}d ago` : `${Math.round(ageHours)}h ago`, color: 'text-red-400' };
}

export default function MapOverlayControls({
  weatherLayer,
  onWeatherLayerChange,
  forecastEnabled,
  onForecastToggle,
  isLoadingWeather,
  onResync,
  layerIngestedAt,
  resyncRunning,
}: MapOverlayControlsProps) {
  const staleness = weatherLayer !== 'none' ? computeStaleness(layerIngestedAt) : null;

  return (
    <div className="absolute top-3 right-3 z-[1000] flex flex-col gap-1.5">
      <OverlayButton
        icon={<Wind className="w-4 h-4" />}
        label="Wind"
        active={weatherLayer === 'wind'}
        onClick={() => onWeatherLayerChange(weatherLayer === 'wind' ? 'none' : 'wind')}
      />
      <OverlayButton
        icon={<Waves className="w-4 h-4" />}
        label="Waves"
        active={weatherLayer === 'waves'}
        onClick={() => onWeatherLayerChange(weatherLayer === 'waves' ? 'none' : 'waves')}
      />
      <OverlayButton
        icon={<Droplets className="w-4 h-4" />}
        label="Currents"
        active={weatherLayer === 'currents'}
        onClick={() => onWeatherLayerChange(weatherLayer === 'currents' ? 'none' : 'currents')}
      />
      <OverlayButton
        icon={<Snowflake className="w-4 h-4" />}
        label="Ice"
        active={weatherLayer === 'ice'}
        onClick={() => onWeatherLayerChange(weatherLayer === 'ice' ? 'none' : 'ice')}
      />
      <OverlayButton
        icon={<CloudFog className="w-4 h-4" />}
        label="Visibility"
        active={weatherLayer === 'visibility'}
        onClick={() => onWeatherLayerChange(weatherLayer === 'visibility' ? 'none' : 'visibility')}
      />
      <OverlayButton
        icon={<Thermometer className="w-4 h-4" />}
        label="SST"
        active={weatherLayer === 'sst'}
        onClick={() => onWeatherLayerChange(weatherLayer === 'sst' ? 'none' : 'sst')}
      />
      <OverlayButton
        icon={<AudioWaveform className="w-4 h-4" />}
        label="Swell"
        active={weatherLayer === 'swell'}
        onClick={() => onWeatherLayerChange(weatherLayer === 'swell' ? 'none' : 'swell')}
      />
      {weatherLayer !== 'none' && (
        <OverlayButton
          icon={<Clock className="w-4 h-4" />}
          label="Timeline"
          active={forecastEnabled}
          onClick={onForecastToggle}
        />
      )}
      {weatherLayer !== 'none' && (
        <button
          onClick={onResync}
          disabled={isLoadingWeather || resyncRunning}
          className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs bg-maritime-dark/90 backdrop-blur-sm border border-white/10 text-gray-400 hover:text-white transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${resyncRunning ? 'animate-spin' : ''}`} />
          <span>{resyncRunning ? 'Resyncing...' : 'Resync'}</span>
        </button>
      )}
      {staleness && (
        <div className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs bg-maritime-dark/90 backdrop-blur-sm border border-white/10 ${staleness.color}`}>
          <Database className="w-3 h-3" />
          <span>{staleness.label}</span>
        </div>
      )}
    </div>
  );
}

function OverlayButton({
  icon,
  label,
  active,
  onClick,
}: {
  icon: React.ReactNode;
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs backdrop-blur-sm border transition-colors ${
        active
          ? 'bg-primary-500/30 border-primary-500/50 text-primary-300'
          : 'bg-maritime-dark/90 border-white/10 text-gray-400 hover:text-white'
      }`}
    >
      {icon}
      <span>{label}</span>
    </button>
  );
}

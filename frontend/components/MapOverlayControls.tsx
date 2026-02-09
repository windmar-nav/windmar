'use client';

import { useEffect, useState } from 'react';
import { Wind, Waves, Droplets, Clock, RefreshCw, Eye, EyeOff, Database, BarChart3 } from 'lucide-react';
import { WeatherLayer } from '@/components/MapComponent';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface MapOverlayControlsProps {
  weatherLayer: WeatherLayer;
  onWeatherLayerChange: (layer: WeatherLayer) => void;
  forecastEnabled: boolean;
  onForecastToggle: () => void;
  isLoadingWeather: boolean;
  onRefresh: () => void;
  analysisOpen?: boolean;
  onAnalysisToggle?: () => void;
}

interface FreshnessInfo {
  age_hours: number;
  latest_ingestion: string;
}

export default function MapOverlayControls({
  weatherLayer,
  onWeatherLayerChange,
  forecastEnabled,
  onForecastToggle,
  isLoadingWeather,
  onRefresh,
  analysisOpen,
  onAnalysisToggle,
}: MapOverlayControlsProps) {
  const [freshness, setFreshness] = useState<FreshnessInfo | null>(null);

  useEffect(() => {
    const fetchFreshness = async () => {
      try {
        const res = await fetch(`${API_URL}/api/weather/ingest/status`);
        if (!res.ok) return;
        const data = await res.json();
        if (data.freshness) {
          setFreshness(data.freshness);
        }
      } catch {
        // Silently ignore — indicator simply won't show
      }
    };

    fetchFreshness();
    const interval = setInterval(fetchFreshness, 5 * 60 * 1000); // every 5 min
    return () => clearInterval(interval);
  }, []);

  const freshnessLabel = freshness
    ? freshness.age_hours < 4
      ? `${Math.round(freshness.age_hours)}h ago`
      : freshness.age_hours < 12
        ? `${Math.round(freshness.age_hours)}h ago`
        : 'stale'
    : null;

  const freshnessColor = freshness
    ? freshness.age_hours < 4
      ? 'text-green-400'
      : freshness.age_hours < 12
        ? 'text-yellow-400'
        : 'text-red-400'
    : 'text-gray-500';

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
      {weatherLayer !== 'none' && (
        <OverlayButton
          icon={<Clock className="w-4 h-4" />}
          label="Timeline"
          active={forecastEnabled}
          onClick={onForecastToggle}
        />
      )}
      {onAnalysisToggle && (
        <OverlayButton
          icon={<BarChart3 className="w-4 h-4" />}
          label="Analysis"
          active={analysisOpen ?? false}
          onClick={onAnalysisToggle}
        />
      )}
      <button
        onClick={onRefresh}
        disabled={isLoadingWeather}
        className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs bg-maritime-dark/90 backdrop-blur-sm border border-white/10 text-gray-400 hover:text-white transition-colors disabled:opacity-50"
      >
        <RefreshCw className={`w-3.5 h-3.5 ${isLoadingWeather ? 'animate-spin' : ''}`} />
        <span>Refresh</span>
      </button>
      {freshnessLabel && (
        <div className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs bg-maritime-dark/90 backdrop-blur-sm border border-white/10 ${freshnessColor}`}>
          <Database className="w-3 h-3" />
          <span>{freshnessLabel}</span>
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
      {active ? (
        <Eye className="w-3 h-3 ml-auto" />
      ) : (
        <EyeOff className="w-3 h-3 ml-auto" />
      )}
    </button>
  );
}

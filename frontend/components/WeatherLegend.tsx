'use client';

interface WeatherLegendProps {
  mode: 'wind' | 'waves' | 'currents' | 'ice' | 'visibility' | 'sst' | 'swell';
  timelineVisible?: boolean;
  dataRange?: { min: number; max: number } | null;
}

const WIND_STOPS = [
  { value: 0, color: 'rgb(30,80,220)' },
  { value: 5, color: 'rgb(0,200,220)' },
  { value: 10, color: 'rgb(0,200,50)' },
  { value: 15, color: 'rgb(240,220,0)' },
  { value: 20, color: 'rgb(240,130,0)' },
  { value: 25, color: 'rgb(220,30,30)' },
];

const WAVE_STOPS = [
  { value: 0, color: 'rgb(0,200,50)' },
  { value: 1, color: 'rgb(240,220,0)' },
  { value: 2, color: 'rgb(240,130,0)' },
  { value: 3, color: 'rgb(220,30,30)' },
  { value: 5, color: 'rgb(128,0,0)' },
];

const CURRENT_STOPS = [
  { value: 0, color: 'rgb(36,104,180)' },
  { value: 0.3, color: 'rgb(24,176,200)' },
  { value: 0.6, color: 'rgb(100,200,160)' },
  { value: 1.0, color: 'rgb(210,220,100)' },
  { value: 1.5, color: 'rgb(250,180,60)' },
  { value: 2.0, color: 'rgb(250,140,40)' },
];

const ICE_STOPS = [
  { value: 0, color: 'rgb(80,160,255)' },
  { value: 10, color: 'rgb(140,210,255)' },
  { value: 30, color: 'rgb(80,200,200)' },
  { value: 50, color: 'rgb(100,220,100)' },
  { value: 70, color: 'rgb(230,220,40)' },
  { value: 80, color: 'rgb(240,170,20)' },
  { value: 90, color: 'rgb(210,80,30)' },
  { value: 95, color: 'rgb(160,30,70)' },
  { value: 100, color: 'rgb(255,255,255)' },
];

const VIS_STOPS = [
  { value: 0, color: 'rgb(20,80,10)' },
  { value: 1, color: 'rgb(40,120,20)' },
  { value: 4, color: 'rgb(80,170,40)' },
  { value: 10, color: 'rgb(130,210,70)' },
  { value: 20, color: 'rgb(180,240,120)' },
];

const SST_STOPS = [
  { value: -2, color: 'rgb(20,30,140)' },
  { value: 2, color: 'rgb(40,80,200)' },
  { value: 8, color: 'rgb(0,180,220)' },
  { value: 14, color: 'rgb(0,200,80)' },
  { value: 20, color: 'rgb(220,220,0)' },
  { value: 26, color: 'rgb(240,130,0)' },
  { value: 32, color: 'rgb(220,30,30)' },
];

const SWELL_STOPS = [
  { value: 0, color: 'rgb(60,120,200)' },
  { value: 1, color: 'rgb(0,200,180)' },
  { value: 2, color: 'rgb(100,200,50)' },
  { value: 3, color: 'rgb(240,200,0)' },
  { value: 5, color: 'rgb(240,100,0)' },
  { value: 8, color: 'rgb(200,30,30)' },
];

function buildGradient(stops: { value: number; color: string }[]): string {
  const min = stops[0].value;
  const max = stops[stops.length - 1].value;
  const range = max - min || 1;
  const parts = stops.map(
    (s) => `${s.color} ${((s.value - min) / range) * 100}%`
  );
  return `linear-gradient(to right, ${parts.join(', ')})`;
}

const LEGEND_CONFIG: Record<string, { stops: typeof WIND_STOPS; unit: string; label: string }> = {
  wind: { stops: WIND_STOPS, unit: 'm/s', label: 'Wind Speed' },
  waves: { stops: WAVE_STOPS, unit: 'm', label: 'Wave Height' },
  currents: { stops: CURRENT_STOPS, unit: 'm/s', label: 'Current Speed' },
  ice: { stops: ICE_STOPS, unit: '%', label: 'Ice Concentration' },
  visibility: { stops: VIS_STOPS, unit: 'km', label: 'Visibility' },
  sst: { stops: SST_STOPS, unit: '°C', label: 'Sea Surface Temp' },
  swell: { stops: SWELL_STOPS, unit: 'm', label: 'Swell Height' },
};

export default function WeatherLegend({ mode, timelineVisible = false, dataRange }: WeatherLegendProps) {
  const config = LEGEND_CONFIG[mode] || LEGEND_CONFIG.wind;
  const { stops, unit, label } = config;
  const gradient = buildGradient(stops);

  // SST auto-range: remap stop labels to actual data bounds
  const displayStops = (mode === 'sst' && dataRange && dataRange.max > dataRange.min)
    ? stops.map(s => {
        const t = (s.value - stops[0].value) / (stops[stops.length - 1].value - stops[0].value);
        return { ...s, displayValue: Math.round((dataRange.min + t * (dataRange.max - dataRange.min)) * 10) / 10 };
      })
    : stops.map(s => ({ ...s, displayValue: s.value }));

  return (
    <div className={`absolute right-4 bg-maritime-dark/90 backdrop-blur-sm rounded-lg p-3 z-[1000] min-w-[180px] transition-all ${timelineVisible ? 'bottom-20' : 'bottom-4'}`}>
      <div className="text-xs text-gray-400 mb-2 font-medium">
        {label} ({unit})
      </div>
      <div
        className="h-3 rounded-sm"
        style={{ background: gradient }}
      />
      <div className="flex justify-between mt-1">
        {displayStops.map((s, i) => (
          <span key={i} className="text-[10px] text-gray-300">
            {s.displayValue}
          </span>
        ))}
      </div>
    </div>
  );
}

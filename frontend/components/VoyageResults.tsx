'use client';

import { VoyageResponse, LegResult, RouteWeatherPoint, apiClient } from '@/lib/api';
import { format } from 'date-fns';
import {
  Navigation,
  Clock,
  Fuel,
  Wind,
  Waves,
  TrendingDown,
  Ship,
  ChevronDown,
  ChevronUp,
  AlertTriangle,
  CloudSun,
  Calendar,
  Loader2,
} from 'lucide-react';
import { useState, useEffect } from 'react';
import {
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts';

interface VoyageResultsProps {
  voyage: VoyageResponse;
}

/**
 * Display voyage calculation results with per-leg details.
 */
export default function VoyageResults({ voyage }: VoyageResultsProps) {
  const [expandedLegs, setExpandedLegs] = useState<Set<number>>(new Set());

  const toggleLeg = (index: number) => {
    const newExpanded = new Set(expandedLegs);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedLegs(newExpanded);
  };

  const formatDuration = (hours: number): string => {
    const days = Math.floor(hours / 24);
    const remainingHours = Math.floor(hours % 24);
    const minutes = Math.round((hours % 1) * 60);

    if (days > 0) {
      return `${days}d ${remainingHours}h`;
    }
    return `${remainingHours}h ${minutes}m`;
  };

  const formatDateTime = (isoString: string): string => {
    return format(new Date(isoString), 'MMM d, HH:mm');
  };

  return (
    <div className="space-y-4">
      {/* Data Source Warning */}
      {voyage.data_sources?.warning && (
        <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-3 flex items-start space-x-3">
          <AlertTriangle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
          <div>
            <div className="text-sm font-medium text-yellow-400">Weather Data Notice</div>
            <div className="text-xs text-yellow-300/80 mt-1">{voyage.data_sources.warning}</div>
            <div className="text-xs text-gray-400 mt-2 flex items-center space-x-4">
              <span className="flex items-center space-x-1">
                <CloudSun className="w-3 h-3" />
                <span>Forecast: {voyage.data_sources.forecast_legs} legs</span>
              </span>
              {voyage.data_sources.blended_legs > 0 && (
                <span className="flex items-center space-x-1">
                  <span className="w-2 h-2 rounded-full bg-yellow-500" />
                  <span>Blended: {voyage.data_sources.blended_legs}</span>
                </span>
              )}
              {voyage.data_sources.climatology_legs > 0 && (
                <span className="flex items-center space-x-1">
                  <Calendar className="w-3 h-3" />
                  <span>Climatology: {voyage.data_sources.climatology_legs}</span>
                </span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Summary Header */}
      <div className="bg-gradient-to-r from-primary-500/20 to-ocean-500/20 rounded-lg p-4 border border-primary-500/30">
        <h3 className="text-lg font-semibold text-white mb-3">Voyage Summary</h3>

        <div className="grid grid-cols-2 gap-4">
          <SummaryItem
            icon={<Navigation className="w-4 h-4" />}
            label="Distance"
            value={`${voyage.total_distance_nm.toFixed(1)} nm`}
          />
          <SummaryItem
            icon={<Clock className="w-4 h-4" />}
            label="Duration"
            value={formatDuration(voyage.total_time_hours)}
          />
          <SummaryItem
            icon={<Ship className="w-4 h-4" />}
            label="Avg SOG"
            value={`${voyage.avg_sog_kts.toFixed(1)} kts`}
          />
          <SummaryItem
            icon={<Fuel className="w-4 h-4" />}
            label="Total Fuel"
            value={`${voyage.total_fuel_mt.toFixed(1)} MT`}
          />
        </div>

        <div className="mt-4 pt-3 border-t border-white/10 grid grid-cols-2 gap-4 text-xs">
          <div>
            <span className="text-gray-400">Departure:</span>
            <div className="text-white">{formatDateTime(voyage.departure_time)}</div>
          </div>
          <div>
            <span className="text-gray-400">Arrival (ETA):</span>
            <div className="text-white font-semibold">{formatDateTime(voyage.arrival_time)}</div>
          </div>
        </div>
      </div>

      {/* Speed Comparison */}
      <div className="bg-maritime-dark rounded-lg p-4">
        <h4 className="text-sm font-medium text-gray-300 mb-3">Speed Analysis</h4>
        <div className="space-y-2">
          <SpeedBar
            label="Calm Speed"
            value={voyage.calm_speed_kts}
            max={20}
            color="bg-green-500"
          />
          <SpeedBar
            label="Avg STW"
            value={voyage.avg_stw_kts}
            max={20}
            color="bg-yellow-500"
          />
          <SpeedBar
            label="Avg SOG"
            value={voyage.avg_sog_kts}
            max={20}
            color="bg-primary-500"
          />
        </div>
        <div className="mt-2 text-xs text-gray-400">
          Weather impact: -{((1 - voyage.avg_sog_kts / voyage.calm_speed_kts) * 100).toFixed(1)}% speed loss
        </div>
      </div>

      {/* Leg Details */}
      <div className="bg-maritime-dark rounded-lg p-4">
        <h4 className="text-sm font-medium text-gray-300 mb-3">
          Leg Details ({voyage.legs.length} legs)
        </h4>

        <div className="space-y-2">
          {voyage.legs.map((leg) => (
            <LegRow
              key={leg.leg_index}
              leg={leg}
              isExpanded={expandedLegs.has(leg.leg_index)}
              onToggle={() => toggleLeg(leg.leg_index)}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

function SummaryItem({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-center space-x-2">
      <div className="text-primary-400">{icon}</div>
      <div>
        <div className="text-xs text-gray-400">{label}</div>
        <div className="text-sm font-semibold text-white">{value}</div>
      </div>
    </div>
  );
}

function SpeedBar({
  label,
  value,
  max,
  color,
}: {
  label: string;
  value: number;
  max: number;
  color: string;
}) {
  const percent = (value / max) * 100;

  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-gray-400">{label}</span>
        <span className="text-white">{value.toFixed(1)} kts</span>
      </div>
      <div className="h-2 bg-maritime-light rounded-full overflow-hidden">
        <div
          className={`h-full ${color} rounded-full transition-all duration-500`}
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}

function LegRow({
  leg,
  isExpanded,
  onToggle,
}: {
  leg: LegResult;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  const speedLossColor =
    leg.speed_loss_pct < 5
      ? 'text-green-400'
      : leg.speed_loss_pct < 15
      ? 'text-yellow-400'
      : 'text-red-400';

  // Data source indicator color
  const getDataSourceStyle = () => {
    if (!leg.data_source) return { bg: 'bg-primary-500/20', text: 'text-primary-400' };
    switch (leg.data_source) {
      case 'forecast':
        return { bg: 'bg-green-500/20', text: 'text-green-400', icon: '☀' };
      case 'blended':
        return { bg: 'bg-yellow-500/20', text: 'text-yellow-400', icon: '◐' };
      case 'climatology':
        return { bg: 'bg-orange-500/20', text: 'text-orange-400', icon: '📊' };
      default:
        return { bg: 'bg-primary-500/20', text: 'text-primary-400' };
    }
  };

  const sourceStyle = getDataSourceStyle();

  return (
    <div className="border border-white/5 rounded-lg overflow-hidden">
      {/* Collapsed Header */}
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between p-3 hover:bg-maritime-light/50 transition-colors"
      >
        <div className="flex items-center space-x-3">
          <span className={`w-6 h-6 rounded-full ${sourceStyle.bg} flex items-center justify-center text-xs ${sourceStyle.text} font-semibold`}
            title={leg.data_source ? `Weather: ${leg.data_source}` : undefined}>
            {leg.leg_index + 1}
          </span>
          <div className="text-left">
            <div className="text-sm text-white">
              {leg.from_wp.name} → {leg.to_wp.name}
            </div>
            <div className="text-xs text-gray-400">
              {leg.distance_nm.toFixed(1)} nm · {formatHours(leg.time_hours)}
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <div className="text-right">
            <div className="text-sm font-semibold text-white">{leg.sog_kts.toFixed(1)} kts</div>
            <div className={`text-xs ${speedLossColor}`}>
              {leg.speed_loss_pct > 0 ? `-${leg.speed_loss_pct.toFixed(1)}%` : 'No loss'}
            </div>
          </div>
          {isExpanded ? (
            <ChevronUp className="w-4 h-4 text-gray-400" />
          ) : (
            <ChevronDown className="w-4 h-4 text-gray-400" />
          )}
        </div>
      </button>

      {/* Expanded Details */}
      {isExpanded && (
        <div className="px-3 pb-3 border-t border-white/5 bg-maritime-light/30">
          <div className="grid grid-cols-2 gap-4 mt-3">
            {/* Weather */}
            <div>
              <div className="text-xs text-gray-400 mb-2 flex items-center space-x-1">
                <Wind className="w-3 h-3" />
                <span>Weather</span>
              </div>
              <div className="space-y-1 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-400">Wind:</span>
                  <span className="text-white">
                    {leg.wind_speed_kts.toFixed(0)} kts @ {leg.wind_dir_deg.toFixed(0)}°
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Waves:</span>
                  <span className="text-white">
                    {leg.wave_height_m.toFixed(1)} m @ {leg.wave_dir_deg.toFixed(0)}°
                  </span>
                </div>
              </div>
            </div>

            {/* Speed */}
            <div>
              <div className="text-xs text-gray-400 mb-2 flex items-center space-x-1">
                <Ship className="w-3 h-3" />
                <span>Speed</span>
              </div>
              <div className="space-y-1 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-400">Calm:</span>
                  <span className="text-white">{leg.calm_speed_kts.toFixed(1)} kts</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">STW:</span>
                  <span className="text-white">{leg.stw_kts.toFixed(1)} kts</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">SOG:</span>
                  <span className="text-white font-semibold">{leg.sog_kts.toFixed(1)} kts</span>
                </div>
              </div>
            </div>

            {/* Navigation */}
            <div>
              <div className="text-xs text-gray-400 mb-2 flex items-center space-x-1">
                <Navigation className="w-3 h-3" />
                <span>Navigation</span>
              </div>
              <div className="space-y-1 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-400">Bearing:</span>
                  <span className="text-white">{leg.bearing_deg.toFixed(1)}°</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Distance:</span>
                  <span className="text-white">{leg.distance_nm.toFixed(1)} nm</span>
                </div>
              </div>
            </div>

            {/* Fuel */}
            <div>
              <div className="text-xs text-gray-400 mb-2 flex items-center space-x-1">
                <Fuel className="w-3 h-3" />
                <span>Fuel</span>
              </div>
              <div className="space-y-1 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-400">Consumption:</span>
                  <span className="text-white">{leg.fuel_mt.toFixed(2)} MT</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Power:</span>
                  <span className="text-white">{leg.power_kw.toFixed(0)} kW</span>
                </div>
              </div>
            </div>
          </div>

          {/* Times */}
          <div className="mt-3 pt-2 border-t border-white/5 flex justify-between text-xs">
            <div>
              <span className="text-gray-400">ETD: </span>
              <span className="text-white">
                {format(new Date(leg.departure_time), 'MMM d, HH:mm')}
              </span>
            </div>
            <div>
              <span className="text-gray-400">ETA: </span>
              <span className="text-white font-semibold">
                {format(new Date(leg.arrival_time), 'MMM d, HH:mm')}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function formatHours(hours: number): string {
  const h = Math.floor(hours);
  const m = Math.round((hours - h) * 60);
  return `${h}h ${m}m`;
}

/**
 * Voyage profile chart — distance-indexed weather with waypoint markers.
 * Uses Recharts ComposedChart with data from the weather-along-route endpoint.
 */
interface VoyageProfileProps {
  voyage: VoyageResponse;
}

export function VoyageProfile({ voyage }: VoyageProfileProps) {
  const [routeWeather, setRouteWeather] = useState<RouteWeatherPoint[] | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  // Fetch weather-along-route data when voyage result exists
  useEffect(() => {
    const waypoints = [
      voyage.legs[0]?.from_wp,
      ...voyage.legs.map((l) => l.to_wp),
    ];
    if (waypoints.length < 2) return;

    setIsLoading(true);
    apiClient
      .getWeatherAlongRoute(
        waypoints.map((wp) => ({ lat: wp.lat, lon: wp.lon })),
        undefined,
        5,
      )
      .then((res) => setRouteWeather(res.points))
      .catch((err) => console.error('Weather along route failed:', err))
      .finally(() => setIsLoading(false));
  }, [voyage]);

  // Build chart data from route weather OR fallback to leg-based data
  const chartData = routeWeather
    ? routeWeather.map((pt) => ({
        distance_nm: pt.distance_nm,
        wind_speed_kts: pt.wind_speed_kts,
        wave_height_m: pt.wave_height_m,
        is_waypoint: pt.is_waypoint,
        waypoint_index: pt.waypoint_index,
      }))
    : buildFallbackChartData(voyage);

  // Interpolate SOG from voyage legs onto the distance axis
  const sogData = buildSogOverlay(voyage);

  // Merge SOG into chart data by nearest distance
  const mergedData = chartData.map((pt) => {
    const nearest = sogData.reduce((best, s) =>
      Math.abs(s.distance_nm - pt.distance_nm) < Math.abs(best.distance_nm - pt.distance_nm) ? s : best,
    );
    return { ...pt, sog_kts: nearest.sog_kts };
  });

  // Waypoint reference lines
  const waypointMarkers = mergedData.filter((d) => d.is_waypoint);

  return (
    <div className="bg-maritime-dark rounded-lg p-4">
      <h4 className="text-sm font-medium text-gray-300 mb-4 flex items-center gap-2">
        Route Weather Profile
        {isLoading && <Loader2 className="w-3.5 h-3.5 animate-spin text-primary-400" />}
      </h4>

      <ResponsiveContainer width="100%" height={160}>
        <ComposedChart data={mergedData} margin={{ top: 5, right: 5, bottom: 5, left: -10 }}>
          <XAxis
            dataKey="distance_nm"
            tick={{ fill: '#9ca3af', fontSize: 10 }}
            tickFormatter={(v: number) => `${Math.round(v)}`}
            label={{ value: 'nm', position: 'insideBottomRight', offset: -2, fill: '#6b7280', fontSize: 10 }}
          />
          <YAxis
            yAxisId="left"
            tick={{ fill: '#9ca3af', fontSize: 10 }}
            domain={[0, 'auto']}
          />
          <YAxis
            yAxisId="right"
            orientation="right"
            tick={{ fill: '#9ca3af', fontSize: 10 }}
            domain={[0, 'auto']}
          />

          <Tooltip
            contentStyle={{ backgroundColor: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontSize: 11 }}
            labelStyle={{ color: '#9ca3af' }}
            labelFormatter={(v: number) => `${Math.round(v)} nm`}
            formatter={(value: number, name: string) => {
              if (name === 'wind_speed_kts') return [`${value.toFixed(1)} kts`, 'Wind'];
              if (name === 'wave_height_m') return [`${value.toFixed(1)} m`, 'Waves'];
              if (name === 'sog_kts') return [`${value.toFixed(1)} kts`, 'SOG'];
              return [value, name];
            }}
          />

          {/* Wind area */}
          <Area
            yAxisId="left"
            dataKey="wind_speed_kts"
            stroke="#3b82f6"
            fill="#3b82f6"
            fillOpacity={0.3}
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />

          {/* Wave area */}
          <Area
            yAxisId="right"
            dataKey="wave_height_m"
            stroke="#22d3ee"
            fill="#22d3ee"
            fillOpacity={0.2}
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />

          {/* SOG line */}
          <Line
            yAxisId="left"
            dataKey="sog_kts"
            stroke="#22c55e"
            strokeWidth={1.5}
            strokeDasharray="4 2"
            dot={false}
            isAnimationActive={false}
          />

          {/* Waypoint reference lines */}
          {waypointMarkers.map((wp) => (
            <ReferenceLine
              key={`wp-${wp.waypoint_index}`}
              yAxisId="left"
              x={wp.distance_nm}
              stroke="rgba(255,255,255,0.25)"
              strokeDasharray="3 3"
              label={{
                value: `WP${(wp.waypoint_index ?? 0) + 1}`,
                position: 'top',
                fill: '#9ca3af',
                fontSize: 9,
              }}
            />
          ))}
        </ComposedChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="flex justify-center space-x-6 mt-2 text-xs">
        <div className="flex items-center space-x-1">
          <div className="w-3 h-3 bg-blue-500/50 rounded" />
          <span className="text-gray-400">Wind (kts)</span>
        </div>
        <div className="flex items-center space-x-1">
          <div className="w-3 h-1 bg-cyan-400" />
          <span className="text-gray-400">Waves (m)</span>
        </div>
        <div className="flex items-center space-x-1">
          <div className="w-3 h-0.5 bg-green-500 border-dashed border" />
          <span className="text-gray-400">SOG (kts)</span>
        </div>
      </div>
    </div>
  );
}

/** Build fallback chart data from voyage legs when weather-along-route unavailable. */
function buildFallbackChartData(voyage: VoyageResponse) {
  let cumDist = 0;
  const data: Array<{
    distance_nm: number;
    wind_speed_kts: number;
    wave_height_m: number;
    is_waypoint: boolean;
    waypoint_index: number | null;
  }> = [];

  // First waypoint
  data.push({
    distance_nm: 0,
    wind_speed_kts: voyage.legs[0]?.wind_speed_kts ?? 0,
    wave_height_m: voyage.legs[0]?.wave_height_m ?? 0,
    is_waypoint: true,
    waypoint_index: 0,
  });

  voyage.legs.forEach((leg, i) => {
    cumDist += leg.distance_nm;
    data.push({
      distance_nm: Math.round(cumDist * 10) / 10,
      wind_speed_kts: leg.wind_speed_kts,
      wave_height_m: leg.wave_height_m,
      is_waypoint: true,
      waypoint_index: i + 1,
    });
  });

  return data;
}

/** Build SOG overlay data indexed by cumulative distance from voyage legs. */
function buildSogOverlay(voyage: VoyageResponse) {
  let cumDist = 0;
  const data: Array<{ distance_nm: number; sog_kts: number }> = [];

  data.push({ distance_nm: 0, sog_kts: voyage.legs[0]?.sog_kts ?? 0 });

  voyage.legs.forEach((leg) => {
    // Leg midpoint
    const mid = cumDist + leg.distance_nm / 2;
    data.push({ distance_nm: mid, sog_kts: leg.sog_kts });
    cumDist += leg.distance_nm;
  });

  // Final point
  data.push({ distance_nm: cumDist, sog_kts: voyage.legs[voyage.legs.length - 1]?.sog_kts ?? 0 });

  return data;
}

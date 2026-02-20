'use client';

import { Suspense, useEffect, useState, useMemo } from 'react';
import { useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { format } from 'date-fns';
import {
  ArrowLeft, Navigation, Clock, Fuel, Ship, Wind, Waves, Droplets,
  AlertTriangle, Database,
} from 'lucide-react';
import Header from '@/components/Header';
import MonteCarloPanel from '@/components/MonteCarloPanel';
import ProfileCharts from '@/components/ProfileCharts';
import { getAnalyses, AnalysisEntry } from '@/lib/analysisStorage';
import { DEMO_MODE } from '@/lib/demoMode';
import { LegResult } from '@/lib/api';

function formatDuration(hours: number): string {
  const days = Math.floor(hours / 24);
  const h = Math.floor(hours % 24);
  const m = Math.round((hours % 1) * 60);
  if (days > 0) return `${days}d ${h}h ${m}m`;
  return `${h}h ${m}m`;
}

function windArrow(deg: number) {
  return (
    <span
      className="inline-block text-[10px]"
      style={{ transform: `rotate(${deg}deg)` }}
      title={`${deg.toFixed(0)}°`}
    >
      ↑
    </span>
  );
}

export default function AnalysisPageWrapper() {
  return (
    <Suspense fallback={
      <div className="min-h-screen bg-gradient-maritime flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-400" />
      </div>
    }>
      <AnalysisPage />
    </Suspense>
  );
}

function AnalysisPage() {
  const searchParams = useSearchParams();
  const id = searchParams.get('id');
  const [analysis, setAnalysis] = useState<AnalysisEntry | null>(null);
  const [notFound, setNotFound] = useState(false);

  useEffect(() => {
    if (!id) { setNotFound(true); return; }
    const all = getAnalyses();
    const found = all.find(a => a.id === id);
    if (found) {
      setAnalysis(found);
    } else {
      setNotFound(true);
    }
  }, [id]);

  // Derived weather insights
  const insights = useMemo(() => {
    if (!analysis) return null;
    const legs = analysis.result.legs;
    if (!legs || legs.length === 0) return null;

    const maxWave = Math.max(...legs.map(l => l.wave_height_m));
    const maxWind = Math.max(...legs.map(l => l.wind_speed_kts));
    const maxCurrent = Math.max(...legs.map(l => ((l.current_speed_ms ?? 0) * 1.9438)));
    const speedLossLegs = legs.filter(l => l.speed_loss_pct > 10).length;
    const speedLossPct = legs.length > 0 ? (speedLossLegs / legs.length * 100).toFixed(0) : '0';

    const ds = analysis.result.data_sources;
    const forecastLegs = ds?.forecast_legs ?? 0;
    const blendedLegs = ds?.blended_legs ?? 0;
    const climLegs = ds?.climatology_legs ?? 0;

    return { maxWave, maxWind, maxCurrent, speedLossLegs, speedLossPct, forecastLegs, blendedLegs, climLegs };
  }, [analysis]);

  if (notFound) {
    return (
      <div className="min-h-screen bg-gradient-maritime">
        <Header />
        <main className="container mx-auto px-6 pt-20">
          <p className="text-gray-400">Analysis not found.</p>
          <Link href="/" className="text-primary-400 hover:text-primary-300 text-sm mt-2 inline-block">
            <ArrowLeft className="w-4 h-4 inline mr-1" />
            Back to Chart
          </Link>
        </main>
      </div>
    );
  }

  if (!analysis) {
    return (
      <div className="min-h-screen bg-gradient-maritime flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-400" />
      </div>
    );
  }

  const r = analysis.result;

  return (
    <div className="min-h-screen bg-gradient-maritime">
      <Header />
      <main className="container mx-auto px-6 pt-20 pb-12">
        {/* Back link */}
        <Link
          href="/"
          className="inline-flex items-center gap-1.5 text-sm text-gray-400 hover:text-white mb-6 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Chart
        </Link>

        {/* Title */}
        <h1 className="text-xl font-bold text-white mb-1">{analysis.routeName}</h1>
        <p className="text-sm text-gray-500 mb-6">
          Calculated {format(new Date(analysis.timestamp), 'MMM d, yyyy HH:mm')}
          {' · '}
          {analysis.parameters.calmSpeed} kts
          {' · '}
          {analysis.parameters.isLaden ? 'Laden' : 'Ballast'}
        </p>

        {/* ── Summary Cards ── */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
          <SummaryCard
            icon={<Clock className="w-5 h-5 text-primary-400" />}
            label="Departure → Arrival"
            value={
              <>
                {format(new Date(r.departure_time), 'MMM d HH:mm')}
                <span className="text-gray-500 mx-1">→</span>
                {format(new Date(r.arrival_time), 'MMM d HH:mm')}
              </>
            }
          />
          <SummaryCard
            icon={<Navigation className="w-5 h-5 text-primary-400" />}
            label="Distance"
            value={`${r.total_distance_nm.toFixed(1)} nm`}
          />
          <SummaryCard
            icon={<Fuel className="w-5 h-5 text-primary-400" />}
            label="Fuel"
            value={`${r.total_fuel_mt.toFixed(1)} MT`}
          />
          <SummaryCard
            icon={<Ship className="w-5 h-5 text-primary-400" />}
            label="Duration / Avg SOG"
            value={`${formatDuration(r.total_time_hours)} / ${r.avg_sog_kts.toFixed(1)} kts`}
          />
        </div>

        {/* ── Per-Waypoint Passage Table ── */}
        <div className="mb-8">
          <h2 className="text-sm font-semibold text-white mb-3">Passage Plan</h2>
          <div className="overflow-x-auto rounded-lg border border-white/10">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-white/5 text-gray-400">
                  <th className="text-left px-3 py-2">#</th>
                  <th className="text-left px-3 py-2">Waypoint</th>
                  <th className="text-left px-3 py-2">ETA</th>
                  <th className="text-right px-3 py-2">Dist (nm)</th>
                  <th className="text-right px-3 py-2">SOG (kts)</th>
                  <th className="text-right px-3 py-2">Wind</th>
                  <th className="text-right px-3 py-2">Waves</th>
                  <th className="text-right px-3 py-2">Current</th>
                  <th className="text-right px-3 py-2">Fuel (MT)</th>
                  <th className="text-right px-3 py-2">Source</th>
                </tr>
              </thead>
              <tbody>
                {/* Origin waypoint */}
                <tr className="border-t border-white/5 text-gray-300">
                  <td className="px-3 py-2 text-gray-500">1</td>
                  <td className="px-3 py-2 font-medium">{r.legs[0]?.from_wp?.name || 'Origin'}</td>
                  <td className="px-3 py-2">{format(new Date(r.departure_time), 'MMM d HH:mm')}</td>
                  <td className="px-3 py-2 text-right text-gray-500">—</td>
                  <td className="px-3 py-2 text-right text-gray-500">—</td>
                  <td className="px-3 py-2 text-right text-gray-500">—</td>
                  <td className="px-3 py-2 text-right text-gray-500">—</td>
                  <td className="px-3 py-2 text-right text-gray-500">—</td>
                  <td className="px-3 py-2 text-right text-gray-500">—</td>
                  <td className="px-3 py-2 text-right text-gray-500">—</td>
                </tr>
                {/* Legs */}
                {r.legs.map((leg, i) => (
                  <LegRow key={i} leg={leg} index={i + 2} />
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* ── Weather Insights ── */}
        {insights && (
          <div className="mb-8">
            <h2 className="text-sm font-semibold text-white mb-3">Weather Insights</h2>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
              <InsightCard
                icon={<Waves className="w-4 h-4 text-blue-400" />}
                label="Max Wave Height"
                value={`${insights.maxWave.toFixed(1)} m`}
                warn={insights.maxWave > 3}
              />
              <InsightCard
                icon={<Wind className="w-4 h-4 text-cyan-400" />}
                label="Max Wind Speed"
                value={`${insights.maxWind.toFixed(0)} kts`}
                warn={insights.maxWind > 30}
              />
              <InsightCard
                icon={<Droplets className="w-4 h-4 text-teal-400" />}
                label="Max Current"
                value={`${insights.maxCurrent.toFixed(1)} kts`}
                warn={insights.maxCurrent > 2}
              />
              <InsightCard
                icon={<AlertTriangle className="w-4 h-4 text-yellow-400" />}
                label="Legs with >10% Speed Loss"
                value={`${insights.speedLossPct}% (${insights.speedLossLegs}/${r.legs.length})`}
                warn={parseInt(insights.speedLossPct) > 50}
              />
              <InsightCard
                icon={<Database className="w-4 h-4 text-purple-400" />}
                label="Data Sources"
                value={`F:${insights.forecastLegs} B:${insights.blendedLegs} C:${insights.climLegs}`}
              />
            </div>
          </div>
        )}

        {/* ── Monte Carlo ── */}
        {analysis.monteCarlo && (
          <div className="mb-8 max-w-md">
            <h2 className="text-sm font-semibold text-white mb-3">Monte Carlo Simulation</h2>
            <MonteCarloPanel result={analysis.monteCarlo} />
          </div>
        )}

        {/* ── SOG Profile & ETA Comparison ── */}
        {!DEMO_MODE && analysis.optimizations && Object.keys(analysis.optimizations).length > 0 && (
          <div className="mb-8">
            <ProfileCharts
              baseline={analysis.result}
              optimizations={analysis.optimizations}
              departureTime={analysis.result.departure_time}
            />
          </div>
        )}
      </main>
    </div>
  );
}

function LegRow({ leg, index }: { leg: LegResult; index: number }) {
  const speedLossHigh = leg.speed_loss_pct > 10;
  return (
    <tr className="border-t border-white/5 text-gray-300 hover:bg-white/[0.02]">
      <td className="px-3 py-2 text-gray-500">{index}</td>
      <td className="px-3 py-2 font-medium">{leg.to_wp?.name || `WP ${index}`}</td>
      <td className="px-3 py-2">{format(new Date(leg.arrival_time), 'MMM d HH:mm')}</td>
      <td className="px-3 py-2 text-right">{leg.distance_nm.toFixed(1)}</td>
      <td className={`px-3 py-2 text-right ${speedLossHigh ? 'text-yellow-400' : ''}`}>
        {leg.sog_kts.toFixed(1)}
      </td>
      <td className="px-3 py-2 text-right">
        {leg.wind_speed_kts.toFixed(0)} kts {windArrow(leg.wind_dir_deg)}
      </td>
      <td className="px-3 py-2 text-right">
        {leg.wave_height_m.toFixed(1)} m {leg.wave_dir_deg != null && windArrow(leg.wave_dir_deg)}
      </td>
      <td className="px-3 py-2 text-right">
        {leg.current_speed_ms != null
          ? <>{((leg.current_speed_ms) * 1.9438).toFixed(1)} kts {windArrow(leg.current_dir_deg ?? 0)}</>
          : <span className="text-gray-600">—</span>}
      </td>
      <td className="px-3 py-2 text-right">{leg.fuel_mt.toFixed(2)}</td>
      <td className="px-3 py-2 text-right">
        <SourceBadge source={leg.data_source} />
      </td>
    </tr>
  );
}

function SourceBadge({ source }: { source?: string }) {
  if (!source) return <span className="text-gray-600">—</span>;
  const colors: Record<string, string> = {
    forecast: 'text-green-400 bg-green-500/10',
    blended: 'text-yellow-400 bg-yellow-500/10',
    climatology: 'text-orange-400 bg-orange-500/10',
  };
  const cls = colors[source] || 'text-gray-400 bg-white/5';
  const label = source === 'forecast' ? 'F' : source === 'blended' ? 'B' : source === 'climatology' ? 'C' : source[0]?.toUpperCase();
  return (
    <span className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-medium ${cls}`}>
      {label}
    </span>
  );
}

function SummaryCard({ icon, label, value }: { icon: React.ReactNode; label: string; value: React.ReactNode }) {
  return (
    <div className="p-4 rounded-lg bg-white/5 border border-white/10">
      <div className="flex items-center gap-2 mb-2">
        {icon}
        <span className="text-xs text-gray-500">{label}</span>
      </div>
      <div className="text-sm font-medium text-white">{value}</div>
    </div>
  );
}

function InsightCard({ icon, label, value, warn }: { icon: React.ReactNode; label: string; value: string; warn?: boolean }) {
  return (
    <div className={`p-3 rounded-lg border ${warn ? 'bg-yellow-500/5 border-yellow-500/20' : 'bg-white/5 border-white/10'}`}>
      <div className="flex items-center gap-2 mb-1">
        {icon}
        <span className="text-[10px] text-gray-500">{label}</span>
      </div>
      <div className={`text-sm font-medium ${warn ? 'text-yellow-400' : 'text-white'}`}>{value}</div>
    </div>
  );
}

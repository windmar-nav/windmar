'use client';

import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine,
} from 'recharts';
import { EngineLogEntryResponse } from '@/lib/api';

const MARITIME_TOOLTIP = {
  backgroundColor: 'rgba(26, 41, 66, 0.9)',
  border: '1px solid rgba(255, 255, 255, 0.1)',
  borderRadius: '8px',
};

const AXIS_STYLE = { fontSize: '12px' };
const GRID_STROKE = 'rgba(255,255,255,0.1)';

// ─── Fuel Timeline ─────────────────────────────────────────────────────────

interface FuelTimelineChartProps {
  entries: EngineLogEntryResponse[];
}

export function FuelTimelineChart({ entries }: FuelTimelineChartProps) {
  const data = entries
    .filter(e => e.hfo_total_mt != null || e.mgo_total_mt != null)
    .map(e => ({
      time: new Date(e.timestamp).toLocaleDateString('en-GB', { day: '2-digit', month: 'short' }),
      ts: new Date(e.timestamp).getTime(),
      HFO: e.hfo_total_mt ?? 0,
      MGO: e.mgo_total_mt ?? 0,
    }))
    .sort((a, b) => a.ts - b.ts);

  if (data.length === 0) {
    return <div className="flex items-center justify-center h-full text-gray-500 text-sm">No fuel data</div>;
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} />
        <XAxis dataKey="time" stroke="#9ca3af" style={AXIS_STYLE} />
        <YAxis
          stroke="#9ca3af"
          style={AXIS_STYLE}
          label={{ value: 'MT', angle: -90, position: 'insideLeft', style: { fill: '#9ca3af' } }}
        />
        <Tooltip contentStyle={MARITIME_TOOLTIP} labelStyle={{ color: '#fff' }} />
        <Legend wrapperStyle={{ paddingTop: '10px' }} iconType="circle" />
        <Line type="monotone" dataKey="HFO" stroke="#0073e6" strokeWidth={2} dot={{ r: 2 }} />
        <Line type="monotone" dataKey="MGO" stroke="#00b4d8" strokeWidth={2} dot={{ r: 2 }} />
      </LineChart>
    </ResponsiveContainer>
  );
}

// ─── RPM Distribution ──────────────────────────────────────────────────────

interface RpmDistributionChartProps {
  entries: EngineLogEntryResponse[];
}

export function RpmDistributionChart({ entries }: RpmDistributionChartProps) {
  const noonRpms = entries
    .filter(e => e.event === 'NOON' && e.rpm != null && e.rpm > 0)
    .map(e => e.rpm!);

  if (noonRpms.length === 0) {
    return <div className="flex items-center justify-center h-full text-gray-500 text-sm">No NOON RPM data</div>;
  }

  const minRpm = Math.floor(Math.min(...noonRpms) / 10) * 10;
  const maxRpm = Math.ceil(Math.max(...noonRpms) / 10) * 10;

  const buckets: Record<string, number> = {};
  for (let r = minRpm; r < maxRpm; r += 10) {
    buckets[`${r}-${r + 10}`] = 0;
  }
  for (const rpm of noonRpms) {
    const bucket = Math.floor(rpm / 10) * 10;
    const key = `${bucket}-${bucket + 10}`;
    if (key in buckets) buckets[key]++;
  }

  const data = Object.entries(buckets).map(([range, count]) => ({ range, count }));

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} />
        <XAxis dataKey="range" stroke="#9ca3af" style={AXIS_STYLE} />
        <YAxis
          stroke="#9ca3af"
          style={AXIS_STYLE}
          label={{ value: 'Count', angle: -90, position: 'insideLeft', style: { fill: '#9ca3af' } }}
        />
        <Tooltip contentStyle={MARITIME_TOOLTIP} labelStyle={{ color: '#fff' }} />
        <Bar dataKey="count" name="Entries" fill="#008ba2" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

// ─── Speed Timeline ────────────────────────────────────────────────────────

interface SpeedTimelineChartProps {
  entries: EngineLogEntryResponse[];
  serviceSpeed?: number;
}

export function SpeedTimelineChart({ entries, serviceSpeed = 13 }: SpeedTimelineChartProps) {
  const data = entries
    .filter(e => e.speed_stw != null && e.speed_stw > 0)
    .map(e => ({
      time: new Date(e.timestamp).toLocaleDateString('en-GB', { day: '2-digit', month: 'short' }),
      ts: new Date(e.timestamp).getTime(),
      speed: e.speed_stw,
    }))
    .sort((a, b) => a.ts - b.ts);

  if (data.length === 0) {
    return <div className="flex items-center justify-center h-full text-gray-500 text-sm">No speed data</div>;
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} />
        <XAxis dataKey="time" stroke="#9ca3af" style={AXIS_STYLE} />
        <YAxis
          stroke="#9ca3af"
          style={AXIS_STYLE}
          label={{ value: 'kts', angle: -90, position: 'insideLeft', style: { fill: '#9ca3af' } }}
        />
        <Tooltip contentStyle={MARITIME_TOOLTIP} labelStyle={{ color: '#fff' }} />
        <Legend wrapperStyle={{ paddingTop: '10px' }} iconType="circle" />
        <ReferenceLine y={serviceSpeed} stroke="#f59e0b" strokeDasharray="6 4" label={{ value: `Service ${serviceSpeed} kts`, fill: '#f59e0b', fontSize: 11, position: 'right' }} />
        <Line type="monotone" dataKey="speed" name="Speed STW" stroke="#22c55e" strokeWidth={2} dot={{ r: 2 }} />
      </LineChart>
    </ResponsiveContainer>
  );
}

// ─── Event Breakdown Pie ───────────────────────────────────────────────────

const EVENT_COLORS: Record<string, string> = {
  NOON: '#3b82f6',
  SOSP: '#f59e0b',
  EOSP: '#f97316',
  ALL_FAST: '#22c55e',
  DRIFTING: '#a855f7',
  ANCHORED: '#06b6d4',
  BUNKERING: '#ec4899',
  UNKNOWN: '#6b7280',
};

interface EventBreakdownChartProps {
  eventsBreakdown: Record<string, number>;
}

export function EventBreakdownChart({ eventsBreakdown }: EventBreakdownChartProps) {
  const data = Object.entries(eventsBreakdown).map(([name, value]) => ({ name, value }));

  if (data.length === 0) {
    return <div className="flex items-center justify-center h-full text-gray-500 text-sm">No event data</div>;
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          innerRadius={50}
          outerRadius={80}
          paddingAngle={3}
          dataKey="value"
          nameKey="name"
          label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
          labelLine={{ stroke: '#9ca3af' }}
        >
          {data.map((entry) => (
            <Cell key={entry.name} fill={EVENT_COLORS[entry.name] || '#6b7280'} />
          ))}
        </Pie>
        <Tooltip contentStyle={MARITIME_TOOLTIP} labelStyle={{ color: '#fff' }} />
        <Legend wrapperStyle={{ paddingTop: '10px' }} iconType="circle" />
      </PieChart>
    </ResponsiveContainer>
  );
}

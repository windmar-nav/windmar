'use client';

import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine,
  ComposedChart, Scatter, ScatterChart,
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

  // Clamp outliers at p95 to avoid spikes compressing the chart
  const allVals = data.flatMap(d => [d.HFO, d.MGO]).filter(v => v > 0).sort((a, b) => a - b);
  const p95 = allVals.length > 0 ? allVals[Math.floor(allVals.length * 0.95)] : 10;
  const cap = p95 * 1.5;
  const clamped = data.map(d => ({
    ...d,
    HFO: Math.min(d.HFO, cap),
    MGO: Math.min(d.MGO, cap),
  }));
  const yMax = Math.ceil(p95 * 1.2);

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={clamped}>
        <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} />
        <XAxis dataKey="time" stroke="#9ca3af" style={AXIS_STYLE} />
        <YAxis
          stroke="#9ca3af"
          style={AXIS_STYLE}
          domain={[0, yMax]}
          allowDecimals={false}
          tickFormatter={(v: number) => Math.round(v).toString()}
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
      <LineChart data={data} margin={{ top: 5, right: 20, bottom: 25, left: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} />
        <XAxis dataKey="time" stroke="#9ca3af" style={AXIS_STYLE} />
        <YAxis
          stroke="#9ca3af"
          style={AXIS_STYLE}
          label={{ value: 'kts', angle: -90, position: 'insideLeft', style: { fill: '#9ca3af' } }}
        />
        <Tooltip contentStyle={MARITIME_TOOLTIP} labelStyle={{ color: '#fff' }} />
        <Legend wrapperStyle={{ paddingTop: '4px' }} iconType="circle" />
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
  const raw = Object.entries(eventsBreakdown).map(([name, value]) => ({ name, value }));

  if (raw.length === 0) {
    return <div className="flex items-center justify-center h-full text-gray-500 text-sm">No event data</div>;
  }

  // Group events below 3% into "Other"
  const total = raw.reduce((s, e) => s + e.value, 0);
  const threshold = total * 0.03;
  let otherSum = 0;
  const data = raw.filter(e => {
    if (e.value >= threshold) return true;
    otherSum += e.value;
    return false;
  });
  if (otherSum > 0) data.push({ name: 'Other', value: otherSum });

  // Sort descending so largest slices are first
  data.sort((a, b) => b.value - a.value);

  return (
    <div className="w-full h-full flex flex-col">
      <div className="flex-1 min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={100}
              paddingAngle={2}
              dataKey="value"
              nameKey="name"
            >
              {data.map((entry) => (
                <Cell key={entry.name} fill={EVENT_COLORS[entry.name] || '#6b7280'} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={MARITIME_TOOLTIP}
              labelStyle={{ color: '#fff' }}
              formatter={(value: number, name: string) => [`${value} (${((value / total) * 100).toFixed(0)}%)`, name]}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>
      {/* Compact inline legend */}
      <div className="flex flex-wrap gap-x-3 gap-y-1 justify-center pt-1 pb-2">
        {data.map((entry) => (
          <div key={entry.name} className="flex items-center gap-1">
            <div className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: EVENT_COLORS[entry.name] || '#6b7280' }} />
            <span className="text-xs text-gray-400">{entry.name}</span>
            <span className="text-xs text-gray-500">{entry.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Performance Charts ───────────────────────────────────────────────────

/** Safely extract numeric value from extended_data. */
function ext(entry: EngineLogEntryResponse, field: string): number | null {
  if (!entry.extended_data) return null;
  const v = entry.extended_data[field];
  if (v == null) return null;
  const n = typeof v === 'number' ? v : parseFloat(String(v));
  return Number.isFinite(n) ? n : null;
}

function fmtDate(ts: string): string {
  return new Date(ts).toLocaleDateString('en-GB', { day: '2-digit', month: 'short' });
}

// ─── Chart 1: Power-Speed Curve ───────────────────────────────────────────

interface PowerSpeedChartProps {
  entries: EngineLogEntryResponse[];
  mcrPower?: number;
  serviceSpeed?: number;
}

export function PowerSpeedChart({ entries, mcrPower = 6600, serviceSpeed = 13 }: PowerSpeedChartProps) {
  const data = entries
    .filter(e => e.event === 'NOON' && e.me_power_kw != null && e.me_power_kw > 0 && e.speed_stw != null && e.speed_stw > 0)
    .map(e => ({
      power: e.me_power_kw!,
      speed: e.speed_stw!,
      date: fmtDate(e.timestamp),
      load: e.me_load_pct,
      slip: e.slip_pct,
    }));

  if (data.length < 3) {
    return <div className="flex items-center justify-center h-full text-gray-500 text-sm">Insufficient power/speed data</div>;
  }

  // Theoretical cubic curve: speed = serviceSpeed * (power / mcrPower)^(1/3)
  const maxPow = Math.max(...data.map(d => d.power));
  const refCurve = Array.from({ length: 30 }, (_, i) => {
    const p = (i + 1) * (maxPow * 1.1) / 30;
    return { power: Math.round(p), theoreticalSpeed: serviceSpeed * Math.pow(p / mcrPower, 1 / 3) };
  });

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={refCurve} margin={{ top: 10, right: 20, bottom: 5, left: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} />
        <XAxis
          dataKey="power"
          type="number"
          stroke="#9ca3af"
          style={AXIS_STYLE}
          label={{ value: 'ME Power (kW)', position: 'insideBottom', offset: -2, style: { fill: '#9ca3af', fontSize: 11 } }}
        />
        <YAxis
          stroke="#9ca3af"
          style={AXIS_STYLE}
          label={{ value: 'Speed (kts)', angle: -90, position: 'insideLeft', style: { fill: '#9ca3af' } }}
        />
        <Tooltip
          contentStyle={MARITIME_TOOLTIP}
          labelStyle={{ color: '#fff' }}
          formatter={(value: number, name: string) => [
            typeof value === 'number' ? value.toFixed(1) : value,
            name === 'theoreticalSpeed' ? 'Theoretical' : name,
          ]}
        />
        <Line
          dataKey="theoreticalSpeed"
          name="Theoretical"
          stroke="#f59e0b"
          strokeDasharray="6 4"
          dot={false}
          strokeWidth={2}
        />
        <Scatter
          data={data}
          dataKey="speed"
          name="Actual"
          fill="#0073e6"
          fillOpacity={0.7}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

// ─── Chart 2: SFOC vs Engine Load ─────────────────────────────────────────

interface SfocLoadChartProps {
  entries: EngineLogEntryResponse[];
  mcrSfoc?: number;
}

export function SfocLoadChart({ entries, mcrSfoc = 171 }: SfocLoadChartProps) {
  const data = entries
    .filter(e =>
      e.event === 'NOON' &&
      e.hfo_me_mt != null && e.hfo_me_mt > 0 &&
      e.me_power_kw != null && e.me_power_kw > 0 &&
      e.lapse_hours != null && e.lapse_hours > 0
    )
    .map(e => {
      const sfoc = (e.hfo_me_mt! * 1e6) / (e.me_power_kw! * e.lapse_hours!);
      return {
        load: e.me_load_pct ?? 0,
        sfoc: Number.isFinite(sfoc) ? Math.round(sfoc * 10) / 10 : null,
        date: fmtDate(e.timestamp),
        hfoMe: e.hfo_me_mt,
      };
    })
    .filter(d => d.sfoc != null && d.sfoc > 50 && d.sfoc < 500);

  if (data.length < 3) {
    return <div className="flex items-center justify-center h-full text-gray-500 text-sm">Insufficient SFOC data</div>;
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ScatterChart margin={{ top: 10, right: 20, bottom: 5, left: 15 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} />
        <XAxis
          dataKey="load"
          type="number"
          stroke="#9ca3af"
          style={AXIS_STYLE}
          name="Load"
          unit="%"
          label={{ value: 'ME Load (%)', position: 'insideBottom', offset: -2, style: { fill: '#9ca3af', fontSize: 11 } }}
        />
        <YAxis
          dataKey="sfoc"
          type="number"
          stroke="#9ca3af"
          style={AXIS_STYLE}
          name="SFOC"
          tickFormatter={(v: number) => Math.round(v).toString()}
          label={{ value: 'SFOC (g/kWh)', angle: -90, position: 'insideLeft', style: { fill: '#9ca3af' } }}
        />
        <Tooltip
          contentStyle={MARITIME_TOOLTIP}
          labelStyle={{ color: '#fff' }}
          formatter={(value: number, name: string) => [
            typeof value === 'number' ? value.toFixed(1) : value,
            name,
          ]}
        />
        <ReferenceLine
          y={mcrSfoc}
          stroke="#f59e0b"
          strokeDasharray="6 4"
          label={{ value: `MCR ${mcrSfoc}`, fill: '#f59e0b', fontSize: 11, position: 'right' }}
        />
        <Scatter name="SFOC" data={data} fill="#008ba2" fillOpacity={0.7} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}

// ─── Chart 3: Fuel Breakdown ──────────────────────────────────────────────

interface FuelBreakdownChartProps {
  entries: EngineLogEntryResponse[];
}

export function FuelBreakdownChart({ entries }: FuelBreakdownChartProps) {
  const data = entries
    .filter(e => e.event === 'NOON' && (e.hfo_me_mt != null || e.hfo_ae_mt != null || e.hfo_boiler_mt != null || e.mgo_total_mt != null))
    .map(e => ({
      date: fmtDate(e.timestamp),
      ts: new Date(e.timestamp).getTime(),
      ME: e.hfo_me_mt ?? 0,
      AE: e.hfo_ae_mt ?? 0,
      Boiler: e.hfo_boiler_mt ?? 0,
      MGO: e.mgo_total_mt ?? 0,
    }))
    .sort((a, b) => a.ts - b.ts);

  if (data.length < 3) {
    return <div className="flex items-center justify-center h-full text-gray-500 text-sm">Insufficient fuel breakdown data</div>;
  }

  // Clamp outliers at p95
  const allVals = data.flatMap(d => [d.ME, d.AE, d.Boiler, d.MGO]).filter(v => v > 0).sort((a, b) => a - b);
  const p95 = allVals.length > 0 ? allVals[Math.floor(allVals.length * 0.95)] : 10;
  const clamp = (v: number) => Math.min(v, p95 * 1.5);
  const clamped = data.map(d => ({
    ...d,
    ME: clamp(d.ME),
    AE: clamp(d.AE),
    Boiler: clamp(d.Boiler),
    MGO: clamp(d.MGO),
  }));

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={clamped} margin={{ top: 10, right: 20, bottom: 5, left: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} />
        <XAxis dataKey="date" stroke="#9ca3af" style={AXIS_STYLE} />
        <YAxis
          stroke="#9ca3af"
          style={AXIS_STYLE}
          label={{ value: 'MT', angle: -90, position: 'insideLeft', style: { fill: '#9ca3af' } }}
        />
        <Tooltip contentStyle={MARITIME_TOOLTIP} labelStyle={{ color: '#fff' }} />
        <Legend wrapperStyle={{ paddingTop: '10px' }} iconType="circle" />
        <Bar dataKey="ME" stackId="fuel" fill="#0073e6" radius={[0, 0, 0, 0]} />
        <Bar dataKey="AE" stackId="fuel" fill="#008ba2" />
        <Bar dataKey="Boiler" stackId="fuel" fill="#f59e0b" />
        <Bar dataKey="MGO" stackId="fuel" fill="#22c55e" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

// ─── Chart 4: Propeller Slip & Speed Timeline ─────────────────────────────

interface SlipSpeedChartProps {
  entries: EngineLogEntryResponse[];
}

export function SlipSpeedChart({ entries }: SlipSpeedChartProps) {
  const data = entries
    .filter(e => e.event === 'NOON' && e.slip_pct != null && e.speed_stw != null && e.speed_stw > 0)
    .map(e => ({
      date: fmtDate(e.timestamp),
      ts: new Date(e.timestamp).getTime(),
      speed: e.speed_stw!,
      slip: e.slip_pct!,
    }))
    .sort((a, b) => a.ts - b.ts);

  if (data.length < 3) {
    return <div className="flex items-center justify-center h-full text-gray-500 text-sm">Insufficient slip/speed data</div>;
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={data} margin={{ top: 10, right: 20, bottom: 5, left: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} />
        <XAxis dataKey="date" stroke="#9ca3af" style={AXIS_STYLE} />
        <YAxis
          yAxisId="left"
          stroke="#22c55e"
          style={AXIS_STYLE}
          label={{ value: 'Speed (kts)', angle: -90, position: 'insideLeft', style: { fill: '#22c55e' } }}
        />
        <YAxis
          yAxisId="right"
          orientation="right"
          stroke="#ef4444"
          style={AXIS_STYLE}
          label={{ value: 'Slip (%)', angle: 90, position: 'insideRight', style: { fill: '#ef4444' } }}
        />
        <Tooltip contentStyle={MARITIME_TOOLTIP} labelStyle={{ color: '#fff' }} />
        <Legend wrapperStyle={{ paddingTop: '10px' }} iconType="circle" />
        <ReferenceLine yAxisId="right" y={2} stroke="#f59e0b" strokeDasharray="4 4" />
        <ReferenceLine yAxisId="right" y={6} stroke="#f59e0b" strokeDasharray="4 4" />
        <Line yAxisId="left" type="monotone" dataKey="speed" name="Speed STW" stroke="#22c55e" strokeWidth={2} dot={{ r: 2 }} />
        <Line yAxisId="right" type="monotone" dataKey="slip" name="Slip %" stroke="#ef4444" strokeWidth={2} dot={{ r: 2 }} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

// ─── Chart 5: Turbocharger Health ─────────────────────────────────────────

interface TurbochargerChartProps {
  entries: EngineLogEntryResponse[];
}

export function TurbochargerChart({ entries }: TurbochargerChartProps) {
  const data = entries
    .filter(e => e.event === 'NOON' && e.tc_rpm != null && e.tc_rpm > 0)
    .map(e => ({
      date: fmtDate(e.timestamp),
      ts: new Date(e.timestamp).getTime(),
      tcRpm: e.tc_rpm!,
      scavAir: e.scav_air_press_bar,
    }))
    .sort((a, b) => a.ts - b.ts);

  if (data.length < 3) {
    return <div className="flex items-center justify-center h-full text-gray-500 text-sm">Insufficient turbocharger data</div>;
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={data} margin={{ top: 10, right: 20, bottom: 5, left: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} />
        <XAxis dataKey="date" stroke="#9ca3af" style={AXIS_STYLE} />
        <YAxis
          yAxisId="left"
          stroke="#0073e6"
          style={AXIS_STYLE}
          label={{ value: 'TC RPM', angle: -90, position: 'insideLeft', style: { fill: '#0073e6' } }}
        />
        <YAxis
          yAxisId="right"
          orientation="right"
          stroke="#008ba2"
          style={AXIS_STYLE}
          label={{ value: 'Scav Air (bar)', angle: 90, position: 'insideRight', style: { fill: '#008ba2' } }}
        />
        <Tooltip contentStyle={MARITIME_TOOLTIP} labelStyle={{ color: '#fff' }} />
        <Legend wrapperStyle={{ paddingTop: '10px' }} iconType="circle" />
        <Line yAxisId="left" type="monotone" dataKey="tcRpm" name="TC RPM" stroke="#0073e6" strokeWidth={2} dot={{ r: 2 }} />
        <Line yAxisId="right" type="monotone" dataKey="scavAir" name="Scav Air" stroke="#008ba2" strokeWidth={2} dot={{ r: 2 }} connectNulls />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

// ─── Chart 6: Thermal Profile ─────────────────────────────────────────────

interface ThermalProfileChartProps {
  entries: EngineLogEntryResponse[];
}

export function ThermalProfileChart({ entries }: ThermalProfileChartProps) {
  const data = entries
    .filter(e => {
      if (e.event !== 'NOON') return false;
      const hasAny = ext(e, 'me_tc_in_c') != null || ext(e, 'me_tc_out_c') != null ||
                     ext(e, 'air_cooler_in_c') != null || ext(e, 'air_cooler_out_c') != null;
      return hasAny;
    })
    .map(e => ({
      date: fmtDate(e.timestamp),
      ts: new Date(e.timestamp).getTime(),
      exhaustIn: ext(e, 'me_tc_in_c'),
      exhaustOut: ext(e, 'me_tc_out_c'),
      chargeAirIn: ext(e, 'air_cooler_in_c'),
      chargeAirOut: ext(e, 'air_cooler_out_c'),
    }))
    .sort((a, b) => a.ts - b.ts);

  if (data.length < 3) {
    return <div className="flex items-center justify-center h-full text-gray-500 text-sm">Insufficient thermal data</div>;
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: 10, right: 20, bottom: 5, left: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} />
        <XAxis dataKey="date" stroke="#9ca3af" style={AXIS_STYLE} />
        <YAxis
          stroke="#9ca3af"
          style={AXIS_STYLE}
          label={{ value: 'Temp (\u00B0C)', angle: -90, position: 'insideLeft', style: { fill: '#9ca3af' } }}
        />
        <Tooltip contentStyle={MARITIME_TOOLTIP} labelStyle={{ color: '#fff' }} />
        <Legend wrapperStyle={{ paddingTop: '10px' }} iconType="circle" />
        <Line type="monotone" dataKey="exhaustIn" name="Exhaust In" stroke="#ef4444" strokeWidth={2} dot={{ r: 2 }} connectNulls />
        <Line type="monotone" dataKey="exhaustOut" name="Exhaust Out" stroke="#f97316" strokeWidth={2} dot={{ r: 2 }} connectNulls />
        <Line type="monotone" dataKey="chargeAirIn" name="Charge Air In" stroke="#3b82f6" strokeWidth={2} dot={{ r: 2 }} connectNulls />
        <Line type="monotone" dataKey="chargeAirOut" name="Charge Air Out" stroke="#22c55e" strokeWidth={2} dot={{ r: 2 }} connectNulls />
      </LineChart>
    </ResponsiveContainer>
  );
}

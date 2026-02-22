'use client';

import { useMemo } from 'react';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceDot,
} from 'recharts';
import type { ParetoSolution } from '@/lib/api';

interface ParetoChartProps {
  solutions: ParetoSolution[];
}

const DARK_TOOLTIP_STYLE: React.CSSProperties = {
  backgroundColor: '#1e293b',
  border: '1px solid rgba(255,255,255,0.1)',
  borderRadius: '6px',
  fontSize: '11px',
  color: '#e2e8f0',
};

function CustomTooltip({ active, payload }: any) {
  if (!active || !payload?.[0]) return null;
  const d = payload[0].payload as ParetoSolution;
  return (
    <div style={DARK_TOOLTIP_STYLE} className="px-3 py-2 space-y-1">
      <div className="text-[10px] text-gray-400">
        {d.is_selected ? 'Selected solution' : `\u03BB = ${d.lambda_value.toFixed(2)}`}
      </div>
      <div className="text-xs text-white">Fuel: {d.fuel_mt.toFixed(1)} MT</div>
      <div className="text-xs text-white">Time: {d.time_hours.toFixed(1)} h</div>
      <div className="text-xs text-gray-400">Dist: {d.distance_nm.toFixed(0)} nm</div>
    </div>
  );
}

export default function ParetoChart({ solutions }: ParetoChartProps) {
  const selected = useMemo(
    () => solutions.find(s => s.is_selected),
    [solutions],
  );

  const fuelDomain = useMemo<[number, number]>(() => {
    const fuels = solutions.map(s => s.fuel_mt);
    const min = Math.min(...fuels);
    const max = Math.max(...fuels);
    const pad = (max - min) * 0.1 || 1;
    return [Math.max(0, min - pad), max + pad];
  }, [solutions]);

  const timeDomain = useMemo<[number, number]>(() => {
    const times = solutions.map(s => s.time_hours);
    const min = Math.min(...times);
    const max = Math.max(...times);
    const pad = (max - min) * 0.1 || 1;
    return [Math.max(0, min - pad), max + pad];
  }, [solutions]);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-medium text-gray-300">Pareto Front</h3>
        <span className="text-[10px] text-gray-500">{solutions.length} solutions</span>
      </div>
      <div className="h-52 bg-white/[0.03] rounded-lg border border-white/10 p-2">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 10, right: 15, bottom: 25, left: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
            <XAxis
              dataKey="time_hours"
              type="number"
              domain={timeDomain}
              name="Time"
              tick={{ fontSize: 9, fill: '#94a3b8' }}
              tickFormatter={(v: number) => v.toFixed(1)}
              label={{ value: 'Time (h)', position: 'insideBottom', offset: -15, style: { fontSize: 10, fill: '#94a3b8' } }}
              stroke="rgba(255,255,255,0.1)"
            />
            <YAxis
              dataKey="fuel_mt"
              type="number"
              domain={fuelDomain}
              name="Fuel"
              tick={{ fontSize: 9, fill: '#94a3b8' }}
              tickFormatter={(v: number) => v.toFixed(1)}
              label={{ value: 'Fuel (MT)', angle: -90, position: 'insideLeft', offset: 10, style: { fontSize: 10, fill: '#94a3b8' } }}
              stroke="rgba(255,255,255,0.1)"
            />
            <Tooltip content={<CustomTooltip />} />
            <Scatter
              data={solutions}
              fill="#38bdf8"
              stroke="#0ea5e9"
              strokeWidth={1}
              r={4}
              shape="circle"
            />
            {selected && (
              <ReferenceDot
                x={selected.time_hours}
                y={selected.fuel_mt}
                r={7}
                fill="#f59e0b"
                stroke="#fbbf24"
                strokeWidth={2}
              />
            )}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      {/* Legend */}
      <div className="flex items-center gap-4 text-[10px] text-gray-500">
        <div className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-sky-400" />
          Non-dominated
        </div>
        {selected && (
          <div className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-full bg-amber-400" />
            Selected ({selected.fuel_mt.toFixed(1)} MT, {selected.time_hours.toFixed(1)} h)
          </div>
        )}
      </div>
    </div>
  );
}

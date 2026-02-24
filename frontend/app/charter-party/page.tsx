'use client';

import { useState, useEffect } from 'react';
import Header from '@/components/Header';
import Card from '@/components/Card';
import {
  Scale,
  Sun,
  CloudRain,
  CheckCircle,
  XCircle,
  Anchor,
  Ship,
  Clock,
  Gauge,
  Fuel,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import {
  apiClient,
  BeaufortEntry,
  GoodWeatherResponse,
  WarrantyVerificationResponse,
  OffHireResponse,
  LegWeatherInput,
} from '@/lib/api';

type TabType = 'weather' | 'warranty' | 'offhire';

export default function CharterPartyPage() {
  const [activeTab, setActiveTab] = useState<TabType>('weather');
  const [beaufortScale, setBeaufortScale] = useState<BeaufortEntry[]>([]);
  const [loading, setLoading] = useState(true);

  // Good Weather state
  const [weatherResult, setWeatherResult] = useState<GoodWeatherResponse | null>(null);
  const [weatherLegs, setWeatherLegs] = useState<LegWeatherInput[]>([
    { wind_speed_kts: 8, wave_height_m: 0.5, time_hours: 12 },
    { wind_speed_kts: 15, wave_height_m: 1.2, time_hours: 8 },
    { wind_speed_kts: 25, wave_height_m: 3.5, time_hours: 6 },
    { wind_speed_kts: 10, wave_height_m: 0.8, time_hours: 10 },
  ]);
  const [bfThreshold, setBfThreshold] = useState(4);
  const [waveThreshold, setWaveThreshold] = useState<string>('');
  const [currentThreshold, setCurrentThreshold] = useState<string>('');

  // Warranty state
  const [warrantyResult, setWarrantyResult] = useState<WarrantyVerificationResponse | null>(null);
  const [warrantyLegs, setWarrantyLegs] = useState<LegWeatherInput[]>([
    { wind_speed_kts: 8, wave_height_m: 0.5, time_hours: 12, distance_nm: 168, sog_kts: 14, fuel_mt: 2.5 },
    { wind_speed_kts: 12, wave_height_m: 1.0, time_hours: 12, distance_nm: 156, sog_kts: 13, fuel_mt: 2.8 },
    { wind_speed_kts: 35, wave_height_m: 4.5, time_hours: 8, distance_nm: 56, sog_kts: 7, fuel_mt: 3.5 },
  ]);
  const [warSpeed, setWarSpeed] = useState(14);
  const [warConsumption, setWarConsumption] = useState(5);
  const [warBfThreshold, setWarBfThreshold] = useState(4);
  const [speedTol, setSpeedTol] = useState(0);
  const [consTol, setConsTol] = useState(0);

  // Off-hire state
  const [offhireResult, setOffhireResult] = useState<OffHireResponse | null>(null);
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');
  const [rpmThreshold, setRpmThreshold] = useState(10);
  const [speedThresholdOH, setSpeedThresholdOH] = useState(1);
  const [gapHours, setGapHours] = useState(6);

  useEffect(() => {
    loadBeaufortScale();
  }, []);

  const loadBeaufortScale = async () => {
    try {
      const data = await apiClient.getBeaufortScale();
      setBeaufortScale(data.scale);
    } catch (error) {
      console.error('Failed to load Beaufort scale:', error);
    } finally {
      setLoading(false);
    }
  };

  const analyzeWeather = async () => {
    try {
      const request: Record<string, unknown> = {
        legs: weatherLegs,
        bf_threshold: bfThreshold,
      };
      if (waveThreshold) request.wave_threshold_m = Number(waveThreshold);
      if (currentThreshold) request.current_threshold_kts = Number(currentThreshold);
      const result = await apiClient.analyzeGoodWeather(request as Parameters<typeof apiClient.analyzeGoodWeather>[0]);
      setWeatherResult(result);
    } catch (error) {
      console.error('Good weather analysis failed:', error);
    }
  };

  const verifyWarranty = async () => {
    try {
      const result = await apiClient.verifyWarranty({
        legs: warrantyLegs,
        warranted_speed_kts: warSpeed,
        warranted_consumption_mt_day: warConsumption,
        bf_threshold: warBfThreshold,
        speed_tolerance_pct: speedTol,
        consumption_tolerance_pct: consTol,
      });
      setWarrantyResult(result);
    } catch (error) {
      console.error('Warranty verification failed:', error);
    }
  };

  const detectOffHire = async () => {
    try {
      const request: Record<string, unknown> = {
        rpm_threshold: rpmThreshold,
        speed_threshold: speedThresholdOH,
        gap_hours: gapHours,
      };
      if (dateFrom) request.date_from = dateFrom;
      if (dateTo) request.date_to = dateTo;
      const result = await apiClient.detectOffHire(request as Parameters<typeof apiClient.detectOffHire>[0]);
      setOffhireResult(result);
    } catch (error) {
      console.error('Off-hire detection failed:', error);
    }
  };

  const bfColor = (force: number) => {
    if (force <= 3) return 'text-green-400';
    if (force <= 5) return 'text-yellow-400';
    if (force <= 7) return 'text-orange-400';
    return 'text-red-400';
  };

  return (
    <div className="min-h-screen bg-gradient-maritime">
      <Header />

      <main className="container mx-auto px-6 pt-20 pb-12">
        {/* Hero */}
        <div className="mb-8">
          <div className="flex items-center space-x-3 mb-3">
            <Scale className="w-10 h-10 text-sky-400" />
            <h2 className="text-4xl font-bold text-white">Charter Party Tools</h2>
          </div>
          <p className="text-gray-300 text-lg">
            Good weather day counting, warranty verification, and off-hire detection for charter party analysis
          </p>
        </div>

        {/* Tabs */}
        <div className="flex flex-wrap gap-2 mb-6">
          <TabButton active={activeTab === 'weather'} onClick={() => setActiveTab('weather')} icon={<Sun className="w-4 h-4" />}>
            Good Weather Days
          </TabButton>
          <TabButton active={activeTab === 'warranty'} onClick={() => setActiveTab('warranty')} icon={<Gauge className="w-4 h-4" />}>
            Warranty Verification
          </TabButton>
          <TabButton active={activeTab === 'offhire'} onClick={() => setActiveTab('offhire')} icon={<Anchor className="w-4 h-4" />}>
            Off-Hire Detection
          </TabButton>
        </div>

        {loading ? (
          <div className="flex items-center justify-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-sky-400" />
          </div>
        ) : (
          <>
            {activeTab === 'weather' && (
              <WeatherTab
                legs={weatherLegs}
                setLegs={setWeatherLegs}
                bfThreshold={bfThreshold}
                setBfThreshold={setBfThreshold}
                waveThreshold={waveThreshold}
                setWaveThreshold={setWaveThreshold}
                currentThreshold={currentThreshold}
                setCurrentThreshold={setCurrentThreshold}
                result={weatherResult}
                onAnalyze={analyzeWeather}
                beaufortScale={beaufortScale}
                bfColor={bfColor}
              />
            )}
            {activeTab === 'warranty' && (
              <WarrantyTab
                legs={warrantyLegs}
                setLegs={setWarrantyLegs}
                warSpeed={warSpeed}
                setWarSpeed={setWarSpeed}
                warConsumption={warConsumption}
                setWarConsumption={setWarConsumption}
                bfThreshold={warBfThreshold}
                setBfThreshold={setWarBfThreshold}
                speedTol={speedTol}
                setSpeedTol={setSpeedTol}
                consTol={consTol}
                setConsTol={setConsTol}
                result={warrantyResult}
                onVerify={verifyWarranty}
                bfColor={bfColor}
              />
            )}
            {activeTab === 'offhire' && (
              <OffHireTab
                dateFrom={dateFrom}
                setDateFrom={setDateFrom}
                dateTo={dateTo}
                setDateTo={setDateTo}
                rpmThreshold={rpmThreshold}
                setRpmThreshold={setRpmThreshold}
                speedThreshold={speedThresholdOH}
                setSpeedThreshold={setSpeedThresholdOH}
                gapHours={gapHours}
                setGapHours={setGapHours}
                result={offhireResult}
                onDetect={detectOffHire}
              />
            )}
          </>
        )}

        {/* Beaufort Scale Reference */}
        {beaufortScale.length > 0 && (
          <div className="mt-12">
            <Card title="Beaufort Scale Reference" icon={<CloudRain className="w-5 h-5" />}>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-white/10">
                      <th className="text-left py-2 px-3 text-xs font-medium text-gray-400">Force</th>
                      <th className="text-left py-2 px-3 text-xs font-medium text-gray-400">Wind (kts)</th>
                      <th className="text-left py-2 px-3 text-xs font-medium text-gray-400">Wave (m)</th>
                      <th className="text-left py-2 px-3 text-xs font-medium text-gray-400">Description</th>
                    </tr>
                  </thead>
                  <tbody>
                    {beaufortScale.map((b) => (
                      <tr key={b.force} className="border-b border-white/5 hover:bg-white/5">
                        <td className={`py-2 px-3 font-bold ${bfColor(b.force)}`}>{b.force}</td>
                        <td className="py-2 px-3 text-sm text-white">{b.wind_min_kts} - {b.wind_max_kts === 999 ? '64+' : b.wind_max_kts}</td>
                        <td className="py-2 px-3 text-sm text-gray-300">{b.wave_height_m}</td>
                        <td className="py-2 px-3 text-sm text-gray-300">{b.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>
        )}
      </main>
    </div>
  );
}

// ============================================================================
// Shared Components
// ============================================================================

function TabButton({ active, onClick, icon, children }: {
  active: boolean; onClick: () => void; icon: React.ReactNode; children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
        active ? 'bg-sky-600 text-white' : 'bg-white/5 text-gray-300 hover:bg-white/10'
      }`}
    >
      {icon}
      <span className="font-medium">{children}</span>
    </button>
  );
}

function ComplianceBadge({ compliant, label }: { compliant: boolean; label: string }) {
  return (
    <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-medium ${
      compliant ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
    }`}>
      {compliant ? <CheckCircle className="w-3.5 h-3.5" /> : <XCircle className="w-3.5 h-3.5" />}
      {label}
    </span>
  );
}

// ============================================================================
// Good Weather Days Tab
// ============================================================================

function WeatherTab({
  legs, setLegs, bfThreshold, setBfThreshold,
  waveThreshold, setWaveThreshold, currentThreshold, setCurrentThreshold,
  result, onAnalyze, beaufortScale, bfColor,
}: {
  legs: LegWeatherInput[];
  setLegs: (l: LegWeatherInput[]) => void;
  bfThreshold: number;
  setBfThreshold: (n: number) => void;
  waveThreshold: string;
  setWaveThreshold: (s: string) => void;
  currentThreshold: string;
  setCurrentThreshold: (s: string) => void;
  result: GoodWeatherResponse | null;
  onAnalyze: () => void;
  beaufortScale: BeaufortEntry[];
  bfColor: (f: number) => string;
}) {
  const updateLeg = (idx: number, field: string, value: number) => {
    const updated = [...legs];
    updated[idx] = { ...updated[idx], [field]: value };
    setLegs(updated);
  };

  const addLeg = () => {
    setLegs([...legs, { wind_speed_kts: 10, wave_height_m: 0.5, time_hours: 6 }]);
  };

  const removeLeg = (idx: number) => {
    setLegs(legs.filter((_, i) => i !== idx));
  };

  const chartData = result?.legs.map((l) => ({
    name: `Leg ${l.leg_index + 1}`,
    hours: l.time_hours,
    good: l.is_good_weather,
    bf: l.bf_force,
  })) ?? [];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Leg Input */}
        <div className="lg:col-span-2">
          <Card title="Voyage Legs" icon={<Ship className="w-5 h-5" />}>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-white/10">
                    <th className="text-left py-2 px-2 text-xs text-gray-400">Leg</th>
                    <th className="text-left py-2 px-2 text-xs text-gray-400">Wind (kts)</th>
                    <th className="text-left py-2 px-2 text-xs text-gray-400">Wave (m)</th>
                    <th className="text-left py-2 px-2 text-xs text-gray-400">Hours</th>
                    <th className="py-2 px-2" />
                  </tr>
                </thead>
                <tbody>
                  {legs.map((leg, idx) => (
                    <tr key={idx} className="border-b border-white/5">
                      <td className="py-2 px-2 text-sm text-gray-300">{idx + 1}</td>
                      <td className="py-2 px-2">
                        <input type="number" value={leg.wind_speed_kts} onChange={(e) => updateLeg(idx, 'wind_speed_kts', Number(e.target.value))}
                          className="w-20 px-2 py-1 bg-maritime-dark border border-white/10 rounded text-white text-sm" />
                      </td>
                      <td className="py-2 px-2">
                        <input type="number" step="0.1" value={leg.wave_height_m ?? 0} onChange={(e) => updateLeg(idx, 'wave_height_m', Number(e.target.value))}
                          className="w-20 px-2 py-1 bg-maritime-dark border border-white/10 rounded text-white text-sm" />
                      </td>
                      <td className="py-2 px-2">
                        <input type="number" value={leg.time_hours} onChange={(e) => updateLeg(idx, 'time_hours', Number(e.target.value))}
                          className="w-20 px-2 py-1 bg-maritime-dark border border-white/10 rounded text-white text-sm" />
                      </td>
                      <td className="py-2 px-2">
                        <button onClick={() => removeLeg(idx)} className="text-red-400 hover:text-red-300 text-xs">Remove</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <button onClick={addLeg} className="mt-3 px-4 py-1.5 bg-white/5 hover:bg-white/10 text-gray-300 rounded-lg text-sm transition-colors">
              + Add Leg
            </button>
          </Card>
        </div>

        {/* Thresholds */}
        <Card title="Thresholds" icon={<CloudRain className="w-5 h-5" />}>
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">Beaufort Threshold (0-12)</label>
              <input type="range" min="0" max="12" value={bfThreshold} onChange={(e) => setBfThreshold(Number(e.target.value))}
                className="w-full" />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>BF 0</span>
                <span className={`font-bold ${bfColor(bfThreshold)}`}>BF {bfThreshold}: {beaufortScale[bfThreshold]?.description ?? ''}</span>
                <span>BF 12</span>
              </div>
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">Wave Height Limit (m, optional)</label>
              <input type="number" step="0.5" value={waveThreshold} placeholder="e.g. 2.0"
                onChange={(e) => setWaveThreshold(e.target.value)}
                className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white" />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">Current Limit (kts, optional)</label>
              <input type="number" step="0.5" value={currentThreshold} placeholder="e.g. 3.0"
                onChange={(e) => setCurrentThreshold(e.target.value)}
                className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white" />
            </div>
            <button onClick={onAnalyze}
              className="w-full py-3 bg-sky-600 hover:bg-sky-700 text-white font-semibold rounded-lg transition-colors">
              Analyze Good Weather
            </button>
          </div>
        </Card>
      </div>

      {/* Results */}
      {result && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card>
              <div className="text-center">
                <p className="text-xs text-gray-400 mb-1">Total Days</p>
                <p className="text-2xl font-bold text-white">{result.total_days.toFixed(2)}</p>
              </div>
            </Card>
            <Card>
              <div className="text-center">
                <p className="text-xs text-gray-400 mb-1">Good Weather</p>
                <p className="text-2xl font-bold text-green-400">{result.good_weather_days.toFixed(2)}</p>
                <p className="text-xs text-gray-500">days</p>
              </div>
            </Card>
            <Card>
              <div className="text-center">
                <p className="text-xs text-gray-400 mb-1">Bad Weather</p>
                <p className="text-2xl font-bold text-red-400">{result.bad_weather_days.toFixed(2)}</p>
                <p className="text-xs text-gray-500">days</p>
              </div>
            </Card>
            <Card>
              <div className="text-center">
                <p className="text-xs text-gray-400 mb-1">Good Weather %</p>
                <p className="text-2xl font-bold text-sky-400">{result.good_weather_pct.toFixed(1)}%</p>
              </div>
            </Card>
          </div>

          {chartData.length > 0 && (
            <Card title="Per-Leg Weather Classification">
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis dataKey="name" stroke="#9ca3af" tick={{ fontSize: 12 }} />
                    <YAxis stroke="#9ca3af" tick={{ fontSize: 12 }} label={{ value: 'Hours', angle: -90, position: 'insideLeft', style: { fill: '#9ca3af', fontSize: 12 } }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                      labelStyle={{ color: '#fff' }}
                      formatter={(value: number, _: string, props: { payload?: { good: boolean; bf: number } }) => [
                        `${value.toFixed(1)}h (BF ${props.payload?.bf ?? 0})`,
                        props.payload?.good ? 'Good Weather' : 'Bad Weather',
                      ]}
                    />
                    <Bar dataKey="hours" radius={[4, 4, 0, 0]}>
                      {chartData.map((entry, idx) => (
                        <Cell key={idx} fill={entry.good ? 'rgba(34,197,94,0.7)' : 'rgba(239,68,68,0.7)'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>
          )}

          <Card title="Leg Details">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-white/10">
                    <th className="text-left py-2 px-3 text-xs text-gray-400">Leg</th>
                    <th className="text-left py-2 px-3 text-xs text-gray-400">Wind (kts)</th>
                    <th className="text-left py-2 px-3 text-xs text-gray-400">Wave (m)</th>
                    <th className="text-left py-2 px-3 text-xs text-gray-400">BF Force</th>
                    <th className="text-left py-2 px-3 text-xs text-gray-400">Hours</th>
                    <th className="text-left py-2 px-3 text-xs text-gray-400">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {result.legs.map((l) => (
                    <tr key={l.leg_index} className="border-b border-white/5 hover:bg-white/5">
                      <td className="py-2 px-3 text-white">{l.leg_index + 1}</td>
                      <td className="py-2 px-3 text-gray-300">{l.wind_speed_kts.toFixed(1)}</td>
                      <td className="py-2 px-3 text-gray-300">{l.wave_height_m.toFixed(1)}</td>
                      <td className={`py-2 px-3 font-bold ${bfColor(l.bf_force)}`}>{l.bf_force}</td>
                      <td className="py-2 px-3 text-gray-300">{l.time_hours.toFixed(1)}</td>
                      <td className="py-2 px-3">
                        {l.is_good_weather
                          ? <span className="text-green-400 text-sm">Good</span>
                          : <span className="text-red-400 text-sm">Bad</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}

// ============================================================================
// Warranty Verification Tab
// ============================================================================

function WarrantyTab({
  legs, setLegs, warSpeed, setWarSpeed, warConsumption, setWarConsumption,
  bfThreshold, setBfThreshold, speedTol, setSpeedTol, consTol, setConsTol,
  result, onVerify, bfColor,
}: {
  legs: LegWeatherInput[];
  setLegs: (l: LegWeatherInput[]) => void;
  warSpeed: number;
  setWarSpeed: (n: number) => void;
  warConsumption: number;
  setWarConsumption: (n: number) => void;
  bfThreshold: number;
  setBfThreshold: (n: number) => void;
  speedTol: number;
  setSpeedTol: (n: number) => void;
  consTol: number;
  setConsTol: (n: number) => void;
  result: WarrantyVerificationResponse | null;
  onVerify: () => void;
  bfColor: (f: number) => string;
}) {
  const updateLeg = (idx: number, field: string, value: number) => {
    const updated = [...legs];
    updated[idx] = { ...updated[idx], [field]: value };
    setLegs(updated);
  };

  const addLeg = () => {
    setLegs([...legs, { wind_speed_kts: 10, wave_height_m: 0.5, time_hours: 12, distance_nm: 140, sog_kts: 12, fuel_mt: 2.5 }]);
  };

  const removeLeg = (idx: number) => {
    setLegs(legs.filter((_, i) => i !== idx));
  };

  return (
    <div className="space-y-6">
      {/* Input */}
      <Card title="Warranty Parameters" icon={<Gauge className="w-5 h-5" />}>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-4">
          <div>
            <label className="block text-sm text-gray-400 mb-1">Warranted Speed (kts)</label>
            <input type="number" step="0.5" value={warSpeed} onChange={(e) => setWarSpeed(Number(e.target.value))}
              className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white" />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">Warranted Cons. (MT/day)</label>
            <input type="number" step="0.5" value={warConsumption} onChange={(e) => setWarConsumption(Number(e.target.value))}
              className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white" />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">BF Threshold</label>
            <select value={bfThreshold} onChange={(e) => setBfThreshold(Number(e.target.value))}
              className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white">
              {Array.from({ length: 13 }, (_, i) => i).map((bf) => (
                <option key={bf} value={bf}>BF {bf}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">Speed Tolerance (%)</label>
            <input type="number" step="1" min="0" max="20" value={speedTol} onChange={(e) => setSpeedTol(Number(e.target.value))}
              className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white" />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">Cons. Tolerance (%)</label>
            <input type="number" step="1" min="0" max="20" value={consTol} onChange={(e) => setConsTol(Number(e.target.value))}
              className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white" />
          </div>
          <div className="flex items-end">
            <button onClick={onVerify}
              className="w-full py-2 bg-sky-600 hover:bg-sky-700 text-white font-semibold rounded-lg transition-colors">
              Verify
            </button>
          </div>
        </div>
      </Card>

      {/* Leg Table */}
      <Card title="Voyage Legs" icon={<Ship className="w-5 h-5" />}>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-white/10">
                <th className="text-left py-2 px-2 text-xs text-gray-400">Leg</th>
                <th className="text-left py-2 px-2 text-xs text-gray-400">Wind (kts)</th>
                <th className="text-left py-2 px-2 text-xs text-gray-400">Wave (m)</th>
                <th className="text-left py-2 px-2 text-xs text-gray-400">Hours</th>
                <th className="text-left py-2 px-2 text-xs text-gray-400">Dist (nm)</th>
                <th className="text-left py-2 px-2 text-xs text-gray-400">SOG (kts)</th>
                <th className="text-left py-2 px-2 text-xs text-gray-400">Fuel (MT)</th>
                <th className="py-2 px-2" />
              </tr>
            </thead>
            <tbody>
              {legs.map((leg, idx) => (
                <tr key={idx} className="border-b border-white/5">
                  <td className="py-2 px-2 text-sm text-gray-300">{idx + 1}</td>
                  {['wind_speed_kts', 'wave_height_m', 'time_hours', 'distance_nm', 'sog_kts', 'fuel_mt'].map((field) => (
                    <td key={field} className="py-2 px-2">
                      <input type="number" step={field === 'wave_height_m' ? '0.1' : '1'}
                        value={(leg as unknown as Record<string, number>)[field] ?? 0}
                        onChange={(e) => updateLeg(idx, field, Number(e.target.value))}
                        className="w-16 px-2 py-1 bg-maritime-dark border border-white/10 rounded text-white text-sm" />
                    </td>
                  ))}
                  <td className="py-2 px-2">
                    <button onClick={() => removeLeg(idx)} className="text-red-400 hover:text-red-300 text-xs">Remove</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <button onClick={addLeg} className="mt-3 px-4 py-1.5 bg-white/5 hover:bg-white/10 text-gray-300 rounded-lg text-sm transition-colors">
          + Add Leg
        </button>
      </Card>

      {/* Results */}
      {result && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card>
              <div className="text-center">
                <p className="text-xs text-gray-400 mb-1">Speed</p>
                <ComplianceBadge compliant={result.speed_compliant} label={result.speed_compliant ? 'Compliant' : 'Non-Compliant'} />
              </div>
            </Card>
            <Card>
              <div className="text-center">
                <p className="text-xs text-gray-400 mb-1">Achieved Speed</p>
                <p className="text-2xl font-bold text-white">{result.achieved_speed_kts.toFixed(2)}</p>
                <p className="text-xs text-gray-500">vs {result.warranted_speed_kts} kts warranted</p>
                <p className={`text-sm mt-1 ${result.speed_margin_kts >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {result.speed_margin_kts >= 0 ? '+' : ''}{result.speed_margin_kts.toFixed(2)} kts
                </p>
              </div>
            </Card>
            <Card>
              <div className="text-center">
                <p className="text-xs text-gray-400 mb-1">Consumption</p>
                <ComplianceBadge compliant={result.consumption_compliant} label={result.consumption_compliant ? 'Compliant' : 'Non-Compliant'} />
              </div>
            </Card>
            <Card>
              <div className="text-center">
                <p className="text-xs text-gray-400 mb-1">Achieved Consumption</p>
                <p className="text-2xl font-bold text-white">{result.achieved_consumption_mt_day.toFixed(2)}</p>
                <p className="text-xs text-gray-500">vs {result.warranted_consumption_mt_day} MT/day warranted</p>
                <p className={`text-sm mt-1 ${result.consumption_margin_mt >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {result.consumption_margin_mt >= 0 ? '+' : ''}{result.consumption_margin_mt.toFixed(2)} MT/day
                </p>
              </div>
            </Card>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <div className="text-center">
                <p className="text-xs text-gray-400 mb-1">Good Weather Hours</p>
                <p className="text-xl font-bold text-white">{result.good_weather_hours.toFixed(1)}</p>
                <p className="text-xs text-gray-500">of {result.total_hours.toFixed(1)} total</p>
              </div>
            </Card>
            <Card>
              <div className="text-center">
                <p className="text-xs text-gray-400 mb-1">Legs Assessed</p>
                <p className="text-xl font-bold text-white">{result.legs_assessed}</p>
              </div>
            </Card>
            <Card>
              <div className="text-center">
                <p className="text-xs text-gray-400 mb-1">Good Weather Legs</p>
                <p className="text-xl font-bold text-green-400">{result.legs_good_weather}</p>
              </div>
            </Card>
          </div>

          <Card title="Per-Leg Detail">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-white/10">
                    <th className="text-left py-2 px-3 text-xs text-gray-400">Leg</th>
                    <th className="text-left py-2 px-3 text-xs text-gray-400">SOG (kts)</th>
                    <th className="text-left py-2 px-3 text-xs text-gray-400">Fuel (MT)</th>
                    <th className="text-left py-2 px-3 text-xs text-gray-400">Hours</th>
                    <th className="text-left py-2 px-3 text-xs text-gray-400">Distance (nm)</th>
                    <th className="text-left py-2 px-3 text-xs text-gray-400">BF</th>
                    <th className="text-left py-2 px-3 text-xs text-gray-400">Weather</th>
                  </tr>
                </thead>
                <tbody>
                  {result.legs.map((l) => (
                    <tr key={l.leg_index} className="border-b border-white/5 hover:bg-white/5">
                      <td className="py-2 px-3 text-white">{l.leg_index + 1}</td>
                      <td className="py-2 px-3 text-gray-300">{l.sog_kts.toFixed(1)}</td>
                      <td className="py-2 px-3 text-gray-300">{l.fuel_mt.toFixed(2)}</td>
                      <td className="py-2 px-3 text-gray-300">{l.time_hours.toFixed(1)}</td>
                      <td className="py-2 px-3 text-gray-300">{l.distance_nm.toFixed(1)}</td>
                      <td className={`py-2 px-3 font-bold ${bfColor(l.bf_force)}`}>{l.bf_force}</td>
                      <td className="py-2 px-3">
                        {l.is_good_weather
                          ? <span className="text-green-400 text-sm">Good</span>
                          : <span className="text-red-400 text-sm">Bad</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}

// ============================================================================
// Off-Hire Detection Tab
// ============================================================================

function OffHireTab({
  dateFrom, setDateFrom, dateTo, setDateTo,
  rpmThreshold, setRpmThreshold, speedThreshold, setSpeedThreshold,
  gapHours, setGapHours, result, onDetect,
}: {
  dateFrom: string;
  setDateFrom: (s: string) => void;
  dateTo: string;
  setDateTo: (s: string) => void;
  rpmThreshold: number;
  setRpmThreshold: (n: number) => void;
  speedThreshold: number;
  setSpeedThreshold: (n: number) => void;
  gapHours: number;
  setGapHours: (n: number) => void;
  result: OffHireResponse | null;
  onDetect: () => void;
}) {
  const chartData = result?.events.map((ev, idx) => ({
    name: `Event ${idx + 1}`,
    hours: ev.duration_hours,
    reason: ev.reason,
  })) ?? [];

  return (
    <div className="space-y-6">
      <Card title="Off-Hire Analysis Parameters" icon={<Anchor className="w-5 h-5" />}>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          <div>
            <label className="block text-sm text-gray-400 mb-1">Date From</label>
            <input type="datetime-local" value={dateFrom} onChange={(e) => setDateFrom(e.target.value)}
              className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white text-sm" />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">Date To</label>
            <input type="datetime-local" value={dateTo} onChange={(e) => setDateTo(e.target.value)}
              className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white text-sm" />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">RPM Threshold</label>
            <input type="number" value={rpmThreshold} onChange={(e) => setRpmThreshold(Number(e.target.value))}
              className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white" />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">Speed Threshold (kts)</label>
            <input type="number" step="0.5" value={speedThreshold} onChange={(e) => setSpeedThreshold(Number(e.target.value))}
              className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white" />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">Gap Hours</label>
            <input type="number" value={gapHours} onChange={(e) => setGapHours(Number(e.target.value))}
              className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white" />
          </div>
          <div className="flex items-end">
            <button onClick={onDetect}
              className="w-full py-2 bg-sky-600 hover:bg-sky-700 text-white font-semibold rounded-lg transition-colors">
              Detect Off-Hire
            </button>
          </div>
        </div>
        <p className="text-xs text-gray-500 mt-3">
          Requires engine log data. Upload engine log first via the Engine Log page.
        </p>
      </Card>

      {result && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card>
              <div className="text-center">
                <p className="text-xs text-gray-400 mb-1">Total Hours</p>
                <p className="text-2xl font-bold text-white">{result.total_hours.toFixed(1)}</p>
              </div>
            </Card>
            <Card>
              <div className="text-center">
                <p className="text-xs text-gray-400 mb-1">On Hire</p>
                <p className="text-2xl font-bold text-green-400">{result.on_hire_hours.toFixed(1)}</p>
                <p className="text-xs text-gray-500">hours</p>
              </div>
            </Card>
            <Card>
              <div className="text-center">
                <p className="text-xs text-gray-400 mb-1">Off Hire</p>
                <p className="text-2xl font-bold text-red-400">{result.off_hire_hours.toFixed(1)}</p>
                <p className="text-xs text-gray-500">hours</p>
              </div>
            </Card>
            <Card>
              <div className="text-center">
                <p className="text-xs text-gray-400 mb-1">Off-Hire %</p>
                <p className="text-2xl font-bold text-sky-400">{result.off_hire_pct.toFixed(1)}%</p>
              </div>
            </Card>
          </div>

          {chartData.length > 0 && (
            <Card title="Off-Hire Events">
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis dataKey="name" stroke="#9ca3af" tick={{ fontSize: 12 }} />
                    <YAxis stroke="#9ca3af" tick={{ fontSize: 12 }} label={{ value: 'Hours', angle: -90, position: 'insideLeft', style: { fill: '#9ca3af', fontSize: 12 } }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                      labelStyle={{ color: '#fff' }}
                      formatter={(value: number, _: string, props: { payload?: { reason: string } }) => [
                        `${value.toFixed(1)} hours`,
                        props.payload?.reason ?? '',
                      ]}
                    />
                    <Bar dataKey="hours" fill="rgba(239,68,68,0.7)" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>
          )}

          {result.events.length > 0 && (
            <Card title="Event Details" icon={<Clock className="w-5 h-5" />}>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-white/10">
                      <th className="text-left py-2 px-3 text-xs text-gray-400">#</th>
                      <th className="text-left py-2 px-3 text-xs text-gray-400">Start</th>
                      <th className="text-left py-2 px-3 text-xs text-gray-400">End</th>
                      <th className="text-left py-2 px-3 text-xs text-gray-400">Duration (h)</th>
                      <th className="text-left py-2 px-3 text-xs text-gray-400">Reason</th>
                      <th className="text-left py-2 px-3 text-xs text-gray-400">Avg Speed</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.events.map((ev, idx) => (
                      <tr key={idx} className="border-b border-white/5 hover:bg-white/5">
                        <td className="py-2 px-3 text-white">{idx + 1}</td>
                        <td className="py-2 px-3 text-sm text-gray-300">{new Date(ev.start_time).toLocaleString()}</td>
                        <td className="py-2 px-3 text-sm text-gray-300">{new Date(ev.end_time).toLocaleString()}</td>
                        <td className="py-2 px-3 text-white font-medium">{ev.duration_hours.toFixed(1)}</td>
                        <td className="py-2 px-3">
                          <span className="px-2 py-0.5 rounded-full text-xs bg-red-500/20 text-red-400">{ev.reason}</span>
                        </td>
                        <td className="py-2 px-3 text-gray-300">{ev.avg_speed_kts != null ? `${ev.avg_speed_kts.toFixed(1)} kts` : '-'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}

          {result.events.length === 0 && (
            <Card>
              <div className="flex items-center justify-center gap-3 py-4">
                <CheckCircle className="w-6 h-6 text-green-400" />
                <p className="text-lg text-green-400 font-medium">No off-hire events detected</p>
              </div>
            </Card>
          )}
        </>
      )}
    </div>
  );
}

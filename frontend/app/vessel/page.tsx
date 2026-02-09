'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import Header from '@/components/Header';
import Card from '@/components/Card';
import FuelChart from '@/components/FuelChart';
import {
  Ship, Save, RotateCcw, Gauge, Fuel, Wind, TrendingDown, TrendingUp,
  Upload, FileSpreadsheet, Trash2, Play, CheckCircle, AlertTriangle,
  Info, Anchor,
} from 'lucide-react';
import {
  apiClient, VesselSpecs, FuelScenario,
  CalibrationStatus, CalibrationResult,
} from '@/lib/api';
import { formatFuel, formatPower } from '@/lib/utils';

type VesselTab = 'specifications' | 'calibration' | 'fuel';

const DEFAULT_SPECS: VesselSpecs = {
  dwt: 49000,
  loa: 183,
  beam: 32,
  draft_laden: 11.8,
  draft_ballast: 6.5,
  mcr_kw: 8840,
  sfoc_at_mcr: 171,
  service_speed_laden: 14.5,
  service_speed_ballast: 15.0,
};

export default function VesselPage() {
  const [activeTab, setActiveTab] = useState<VesselTab>('specifications');

  return (
    <div className="min-h-screen bg-gradient-maritime">
      <Header />

      <main className="container mx-auto px-4 pt-20 pb-12">
        {/* Tab bar */}
        <div className="flex space-x-1 mb-6 bg-maritime-medium/50 backdrop-blur-sm rounded-lg p-1 max-w-xl">
          <TabButton label="Specifications" active={activeTab === 'specifications'} onClick={() => setActiveTab('specifications')} />
          <TabButton label="Calibration" active={activeTab === 'calibration'} onClick={() => setActiveTab('calibration')} />
          <TabButton label="Fuel Analysis" active={activeTab === 'fuel'} onClick={() => setActiveTab('fuel')} />
        </div>

        {activeTab === 'specifications' && <SpecificationsSection />}
        {activeTab === 'calibration' && <CalibrationSection />}
        {activeTab === 'fuel' && <FuelAnalysisSection />}
      </main>
    </div>
  );
}

function TabButton({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`flex-1 px-4 py-2.5 rounded-lg text-sm font-medium transition-colors ${
        active
          ? 'bg-primary-500 text-white shadow-md'
          : 'text-gray-400 hover:text-white hover:bg-white/5'
      }`}
    >
      {label}
    </button>
  );
}

// ─── Specifications ──────────────────────────────────────────────────────────

function SpecificationsSection() {
  const [specs, setSpecs] = useState<VesselSpecs>(DEFAULT_SPECS);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const data = await apiClient.getVesselSpecs();
        setSpecs(data);
      } catch (error) {
        console.error('Failed to load specs:', error);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const handleSave = async () => {
    setSaving(true);
    setMessage(null);
    try {
      await apiClient.updateVesselSpecs(specs);
      setMessage({ type: 'success', text: 'Vessel specifications updated successfully!' });
    } catch {
      setMessage({ type: 'error', text: 'Failed to update vessel specifications.' });
    } finally {
      setSaving(false);
    }
  };

  const handleReset = () => {
    setSpecs(DEFAULT_SPECS);
    setMessage(null);
  };

  const updateSpec = (key: keyof VesselSpecs, value: number) => {
    setSpecs((prev) => ({ ...prev, [key]: value }));
    setMessage(null);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-400" />
      </div>
    );
  }

  return (
    <div className="max-w-4xl">
      {message && (
        <div className={`mb-6 p-4 rounded-lg ${
          message.type === 'success'
            ? 'bg-green-500/10 border border-green-500/20 text-green-300'
            : 'bg-red-500/10 border border-red-500/20 text-red-300'
        }`}>
          {message.text}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Vessel Dimensions" icon={<Ship className="w-5 h-5" />}>
          <div className="space-y-4">
            <InputField label="Deadweight Tonnage (DWT)" value={specs.dwt} onChange={(v) => updateSpec('dwt', v)} unit="MT" />
            <InputField label="Length Overall (LOA)" value={specs.loa} onChange={(v) => updateSpec('loa', v)} unit="m" />
            <InputField label="Beam" value={specs.beam} onChange={(v) => updateSpec('beam', v)} unit="m" />
            <InputField label="Draft (Laden)" value={specs.draft_laden} onChange={(v) => updateSpec('draft_laden', v)} unit="m" />
            <InputField label="Draft (Ballast)" value={specs.draft_ballast} onChange={(v) => updateSpec('draft_ballast', v)} unit="m" />
          </div>
        </Card>

        <Card title="Engine & Performance" icon={<Ship className="w-5 h-5" />}>
          <div className="space-y-4">
            <InputField label="Main Engine MCR" value={specs.mcr_kw} onChange={(v) => updateSpec('mcr_kw', v)} unit="kW" />
            <InputField label="SFOC at MCR" value={specs.sfoc_at_mcr} onChange={(v) => updateSpec('sfoc_at_mcr', v)} unit="g/kWh" />
            <InputField label="Service Speed (Laden)" value={specs.service_speed_laden} onChange={(v) => updateSpec('service_speed_laden', v)} unit="kts" step={0.1} />
            <InputField label="Service Speed (Ballast)" value={specs.service_speed_ballast} onChange={(v) => updateSpec('service_speed_ballast', v)} unit="kts" step={0.1} />
          </div>
        </Card>
      </div>

      <div className="mt-8 flex items-center justify-between">
        <button
          onClick={handleReset}
          className="flex items-center space-x-2 px-6 py-3 bg-maritime-dark text-gray-300 rounded-lg hover:bg-maritime-light transition-colors"
        >
          <RotateCcw className="w-5 h-5" />
          <span>Reset to Default</span>
        </button>
        <button
          onClick={handleSave}
          disabled={saving}
          className="flex items-center space-x-2 px-8 py-3 bg-gradient-ocean text-white font-semibold rounded-lg shadow-ocean hover:opacity-90 transition-opacity disabled:opacity-50"
        >
          <Save className="w-5 h-5" />
          <span>{saving ? 'Saving...' : 'Save Changes'}</span>
        </button>
      </div>
    </div>
  );
}

// ─── Calibration ─────────────────────────────────────────────────────────────

function CalibrationSection() {
  const [calibration, setCalibration] = useState<CalibrationStatus | null>(null);
  const [reportsCount, setReportsCount] = useState(0);
  const [loading, setLoading] = useState(false);
  const [calibrating, setCalibrating] = useState(false);
  const [result, setResult] = useState<CalibrationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [daysSinceDrydock, setDaysSinceDrydock] = useState(180);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadCalibration = useCallback(async () => {
    try {
      const [cal, reports] = await Promise.all([
        apiClient.getCalibration(),
        apiClient.getNoonReports(),
      ]);
      setCalibration(cal);
      setReportsCount(reports.count);
    } catch (err) {
      console.error('Failed to load calibration:', err);
    }
  }, []);

  useEffect(() => { loadCalibration(); }, [loadCalibration]);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const r = await apiClient.uploadNoonReportsCSV(file);
      setReportsCount(r.total_reports);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to upload CSV');
    } finally {
      setLoading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleClearReports = async () => {
    if (!confirm('Clear all noon reports?')) return;
    setLoading(true);
    try {
      await apiClient.clearNoonReports();
      setReportsCount(0);
      setResult(null);
    } catch (err) {
      console.error('Failed to clear reports:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleCalibrate = async () => {
    if (reportsCount < 5) { setError('Need at least 5 noon reports for calibration'); return; }
    setCalibrating(true);
    setError(null);
    try {
      const r = await apiClient.calibrateVessel(daysSinceDrydock);
      setResult(r);
      await loadCalibration();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Calibration failed');
    } finally {
      setCalibrating(false);
    }
  };

  const formatFactor = (factor: number): string => {
    const pct = (factor - 1) * 100;
    if (pct === 0) return '0%';
    return pct > 0 ? `+${pct.toFixed(1)}%` : `${pct.toFixed(1)}%`;
  };

  return (
    <div className="max-w-2xl space-y-6">
      {/* Status card */}
      <Card>
        <div className="flex items-center gap-3 mb-4">
          <Gauge className="w-5 h-5 text-primary-400" />
          <div>
            <h3 className="text-white font-medium">Calibration Status</h3>
            <p className="text-xs text-gray-400">
              {calibration?.calibrated
                ? `Calibrated with ${calibration.num_reports_used} reports`
                : 'Using theoretical model'}
            </p>
          </div>
          {calibration?.calibrated ? (
            <CheckCircle className="w-5 h-5 text-green-400 ml-auto" />
          ) : (
            <Info className="w-5 h-5 text-gray-400 ml-auto" />
          )}
        </div>

        {/* Current factors */}
        {calibration && (
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div className="bg-maritime-dark rounded p-2">
              <div className="text-gray-400 text-xs">Hull Fouling</div>
              <div className={`font-medium ${calibration.factors.calm_water > 1.1 ? 'text-amber-400' : 'text-white'}`}>
                {formatFactor(calibration.factors.calm_water)}
              </div>
            </div>
            <div className="bg-maritime-dark rounded p-2">
              <div className="text-gray-400 text-xs">Wind Response</div>
              <div className="text-white font-medium">{formatFactor(calibration.factors.wind)}</div>
            </div>
            <div className="bg-maritime-dark rounded p-2">
              <div className="text-gray-400 text-xs">Wave Response</div>
              <div className="text-white font-medium">{formatFactor(calibration.factors.waves)}</div>
            </div>
            <div className="bg-maritime-dark rounded p-2">
              <div className="text-gray-400 text-xs">SFOC Factor</div>
              <div className="text-white font-medium">{formatFactor(calibration.factors.sfoc_factor)}</div>
            </div>
          </div>
        )}
      </Card>

      {/* Noon Reports */}
      <Card title="Noon Reports" icon={<FileSpreadsheet className="w-5 h-5" />}>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-300">
              Reports: <span className="text-white font-medium">{reportsCount}</span>
            </span>
            {reportsCount > 0 && (
              <button
                onClick={handleClearReports}
                className="text-xs text-red-400 hover:text-red-300 flex items-center gap-1"
                disabled={loading}
              >
                <Trash2 className="w-3 h-3" />
                Clear
              </button>
            )}
          </div>

          <input ref={fileInputRef} type="file" accept=".csv" onChange={handleFileUpload} className="hidden" />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={loading}
            className="w-full py-2 px-3 bg-maritime-dark border border-white/10 rounded text-sm text-gray-300 hover:text-white hover:border-primary-500/50 transition-colors flex items-center justify-center gap-2"
          >
            <FileSpreadsheet className="w-4 h-4" />
            {loading ? 'Uploading...' : 'Upload Noon Reports CSV'}
          </button>
          <p className="text-xs text-gray-500">CSV with: timestamp, latitude, longitude, speed_over_ground_kts, fuel_consumption_mt</p>
        </div>
      </Card>

      {/* Calibrate */}
      <Card>
        <div className="space-y-4">
          <div>
            <label className="block text-sm text-gray-300 mb-1">
              <div className="flex items-center gap-1">
                <Anchor className="w-3 h-3" />
                Days Since Drydock: {daysSinceDrydock}
              </div>
            </label>
            <input
              type="range" min="0" max="730" step="30"
              value={daysSinceDrydock}
              onChange={(e) => setDaysSinceDrydock(parseInt(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-500">
              <span>0</span><span>1 year</span><span>2 years</span>
            </div>
          </div>

          <button
            onClick={handleCalibrate}
            disabled={reportsCount < 5 || calibrating}
            className="w-full py-2 px-3 bg-primary-500 text-white rounded text-sm font-medium hover:bg-primary-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {calibrating ? (
              <><div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />Calibrating...</>
            ) : (
              <><Play className="w-4 h-4" />Run Calibration</>
            )}
          </button>

          {reportsCount < 5 && (
            <p className="text-xs text-amber-400 flex items-center gap-1">
              <AlertTriangle className="w-3 h-3" />Need at least 5 noon reports
            </p>
          )}

          {error && (
            <div className="p-2 bg-red-500/20 border border-red-500/30 rounded text-sm text-red-400">{error}</div>
          )}

          {result && (
            <div className="p-3 bg-green-500/10 border border-green-500/30 rounded">
              <div className="flex items-center gap-2 text-green-400 font-medium mb-2">
                <CheckCircle className="w-4 h-4" />Calibration Complete
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div><span className="text-gray-400">Reports used:</span> <span className="text-white">{result.reports_used}</span></div>
                <div><span className="text-gray-400">Skipped:</span> <span className="text-white">{result.reports_skipped}</span></div>
                <div><span className="text-gray-400">Error before:</span> <span className="text-white">{result.mean_error_before_mt.toFixed(2)} MT</span></div>
                <div><span className="text-gray-400">Error after:</span> <span className="text-green-400">{result.mean_error_after_mt.toFixed(2)} MT</span></div>
              </div>
              <div className="mt-2 text-sm text-green-400">Improvement: {result.improvement_pct.toFixed(1)}%</div>
            </div>
          )}
        </div>
      </Card>

      <div className="text-xs text-gray-500">
        Calibration adjusts the theoretical Holtrop-Mennen model to match your vessel's
        actual performance. Upload noon reports with actual fuel consumption to derive
        calibration factors for hull fouling, wind, and wave response.
      </div>
    </div>
  );
}

// ─── Fuel Analysis ───────────────────────────────────────────────────────────

function FuelAnalysisSection() {
  const [scenarios, setScenarios] = useState<FuelScenario[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const data = await apiClient.getFuelScenarios();
        setScenarios(data.scenarios);
      } catch (error) {
        console.error('Failed to load scenarios:', error);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const chartData = scenarios.map((s) => ({
    name: s.name.replace(' (Laden)', '').replace(' (Ballast)', ''),
    calm_water: s.fuel_mt * 0.6,
    wind: s.fuel_mt * 0.25,
    waves: s.fuel_mt * 0.15,
  }));

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-400" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Scenarios Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {scenarios.map((scenario, idx) => (
          <ScenarioCard key={idx} scenario={scenario} />
        ))}
      </div>

      {/* Chart */}
      <Card title="Fuel Consumption Comparison" className="h-96">
        <FuelChart data={chartData} />
      </Card>

      {/* Impact Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Weather Impact Analysis" icon={<Wind className="w-5 h-5" />}>
          <div className="space-y-4">
            <ImpactRow label="Head Wind (20 kts)" baseline={scenarios[0]?.fuel_mt || 0} actual={scenarios[1]?.fuel_mt || 0} />
            <ImpactRow label="Rough Seas (3m waves)" baseline={scenarios[0]?.fuel_mt || 0} actual={scenarios[2]?.fuel_mt || 0} />
            <ImpactRow label="Ballast Condition" baseline={scenarios[0]?.fuel_mt || 0} actual={scenarios[3]?.fuel_mt || 0} />
          </div>
        </Card>

        <Card title="Optimization Opportunities" icon={<TrendingDown className="w-5 h-5" />}>
          <div className="space-y-4">
            <OpportunityItem title="Weather Routing" description="Avoid head winds and rough seas" savings="15-25%" />
            <OpportunityItem title="Speed Optimization" description="Adjust speed based on conditions" savings="8-12%" />
            <OpportunityItem title="Route Planning" description="Choose fuel-optimal waypoints" savings="5-10%" />
          </div>
        </Card>
      </div>
    </div>
  );
}

// ─── Shared helpers ──────────────────────────────────────────────────────────

function InputField({ label, value, onChange, unit, step = 1 }: {
  label: string; value: number; onChange: (value: number) => void; unit: string; step?: number;
}) {
  return (
    <div>
      <label className="block text-sm font-medium text-gray-300 mb-2">{label}</label>
      <div className="relative">
        <input
          type="number"
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          step={step}
          className="w-full bg-maritime-dark border border-white/10 rounded-lg px-4 py-3 pr-16 text-white focus:outline-none focus:border-primary-400 transition-colors"
        />
        <span className="absolute right-4 top-1/2 -translate-y-1/2 text-sm text-gray-400">{unit}</span>
      </div>
    </div>
  );
}

function ScenarioCard({ scenario }: { scenario: FuelScenario }) {
  const isLaden = scenario.name.includes('Laden');
  const hasWeather = !scenario.name.includes('Calm');

  return (
    <Card>
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <h3 className="font-semibold text-white mb-1">{scenario.name}</h3>
          <p className="text-xs text-gray-400">{scenario.conditions}</p>
        </div>
        <div className="flex space-x-1">
          {isLaden && <span className="px-2 py-1 bg-ocean-500/20 text-ocean-300 text-xs rounded">Laden</span>}
          {hasWeather && <Wind className="w-4 h-4 text-primary-400" />}
        </div>
      </div>
      <div className="space-y-3">
        <div>
          <p className="text-xs text-gray-400 mb-1">Daily Fuel</p>
          <p className="text-2xl font-bold text-white">{formatFuel(scenario.fuel_mt)}</p>
        </div>
        <div>
          <p className="text-xs text-gray-400 mb-1">Power</p>
          <p className="text-lg font-semibold text-gray-300">{formatPower(scenario.power_kw)}</p>
        </div>
      </div>
    </Card>
  );
}

function ImpactRow({ label, baseline, actual }: { label: string; baseline: number; actual: number }) {
  const impact = baseline > 0 ? ((actual - baseline) / baseline) * 100 : 0;
  const isNegative = impact < 0;
  return (
    <div className="flex items-center justify-between py-3 border-b border-white/5 last:border-0">
      <span className="text-sm text-gray-300">{label}</span>
      <div className="flex items-center space-x-2">
        {isNegative ? <TrendingDown className="w-4 h-4 text-green-400" /> : <TrendingUp className="w-4 h-4 text-red-400" />}
        <span className={`text-sm font-semibold ${isNegative ? 'text-green-400' : 'text-red-400'}`}>
          {isNegative ? '' : '+'}{impact.toFixed(1)}%
        </span>
      </div>
    </div>
  );
}

function OpportunityItem({ title, description, savings }: { title: string; description: string; savings: string }) {
  return (
    <div className="flex items-start space-x-4 p-4 bg-maritime-dark rounded-lg hover:bg-maritime-light transition-colors">
      <div className="flex-shrink-0 w-12 h-12 bg-green-500/10 rounded-lg flex items-center justify-center">
        <TrendingDown className="w-6 h-6 text-green-400" />
      </div>
      <div className="flex-1 min-w-0">
        <h4 className="text-sm font-semibold text-white mb-1">{title}</h4>
        <p className="text-xs text-gray-400">{description}</p>
      </div>
      <span className="px-3 py-1 bg-green-500/20 text-green-300 text-xs font-semibold rounded-full">{savings}</span>
    </div>
  );
}

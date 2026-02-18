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
  CalibrationStatus, CalibrationResult, VesselModelStatus,
  PerformancePredictionRequest, PerformancePredictionResult,
  ModelCurvesResponse,
} from '@/lib/api';
import { formatFuel, formatPower } from '@/lib/utils';
import { useVoyage } from '@/components/VoyageContext';

import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer,
} from 'recharts';

type VesselTab = 'specifications' | 'calibration' | 'fuel' | 'model';

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

      <main className="container mx-auto px-6 pt-20 pb-12">
        {/* Tab bar */}
        <div className="flex space-x-1 mb-6 bg-maritime-medium/50 backdrop-blur-sm rounded-lg p-1 max-w-2xl">
          <TabButton label="Specifications" active={activeTab === 'specifications'} onClick={() => setActiveTab('specifications')} />
          <TabButton label="Calibration" active={activeTab === 'calibration'} onClick={() => setActiveTab('calibration')} />
          <TabButton label="Model" active={activeTab === 'model'} onClick={() => setActiveTab('model')} />
          <TabButton label="Fuel Analysis" active={activeTab === 'fuel'} onClick={() => setActiveTab('fuel')} />
        </div>

        {activeTab === 'specifications' && <SpecificationsSection />}
        {activeTab === 'calibration' && <CalibrationSection />}
        {activeTab === 'model' && <ModelCurvesSection />}
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
  const { refreshSpecs } = useVoyage();
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
      await refreshSpecs();
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
    <div>
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

      {/* Model Parameters (read-only derived values) */}
      <ModelParametersCard />

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

function ModelParametersCard() {
  const [model, setModel] = useState<VesselModelStatus | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const data = await apiClient.getVesselModelStatus();
        setModel(data);
      } catch (error) {
        console.error('Failed to load model status:', error);
      }
    })();
  }, []);

  if (!model) return null;

  const { hull_form, areas } = model.specifications;
  const { computed } = model;

  return (
    <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
      <Card title="Hull Form" icon={<Anchor className="w-5 h-5" />}>
        <div className="space-y-2 text-sm">
          <ReadOnlyField label="Block Coefficient (Laden)" value={hull_form.cb_laden.toFixed(3)} />
          <ReadOnlyField label="Block Coefficient (Ballast)" value={hull_form.cb_ballast.toFixed(3)} />
          <ReadOnlyField label="Wetted Surface (Laden)" value={`${hull_form.wetted_surface_laden.toLocaleString()} m\u00B2`} />
          <ReadOnlyField label="Wetted Surface (Ballast)" value={`${hull_form.wetted_surface_ballast.toLocaleString()} m\u00B2`} />
        </div>
      </Card>

      <Card title="Propulsion Areas & Performance" icon={<Gauge className="w-5 h-5" />}>
        <div className="space-y-2 text-sm">
          <ReadOnlyField label="Frontal Area (Laden / Ballast)" value={`${areas.frontal_area_laden} / ${areas.frontal_area_ballast} m\u00B2`} />
          <ReadOnlyField label="Lateral Area (Laden / Ballast)" value={`${areas.lateral_area_laden} / ${areas.lateral_area_ballast} m\u00B2`} />
          <ReadOnlyField label="Optimal Speed (Laden)" value={`${computed.optimal_speed_laden_kts} kts`} />
          <ReadOnlyField label="Optimal Speed (Ballast)" value={`${computed.optimal_speed_ballast_kts} kts`} />
          <ReadOnlyField label="Daily Fuel at Service (Laden)" value={`${computed.daily_fuel_service_laden_mt} MT`} />
          <ReadOnlyField label="Daily Fuel at Service (Ballast)" value={`${computed.daily_fuel_service_ballast_mt} MT`} />
          <ReadOnlyField label="Wave Method" value={model.wave_method === 'kwon' ? "Kwon's Method" : 'STAWAVE-1 (ISO 15016)'} />
        </div>
      </Card>
    </div>
  );
}

function ReadOnlyField({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-white/5 last:border-0">
      <span className="text-gray-400">{label}</span>
      <span className="text-white font-medium">{value}</span>
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
      const ext = file.name.split('.').pop()?.toLowerCase();
      const isExcel = ext === 'xlsx' || ext === 'xls';
      const r = isExcel
        ? await apiClient.uploadNoonReportsExcel(file)
        : await apiClient.uploadNoonReportsCSV(file);
      setReportsCount(r.total_reports);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to upload file');
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

        {/* Calibration timestamp */}
        {calibration?.calibrated && calibration.calibrated_at && (
          <div className="mb-3 px-2 py-1.5 bg-maritime-dark rounded text-xs text-gray-300">
            <span className="text-gray-500">Last calibrated:</span>{' '}
            {(() => {
              const d = new Date(calibration.calibrated_at);
              const ago = Math.round((Date.now() - d.getTime()) / 60000);
              const relative = ago < 60 ? `${ago}m ago` : ago < 1440 ? `${Math.round(ago / 60)}h ago` : `${Math.round(ago / 1440)}d ago`;
              return `${d.toLocaleString()} (${relative})`;
            })()}
            {calibration.days_since_drydock !== undefined && calibration.days_since_drydock > 0 && (
              <span className="ml-3 text-gray-500">Drydock: <span className="text-gray-300">{calibration.days_since_drydock}d ago</span></span>
            )}
            {calibration.calibration_error_mt !== undefined && calibration.calibration_error_mt > 0 && (
              <span className="ml-3 text-gray-500">Error: <span className="text-gray-300">{calibration.calibration_error_mt.toFixed(2)} MT</span></span>
            )}
          </div>
        )}

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

          <input ref={fileInputRef} type="file" accept=".csv,.xlsx,.xls" onChange={handleFileUpload} className="hidden" />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={loading}
            className="w-full py-2 px-3 bg-maritime-dark border border-white/10 rounded text-sm text-gray-300 hover:text-white hover:border-primary-500/50 transition-colors flex items-center justify-center gap-2"
          >
            <FileSpreadsheet className="w-4 h-4" />
            {loading ? 'Uploading...' : 'Upload Noon Reports (CSV/Excel)'}
          </button>
          <p className="text-xs text-gray-500">CSV or Excel with: date, latitude, longitude, speed, fuel_consumption</p>
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

// ─── Model Curves ────────────────────────────────────────────────────────────

const CHART_TOOLTIP_STYLE = {
  contentStyle: {
    backgroundColor: 'rgba(26, 41, 66, 0.95)',
    border: '1px solid rgba(255, 255, 255, 0.1)',
    borderRadius: '8px',
    fontSize: '12px',
  },
  labelStyle: { color: '#fff', fontWeight: 600 },
};

function ModelCurvesSection() {
  const [curves, setCurves] = useState<ModelCurvesResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const data = await apiClient.getModelCurves();
        setCurves(data);
      } catch (error) {
        console.error('Failed to load model curves:', error);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-400" />
      </div>
    );
  }

  if (!curves) {
    return <div className="text-gray-400 text-sm">Failed to load model curves</div>;
  }

  const cal = curves.calibration;

  // Prepare chart data
  const resistanceData = curves.speed_range_kts.map((spd, i) => ({
    speed: spd,
    theoretical: curves.resistance_theoretical_kn[i],
    calibrated: curves.resistance_calibrated_kn[i],
  }));

  const powerFuelData = curves.speed_range_kts.map((spd, i) => ({
    speed: spd,
    power: curves.power_kw[i],
    fuel: curves.fuel_mt_per_day[i],
  }));

  const sfocData = curves.sfoc_curve.load_pct.map((load, i) => ({
    load,
    theoretical: curves.sfoc_curve.sfoc_theoretical_gkwh[i],
    calibrated: curves.sfoc_curve.sfoc_calibrated_gkwh[i],
  }));

  return (
    <div className="space-y-6">
      {/* Calibration banner */}
      <div className={`p-3 rounded-lg border text-sm flex items-center gap-3 ${
        cal.calibrated
          ? 'bg-green-500/10 border-green-500/20 text-green-300'
          : 'bg-white/5 border-white/10 text-gray-400'
      }`}>
        {cal.calibrated ? (
          <CheckCircle className="w-4 h-4 flex-shrink-0" />
        ) : (
          <Info className="w-4 h-4 flex-shrink-0" />
        )}
        <span>
          {cal.calibrated
            ? `Model calibrated on ${new Date(cal.calibrated_at!).toLocaleDateString()} from ${cal.num_reports_used} reports — MAE: ${cal.calibration_error_mt.toFixed(2)} MT/day`
            : 'Using theoretical (uncalibrated) model'}
        </span>
      </div>

      {/* Calibration factors summary */}
      {cal.calibrated && (
        <div className="grid grid-cols-4 gap-3">
          {(['calm_water', 'wind', 'waves', 'sfoc_factor'] as const).map(key => {
            const val = cal.factors[key];
            const pct = ((val - 1) * 100);
            const label = key === 'calm_water' ? 'Hull' : key === 'sfoc_factor' ? 'SFOC' : key.charAt(0).toUpperCase() + key.slice(1);
            return (
              <div key={key} className="bg-maritime-dark rounded-lg p-3 text-center">
                <div className="text-gray-500 text-xs mb-1">{label}</div>
                <div className={`text-lg font-bold ${Math.abs(pct) < 0.5 ? 'text-white' : pct > 0 ? 'text-amber-400' : 'text-green-400'}`}>
                  {pct > 0 ? '+' : ''}{pct.toFixed(1)}%
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Note when uncalibrated — theoretical = calibrated */}
      {!cal.calibrated && (
        <div className="text-xs text-gray-500 italic">
          Theoretical and calibrated curves overlap — calibrate from noon reports or engine log to see the difference.
        </div>
      )}

      {/* Charts — 2x2 grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Resistance vs Speed */}
        <Card title="Resistance vs Speed (Calm Water, Laden)" icon={<TrendingUp className="w-5 h-5" />}>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={resistanceData} margin={{ top: 5, right: 20, bottom: 20, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="speed" stroke="#9ca3af" tick={{ fontSize: 12 }} label={{ value: 'Speed (kts)', position: 'insideBottom', offset: -10, style: { fill: '#9ca3af', fontSize: 12 } }} />
                <YAxis stroke="#9ca3af" tick={{ fontSize: 12 }} label={{ value: 'Resistance (kN)', angle: -90, position: 'insideLeft', offset: 5, style: { fill: '#9ca3af', fontSize: 12 } }} />
                <Tooltip {...CHART_TOOLTIP_STYLE} formatter={(v: number) => `${v.toFixed(1)} kN`} />
                <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
                <Line type="monotone" dataKey="calibrated" name="Calibrated" stroke="#3b82f6" dot={false} strokeWidth={2.5} />
                <Line type="monotone" dataKey="theoretical" name="Theoretical" stroke="#94a3b8" strokeDasharray="8 4" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* SFOC vs Engine Load */}
        <Card title="SFOC vs Engine Load" icon={<Fuel className="w-5 h-5" />}>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={sfocData} margin={{ top: 5, right: 20, bottom: 20, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="load" stroke="#9ca3af" tick={{ fontSize: 12 }} label={{ value: 'Engine Load (% MCR)', position: 'insideBottom', offset: -10, style: { fill: '#9ca3af', fontSize: 12 } }} />
                <YAxis stroke="#9ca3af" tick={{ fontSize: 12 }} label={{ value: 'SFOC (g/kWh)', angle: -90, position: 'insideLeft', offset: 5, style: { fill: '#9ca3af', fontSize: 12 } }} domain={['auto', 'auto']} />
                <Tooltip {...CHART_TOOLTIP_STYLE} formatter={(v: number) => `${v.toFixed(1)} g/kWh`} />
                <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
                <Line type="monotone" dataKey="calibrated" name="Calibrated" stroke="#f97316" dot={false} strokeWidth={2.5} />
                <Line type="monotone" dataKey="theoretical" name="Theoretical" stroke="#94a3b8" strokeDasharray="8 4" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* Power vs Speed */}
        <Card title="Brake Power vs Speed (Laden)" icon={<Gauge className="w-5 h-5" />}>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={powerFuelData} margin={{ top: 5, right: 20, bottom: 20, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="speed" stroke="#9ca3af" tick={{ fontSize: 12 }} label={{ value: 'Speed (kts)', position: 'insideBottom', offset: -10, style: { fill: '#9ca3af', fontSize: 12 } }} />
                <YAxis stroke="#9ca3af" tick={{ fontSize: 12 }} label={{ value: 'Power (kW)', angle: -90, position: 'insideLeft', offset: 5, style: { fill: '#9ca3af', fontSize: 12 } }} />
                <Tooltip {...CHART_TOOLTIP_STYLE} formatter={(v: number) => `${v.toLocaleString()} kW`} />
                <Line type="monotone" dataKey="power" name="Brake Power" stroke="#22c55e" dot={false} strokeWidth={2.5} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* Fuel vs Speed */}
        <Card title="Daily Fuel vs Speed (Laden)" icon={<Fuel className="w-5 h-5" />}>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={powerFuelData} margin={{ top: 5, right: 20, bottom: 20, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="speed" stroke="#9ca3af" tick={{ fontSize: 12 }} label={{ value: 'Speed (kts)', position: 'insideBottom', offset: -10, style: { fill: '#9ca3af', fontSize: 12 } }} />
                <YAxis stroke="#9ca3af" tick={{ fontSize: 12 }} label={{ value: 'Fuel (MT/day)', angle: -90, position: 'insideLeft', offset: 5, style: { fill: '#9ca3af', fontSize: 12 } }} />
                <Tooltip {...CHART_TOOLTIP_STYLE} formatter={(v: number) => `${v.toFixed(2)} MT/day`} />
                <Line type="monotone" dataKey="fuel" name="Daily Fuel" stroke="#ef4444" dot={false} strokeWidth={2.5} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
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

  const chartData = scenarios.map((s) => {
    const isCalm = s.name.includes('Calm');
    const hasWind = s.name.includes('Wind');
    const hasWaves = s.name.includes('Rough');
    // Approximate breakdown: calm scenarios are all calm_water;
    // wind scenario splits ~65/30/5; rough seas ~55/15/30
    const cw = isCalm ? s.fuel_mt : hasWind ? s.fuel_mt * 0.65 : s.fuel_mt * 0.55;
    const wi = isCalm ? 0 : hasWind ? s.fuel_mt * 0.30 : s.fuel_mt * 0.15;
    const wa = isCalm ? 0 : hasWind ? s.fuel_mt * 0.05 : s.fuel_mt * 0.30;
    return {
      name: s.name.replace(' (Laden)', '').replace(' (Ballast)', ''),
      calm_water: cw,
      wind: wi,
      waves: wa,
    };
  });

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-400" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Performance Predictor */}
      <PerformancePredictor />

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

// ─── Performance Predictor ────────────────────────────────────────────────────

function PerformancePredictor() {
  const [isLaden, setIsLaden] = useState(true);
  const [mode, setMode] = useState<'engine_load' | 'calm_speed'>('engine_load');
  const [engineLoad, setEngineLoad] = useState(85);
  const [calmSpeed, setCalmSpeed] = useState(14.5);
  const [windSpeed, setWindSpeed] = useState(0);
  const [windRelDir, setWindRelDir] = useState(0);
  const [waveHeight, setWaveHeight] = useState(0);
  const [waveRelDir, setWaveRelDir] = useState(0);
  const [currentSpeed, setCurrentSpeed] = useState(0);
  const [currentRelDir, setCurrentRelDir] = useState(0);
  const [result, setResult] = useState<PerformancePredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const relDirLabel = (deg: number) => {
    if (deg === 0) return 'Head';
    if (deg <= 30) return 'Bow quarter';
    if (deg <= 60) return 'Fwd beam';
    if (deg <= 90) return 'Beam';
    if (deg <= 120) return 'Aft beam';
    if (deg <= 150) return 'Quarter';
    return 'Following';
  };

  const predict = useCallback(async () => {
    setLoading(true);
    try {
      const req: PerformancePredictionRequest = {
        is_laden: isLaden,
        wind_speed_kts: windSpeed,
        wind_relative_deg: windRelDir,
        wave_height_m: waveHeight,
        wave_relative_deg: waveRelDir,
        current_speed_kts: currentSpeed,
        current_relative_deg: currentRelDir,
      };
      if (mode === 'engine_load') {
        req.engine_load_pct = engineLoad;
      } else {
        req.calm_speed_kts = calmSpeed;
      }
      const r = await apiClient.predictPerformance(req);
      setResult(r);
    } catch (err) {
      console.error('Prediction failed:', err);
    } finally {
      setLoading(false);
    }
  }, [isLaden, mode, engineLoad, calmSpeed, windSpeed, windRelDir, waveHeight, waveRelDir, currentSpeed, currentRelDir]);

  // Auto-predict on input change with debounce
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(predict, 400);
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
  }, [predict]);

  return (
    <Card title="Performance Predictor" icon={<Gauge className="w-5 h-5" />}>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Inputs */}
        <div className="space-y-4">
          {/* Loading condition toggle */}
          <div className="flex items-center gap-3">
            <span className="text-sm text-gray-400">Condition:</span>
            <button
              onClick={() => setIsLaden(true)}
              className={`px-3 py-1 rounded text-xs font-medium transition-colors ${isLaden ? 'bg-primary-500 text-white' : 'bg-maritime-dark text-gray-400 hover:text-white'}`}
            >Laden</button>
            <button
              onClick={() => setIsLaden(false)}
              className={`px-3 py-1 rounded text-xs font-medium transition-colors ${!isLaden ? 'bg-primary-500 text-white' : 'bg-maritime-dark text-gray-400 hover:text-white'}`}
            >Ballast</button>
          </div>

          {/* Mode toggle: Engine Load vs Target Speed */}
          <div className="flex items-center gap-3">
            <span className="text-sm text-gray-400">Mode:</span>
            <button
              onClick={() => setMode('engine_load')}
              className={`px-3 py-1 rounded text-xs font-medium transition-colors ${mode === 'engine_load' ? 'bg-primary-500 text-white' : 'bg-maritime-dark text-gray-400 hover:text-white'}`}
            >Engine Load</button>
            <button
              onClick={() => setMode('calm_speed')}
              className={`px-3 py-1 rounded text-xs font-medium transition-colors ${mode === 'calm_speed' ? 'bg-primary-500 text-white' : 'bg-maritime-dark text-gray-400 hover:text-white'}`}
            >Target Speed</button>
          </div>

          {mode === 'engine_load' ? (
            <SliderField label="Engine Load" value={engineLoad} onChange={setEngineLoad} min={15} max={100} step={5} unit="% MCR" />
          ) : (
            <SliderField label="Calm Water Speed" value={calmSpeed} onChange={setCalmSpeed} min={5} max={20} step={0.5} unit="kts" />
          )}

          <div className="border-t border-white/5 pt-3">
            <div className="text-xs text-gray-500 mb-2 flex items-center gap-1"><Wind className="w-3 h-3" /> Wind & Waves</div>
            <p className="text-[10px] text-gray-600 mb-2">Directions relative to bow: 0°=ahead, 90°=beam, 180°=astern</p>
            <div className="space-y-3">
              <SliderField label="Wind Speed" value={windSpeed} onChange={setWindSpeed} min={0} max={60} step={5} unit="kts" />
              <SliderField label="Wind Direction" value={windRelDir} onChange={setWindRelDir} min={0} max={180} step={15} unit={`° ${relDirLabel(windRelDir)}`} />
              <SliderField label="Wave Height" value={waveHeight} onChange={setWaveHeight} min={0} max={8} step={0.5} unit="m" />
              <SliderField label="Wave Direction" value={waveRelDir} onChange={setWaveRelDir} min={0} max={180} step={15} unit={`° ${relDirLabel(waveRelDir)}`} />
            </div>
          </div>

          <div className="border-t border-white/5 pt-3">
            <div className="text-xs text-gray-500 mb-2">Current</div>
            <div className="space-y-3">
              <SliderField label="Current Speed" value={currentSpeed} onChange={setCurrentSpeed} min={0} max={5} step={0.5} unit="kts" />
              <SliderField label="Current Direction" value={currentRelDir} onChange={setCurrentRelDir} min={0} max={180} step={15} unit={`° ${relDirLabel(currentRelDir)}`} />
            </div>
          </div>
        </div>

        {/* Results */}
        <div>
          {loading && !result && (
            <div className="flex items-center justify-center h-full">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-400" />
            </div>
          )}
          {result && (
            <div className="space-y-3">
              {/* Primary metrics */}
              <div className="grid grid-cols-2 gap-3">
                <MetricBox label="Speed Through Water" value={`${result.stw_kts}`} unit="kts" large />
                <MetricBox label="Speed Over Ground" value={`${result.sog_kts}`} unit="kts" large />
                <MetricBox label="Fuel / Day" value={`${result.fuel_per_day_mt}`} unit="MT" large />
                <MetricBox label="Fuel / NM" value={`${result.fuel_per_nm_mt}`} unit="MT" />
              </div>

              {/* Engine state */}
              <div className="bg-maritime-dark rounded-lg p-3 space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Power</span>
                  <span className="text-white font-medium">{formatPower(result.power_kw)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Engine Load</span>
                  <span className="text-white font-medium">{result.load_pct}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">SFOC</span>
                  <span className="text-white font-medium">{result.sfoc_gkwh} g/kWh</span>
                </div>
              </div>

              {/* MCR exceeded warning (calm speed mode) */}
              {result.mcr_exceeded && (
                <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-3 text-sm">
                  <div className="flex items-center gap-2 text-red-400 mb-1">
                    <AlertTriangle className="w-4 h-4" />
                    <span className="font-medium">MCR Exceeded</span>
                  </div>
                  <div className="text-gray-300">
                    Target speed requires <span className="text-red-400 font-medium">{result.required_power_kw?.toLocaleString()} kW</span>
                    <span className="text-gray-500 ml-1">(MCR = {(result.power_kw).toLocaleString()} kW)</span>
                    <br />
                    <span className="text-gray-400">Speed capped to {result.stw_kts} kts at 100% MCR</span>
                  </div>
                </div>
              )}

              {/* Weather impact */}
              {result.speed_loss_from_weather_pct > 0 && (
                <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg p-3 text-sm">
                  <div className="flex items-center gap-2 text-amber-400 mb-1">
                    <Wind className="w-4 h-4" />
                    <span className="font-medium">Weather Impact</span>
                  </div>
                  <div className="text-gray-300">
                    Speed loss: <span className="text-amber-400 font-medium">{result.speed_loss_from_weather_pct}%</span>
                    <span className="text-gray-500 ml-2">({result.calm_water_speed_kts} kts in calm water)</span>
                  </div>
                </div>
              )}

              {/* Current effect */}
              {result.current_effect_kts !== 0 && (
                <div className={`${result.current_effect_kts > 0 ? 'bg-green-500/10 border-green-500/20' : 'bg-red-500/10 border-red-500/20'} border rounded-lg p-3 text-sm`}>
                  <span className="text-gray-300">Current effect: </span>
                  <span className={`font-medium ${result.current_effect_kts > 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {result.current_effect_kts > 0 ? '+' : ''}{result.current_effect_kts} kts
                  </span>
                </div>
              )}

              {/* Resistance breakdown */}
              <div className="bg-maritime-dark rounded-lg p-3 text-sm">
                <div className="text-gray-500 text-xs mb-2">Resistance Breakdown</div>
                <ResistanceBar
                  calm={result.resistance_breakdown_kn.calm_water}
                  wind={result.resistance_breakdown_kn.wind}
                  waves={result.resistance_breakdown_kn.waves}
                  total={result.resistance_breakdown_kn.total}
                />
              </div>
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}

function SliderField({ label, value, onChange, min, max, step, unit }: {
  label: string; value: number; onChange: (v: number) => void;
  min: number; max: number; step: number; unit: string;
}) {
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-gray-400 w-28 shrink-0">{label}</span>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="flex-1 h-1.5 accent-primary-500"
      />
      <span className="text-xs text-white w-16 text-right font-mono">{value} {unit}</span>
    </div>
  );
}

function MetricBox({ label, value, unit, large }: {
  label: string; value: string; unit: string; large?: boolean;
}) {
  return (
    <div className="bg-maritime-dark rounded-lg p-3">
      <div className="text-gray-500 text-xs mb-1">{label}</div>
      <div className="flex items-baseline gap-1">
        <span className={`text-white font-bold ${large ? 'text-xl' : 'text-base'}`}>{value}</span>
        <span className="text-gray-400 text-xs">{unit}</span>
      </div>
    </div>
  );
}

function ResistanceBar({ calm, wind, waves, total }: {
  calm: number; wind: number; waves: number; total: number;
}) {
  if (total <= 0) return null;
  const cPct = (calm / total) * 100;
  const wPct = (wind / total) * 100;
  const vPct = (waves / total) * 100;
  return (
    <div>
      <div className="flex h-3 rounded-full overflow-hidden mb-2">
        <div className="bg-blue-500" style={{ width: `${cPct}%` }} title={`Calm: ${calm.toFixed(1)} kN`} />
        <div className="bg-cyan-400" style={{ width: `${wPct}%` }} title={`Wind: ${wind.toFixed(1)} kN`} />
        <div className="bg-teal-400" style={{ width: `${vPct}%` }} title={`Waves: ${waves.toFixed(1)} kN`} />
      </div>
      <div className="flex justify-between text-xs">
        <span className="text-blue-400">Calm {cPct.toFixed(0)}%</span>
        <span className="text-cyan-400">Wind {wPct.toFixed(0)}%</span>
        <span className="text-teal-400">Waves {vPct.toFixed(0)}%</span>
        <span className="text-gray-400">Total: {total.toFixed(1)} kN</span>
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

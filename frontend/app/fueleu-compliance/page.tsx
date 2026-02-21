'use client';

import { useState, useEffect } from 'react';
import Header from '@/components/Header';
import Card from '@/components/Card';
import {
  Fuel,
  TrendingDown,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Calendar,
  Target,
  Users,
  Plus,
  Trash2,
  DollarSign,
} from 'lucide-react';
import {
  AreaChart,
  Area,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
} from 'recharts';
import {
  apiClient,
  FuelEUFuelInfo,
  FuelEUCalculateResponse,
  FuelEUComplianceResponse,
  FuelEUPenaltyResponse,
  FuelEUPoolingResponse,
  FuelEUPoolingVessel,
  FuelEUProjectResponse,
  FuelEULimitsResponse,
} from '@/lib/api';

type TabType = 'calculator' | 'compliance' | 'projection' | 'pooling';

export default function FuelEUCompliancePage() {
  const [activeTab, setActiveTab] = useState<TabType>('calculator');
  const [fuelTypes, setFuelTypes] = useState<FuelEUFuelInfo[]>([]);
  const [loading, setLoading] = useState(true);

  // Calculator state
  const [calcResult, setCalcResult] = useState<FuelEUCalculateResponse | null>(null);
  const [calcForm, setCalcForm] = useState({
    hfo: 3000,
    vlsfo: 2000,
    lng: 0,
    mgo: 500,
    year: 2025,
  });

  // Compliance state
  const [compResult, setCompResult] = useState<FuelEUComplianceResponse | null>(null);
  const [penResult, setPenResult] = useState<FuelEUPenaltyResponse | null>(null);
  const [compForm, setCompForm] = useState({
    hfo: 3000,
    vlsfo: 2000,
    lng: 0,
    mgo: 500,
    year: 2025,
    consecutiveDeficitYears: 0,
  });

  // Projection state
  const [projResult, setProjResult] = useState<FuelEUProjectResponse | null>(null);
  const [limitsData, setLimitsData] = useState<FuelEULimitsResponse | null>(null);
  const [projForm, setProjForm] = useState({
    hfo: 3000,
    vlsfo: 2000,
    lng: 0,
    mgo: 500,
    startYear: 2025,
    endYear: 2050,
    efficiencyImprovement: 0,
  });

  // Pooling state
  const [poolResult, setPoolResult] = useState<FuelEUPoolingResponse | null>(null);
  const [poolYear, setPoolYear] = useState(2025);
  const [poolVessels, setPoolVessels] = useState<FuelEUPoolingVessel[]>([
    { name: 'Vessel A', fuel_mt: { hfo: 5000 } },
    { name: 'Vessel B', fuel_mt: { lng: 4000 } },
  ]);

  useEffect(() => {
    loadReferenceData();
  }, []);

  const loadReferenceData = async () => {
    try {
      const data = await apiClient.getFuelEUFuelTypes();
      setFuelTypes(data.fuel_types);
    } catch (error) {
      console.error('Failed to load FuelEU fuel types:', error);
    } finally {
      setLoading(false);
    }
  };

  const buildFuelDict = (form: { hfo: number; vlsfo: number; lng: number; mgo: number }) => {
    const d: Record<string, number> = {};
    if (form.hfo > 0) d.hfo = form.hfo;
    if (form.vlsfo > 0) d.vlsfo = form.vlsfo;
    if (form.lng > 0) d.lng = form.lng;
    if (form.mgo > 0) d.mgo = form.mgo;
    return d;
  };

  const calculateGHG = async () => {
    try {
      const result = await apiClient.calculateFuelEU({
        fuel_consumption_mt: buildFuelDict(calcForm),
        year: calcForm.year,
      });
      setCalcResult(result);
    } catch (error) {
      console.error('FuelEU calculation failed:', error);
    }
  };

  const calculateCompliance = async () => {
    try {
      const [comp, pen] = await Promise.all([
        apiClient.calculateFuelEUCompliance({
          fuel_consumption_mt: buildFuelDict(compForm),
          year: compForm.year,
        }),
        apiClient.calculateFuelEUPenalty({
          fuel_consumption_mt: buildFuelDict(compForm),
          year: compForm.year,
          consecutive_deficit_years: compForm.consecutiveDeficitYears,
        }),
      ]);
      setCompResult(comp);
      setPenResult(pen);
    } catch (error) {
      console.error('FuelEU compliance failed:', error);
    }
  };

  const projectCompliance = async () => {
    try {
      const [proj, limits] = await Promise.all([
        apiClient.projectFuelEU({
          fuel_consumption_mt: buildFuelDict(projForm),
          start_year: projForm.startYear,
          end_year: projForm.endYear,
          annual_efficiency_improvement_pct: projForm.efficiencyImprovement,
        }),
        apiClient.getFuelEULimits(),
      ]);
      setProjResult(proj);
      setLimitsData(limits);
    } catch (error) {
      console.error('FuelEU projection failed:', error);
    }
  };

  const simulatePooling = async () => {
    try {
      const result = await apiClient.simulateFuelEUPooling({
        vessels: poolVessels,
        year: poolYear,
      });
      setPoolResult(result);
    } catch (error) {
      console.error('FuelEU pooling failed:', error);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-maritime">
      <Header />

      <main className="container mx-auto px-6 pt-20 pb-12">
        {/* Hero Section */}
        <div className="mb-8">
          <div className="flex items-center space-x-3 mb-3">
            <Fuel className="w-10 h-10 text-emerald-400" />
            <h2 className="text-4xl font-bold text-white">FuelEU Maritime</h2>
          </div>
          <p className="text-gray-300 text-lg">
            EU 2023/1805 GHG intensity calculator, compliance assessment, and fleet pooling simulation
          </p>
        </div>

        {/* Tabs */}
        <div className="flex flex-wrap gap-2 mb-6">
          <TabButton
            active={activeTab === 'calculator'}
            onClick={() => setActiveTab('calculator')}
            icon={<Target className="w-4 h-4" />}
          >
            GHG Calculator
          </TabButton>
          <TabButton
            active={activeTab === 'compliance'}
            onClick={() => setActiveTab('compliance')}
            icon={<CheckCircle className="w-4 h-4" />}
          >
            Compliance
          </TabButton>
          <TabButton
            active={activeTab === 'projection'}
            onClick={() => setActiveTab('projection')}
            icon={<Calendar className="w-4 h-4" />}
          >
            Projection
          </TabButton>
          <TabButton
            active={activeTab === 'pooling'}
            onClick={() => setActiveTab('pooling')}
            icon={<Users className="w-4 h-4" />}
          >
            Fleet Pooling
          </TabButton>
        </div>

        {loading ? (
          <LoadingSpinner />
        ) : (
          <>
            {activeTab === 'calculator' && (
              <CalculatorTab
                form={calcForm}
                setForm={setCalcForm}
                result={calcResult}
                onCalculate={calculateGHG}
              />
            )}
            {activeTab === 'compliance' && (
              <ComplianceTab
                form={compForm}
                setForm={setCompForm}
                compResult={compResult}
                penResult={penResult}
                onCalculate={calculateCompliance}
              />
            )}
            {activeTab === 'projection' && (
              <ProjectionTab
                form={projForm}
                setForm={setProjForm}
                result={projResult}
                limits={limitsData}
                onProject={projectCompliance}
              />
            )}
            {activeTab === 'pooling' && (
              <PoolingTab
                vessels={poolVessels}
                setVessels={setPoolVessels}
                year={poolYear}
                setYear={setPoolYear}
                result={poolResult}
                onSimulate={simulatePooling}
              />
            )}
          </>
        )}
      </main>
    </div>
  );
}

// ============================================================================
// Shared Components
// ============================================================================

function TabButton({
  active,
  onClick,
  icon,
  children,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
        active
          ? 'bg-emerald-600 text-white'
          : 'bg-white/5 text-gray-300 hover:bg-white/10'
      }`}
    >
      {icon}
      <span className="font-medium">{children}</span>
    </button>
  );
}

function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center h-64">
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-emerald-400" />
    </div>
  );
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-sm text-gray-400">{label}</span>
      <span className="text-sm font-medium text-white">{value}</span>
    </div>
  );
}

function FuelInputs({
  form,
  setForm,
}: {
  form: { hfo: number; vlsfo: number; lng: number; mgo: number };
  setForm: (f: any) => void;
}) {
  return (
    <div className="grid grid-cols-2 gap-4">
      <div>
        <label className="block text-sm text-gray-400 mb-1">HFO (MT)</label>
        <input
          type="number"
          value={form.hfo}
          onChange={(e) => setForm({ ...form, hfo: Number(e.target.value) })}
          className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white"
        />
      </div>
      <div>
        <label className="block text-sm text-gray-400 mb-1">VLSFO (MT)</label>
        <input
          type="number"
          value={form.vlsfo}
          onChange={(e) => setForm({ ...form, vlsfo: Number(e.target.value) })}
          className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white"
        />
      </div>
      <div>
        <label className="block text-sm text-gray-400 mb-1">LNG (MT)</label>
        <input
          type="number"
          value={form.lng}
          onChange={(e) => setForm({ ...form, lng: Number(e.target.value) })}
          className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white"
        />
      </div>
      <div>
        <label className="block text-sm text-gray-400 mb-1">MGO (MT)</label>
        <input
          type="number"
          value={form.mgo}
          onChange={(e) => setForm({ ...form, mgo: Number(e.target.value) })}
          className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white"
        />
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const isCompliant = status === 'compliant';
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-medium ${
        isCompliant ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
      }`}
    >
      {isCompliant ? <CheckCircle className="w-3.5 h-3.5" /> : <XCircle className="w-3.5 h-3.5" />}
      {isCompliant ? 'Compliant' : 'Deficit'}
    </span>
  );
}

// ============================================================================
// Calculator Tab
// ============================================================================

function CalculatorTab({
  form,
  setForm,
  result,
  onCalculate,
}: {
  form: any;
  setForm: (f: any) => void;
  result: FuelEUCalculateResponse | null;
  onCalculate: () => void;
}) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Input Form */}
      <Card title="Fuel Consumption" icon={<Fuel className="w-5 h-5" />}>
        <div className="space-y-4">
          <FuelInputs form={form} setForm={setForm} />
          <div>
            <label className="block text-sm text-gray-400 mb-1">Year</label>
            <select
              value={form.year}
              onChange={(e) => setForm({ ...form, year: Number(e.target.value) })}
              className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white"
            >
              {Array.from({ length: 26 }, (_, i) => 2025 + i).map((y) => (
                <option key={y} value={y}>{y}</option>
              ))}
            </select>
          </div>
          <button
            onClick={onCalculate}
            className="w-full py-3 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold rounded-lg transition-colors"
          >
            Calculate GHG Intensity
          </button>
        </div>
      </Card>

      {/* Results */}
      {result && (
        <div className="space-y-6">
          <Card>
            <div className="text-center">
              <p className="text-sm text-gray-400 mb-2">Well-to-Wake GHG Intensity</p>
              <p className="text-5xl font-bold text-emerald-400 mb-1">
                {result.ghg_intensity.toFixed(2)}
              </p>
              <p className="text-gray-400 text-sm">gCO2eq/MJ</p>
              <div className="grid grid-cols-2 gap-4 mt-6">
                <div className="p-3 bg-maritime-dark rounded-lg">
                  <p className="text-xs text-gray-400">Total Energy</p>
                  <p className="text-lg font-bold text-white">{(result.total_energy_mj / 1e9).toFixed(2)} GJ</p>
                </div>
                <div className="p-3 bg-maritime-dark rounded-lg">
                  <p className="text-xs text-gray-400">Total CO2eq</p>
                  <p className="text-lg font-bold text-white">{(result.total_co2eq_g / 1e6).toFixed(0)} MT</p>
                </div>
              </div>
            </div>
          </Card>

          {result.fuel_breakdown.length > 0 && (
            <Card title="Fuel Breakdown" icon={<TrendingDown className="w-5 h-5" />}>
              <div className="space-y-3">
                {result.fuel_breakdown.map((fb) => (
                  <div key={fb.fuel_type} className="p-3 bg-maritime-dark rounded-lg">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium text-white">{fb.fuel_type.toUpperCase()}</span>
                      <span className="text-sm text-emerald-400">{fb.wtw_intensity.toFixed(2)} gCO2eq/MJ</span>
                    </div>
                    <div className="grid grid-cols-3 gap-2 text-xs text-gray-400">
                      <div>
                        <p>Mass</p>
                        <p className="text-white">{fb.mass_mt.toLocaleString()} MT</p>
                      </div>
                      <div>
                        <p>WtT</p>
                        <p className="text-white">{(fb.wtt_gco2eq / 1e6).toFixed(0)} MT CO2</p>
                      </div>
                      <div>
                        <p>TtW</p>
                        <p className="text-white">{(fb.ttw_gco2eq / 1e6).toFixed(0)} MT CO2</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Compliance Tab
// ============================================================================

function ComplianceTab({
  form,
  setForm,
  compResult,
  penResult,
  onCalculate,
}: {
  form: any;
  setForm: (f: any) => void;
  compResult: FuelEUComplianceResponse | null;
  penResult: FuelEUPenaltyResponse | null;
  onCalculate: () => void;
}) {
  return (
    <div className="space-y-6">
      {/* Input Form */}
      <Card title="Compliance Assessment" icon={<CheckCircle className="w-5 h-5" />}>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="lg:col-span-2">
            <FuelInputs form={form} setForm={setForm} />
          </div>
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">Year</label>
              <select
                value={form.year}
                onChange={(e) => setForm({ ...form, year: Number(e.target.value) })}
                className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white"
              >
                {Array.from({ length: 26 }, (_, i) => 2025 + i).map((y) => (
                  <option key={y} value={y}>{y}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">Consecutive Deficit Years</label>
              <input
                type="number"
                min="0"
                max="20"
                value={form.consecutiveDeficitYears}
                onChange={(e) => setForm({ ...form, consecutiveDeficitYears: Number(e.target.value) })}
                className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white"
              />
            </div>
          </div>
          <div className="flex items-end">
            <button
              onClick={onCalculate}
              className="w-full py-3 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold rounded-lg transition-colors"
            >
              Assess Compliance
            </button>
          </div>
        </div>
      </Card>

      {/* Results */}
      {compResult && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card>
            <div className="text-center">
              <p className="text-xs text-gray-400 mb-1">Status</p>
              <div className="mt-2"><StatusBadge status={compResult.status} /></div>
            </div>
          </Card>
          <Card>
            <div className="text-center">
              <p className="text-xs text-gray-400 mb-1">GHG Intensity</p>
              <p className="text-2xl font-bold text-white">{compResult.ghg_intensity.toFixed(2)}</p>
              <p className="text-xs text-gray-500">gCO2eq/MJ</p>
            </div>
          </Card>
          <Card>
            <div className="text-center">
              <p className="text-xs text-gray-400 mb-1">Limit ({compResult.year})</p>
              <p className="text-2xl font-bold text-emerald-400">{compResult.ghg_limit.toFixed(2)}</p>
              <p className="text-xs text-gray-500">-{compResult.reduction_target_pct}% from baseline</p>
            </div>
          </Card>
          <Card>
            <div className="text-center">
              <p className="text-xs text-gray-400 mb-1">Balance</p>
              <p className={`text-2xl font-bold ${compResult.compliance_balance_gco2eq >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {compResult.compliance_balance_gco2eq >= 0 ? '+' : ''}{(compResult.compliance_balance_gco2eq / 1e9).toFixed(2)}
              </p>
              <p className="text-xs text-gray-500">GtCO2eq</p>
            </div>
          </Card>
        </div>
      )}

      {/* Penalty Section */}
      {penResult && penResult.penalty_eur > 0 && (
        <Card title="Penalty Exposure" icon={<DollarSign className="w-5 h-5" />}>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg text-center">
              <p className="text-xs text-gray-400 mb-1">Penalty</p>
              <p className="text-2xl font-bold text-red-400">{penResult.penalty_eur.toLocaleString(undefined, { maximumFractionDigits: 0 })}</p>
              <p className="text-xs text-gray-500">EUR</p>
            </div>
            <div className="p-4 bg-maritime-dark rounded-lg text-center">
              <p className="text-xs text-gray-400 mb-1">Per MT Fuel</p>
              <p className="text-xl font-bold text-white">{penResult.penalty_per_mt_fuel.toFixed(2)}</p>
              <p className="text-xs text-gray-500">EUR/MT</p>
            </div>
            <div className="p-4 bg-maritime-dark rounded-lg text-center">
              <p className="text-xs text-gray-400 mb-1">VLSFO Equivalent</p>
              <p className="text-xl font-bold text-white">{penResult.vlsfo_equivalent_mt.toFixed(1)}</p>
              <p className="text-xs text-gray-500">MT</p>
            </div>
            <div className="p-4 bg-maritime-dark rounded-lg text-center">
              <p className="text-xs text-gray-400 mb-1">Non-Compliant Energy</p>
              <p className="text-xl font-bold text-white">{(penResult.non_compliant_energy_mj / 1e9).toFixed(2)}</p>
              <p className="text-xs text-gray-500">GJ</p>
            </div>
          </div>
        </Card>
      )}

      {penResult && penResult.penalty_eur === 0 && compResult && (
        <Card>
          <div className="flex items-center justify-center gap-3 py-4">
            <CheckCircle className="w-6 h-6 text-green-400" />
            <p className="text-lg text-green-400 font-medium">No penalty — vessel is compliant for {compResult.year}</p>
          </div>
        </Card>
      )}
    </div>
  );
}

// ============================================================================
// Projection Tab
// ============================================================================

function ProjectionTab({
  form,
  setForm,
  result,
  limits,
  onProject,
}: {
  form: any;
  setForm: (f: any) => void;
  result: FuelEUProjectResponse | null;
  limits: FuelEULimitsResponse | null;
  onProject: () => void;
}) {
  // Build chart data
  const chartData = result?.projections.map((p) => ({
    year: p.year,
    intensity: p.ghg_intensity,
    limit: p.ghg_limit,
    penalty: p.penalty_eur,
  })) ?? [];

  return (
    <div className="space-y-6">
      {/* Input Form */}
      <Card title="Projection Parameters" icon={<Calendar className="w-5 h-5" />}>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4">
          <div>
            <label className="block text-sm text-gray-400 mb-1">HFO (MT)</label>
            <input
              type="number"
              value={form.hfo}
              onChange={(e) => setForm({ ...form, hfo: Number(e.target.value) })}
              className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">VLSFO (MT)</label>
            <input
              type="number"
              value={form.vlsfo}
              onChange={(e) => setForm({ ...form, vlsfo: Number(e.target.value) })}
              className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">LNG (MT)</label>
            <input
              type="number"
              value={form.lng}
              onChange={(e) => setForm({ ...form, lng: Number(e.target.value) })}
              className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">MGO (MT)</label>
            <input
              type="number"
              value={form.mgo}
              onChange={(e) => setForm({ ...form, mgo: Number(e.target.value) })}
              className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">Start Year</label>
            <select
              value={form.startYear}
              onChange={(e) => setForm({ ...form, startYear: Number(e.target.value) })}
              className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white"
            >
              {Array.from({ length: 26 }, (_, i) => 2025 + i).map((y) => (
                <option key={y} value={y}>{y}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">Efficiency (%/yr)</label>
            <input
              type="number"
              step="0.5"
              min="0"
              max="10"
              value={form.efficiencyImprovement}
              onChange={(e) => setForm({ ...form, efficiencyImprovement: Number(e.target.value) })}
              className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white"
            />
          </div>
          <div className="flex items-end">
            <button
              onClick={onProject}
              className="w-full py-2 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold rounded-lg transition-colors"
            >
              Project
            </button>
          </div>
        </div>
      </Card>

      {/* Projection Chart */}
      {result && chartData.length > 0 && (
        <>
          <Card title="GHG Intensity vs. Regulatory Limit">
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="year" stroke="#9ca3af" tick={{ fontSize: 12 }} />
                  <YAxis stroke="#9ca3af" tick={{ fontSize: 12 }} domain={['auto', 'auto']} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                    labelStyle={{ color: '#fff' }}
                    formatter={(value: number, name: string) => {
                      const labels: Record<string, string> = {
                        intensity: 'GHG Intensity',
                        limit: 'Regulatory Limit',
                      };
                      return [value?.toFixed(2) ?? '-', labels[name] ?? name];
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="limit"
                    fill="rgba(16,185,129,0.1)"
                    stroke="rgba(16,185,129,0.6)"
                    strokeDasharray="6 3"
                    name="limit"
                  />
                  <Line
                    type="monotone"
                    dataKey="intensity"
                    stroke="#f472b6"
                    strokeWidth={2.5}
                    dot={{ r: 3, fill: '#f472b6' }}
                    name="intensity"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <div className="flex gap-4 mt-3 px-2">
              <span className="flex items-center gap-1.5 text-xs text-gray-400">
                <span className="w-3 h-0.5 bg-pink-400 inline-block" /> Vessel GHG Intensity
              </span>
              <span className="flex items-center gap-1.5 text-xs text-gray-400">
                <span className="w-3 h-3 rounded bg-emerald-500/20" /> Regulatory Limit
              </span>
            </div>
          </Card>

          {/* Penalty Chart */}
          {result.projections.some(p => p.penalty_eur > 0) && (
            <Card title="Projected Annual Penalty">
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis dataKey="year" stroke="#9ca3af" tick={{ fontSize: 12 }} />
                    <YAxis stroke="#9ca3af" tick={{ fontSize: 12 }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                      labelStyle={{ color: '#fff' }}
                      formatter={(value: number) => [`${value.toLocaleString()} EUR`, 'Penalty']}
                    />
                    <Bar dataKey="penalty" radius={[4, 4, 0, 0]}>
                      {chartData.map((entry, idx) => (
                        <Cell key={idx} fill={entry.penalty > 0 ? 'rgba(239,68,68,0.7)' : 'rgba(16,185,129,0.3)'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>
          )}

          {/* Table */}
          <Card title="Year-by-Year Projection">
            <div className="overflow-x-auto max-h-96">
              <table className="w-full">
                <thead className="sticky top-0 bg-maritime-navy">
                  <tr className="border-b border-white/10">
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Year</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">GHG Intensity</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Limit</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Reduction %</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Status</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Penalty (EUR)</th>
                  </tr>
                </thead>
                <tbody>
                  {result.projections.map((p) => (
                    <tr key={p.year} className="border-b border-white/5 hover:bg-white/5">
                      <td className="py-3 px-4 text-white font-medium">{p.year}</td>
                      <td className="py-3 px-4 text-white">{p.ghg_intensity.toFixed(2)}</td>
                      <td className="py-3 px-4 text-gray-300">{p.ghg_limit.toFixed(2)}</td>
                      <td className="py-3 px-4 text-gray-300">{p.reduction_target_pct.toFixed(1)}%</td>
                      <td className="py-3 px-4"><StatusBadge status={p.status} /></td>
                      <td className="py-3 px-4">
                        {p.penalty_eur > 0 ? (
                          <span className="text-red-400">{p.penalty_eur.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
                        ) : (
                          <span className="text-green-400">0</span>
                        )}
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
// Pooling Tab
// ============================================================================

function PoolingTab({
  vessels,
  setVessels,
  year,
  setYear,
  result,
  onSimulate,
}: {
  vessels: FuelEUPoolingVessel[];
  setVessels: (v: FuelEUPoolingVessel[]) => void;
  year: number;
  setYear: (y: number) => void;
  result: FuelEUPoolingResponse | null;
  onSimulate: () => void;
}) {
  const addVessel = () => {
    if (vessels.length >= 20) return;
    setVessels([
      ...vessels,
      { name: `Vessel ${String.fromCharCode(65 + vessels.length)}`, fuel_mt: { vlsfo: 3000 } },
    ]);
  };

  const removeVessel = (idx: number) => {
    setVessels(vessels.filter((_, i) => i !== idx));
  };

  const updateVessel = (idx: number, field: string, value: any) => {
    const updated = [...vessels];
    if (['hfo', 'vlsfo', 'lng', 'mgo'].includes(field)) {
      const fuel = { ...updated[idx].fuel_mt };
      if (Number(value) > 0) {
        fuel[field] = Number(value);
      } else {
        delete fuel[field];
      }
      updated[idx] = { ...updated[idx], fuel_mt: fuel };
    } else {
      updated[idx] = { ...updated[idx], [field]: value };
    }
    setVessels(updated);
  };

  return (
    <div className="space-y-6">
      {/* Vessel Input Table */}
      <Card title="Fleet Vessels" icon={<Users className="w-5 h-5" />}>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-white/10">
                <th className="text-left py-2 px-3 text-xs font-medium text-gray-400">Name</th>
                <th className="text-left py-2 px-3 text-xs font-medium text-gray-400">HFO (MT)</th>
                <th className="text-left py-2 px-3 text-xs font-medium text-gray-400">VLSFO (MT)</th>
                <th className="text-left py-2 px-3 text-xs font-medium text-gray-400">LNG (MT)</th>
                <th className="text-left py-2 px-3 text-xs font-medium text-gray-400">MGO (MT)</th>
                <th className="py-2 px-3" />
              </tr>
            </thead>
            <tbody>
              {vessels.map((v, idx) => (
                <tr key={idx} className="border-b border-white/5">
                  <td className="py-2 px-3">
                    <input
                      type="text"
                      value={v.name}
                      onChange={(e) => updateVessel(idx, 'name', e.target.value)}
                      className="w-full px-2 py-1 bg-maritime-dark border border-white/10 rounded text-white text-sm"
                    />
                  </td>
                  <td className="py-2 px-3">
                    <input
                      type="number"
                      value={v.fuel_mt.hfo ?? 0}
                      onChange={(e) => updateVessel(idx, 'hfo', e.target.value)}
                      className="w-24 px-2 py-1 bg-maritime-dark border border-white/10 rounded text-white text-sm"
                    />
                  </td>
                  <td className="py-2 px-3">
                    <input
                      type="number"
                      value={v.fuel_mt.vlsfo ?? 0}
                      onChange={(e) => updateVessel(idx, 'vlsfo', e.target.value)}
                      className="w-24 px-2 py-1 bg-maritime-dark border border-white/10 rounded text-white text-sm"
                    />
                  </td>
                  <td className="py-2 px-3">
                    <input
                      type="number"
                      value={v.fuel_mt.lng ?? 0}
                      onChange={(e) => updateVessel(idx, 'lng', e.target.value)}
                      className="w-24 px-2 py-1 bg-maritime-dark border border-white/10 rounded text-white text-sm"
                    />
                  </td>
                  <td className="py-2 px-3">
                    <input
                      type="number"
                      value={v.fuel_mt.mgo ?? 0}
                      onChange={(e) => updateVessel(idx, 'mgo', e.target.value)}
                      className="w-24 px-2 py-1 bg-maritime-dark border border-white/10 rounded text-white text-sm"
                    />
                  </td>
                  <td className="py-2 px-3">
                    <button
                      onClick={() => removeVessel(idx)}
                      className="text-red-400 hover:text-red-300"
                      title="Remove vessel"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="flex items-center gap-3 mt-4">
          <button
            onClick={addVessel}
            disabled={vessels.length >= 20}
            className="flex items-center gap-1.5 px-4 py-2 bg-white/5 hover:bg-white/10 text-gray-300 rounded-lg transition-colors disabled:opacity-40"
          >
            <Plus className="w-4 h-4" /> Add Vessel
          </button>
          <div>
            <select
              value={year}
              onChange={(e) => setYear(Number(e.target.value))}
              className="px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white text-sm"
            >
              {Array.from({ length: 26 }, (_, i) => 2025 + i).map((y) => (
                <option key={y} value={y}>{y}</option>
              ))}
            </select>
          </div>
          <button
            onClick={onSimulate}
            disabled={vessels.length === 0}
            className="px-6 py-2 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold rounded-lg transition-colors disabled:opacity-40"
          >
            Simulate Pooling
          </button>
        </div>
      </Card>

      {/* Results */}
      {result && (
        <>
          {/* Fleet Summary */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card>
              <div className="text-center">
                <p className="text-xs text-gray-400 mb-1">Fleet Status</p>
                <div className="mt-2"><StatusBadge status={result.status} /></div>
              </div>
            </Card>
            <Card>
              <div className="text-center">
                <p className="text-xs text-gray-400 mb-1">Fleet GHG Intensity</p>
                <p className="text-2xl font-bold text-white">{result.fleet_ghg_intensity.toFixed(2)}</p>
                <p className="text-xs text-gray-500">gCO2eq/MJ</p>
              </div>
            </Card>
            <Card>
              <div className="text-center">
                <p className="text-xs text-gray-400 mb-1">Fleet Energy</p>
                <p className="text-2xl font-bold text-white">{(result.fleet_total_energy_mj / 1e9).toFixed(2)}</p>
                <p className="text-xs text-gray-500">GJ</p>
              </div>
            </Card>
            <Card>
              <div className="text-center">
                <p className="text-xs text-gray-400 mb-1">Fleet Balance</p>
                <p className={`text-2xl font-bold ${result.fleet_balance_gco2eq >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {result.fleet_balance_gco2eq >= 0 ? '+' : ''}{(result.fleet_balance_gco2eq / 1e9).toFixed(2)}
                </p>
                <p className="text-xs text-gray-500">GtCO2eq</p>
              </div>
            </Card>
          </div>

          {/* Per-Vessel Results */}
          <Card title="Per-Vessel Results">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-white/10">
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Vessel</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">GHG Intensity</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Energy (GJ)</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Balance</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {result.per_vessel.map((v) => (
                    <tr key={v.name} className="border-b border-white/5 hover:bg-white/5">
                      <td className="py-3 px-4 text-white font-medium">{v.name}</td>
                      <td className="py-3 px-4 text-white">{v.ghg_intensity.toFixed(2)}</td>
                      <td className="py-3 px-4 text-gray-300">{(v.total_energy_mj / 1e9).toFixed(2)}</td>
                      <td className="py-3 px-4">
                        <span className={v.individual_balance_gco2eq >= 0 ? 'text-green-400' : 'text-red-400'}>
                          {v.individual_balance_gco2eq >= 0 ? '+' : ''}{(v.individual_balance_gco2eq / 1e9).toFixed(3)}
                        </span>
                      </td>
                      <td className="py-3 px-4"><StatusBadge status={v.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Pooling benefit message */}
          {(() => {
            const deficitVessels = result.per_vessel.filter(v => v.status === 'deficit');
            if (deficitVessels.length > 0 && result.status === 'compliant') {
              return (
                <Card>
                  <div className="flex items-center justify-center gap-3 py-4">
                    <TrendingUp className="w-6 h-6 text-emerald-400" />
                    <p className="text-lg text-emerald-400 font-medium">
                      Pooling benefit: {deficitVessels.length} deficit vessel{deficitVessels.length > 1 ? 's' : ''} offset by fleet surplus
                    </p>
                  </div>
                </Card>
              );
            }
            return null;
          })()}
        </>
      )}
    </div>
  );
}

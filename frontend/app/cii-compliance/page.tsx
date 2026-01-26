'use client';

import { useState, useEffect } from 'react';
import Header from '@/components/Header';
import Card from '@/components/Card';
import {
  Leaf,
  TrendingDown,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Calendar,
  Target,
  Fuel,
  Ship,
} from 'lucide-react';
import {
  apiClient,
  CIICalculationResponse,
  CIIProjectionResponse,
  CIIReductionResponse,
  VesselTypeInfo,
  FuelTypeInfo,
} from '@/lib/api';

type TabType = 'calculator' | 'projection' | 'reduction';

export default function CIICompliancePage() {
  const [activeTab, setActiveTab] = useState<TabType>('calculator');
  const [vesselTypes, setVesselTypes] = useState<VesselTypeInfo[]>([]);
  const [fuelTypes, setFuelTypes] = useState<FuelTypeInfo[]>([]);
  const [loading, setLoading] = useState(true);

  // Calculator state
  const [calcResult, setCalcResult] = useState<CIICalculationResponse | null>(null);
  const [calcForm, setCalcForm] = useState({
    vlsfo: 5000,
    mgo: 500,
    distance: 50000,
    dwt: 49000,
    vesselType: 'tanker',
    year: 2024,
  });

  // Projection state
  const [projResult, setProjResult] = useState<CIIProjectionResponse | null>(null);
  const [projForm, setProjForm] = useState({
    annualFuel: 7000,
    annualDistance: 60000,
    dwt: 49000,
    vesselType: 'tanker',
    startYear: 2024,
    endYear: 2030,
    efficiencyImprovement: 0,
  });

  // Reduction state
  const [reductionResult, setReductionResult] = useState<CIIReductionResponse | null>(null);
  const [reductionForm, setReductionForm] = useState({
    currentFuel: 7000,
    currentDistance: 60000,
    dwt: 49000,
    vesselType: 'tanker',
    targetRating: 'C',
    targetYear: 2026,
  });

  useEffect(() => {
    loadReferenceData();
  }, []);

  const loadReferenceData = async () => {
    try {
      const [vesselData, fuelData] = await Promise.all([
        apiClient.getVesselTypes(),
        apiClient.getFuelTypes(),
      ]);
      setVesselTypes(vesselData.vessel_types);
      setFuelTypes(fuelData.fuel_types);
    } catch (error) {
      console.error('Failed to load reference data:', error);
    } finally {
      setLoading(false);
    }
  };

  const calculateCII = async () => {
    try {
      const result = await apiClient.calculateCII({
        fuel_consumption_mt: {
          vlsfo: calcForm.vlsfo,
          mgo: calcForm.mgo,
        },
        total_distance_nm: calcForm.distance,
        dwt: calcForm.dwt,
        vessel_type: calcForm.vesselType,
        year: calcForm.year,
      });
      setCalcResult(result);
    } catch (error) {
      console.error('CII calculation failed:', error);
    }
  };

  const projectCII = async () => {
    try {
      const result = await apiClient.projectCII({
        annual_fuel_mt: { vlsfo: projForm.annualFuel },
        annual_distance_nm: projForm.annualDistance,
        dwt: projForm.dwt,
        vessel_type: projForm.vesselType,
        start_year: projForm.startYear,
        end_year: projForm.endYear,
        fuel_efficiency_improvement_pct: projForm.efficiencyImprovement,
      });
      setProjResult(result);
    } catch (error) {
      console.error('CII projection failed:', error);
    }
  };

  const calculateReduction = async () => {
    try {
      const result = await apiClient.calculateCIIReduction({
        current_fuel_mt: { vlsfo: reductionForm.currentFuel },
        current_distance_nm: reductionForm.currentDistance,
        dwt: reductionForm.dwt,
        vessel_type: reductionForm.vesselType,
        target_rating: reductionForm.targetRating,
        target_year: reductionForm.targetYear,
      });
      setReductionResult(result);
    } catch (error) {
      console.error('CII reduction calculation failed:', error);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-maritime">
      <Header />

      <main className="container mx-auto px-4 pt-24 pb-12">
        {/* Hero Section */}
        <div className="mb-8">
          <div className="flex items-center space-x-3 mb-3">
            <Leaf className="w-10 h-10 text-green-400" />
            <h2 className="text-4xl font-bold text-white">CII Compliance</h2>
          </div>
          <p className="text-gray-300 text-lg">
            Carbon Intensity Indicator calculator and projections for IMO 2023 regulations
          </p>
        </div>

        {/* Tabs */}
        <div className="flex space-x-2 mb-6">
          <TabButton
            active={activeTab === 'calculator'}
            onClick={() => setActiveTab('calculator')}
            icon={<Target className="w-4 h-4" />}
          >
            Rating Calculator
          </TabButton>
          <TabButton
            active={activeTab === 'projection'}
            onClick={() => setActiveTab('projection')}
            icon={<Calendar className="w-4 h-4" />}
          >
            Future Projection
          </TabButton>
          <TabButton
            active={activeTab === 'reduction'}
            onClick={() => setActiveTab('reduction')}
            icon={<TrendingDown className="w-4 h-4" />}
          >
            Reduction Planner
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
                onCalculate={calculateCII}
                vesselTypes={vesselTypes}
              />
            )}
            {activeTab === 'projection' && (
              <ProjectionTab
                form={projForm}
                setForm={setProjForm}
                result={projResult}
                onProject={projectCII}
                vesselTypes={vesselTypes}
              />
            )}
            {activeTab === 'reduction' && (
              <ReductionTab
                form={reductionForm}
                setForm={setReductionForm}
                result={reductionResult}
                onCalculate={calculateReduction}
                vesselTypes={vesselTypes}
              />
            )}
          </>
        )}
      </main>
    </div>
  );
}

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
          ? 'bg-primary-500 text-white'
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
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-400" />
    </div>
  );
}

function CalculatorTab({
  form,
  setForm,
  result,
  onCalculate,
  vesselTypes,
}: {
  form: any;
  setForm: (f: any) => void;
  result: CIICalculationResponse | null;
  onCalculate: () => void;
  vesselTypes: VesselTypeInfo[];
}) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Input Form */}
      <Card title="Input Parameters" icon={<Fuel className="w-5 h-5" />}>
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
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
              <label className="block text-sm text-gray-400 mb-1">MGO (MT)</label>
              <input
                type="number"
                value={form.mgo}
                onChange={(e) => setForm({ ...form, mgo: Number(e.target.value) })}
                className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-1">Total Distance (nm)</label>
            <input
              type="number"
              value={form.distance}
              onChange={(e) => setForm({ ...form, distance: Number(e.target.value) })}
              className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">DWT</label>
              <input
                type="number"
                value={form.dwt}
                onChange={(e) => setForm({ ...form, dwt: Number(e.target.value) })}
                className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">Year</label>
              <select
                value={form.year}
                onChange={(e) => setForm({ ...form, year: Number(e.target.value) })}
                className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white"
              >
                {[2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030].map((y) => (
                  <option key={y} value={y}>{y}</option>
                ))}
              </select>
            </div>
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-1">Vessel Type</label>
            <select
              value={form.vesselType}
              onChange={(e) => setForm({ ...form, vesselType: e.target.value })}
              className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white"
            >
              {vesselTypes.map((vt) => (
                <option key={vt.id} value={vt.id}>{vt.name}</option>
              ))}
            </select>
          </div>

          <button
            onClick={onCalculate}
            className="w-full py-3 bg-primary-500 hover:bg-primary-600 text-white font-semibold rounded-lg transition-colors"
          >
            Calculate CII Rating
          </button>
        </div>
      </Card>

      {/* Results */}
      {result && (
        <div className="space-y-6">
          <RatingCard result={result} />
          <CIIDetailsCard result={result} />
        </div>
      )}
    </div>
  );
}

function RatingCard({ result }: { result: CIICalculationResponse }) {
  const ratingColors: Record<string, string> = {
    A: 'from-green-500 to-green-600',
    B: 'from-lime-500 to-lime-600',
    C: 'from-yellow-500 to-yellow-600',
    D: 'from-orange-500 to-orange-600',
    E: 'from-red-500 to-red-600',
  };

  const ratingIcons: Record<string, React.ReactNode> = {
    A: <CheckCircle className="w-8 h-8" />,
    B: <CheckCircle className="w-8 h-8" />,
    C: <AlertTriangle className="w-8 h-8" />,
    D: <XCircle className="w-8 h-8" />,
    E: <XCircle className="w-8 h-8" />,
  };

  return (
    <Card>
      <div className="text-center">
        <p className="text-sm text-gray-400 mb-2">CII Rating {result.year}</p>
        <div
          className={`inline-flex items-center justify-center w-24 h-24 rounded-full bg-gradient-to-br ${
            ratingColors[result.rating]
          } mb-4`}
        >
          <span className="text-5xl font-bold text-white">{result.rating}</span>
        </div>
        <div className="flex items-center justify-center space-x-2 mb-4">
          {ratingIcons[result.rating]}
          <span className="text-lg font-medium text-white">{result.compliance_status}</span>
        </div>
        <div className="grid grid-cols-2 gap-4 mt-6">
          <div className="p-3 bg-maritime-dark rounded-lg">
            <p className="text-xs text-gray-400">Attained CII</p>
            <p className="text-xl font-bold text-white">{result.attained_cii.toFixed(2)}</p>
          </div>
          <div className="p-3 bg-maritime-dark rounded-lg">
            <p className="text-xs text-gray-400">Required CII</p>
            <p className="text-xl font-bold text-white">{result.required_cii.toFixed(2)}</p>
          </div>
        </div>
      </div>
    </Card>
  );
}

function CIIDetailsCard({ result }: { result: CIICalculationResponse }) {
  return (
    <Card title="Details" icon={<Ship className="w-5 h-5" />}>
      <div className="space-y-3">
        <DetailRow label="Total CO2 Emissions" value={`${result.total_co2_mt.toLocaleString()} MT`} />
        <DetailRow label="Total Distance" value={`${result.total_distance_nm.toLocaleString()} nm`} />
        <DetailRow label="Capacity (DWT)" value={result.capacity.toLocaleString()} />
        <DetailRow label="Reduction Factor" value={`${result.reduction_factor}%`} />
        <div className="pt-3 border-t border-white/10">
          <p className="text-xs text-gray-400 mb-2">Rating Boundaries</p>
          <div className="grid grid-cols-5 gap-1">
            {['A', 'B', 'C', 'D', 'E'].map((r) => (
              <div
                key={r}
                className={`text-center py-1 rounded ${
                  result.rating === r ? 'bg-primary-500' : 'bg-white/5'
                }`}
              >
                <span className="text-xs font-semibold text-white">{r}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </Card>
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

function ProjectionTab({
  form,
  setForm,
  result,
  onProject,
  vesselTypes,
}: {
  form: any;
  setForm: (f: any) => void;
  result: CIIProjectionResponse | null;
  onProject: () => void;
  vesselTypes: VesselTypeInfo[];
}) {
  return (
    <div className="space-y-6">
      {/* Input Form */}
      <Card title="Projection Parameters" icon={<Calendar className="w-5 h-5" />}>
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4">
          <div>
            <label className="block text-sm text-gray-400 mb-1">Annual Fuel (MT)</label>
            <input
              type="number"
              value={form.annualFuel}
              onChange={(e) => setForm({ ...form, annualFuel: Number(e.target.value) })}
              className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">Annual Distance (nm)</label>
            <input
              type="number"
              value={form.annualDistance}
              onChange={(e) => setForm({ ...form, annualDistance: Number(e.target.value) })}
              className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">DWT</label>
            <input
              type="number"
              value={form.dwt}
              onChange={(e) => setForm({ ...form, dwt: Number(e.target.value) })}
              className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">Vessel Type</label>
            <select
              value={form.vesselType}
              onChange={(e) => setForm({ ...form, vesselType: e.target.value })}
              className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white"
            >
              {vesselTypes.map((vt) => (
                <option key={vt.id} value={vt.id}>{vt.name}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">Efficiency Gain (%/yr)</label>
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
              className="w-full py-2 bg-primary-500 hover:bg-primary-600 text-white font-semibold rounded-lg transition-colors"
            >
              Project
            </button>
          </div>
        </div>
      </Card>

      {/* Results */}
      {result && (
        <>
          {/* Summary */}
          <Card title="Projection Summary" icon={<AlertTriangle className="w-5 h-5" />}>
            <div className="p-4 bg-maritime-dark rounded-lg">
              <p className="text-lg text-white">{result.summary.recommendation}</p>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
                <div>
                  <p className="text-xs text-gray-400">Current Rating</p>
                  <p className="text-2xl font-bold text-white">{result.summary.current_rating}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-400">Final Rating ({form.endYear})</p>
                  <p className="text-2xl font-bold text-white">{result.summary.final_rating}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-400">Years to D Rating</p>
                  <p className="text-2xl font-bold text-orange-400">{result.summary.years_until_d_rating}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-400">Years to E Rating</p>
                  <p className="text-2xl font-bold text-red-400">{result.summary.years_until_e_rating}</p>
                </div>
              </div>
            </div>
          </Card>

          {/* Timeline */}
          <Card title="Year-by-Year Projection">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-white/10">
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Year</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Rating</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Attained CII</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Required CII</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Reduction %</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {result.projections.map((p) => (
                    <tr key={p.year} className="border-b border-white/5 hover:bg-white/5">
                      <td className="py-3 px-4 text-white font-medium">{p.year}</td>
                      <td className="py-3 px-4">
                        <RatingBadge rating={p.rating} />
                      </td>
                      <td className="py-3 px-4 text-white">{p.attained_cii.toFixed(2)}</td>
                      <td className="py-3 px-4 text-gray-300">{p.required_cii.toFixed(2)}</td>
                      <td className="py-3 px-4 text-gray-300">{p.reduction_factor}%</td>
                      <td className="py-3 px-4">
                        <StatusBadge status={p.status} />
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

function RatingBadge({ rating }: { rating: string }) {
  const colors: Record<string, string> = {
    A: 'bg-green-500/20 text-green-400',
    B: 'bg-lime-500/20 text-lime-400',
    C: 'bg-yellow-500/20 text-yellow-400',
    D: 'bg-orange-500/20 text-orange-400',
    E: 'bg-red-500/20 text-red-400',
  };
  return (
    <span className={`px-3 py-1 rounded-full text-sm font-bold ${colors[rating]}`}>
      {rating}
    </span>
  );
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    compliant: 'bg-green-500/20 text-green-400',
    at_risk: 'bg-yellow-500/20 text-yellow-400',
    non_compliant: 'bg-red-500/20 text-red-400',
  };
  const labels: Record<string, string> = {
    compliant: 'Compliant',
    at_risk: 'At Risk',
    non_compliant: 'Non-Compliant',
  };
  return (
    <span className={`px-3 py-1 rounded-full text-xs font-medium ${colors[status]}`}>
      {labels[status]}
    </span>
  );
}

function ReductionTab({
  form,
  setForm,
  result,
  onCalculate,
  vesselTypes,
}: {
  form: any;
  setForm: (f: any) => void;
  result: CIIReductionResponse | null;
  onCalculate: () => void;
  vesselTypes: VesselTypeInfo[];
}) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Input Form */}
      <Card title="Reduction Calculator" icon={<Target className="w-5 h-5" />}>
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">Current Annual Fuel (MT)</label>
              <input
                type="number"
                value={form.currentFuel}
                onChange={(e) => setForm({ ...form, currentFuel: Number(e.target.value) })}
                className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">Annual Distance (nm)</label>
              <input
                type="number"
                value={form.currentDistance}
                onChange={(e) => setForm({ ...form, currentDistance: Number(e.target.value) })}
                className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">DWT</label>
              <input
                type="number"
                value={form.dwt}
                onChange={(e) => setForm({ ...form, dwt: Number(e.target.value) })}
                className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">Vessel Type</label>
              <select
                value={form.vesselType}
                onChange={(e) => setForm({ ...form, vesselType: e.target.value })}
                className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white"
              >
                {vesselTypes.map((vt) => (
                  <option key={vt.id} value={vt.id}>{vt.name}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">Target Rating</label>
              <select
                value={form.targetRating}
                onChange={(e) => setForm({ ...form, targetRating: e.target.value })}
                className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white"
              >
                <option value="A">A - Superior</option>
                <option value="B">B - Good</option>
                <option value="C">C - Compliant</option>
                <option value="D">D - Needs Improvement</option>
              </select>
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">Target Year</label>
              <select
                value={form.targetYear}
                onChange={(e) => setForm({ ...form, targetYear: Number(e.target.value) })}
                className="w-full px-3 py-2 bg-maritime-dark border border-white/10 rounded-lg text-white"
              >
                {[2024, 2025, 2026, 2027, 2028, 2029, 2030].map((y) => (
                  <option key={y} value={y}>{y}</option>
                ))}
              </select>
            </div>
          </div>

          <button
            onClick={onCalculate}
            className="w-full py-3 bg-primary-500 hover:bg-primary-600 text-white font-semibold rounded-lg transition-colors"
          >
            Calculate Required Reduction
          </button>
        </div>
      </Card>

      {/* Results */}
      {result && (
        <div className="space-y-6">
          <Card>
            <div className="text-center">
              <p className="text-sm text-gray-400 mb-2">Required Fuel Reduction</p>
              <p className="text-5xl font-bold text-primary-400 mb-2">
                {result.reduction_needed_pct.toFixed(1)}%
              </p>
              <p className="text-gray-300">{result.message}</p>
            </div>
          </Card>

          <Card title="Analysis" icon={<TrendingDown className="w-5 h-5" />}>
            <div className="space-y-4">
              <div className="flex justify-between items-center p-3 bg-maritime-dark rounded-lg">
                <span className="text-gray-400">Current Rating</span>
                <RatingBadge rating={result.current_rating} />
              </div>
              <div className="flex justify-between items-center p-3 bg-maritime-dark rounded-lg">
                <span className="text-gray-400">Target Rating</span>
                <RatingBadge rating={result.target_rating} />
              </div>
              <div className="flex justify-between items-center p-3 bg-maritime-dark rounded-lg">
                <span className="text-gray-400">Current CII</span>
                <span className="text-white font-medium">{result.current_cii.toFixed(2)}</span>
              </div>
              <div className="flex justify-between items-center p-3 bg-maritime-dark rounded-lg">
                <span className="text-gray-400">Target CII</span>
                <span className="text-white font-medium">{result.target_cii.toFixed(2)}</span>
              </div>
              <div className="flex justify-between items-center p-3 bg-green-500/10 rounded-lg border border-green-500/20">
                <span className="text-green-400">Potential Fuel Savings</span>
                <span className="text-green-400 font-bold">{result.fuel_savings_mt.toLocaleString()} MT/year</span>
              </div>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}

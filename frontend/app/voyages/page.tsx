'use client';

import { useState, useEffect, useCallback } from 'react';
import Header from '@/components/Header';
import Card from '@/components/Card';
import { StatCard } from '@/components/Card';
import {
  BookOpen, Trash2, Download, ChevronLeft, ChevronRight, Search,
  Anchor, Ship, Fuel, Clock, Navigation, Wind, Waves, FileText,
  Upload, MapPin, Map,
} from 'lucide-react';
import Link from 'next/link';
import RouteImport from '@/components/RouteImport';
import { useVoyage } from '@/components/VoyageContext';
import {
  apiClient,
  Position,
  VoyageSummary,
  VoyageDetail,
  VoyageListResponse,
  NoonReportEntry,
  NoonReportsResponse,
  DepartureReportData,
  ArrivalReportData,
  VoyageReportsResponse,
} from '@/lib/api';
import { isDemoUser } from '@/lib/demoMode';

type VoyageTab = 'setup' | 'history' | 'detail' | 'reports';

export default function VoyagesPage() {
  const [activeTab, setActiveTab] = useState<VoyageTab>('setup');
  const [selectedVoyageId, setSelectedVoyageId] = useState<string | null>(null);

  const selectVoyage = (id: string) => {
    setSelectedVoyageId(id);
    setActiveTab('detail');
  };

  return (
    <div className="min-h-screen bg-gradient-maritime">
      <Header />

      <main className="container mx-auto px-6 pt-20 pb-12">
        {/* Tab bar */}
        <div className="flex space-x-1 mb-6 bg-maritime-medium/50 backdrop-blur-sm rounded-lg p-1 max-w-xl">
          <TabButton label="Route Setup" active={activeTab === 'setup'} onClick={() => setActiveTab('setup')} />
          <TabButton label="History" active={activeTab === 'history'} onClick={() => setActiveTab('history')} />
          <TabButton
            label="Detail"
            active={activeTab === 'detail'}
            onClick={() => setActiveTab('detail')}
            disabled={!selectedVoyageId}
          />
          <TabButton
            label="Reports"
            active={activeTab === 'reports'}
            onClick={() => setActiveTab('reports')}
            disabled={!selectedVoyageId}
          />
        </div>

        {activeTab === 'setup' && <SetupSection />}
        {activeTab === 'history' && <HistorySection onSelectVoyage={selectVoyage} />}
        {activeTab === 'detail' && selectedVoyageId && (
          <DetailSection voyageId={selectedVoyageId} />
        )}
        {activeTab === 'reports' && selectedVoyageId && (
          <ReportsSection voyageId={selectedVoyageId} />
        )}
      </main>
    </div>
  );
}

function TabButton({ label, active, onClick, disabled }: {
  label: string;
  active: boolean;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`flex-1 px-4 py-2.5 rounded-lg text-sm font-medium transition-colors ${
        active
          ? 'bg-primary-500 text-white shadow-md'
          : disabled
            ? 'text-gray-600 cursor-not-allowed'
            : 'text-gray-400 hover:text-white hover:bg-white/5'
      }`}
    >
      {label}
    </button>
  );
}

// ─── Setup Section ────────────────────────────────────────────────────────

const SAMPLE_WAYPOINTS: Position[] = [
  { lat: 51.9225, lon: 4.4792, name: 'Rotterdam' },
  { lat: 51.0500, lon: 1.5000, name: 'Dover Strait' },
  { lat: 48.4500, lon: -5.1000, name: 'Ushant' },
  { lat: 42.8800, lon: -9.2700, name: 'Finisterre' },
  { lat: 36.1408, lon: -5.3536, name: 'Gibraltar' },
  { lat: 38.8000, lon: 8.4000, name: 'Sardinia South' },
  { lat: 37.2333, lon: 15.2167, name: 'Augusta' },
];

function SetupSection() {
  const {
    departureTime, setDepartureTime,
    calmSpeed, setCalmSpeed,
    isLaden, setIsLaden,
    useWeather, setUseWeather,
    waypoints, setWaypoints,
    routeName, setRouteName,
    setViewMode,
  } = useVoyage();

  const handleImport = (importedWaypoints: Position[], name: string) => {
    setWaypoints(importedWaypoints);
    setRouteName(name);
  };

  const handleClearRoute = () => {
    setWaypoints([]);
    setRouteName('Custom Route');
  };

  const totalDistance = waypoints.reduce((sum, wp, i) => {
    if (i === 0) return 0;
    const prev = waypoints[i - 1];
    const R = 3440.065;
    const lat1 = (prev.lat * Math.PI) / 180;
    const lat2 = (wp.lat * Math.PI) / 180;
    const dlat = ((wp.lat - prev.lat) * Math.PI) / 180;
    const dlon = ((wp.lon - prev.lon) * Math.PI) / 180;
    const a = Math.sin(dlat / 2) ** 2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dlon / 2) ** 2;
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return sum + R * c;
  }, 0);

  return (
    <div className="max-w-2xl space-y-6">
      {/* Voyage Parameters */}
      <Card title="Voyage Parameters" icon={<Ship className="w-5 h-5" />}>
        <div className="space-y-4">
          {/* Departure Time */}
          <div>
            <label className="block text-sm text-gray-400 mb-1">Departure Time</label>
            <input
              type="datetime-local"
              value={departureTime}
              onChange={(e) => setDepartureTime(e.target.value)}
              className="w-full px-3 py-2 rounded-lg text-sm bg-white/5 border border-white/10 text-white focus:border-primary-500/50 focus:outline-none"
            />
          </div>

          {/* Speed */}
          <div>
            <label className="block text-sm text-gray-400 mb-1">Calm Water Speed</label>
            <div className="flex items-center gap-3">
              <input
                type="range"
                min="8"
                max="18"
                step="0.5"
                value={calmSpeed}
                onChange={(e) => setCalmSpeed(parseFloat(e.target.value))}
                className="flex-1"
              />
              <span className="w-16 text-right text-white font-semibold text-sm">{calmSpeed} kts</span>
            </div>
          </div>

          {/* Laden / Ballast */}
          <div>
            <label className="block text-sm text-gray-400 mb-1">Loading Condition</label>
            <div className="flex gap-2">
              <button
                onClick={() => setIsLaden(true)}
                className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors ${
                  isLaden ? 'bg-primary-500 text-white' : 'bg-maritime-medium text-gray-400 hover:text-white'
                }`}
              >
                Laden
              </button>
              <button
                onClick={() => setIsLaden(false)}
                className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors ${
                  !isLaden ? 'bg-primary-500 text-white' : 'bg-maritime-medium text-gray-400 hover:text-white'
                }`}
              >
                Ballast
              </button>
            </div>
          </div>

          {/* Weather Toggle */}
          <div className="flex items-center justify-between p-3 bg-maritime-medium rounded-lg">
            <div className="flex items-center gap-2">
              <Wind className="w-4 h-4 text-primary-400" />
              <span className="text-sm text-white">Use Weather</span>
            </div>
            <button
              onClick={() => setUseWeather(!useWeather)}
              className={`relative w-10 h-6 rounded-full transition-colors ${useWeather ? 'bg-primary-500' : 'bg-gray-600'}`}
            >
              <span className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-transform ${useWeather ? 'left-5' : 'left-1'}`} />
            </button>
          </div>
        </div>
      </Card>

      {/* Route */}
      <Card title="Route" icon={<Navigation className="w-5 h-5" />}>
        {waypoints.length > 0 ? (
          <div className="space-y-4">
            {/* Route summary */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <MapPin className="w-4 h-4 text-primary-400" />
                <span className="text-sm text-white font-medium">{routeName}</span>
              </div>
              <button
                onClick={handleClearRoute}
                className="text-gray-400 hover:text-red-400 transition-colors"
                title="Clear Route"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
            <div className="text-xs text-gray-400">
              {waypoints.length} waypoints &middot; {totalDistance.toFixed(1)} nm
            </div>

            {/* Waypoint table */}
            <div className="max-h-[300px] overflow-y-auto rounded-lg border border-white/5">
              <table className="w-full text-xs">
                <thead className="sticky top-0 bg-maritime-dark/95">
                  <tr className="text-gray-400 border-b border-white/5">
                    <th className="px-2 py-1.5 text-left w-8">#</th>
                    <th className="px-2 py-1.5 text-left">Name</th>
                    <th className="px-2 py-1.5 text-right">Lat</th>
                    <th className="px-2 py-1.5 text-right">Lon</th>
                  </tr>
                </thead>
                <tbody>
                  {waypoints.map((wp, i) => (
                    <tr key={i} className="text-gray-300 border-b border-white/5 last:border-0 hover:bg-white/5">
                      <td className="px-2 py-1 text-gray-500">{i + 1}</td>
                      <td className="px-2 py-1 truncate max-w-[200px]">{wp.name || `WP ${i + 1}`}</td>
                      <td className="px-2 py-1 text-right font-mono">{wp.lat.toFixed(4)}</td>
                      <td className="px-2 py-1 text-right font-mono">{wp.lon.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            <p className="text-sm text-gray-500">No route loaded. Import an RTZ file or load the sample route.</p>
            <RouteImport onImport={handleImport} />
            <button
              onClick={() => handleImport(SAMPLE_WAYPOINTS, 'Rotterdam to Augusta')}
              className="text-xs text-primary-400 hover:text-primary-300 underline"
            >
              Load sample route (Rotterdam — Augusta)
            </button>
          </div>
        )}
      </Card>

      {/* Open in Chart */}
      <Link
        href="/"
        onClick={() => setViewMode('analysis')}
        className="flex items-center justify-center gap-2 w-full px-4 py-3 rounded-lg text-sm font-medium bg-primary-500/20 text-primary-400 hover:bg-primary-500/30 transition-colors"
      >
        <Map className="w-4 h-4" />
        Open in Chart
      </Link>
    </div>
  );
}

// ─── History Section ───────────────────────────────────────────────────────

function HistorySection({ onSelectVoyage }: { onSelectVoyage: (id: string) => void }) {
  const [data, setData] = useState<VoyageListResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [searchName, setSearchName] = useState('');
  const [page, setPage] = useState(0);
  const [deleting, setDeleting] = useState<string | null>(null);
  const pageSize = 20;

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const params: Record<string, string | number> = { limit: pageSize, offset: page * pageSize };
      if (searchName.trim()) params.name = searchName.trim();
      const result = await apiClient.listVoyages(params);
      setData(result);
    } catch (err) {
      console.error('Failed to load voyages:', err);
    } finally {
      setLoading(false);
    }
  }, [page, searchName]);

  useEffect(() => { load(); }, [load]);

  const handleDelete = async (id: string) => {
    if (!confirm('Delete this voyage and all its data?')) return;
    setDeleting(id);
    try {
      await apiClient.deleteVoyage(id);
      load();
    } catch (err) {
      console.error('Delete failed:', err);
    } finally {
      setDeleting(null);
    }
  };

  const handleDownloadPDF = async (id: string, name?: string) => {
    try {
      const blob = await apiClient.downloadVoyagePDF(id);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `voyage-report-${name || id.slice(0, 8)}.pdf`.replace(/ /g, '_');
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('PDF download failed:', err);
    }
  };

  const totalPages = data ? Math.ceil(data.total / pageSize) : 0;

  return (
    <div className="space-y-6">
      {/* Search */}
      <div className="flex items-center gap-3">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search voyages..."
            value={searchName}
            onChange={(e) => { setSearchName(e.target.value); setPage(0); }}
            className="w-full pl-9 pr-4 py-2.5 bg-maritime-dark border border-white/10 rounded-lg text-sm text-white placeholder-gray-500 focus:outline-none focus:border-primary-500/50"
          />
        </div>
        <span className="text-sm text-gray-400">
          {data ? `${data.total} voyage${data.total !== 1 ? 's' : ''}` : '...'}
        </span>
      </div>

      {/* Voyage list */}
      {loading ? (
        <div className="flex items-center justify-center py-12">
          <div className="w-6 h-6 border-2 border-white/30 border-t-white rounded-full animate-spin" />
        </div>
      ) : !data || data.voyages.length === 0 ? (
        <Card>
          <p className="text-sm text-gray-500 text-center py-8">
            No saved voyages yet. Calculate a voyage and save it to see it here.
          </p>
        </Card>
      ) : (
        <div className="space-y-3">
          {data.voyages.map((v) => (
            <div
              key={v.id}
              className="glass rounded-xl p-4 hover:bg-white/5 transition-colors cursor-pointer group"
              onClick={() => onSelectVoyage(v.id)}
            >
              <div className="flex items-center justify-between">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-white font-medium truncate">
                      {v.name || 'Unnamed Voyage'}
                    </span>
                    <CIIBadge cii={v.cii_estimate} />
                    <span className={`px-1.5 py-0.5 text-xs rounded ${
                      v.is_laden ? 'bg-blue-500/20 text-blue-300' : 'bg-amber-500/20 text-amber-300'
                    }`}>
                      {v.is_laden ? 'Laden' : 'Ballast'}
                    </span>
                  </div>
                  <div className="flex items-center gap-4 text-xs text-gray-400">
                    <span>{v.departure_port || '?'} → {v.arrival_port || '?'}</span>
                    <span>{new Date(v.departure_time).toLocaleDateString()}</span>
                    <span>{v.total_distance_nm.toFixed(0)} NM</span>
                    <span>{v.total_fuel_mt.toFixed(1)} MT</span>
                    <span>{v.total_time_hours.toFixed(0)}h</span>
                  </div>
                </div>
                <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button
                    onClick={(e) => { e.stopPropagation(); handleDownloadPDF(v.id, v.name || undefined); }}
                    className="p-2 text-gray-400 hover:text-primary-400 hover:bg-primary-500/10 rounded transition-colors"
                    title="Download PDF"
                  >
                    <Download className="w-4 h-4" />
                  </button>
                  {!isDemoUser() && (
                    <button
                      onClick={(e) => { e.stopPropagation(); handleDelete(v.id); }}
                      disabled={deleting === v.id}
                      className="p-2 text-gray-400 hover:text-red-400 hover:bg-red-500/10 rounded transition-colors disabled:opacity-40"
                      title="Delete voyage"
                    >
                      {deleting === v.id ? (
                        <div className="w-4 h-4 border-2 border-red-400/30 border-t-red-400 rounded-full animate-spin" />
                      ) : (
                        <Trash2 className="w-4 h-4" />
                      )}
                    </button>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-4">
          <button
            onClick={() => setPage(Math.max(0, page - 1))}
            disabled={page === 0}
            className="p-2 text-gray-400 hover:text-white disabled:opacity-30 transition-colors"
          >
            <ChevronLeft className="w-5 h-5" />
          </button>
          <span className="text-sm text-gray-400">
            Page {page + 1} of {totalPages}
          </span>
          <button
            onClick={() => setPage(Math.min(totalPages - 1, page + 1))}
            disabled={page >= totalPages - 1}
            className="p-2 text-gray-400 hover:text-white disabled:opacity-30 transition-colors"
          >
            <ChevronRight className="w-5 h-5" />
          </button>
        </div>
      )}
    </div>
  );
}

// ─── Detail Section ────────────────────────────────────────────────────────

function DetailSection({ voyageId }: { voyageId: string }) {
  const [voyage, setVoyage] = useState<VoyageDetail | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    apiClient.getVoyage(voyageId)
      .then(setVoyage)
      .catch((err) => console.error('Failed to load voyage:', err))
      .finally(() => setLoading(false));
  }, [voyageId]);

  const handleDownloadPDF = async () => {
    try {
      const blob = await apiClient.downloadVoyagePDF(voyageId);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `voyage-report-${voyage?.name || voyageId.slice(0, 8)}.pdf`.replace(/ /g, '_');
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('PDF download failed:', err);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="w-6 h-6 border-2 border-white/30 border-t-white rounded-full animate-spin" />
      </div>
    );
  }

  if (!voyage) {
    return <Card><p className="text-sm text-gray-500">Voyage not found.</p></Card>;
  }

  return (
    <div className="space-y-6">
      {/* Header with PDF button */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-white">{voyage.name || 'Unnamed Voyage'}</h2>
          <p className="text-sm text-gray-400">
            {voyage.departure_port || '?'} → {voyage.arrival_port || '?'}
          </p>
        </div>
        <button
          onClick={handleDownloadPDF}
          className="flex items-center gap-2 px-4 py-2 bg-primary-500/20 text-primary-400 rounded-lg hover:bg-primary-500/30 transition-colors"
        >
          <Download className="w-4 h-4" /> Download PDF
        </button>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Distance" value={voyage.total_distance_nm.toFixed(0)} unit="NM" icon={<Navigation className="w-4 h-4" />} />
        <StatCard label="Duration" value={voyage.total_time_hours.toFixed(1)} unit="hours" icon={<Clock className="w-4 h-4" />} />
        <StatCard label="Fuel" value={voyage.total_fuel_mt.toFixed(2)} unit="MT" icon={<Fuel className="w-4 h-4" />} />
        <StatCard label="Avg SOG" value={(voyage.avg_sog_kts ?? 0).toFixed(1)} unit="kts" icon={<Ship className="w-4 h-4" />} />
      </div>

      {/* CII card */}
      {voyage.cii_estimate && (
        <Card title="CII Estimate" icon={<Anchor className="w-5 h-5" />}>
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <span className="text-gray-400">Rating</span>
              <div className="text-white font-semibold text-lg mt-1">
                <CIIBadge cii={voyage.cii_estimate} large />
              </div>
            </div>
            <div>
              <span className="text-gray-400">Attained CII</span>
              <div className="text-white font-medium mt-1">
                {(voyage.cii_estimate.attained_cii as number)?.toFixed(4) ?? 'N/A'}
              </div>
            </div>
            <div>
              <span className="text-gray-400">Required CII</span>
              <div className="text-white font-medium mt-1">
                {(voyage.cii_estimate.required_cii as number)?.toFixed(4) ?? 'N/A'}
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Legs table */}
      <Card title="Leg Details" subtitle={`${voyage.legs.length} legs`} icon={<Navigation className="w-5 h-5" />}>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-400 border-b border-white/10">
                <th className="text-left py-2 px-2">#</th>
                <th className="text-left py-2 px-2">From</th>
                <th className="text-left py-2 px-2">To</th>
                <th className="text-right py-2 px-2">Dist (NM)</th>
                <th className="text-right py-2 px-2">SOG (kts)</th>
                <th className="text-right py-2 px-2">Fuel (MT)</th>
                <th className="text-right py-2 px-2">Time (h)</th>
                <th className="text-right py-2 px-2">Wind (kts)</th>
                <th className="text-right py-2 px-2">Wave (m)</th>
                <th className="text-right py-2 px-2">Loss (%)</th>
              </tr>
            </thead>
            <tbody>
              {voyage.legs.map((leg) => (
                <tr key={leg.id} className="border-b border-white/5 hover:bg-white/5">
                  <td className="py-2 px-2 text-gray-400">{leg.leg_index}</td>
                  <td className="py-2 px-2 text-white text-xs">
                    {leg.from_name || `${leg.from_lat.toFixed(2)},${leg.from_lon.toFixed(2)}`}
                  </td>
                  <td className="py-2 px-2 text-white text-xs">
                    {leg.to_name || `${leg.to_lat.toFixed(2)},${leg.to_lon.toFixed(2)}`}
                  </td>
                  <td className="py-2 px-2 text-right text-white">{leg.distance_nm.toFixed(1)}</td>
                  <td className="py-2 px-2 text-right text-white">{leg.sog_kts?.toFixed(1) ?? '-'}</td>
                  <td className="py-2 px-2 text-right text-white">{leg.fuel_mt.toFixed(2)}</td>
                  <td className="py-2 px-2 text-right text-white">{leg.time_hours.toFixed(1)}</td>
                  <td className="py-2 px-2 text-right text-gray-300">{leg.wind_speed_kts?.toFixed(1) ?? '-'}</td>
                  <td className="py-2 px-2 text-right text-gray-300">{leg.wave_height_m?.toFixed(1) ?? '-'}</td>
                  <td className="py-2 px-2 text-right text-gray-300">{leg.speed_loss_pct?.toFixed(1) ?? '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Notes */}
      {voyage.notes && (
        <Card title="Notes" icon={<FileText className="w-5 h-5" />}>
          <p className="text-sm text-gray-300 whitespace-pre-wrap">{voyage.notes}</p>
        </Card>
      )}
    </div>
  );
}

// ─── Reports Section ───────────────────────────────────────────────────────

function ReportsSection({ voyageId }: { voyageId: string }) {
  const [reports, setReports] = useState<VoyageReportsResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    apiClient.getVoyageReports(voyageId)
      .then(setReports)
      .catch((err) => console.error('Failed to load reports:', err))
      .finally(() => setLoading(false));
  }, [voyageId]);

  const handleDownloadPDF = async () => {
    try {
      const blob = await apiClient.downloadVoyagePDF(voyageId);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `voyage-report-${voyageId.slice(0, 8)}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('PDF download failed:', err);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="w-6 h-6 border-2 border-white/30 border-t-white rounded-full animate-spin" />
      </div>
    );
  }

  if (!reports) {
    return <Card><p className="text-sm text-gray-500">No reports available.</p></Card>;
  }

  const dep = reports.departure_report;
  const arr = reports.arrival_report;

  return (
    <div className="space-y-6">
      {/* Download button */}
      <div className="flex justify-end">
        <button
          onClick={handleDownloadPDF}
          className="flex items-center gap-2 px-4 py-2 bg-primary-500/20 text-primary-400 rounded-lg hover:bg-primary-500/30 transition-colors"
        >
          <Download className="w-4 h-4" /> Download Full PDF
        </button>
      </div>

      {/* Departure Report */}
      <Card title="Departure Report" icon={<Anchor className="w-5 h-5" />}>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <KVRow label="Vessel" value={dep.vessel_name || 'Default Vessel'} />
          <KVRow label="DWT" value={dep.dwt ? `${dep.dwt.toLocaleString()} MT` : 'N/A'} />
          <KVRow label="Departure Port" value={dep.departure_port || 'N/A'} />
          <KVRow label="Departure Time" value={new Date(dep.departure_time).toLocaleString()} />
          <KVRow label="Condition" value={dep.loading_condition} />
          <KVRow label="Destination" value={dep.destination || 'N/A'} />
          <KVRow label="ETA" value={new Date(dep.eta).toLocaleString()} />
          <KVRow label="Planned Distance" value={`${dep.planned_distance_nm.toFixed(0)} NM`} />
          <KVRow label="Planned Speed" value={`${dep.planned_speed_kts.toFixed(1)} kts`} />
          <KVRow label="Est. Fuel" value={`${dep.estimated_fuel_mt.toFixed(2)} MT`} />
        </div>
        {dep.weather_at_departure && (
          <div className="mt-4 pt-3 border-t border-white/10">
            <p className="text-xs text-gray-400 mb-2">Weather at Departure</p>
            <div className="flex gap-4 text-xs text-gray-300">
              <span><Wind className="w-3 h-3 inline mr-1" />{(dep.weather_at_departure.wind_speed_kts as number)?.toFixed(1) ?? '-'} kts</span>
              <span><Waves className="w-3 h-3 inline mr-1" />{(dep.weather_at_departure.wave_height_m as number)?.toFixed(1) ?? '-'} m</span>
            </div>
          </div>
        )}
      </Card>

      {/* Arrival Report */}
      <Card title="Arrival Report" icon={<Anchor className="w-5 h-5" />}>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <KVRow label="Vessel" value={arr.vessel_name || 'Default Vessel'} />
          <KVRow label="Arrival Port" value={arr.arrival_port || 'N/A'} />
          <KVRow label="Arrival Time" value={new Date(arr.arrival_time).toLocaleString()} />
          <KVRow label="Voyage Time" value={`${arr.actual_voyage_time_hours.toFixed(1)} hours`} />
          <KVRow label="Total Distance" value={`${arr.total_distance_nm.toFixed(0)} NM`} />
          <KVRow label="Total Fuel" value={`${arr.total_fuel_consumed_mt.toFixed(2)} MT`} />
          <KVRow label="Avg Speed" value={`${arr.average_speed_kts.toFixed(1)} kts`} />
        </div>
        {arr.cii_estimate && (
          <div className="mt-4 pt-3 border-t border-white/10">
            <p className="text-xs text-gray-400 mb-2">CII Estimate</p>
            <CIIBadge cii={arr.cii_estimate} large />
          </div>
        )}
      </Card>

      {/* Noon Reports */}
      <Card title="Noon Reports" subtitle={`${reports.noon_reports.length} reports at 24h intervals`} icon={<Clock className="w-5 h-5" />}>
        {reports.noon_reports.length === 0 ? (
          <p className="text-sm text-gray-500">Voyage too short for noon reports (less than 24 hours).</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-400 border-b border-white/10">
                  <th className="text-left py-2 px-2">#</th>
                  <th className="text-left py-2 px-2">Date/Time</th>
                  <th className="text-right py-2 px-2">Lat</th>
                  <th className="text-right py-2 px-2">Lon</th>
                  <th className="text-right py-2 px-2">SOG</th>
                  <th className="text-right py-2 px-2">Dist (NM)</th>
                  <th className="text-right py-2 px-2">Fuel (MT)</th>
                  <th className="text-right py-2 px-2">Wind</th>
                  <th className="text-right py-2 px-2">Wave</th>
                </tr>
              </thead>
              <tbody>
                {reports.noon_reports.map((nr) => (
                  <tr key={nr.report_number} className="border-b border-white/5 hover:bg-white/5">
                    <td className="py-2 px-2 text-gray-400">{nr.report_number}</td>
                    <td className="py-2 px-2 text-white text-xs">{new Date(nr.timestamp).toLocaleString()}</td>
                    <td className="py-2 px-2 text-right text-white">{nr.lat.toFixed(2)}</td>
                    <td className="py-2 px-2 text-right text-white">{nr.lon.toFixed(2)}</td>
                    <td className="py-2 px-2 text-right text-white">{nr.sog_kts?.toFixed(1) ?? '-'}</td>
                    <td className="py-2 px-2 text-right text-white">{nr.cumulative_distance_nm.toFixed(0)}</td>
                    <td className="py-2 px-2 text-right text-white">{nr.cumulative_fuel_mt.toFixed(2)}</td>
                    <td className="py-2 px-2 text-right text-gray-300">{nr.wind_speed_kts?.toFixed(1) ?? '-'}</td>
                    <td className="py-2 px-2 text-right text-gray-300">{nr.wave_height_m?.toFixed(1) ?? '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}

// ─── Shared Components ─────────────────────────────────────────────────────

function KVRow({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span className="text-gray-400">{label}</span>
      <div className="text-white font-medium mt-0.5">{value}</div>
    </div>
  );
}

const CII_COLORS: Record<string, string> = {
  A: 'bg-green-500/20 text-green-400 border-green-500/30',
  B: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
  C: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
  D: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
  E: 'bg-red-500/20 text-red-400 border-red-500/30',
};

function CIIBadge({ cii, large }: { cii?: Record<string, unknown> | null; large?: boolean }) {
  if (!cii) return null;
  const rating = String(cii.rating ?? '');
  if (!rating) return null;
  const colors = CII_COLORS[rating] || 'bg-gray-500/20 text-gray-400 border-gray-500/30';
  return (
    <span className={`inline-flex items-center justify-center border rounded ${colors} ${
      large ? 'px-3 py-1 text-lg font-bold' : 'px-1.5 py-0.5 text-xs font-semibold'
    }`}>
      {rating}
    </span>
  );
}

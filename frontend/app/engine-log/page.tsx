'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import Header from '@/components/Header';
import Card from '@/components/Card';
import { StatCard } from '@/components/Card';
import {
  FuelTimelineChart, RpmDistributionChart,
  SpeedTimelineChart, EventBreakdownChart,
} from '@/components/EngineLogCharts';
import {
  Upload, FileSpreadsheet, Trash2, Filter, ChevronLeft, ChevronRight,
  Database, Calendar, Gauge, Fuel, Activity,
} from 'lucide-react';
import {
  apiClient, EngineLogEntryResponse, EngineLogSummaryResponse,
  EngineLogEntriesParams, EngineLogUploadResponse, EngineLogCalibrateResponse,
} from '@/lib/api';

type EngineLogTab = 'upload' | 'entries' | 'analytics';

export default function EngineLogPage() {
  const [activeTab, setActiveTab] = useState<EngineLogTab>('upload');
  const [summary, setSummary] = useState<EngineLogSummaryResponse | null>(null);

  const loadSummary = useCallback(async () => {
    try {
      const data = await apiClient.getEngineLogSummary();
      setSummary(data);
    } catch (err) {
      console.error('Failed to load summary:', err);
    }
  }, []);

  useEffect(() => { loadSummary(); }, [loadSummary]);

  return (
    <div className="min-h-screen bg-gradient-maritime">
      <Header />

      <main className="container mx-auto px-4 pt-20 pb-12">
        {/* Tab bar */}
        <div className="flex space-x-1 mb-6 bg-maritime-medium/50 backdrop-blur-sm rounded-lg p-1 max-w-xl">
          <TabButton label="Upload" active={activeTab === 'upload'} onClick={() => setActiveTab('upload')} />
          <TabButton label="Entries" active={activeTab === 'entries'} onClick={() => setActiveTab('entries')} />
          <TabButton label="Analytics" active={activeTab === 'analytics'} onClick={() => setActiveTab('analytics')} />
        </div>

        {activeTab === 'upload' && <UploadSection summary={summary} onRefresh={loadSummary} />}
        {activeTab === 'entries' && <EntriesSection summary={summary} />}
        {activeTab === 'analytics' && <AnalyticsSection summary={summary} />}
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

// ─── Upload Section ─────────────────────────────────────────────────────────

function UploadSection({
  summary,
  onRefresh,
}: {
  summary: EngineLogSummaryResponse | null;
  onRefresh: () => void;
}) {
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<EngineLogUploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setError(null);
    setUploadResult(null);
    try {
      const result = await apiClient.uploadEngineLog(file);
      setUploadResult(result);
      onRefresh();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Upload failed';
      const axiosErr = err as { response?: { data?: { detail?: string } } };
      setError(axiosErr.response?.data?.detail || msg);
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleDeleteBatch = async (batchId: string) => {
    if (!confirm('Delete this batch and all its entries?')) return;
    setDeleting(batchId);
    try {
      await apiClient.deleteEngineLogBatch(batchId);
      onRefresh();
    } catch (err) {
      console.error('Delete failed:', err);
    } finally {
      setDeleting(null);
    }
  };

  return (
    <div className="max-w-2xl space-y-6">
      {/* Upload card */}
      <Card title="Upload Engine Log" icon={<Upload className="w-5 h-5" />}>
        <div className="space-y-4">
          <input
            ref={fileInputRef}
            type="file"
            accept=".xlsx,.xls"
            onChange={handleUpload}
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            className="w-full py-3 px-4 bg-maritime-dark border border-dashed border-white/20 rounded-lg text-sm text-gray-300 hover:text-white hover:border-primary-500/50 transition-colors flex items-center justify-center gap-2 disabled:opacity-40"
          >
            {uploading ? (
              <><div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> Uploading...</>
            ) : (
              <><FileSpreadsheet className="w-5 h-5" /> Select Excel File (.xlsx)</>
            )}
          </button>
          <p className="text-xs text-gray-500">
            Upload engine log workbook. Parser expects &quot;E log&quot; sheet with timestamped rows.
          </p>
        </div>
      </Card>

      {/* Upload result */}
      {uploadResult && (
        <Card>
          <div className="p-3 bg-green-500/10 border border-green-500/30 rounded-lg">
            <div className="flex items-center gap-2 text-green-400 font-medium mb-3">
              <Database className="w-4 h-4" /> Upload Successful
            </div>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div><span className="text-gray-400">Batch:</span> <span className="text-white font-mono text-xs">{uploadResult.batch_id.slice(0, 8)}...</span></div>
              <div><span className="text-gray-400">Imported:</span> <span className="text-white">{uploadResult.imported}</span></div>
              {uploadResult.date_range && (
                <>
                  <div><span className="text-gray-400">From:</span> <span className="text-white">{uploadResult.date_range.start?.slice(0, 10)}</span></div>
                  <div><span className="text-gray-400">To:</span> <span className="text-white">{uploadResult.date_range.end?.slice(0, 10)}</span></div>
                </>
              )}
            </div>
            {uploadResult.events_summary && (
              <div className="mt-3 flex flex-wrap gap-1.5">
                {Object.entries(uploadResult.events_summary).map(([ev, cnt]) => (
                  <span key={ev} className="px-2 py-0.5 bg-primary-500/20 text-primary-300 text-xs rounded-full">
                    {ev}: {cnt}
                  </span>
                ))}
              </div>
            )}
          </div>
        </Card>
      )}

      {/* Error */}
      {error && (
        <div className="p-3 bg-red-500/10 border border-red-500/30 rounded text-sm text-red-400">{error}</div>
      )}

      {/* Batch list */}
      <Card title="Uploaded Batches" icon={<Database className="w-5 h-5" />}>
        {!summary?.batches || summary.batches.length === 0 ? (
          <p className="text-sm text-gray-500">No batches uploaded yet.</p>
        ) : (
          <div className="space-y-3">
            {summary.batches.map((batch) => (
              <div
                key={batch.batch_id}
                className="flex items-center justify-between p-3 bg-maritime-dark rounded-lg"
              >
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-white text-sm font-medium truncate">
                      {batch.source_file || batch.batch_id.slice(0, 8)}
                    </span>
                    <span className="px-1.5 py-0.5 bg-primary-500/20 text-primary-300 text-xs rounded">
                      {batch.count} entries
                    </span>
                  </div>
                  <div className="text-xs text-gray-400">
                    {batch.date_start?.slice(0, 10)} &mdash; {batch.date_end?.slice(0, 10)}
                  </div>
                </div>
                <button
                  onClick={() => handleDeleteBatch(batch.batch_id)}
                  disabled={deleting === batch.batch_id}
                  className="ml-3 p-2 text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded transition-colors disabled:opacity-40"
                  title="Delete batch"
                >
                  {deleting === batch.batch_id ? (
                    <div className="w-4 h-4 border-2 border-red-400/30 border-t-red-400 rounded-full animate-spin" />
                  ) : (
                    <Trash2 className="w-4 h-4" />
                  )}
                </button>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}

// ─── Entries Section ────────────────────────────────────────────────────────

const EVENT_BADGE_COLORS: Record<string, string> = {
  NOON: 'bg-blue-500/20 text-blue-300',
  SOSP: 'bg-amber-500/20 text-amber-300',
  EOSP: 'bg-orange-500/20 text-orange-300',
  ALL_FAST: 'bg-green-500/20 text-green-300',
  DRIFTING: 'bg-purple-500/20 text-purple-300',
  ANCHORED: 'bg-cyan-500/20 text-cyan-300',
  BUNKERING: 'bg-pink-500/20 text-pink-300',
};

function EntriesSection({ summary }: { summary: EngineLogSummaryResponse | null }) {
  const [entries, setEntries] = useState<EngineLogEntryResponse[]>([]);
  const [loading, setLoading] = useState(false);
  const [pageSize, setPageSize] = useState(25);
  const [offset, setOffset] = useState(0);

  // Filters
  const [filterEvent, setFilterEvent] = useState('');
  const [filterDateFrom, setFilterDateFrom] = useState('');
  const [filterDateTo, setFilterDateTo] = useState('');
  const [filterMinRpm, setFilterMinRpm] = useState('');
  const [filterBatch, setFilterBatch] = useState('');

  const fetchEntries = useCallback(async (newOffset: number) => {
    setLoading(true);
    try {
      const params: EngineLogEntriesParams = {
        limit: pageSize,
        offset: newOffset,
      };
      if (filterEvent) params.event = filterEvent;
      if (filterDateFrom) params.date_from = filterDateFrom;
      if (filterDateTo) params.date_to = filterDateTo;
      if (filterMinRpm) params.min_rpm = parseFloat(filterMinRpm);
      if (filterBatch) params.batch_id = filterBatch;

      const data = await apiClient.getEngineLogEntries(params);
      setEntries(data);
      setOffset(newOffset);
    } catch (err) {
      console.error('Failed to fetch entries:', err);
    } finally {
      setLoading(false);
    }
  }, [pageSize, filterEvent, filterDateFrom, filterDateTo, filterMinRpm, filterBatch]);

  useEffect(() => { fetchEntries(0); }, [fetchEntries]);

  const handleApply = () => fetchEntries(0);
  const handleClear = () => {
    setFilterEvent('');
    setFilterDateFrom('');
    setFilterDateTo('');
    setFilterMinRpm('');
    setFilterBatch('');
  };

  const eventOptions = summary?.events_breakdown ? Object.keys(summary.events_breakdown) : [];
  const batchOptions = summary?.batches || [];
  const page = Math.floor(offset / pageSize) + 1;

  const fmtVal = (v: number | null, decimals = 1): string =>
    v != null ? v.toFixed(decimals) : '\u2014';

  const fmtTs = (ts: string): string => {
    const d = new Date(ts);
    return d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: '2-digit' }) +
      ' ' + d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="space-y-4">
      {/* Filter bar */}
      <Card>
        <div className="flex items-center gap-2 mb-3">
          <Filter className="w-4 h-4 text-primary-400" />
          <span className="text-sm font-medium text-white">Filters</span>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          <select
            value={filterEvent}
            onChange={(e) => setFilterEvent(e.target.value)}
            className="bg-maritime-dark border border-white/10 rounded px-2 py-1.5 text-sm text-white"
          >
            <option value="">All Events</option>
            {eventOptions.map((ev) => (
              <option key={ev} value={ev}>{ev}</option>
            ))}
          </select>
          <input
            type="date"
            value={filterDateFrom}
            onChange={(e) => setFilterDateFrom(e.target.value)}
            placeholder="From"
            className="bg-maritime-dark border border-white/10 rounded px-2 py-1.5 text-sm text-white"
          />
          <input
            type="date"
            value={filterDateTo}
            onChange={(e) => setFilterDateTo(e.target.value)}
            placeholder="To"
            className="bg-maritime-dark border border-white/10 rounded px-2 py-1.5 text-sm text-white"
          />
          <input
            type="number"
            value={filterMinRpm}
            onChange={(e) => setFilterMinRpm(e.target.value)}
            placeholder="Min RPM"
            className="bg-maritime-dark border border-white/10 rounded px-2 py-1.5 text-sm text-white"
          />
          <select
            value={filterBatch}
            onChange={(e) => setFilterBatch(e.target.value)}
            className="bg-maritime-dark border border-white/10 rounded px-2 py-1.5 text-sm text-white"
          >
            <option value="">All Batches</option>
            {batchOptions.map((b) => (
              <option key={b.batch_id} value={b.batch_id}>
                {b.source_file || b.batch_id.slice(0, 8)}
              </option>
            ))}
          </select>
          <div className="flex gap-2">
            <button
              onClick={handleApply}
              className="flex-1 py-1.5 px-3 bg-primary-500 text-white text-sm rounded hover:bg-primary-600 transition-colors"
            >
              Apply
            </button>
            <button
              onClick={handleClear}
              className="py-1.5 px-3 bg-maritime-dark text-gray-300 text-sm rounded hover:bg-maritime-light transition-colors"
            >
              Clear
            </button>
          </div>
        </div>
      </Card>

      {/* Data table */}
      <Card>
        {loading ? (
          <div className="flex items-center justify-center h-32">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-400" />
          </div>
        ) : entries.length === 0 ? (
          <p className="text-sm text-gray-500 text-center py-8">No entries found.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  {['Timestamp', 'Event', 'Place', 'RPM', 'Speed STW', 'ME Power', 'ME Load%', 'HFO Total', 'MGO Total', 'Slip%'].map((h) => (
                    <th key={h} className="text-left py-2 px-3 text-xs font-medium text-gray-400 whitespace-nowrap">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {entries.map((e) => (
                  <tr key={e.id} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                    <td className="py-2 px-3 text-gray-300 whitespace-nowrap">{fmtTs(e.timestamp)}</td>
                    <td className="py-2 px-3">
                      {e.event ? (
                        <span className={`px-2 py-0.5 text-xs rounded-full ${EVENT_BADGE_COLORS[e.event] || 'bg-gray-500/20 text-gray-300'}`}>
                          {e.event}
                        </span>
                      ) : '\u2014'}
                    </td>
                    <td className="py-2 px-3 text-gray-300 max-w-[120px] truncate">{e.place || '\u2014'}</td>
                    <td className="py-2 px-3 text-white font-mono">{fmtVal(e.rpm, 0)}</td>
                    <td className="py-2 px-3 text-white font-mono">{fmtVal(e.speed_stw)}</td>
                    <td className="py-2 px-3 text-white font-mono">{fmtVal(e.me_power_kw, 0)}</td>
                    <td className="py-2 px-3 text-white font-mono">{fmtVal(e.me_load_pct)}</td>
                    <td className="py-2 px-3 text-white font-mono">{fmtVal(e.hfo_total_mt, 2)}</td>
                    <td className="py-2 px-3 text-white font-mono">{fmtVal(e.mgo_total_mt, 2)}</td>
                    <td className="py-2 px-3 text-white font-mono">{fmtVal(e.slip_pct)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Pagination */}
        <div className="flex items-center justify-between mt-4 pt-3 border-t border-white/10">
          <div className="flex items-center gap-2 text-sm text-gray-400">
            <span>Rows:</span>
            <select
              value={pageSize}
              onChange={(e) => { setPageSize(Number(e.target.value)); setOffset(0); }}
              className="bg-maritime-dark border border-white/10 rounded px-2 py-1 text-sm text-white"
            >
              {[25, 50, 100].map((n) => (
                <option key={n} value={n}>{n}</option>
              ))}
            </select>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => fetchEntries(Math.max(0, offset - pageSize))}
              disabled={offset === 0}
              className="p-1.5 rounded hover:bg-white/5 disabled:opacity-30 disabled:cursor-not-allowed text-gray-300"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <span className="text-sm text-gray-400">Page {page}</span>
            <button
              onClick={() => fetchEntries(offset + pageSize)}
              disabled={entries.length < pageSize}
              className="p-1.5 rounded hover:bg-white/5 disabled:opacity-30 disabled:cursor-not-allowed text-gray-300"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </Card>
    </div>
  );
}

// ─── Analytics Section ──────────────────────────────────────────────────────

function AnalyticsSection({ summary }: { summary: EngineLogSummaryResponse | null }) {
  const [entries, setEntries] = useState<EngineLogEntryResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [calibrating, setCalibrating] = useState(false);
  const [calResult, setCalResult] = useState<EngineLogCalibrateResponse | null>(null);
  const [calError, setCalError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const data = await apiClient.getEngineLogEntries({ limit: 1000 });
        setEntries(data);
      } catch (err) {
        console.error('Failed to load chart data:', err);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const handleCalibrate = async () => {
    setCalibrating(true);
    setCalError(null);
    setCalResult(null);
    try {
      const result = await apiClient.calibrateFromEngineLog();
      setCalResult(result);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Calibration failed';
      const axiosErr = err as { response?: { data?: { detail?: string } } };
      setCalError(axiosErr.response?.data?.detail || msg);
    } finally {
      setCalibrating(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-400" />
      </div>
    );
  }

  const fmtShortDate = (iso: string | null | undefined): string => {
    if (!iso) return '?';
    const d = new Date(iso);
    return d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: '2-digit' });
  };

  const dateStart = summary?.date_range ? fmtShortDate(summary.date_range.start) : '\u2014';
  const dateEnd = summary?.date_range ? fmtShortDate(summary.date_range.end) : '';

  const totalFuel = summary?.fuel_summary
    ? (summary.fuel_summary.hfo_mt + summary.fuel_summary.mgo_mt + summary.fuel_summary.methanol_mt).toFixed(1)
    : '\u2014';

  return (
    <div className="space-y-6">
      {/* Stat cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
        <StatCard
          label="Total Entries"
          value={summary?.total_entries ?? 0}
          icon={<Database className="w-5 h-5" />}
        />
        <Card>
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <p className="text-sm text-gray-400 mb-2">Date Range</p>
              <p className="text-lg font-bold text-white leading-tight">{dateStart}</p>
              {dateEnd && <p className="text-lg font-bold text-white leading-tight">{dateEnd}</p>}
            </div>
            <div className="p-3 rounded-lg bg-white/5">
              <Calendar className="w-5 h-5" />
            </div>
          </div>
        </Card>
        <StatCard
          label="Avg RPM at Sea"
          value={summary?.avg_rpm_at_sea?.toFixed(0) ?? '\u2014'}
          icon={<Gauge className="w-5 h-5" />}
        />
        <StatCard
          label="Avg Speed STW"
          value={summary?.avg_speed_stw?.toFixed(1) ?? '\u2014'}
          unit="kts"
          icon={<Activity className="w-5 h-5" />}
        />
        <StatCard
          label="Total Fuel"
          value={totalFuel}
          unit="MT"
          icon={<Fuel className="w-5 h-5" />}
        />
      </div>

      {/* Calibration Card */}
      <Card className={`border ${calResult ? 'border-green-500/40' : calError ? 'border-red-500/40' : 'border-white/10'}`}>
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-semibold text-white">Vessel Calibration</h3>
            <p className="text-sm text-gray-400">
              Calibrate vessel model using engine log NOON entries (speed, fuel, power)
            </p>
          </div>
          <button
            onClick={handleCalibrate}
            disabled={calibrating || !summary?.total_entries}
            className="flex items-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-500 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg transition-colors text-sm font-medium"
          >
            {calibrating ? (
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" />
            ) : (
              <Activity className="w-4 h-4" />
            )}
            {calibrating ? 'Calibrating...' : 'Calibrate from Engine Log'}
          </button>
        </div>

        {calResult && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
            <div className="bg-white/5 rounded-lg p-3">
              <p className="text-gray-400 mb-1">Calm Water Factor</p>
              <p className="text-xl font-bold text-white">{calResult.factors.calm_water.toFixed(3)}</p>
            </div>
            <div className="bg-white/5 rounded-lg p-3">
              <p className="text-gray-400 mb-1">SFOC Factor</p>
              <p className="text-xl font-bold text-white">{calResult.factors.sfoc_factor.toFixed(3)}</p>
            </div>
            <div className="bg-white/5 rounded-lg p-3">
              <p className="text-gray-400 mb-1">Entries Used</p>
              <p className="text-xl font-bold text-white">{calResult.entries_used}</p>
              <p className="text-xs text-gray-500">{calResult.entries_skipped} skipped</p>
            </div>
            <div className="bg-white/5 rounded-lg p-3">
              <p className="text-gray-400 mb-1">Improvement</p>
              <p className="text-xl font-bold text-green-400">{calResult.improvement_pct.toFixed(1)}%</p>
              <p className="text-xs text-gray-500">{calResult.mean_error_before_mt.toFixed(2)} → {calResult.mean_error_after_mt.toFixed(2)} MT</p>
            </div>
          </div>
        )}

        {calError && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-sm text-red-300">
            {calError}
          </div>
        )}
      </Card>

      {/* Charts 2x2 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Fuel Consumption Timeline" className="h-96">
          <FuelTimelineChart entries={entries} />
        </Card>
        <Card title="RPM Distribution (NOON)" className="h-96">
          <RpmDistributionChart entries={entries} />
        </Card>
        <Card title="Speed Through Water" className="h-96">
          <SpeedTimelineChart entries={entries} />
        </Card>
        <Card title="Event Breakdown" className="h-96">
          <EventBreakdownChart eventsBreakdown={summary?.events_breakdown || {}} />
        </Card>
      </div>
    </div>
  );
}

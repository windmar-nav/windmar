'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { apiClient, CalibrationStatus, CalibrationResult } from '@/lib/api';
import {
  Gauge,
  Upload,
  FileSpreadsheet,
  Trash2,
  Play,
  CheckCircle,
  AlertTriangle,
  Info,
  ChevronDown,
  ChevronUp,
  Anchor,
} from 'lucide-react';

interface CalibrationPanelProps {
  onCalibrationChange?: () => void;
}

export default function CalibrationPanel({ onCalibrationChange }: CalibrationPanelProps) {
  const [calibration, setCalibration] = useState<CalibrationStatus | null>(null);
  const [reportsCount, setReportsCount] = useState(0);
  const [loading, setLoading] = useState(false);
  const [calibrating, setCalibrating] = useState(false);
  const [result, setResult] = useState<CalibrationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [daysSinceDrydock, setDaysSinceDrydock] = useState(180);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load calibration status
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

  useEffect(() => {
    loadCalibration();
  }, [loadCalibration]);

  // Handle CSV upload
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setError(null);

    try {
      const result = await apiClient.uploadNoonReportsCSV(file);
      setReportsCount(result.total_reports);
      setError(null);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to upload CSV');
    } finally {
      setLoading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  // Clear noon reports
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

  // Run calibration
  const handleCalibrate = async () => {
    if (reportsCount < 5) {
      setError('Need at least 5 noon reports for calibration');
      return;
    }

    setCalibrating(true);
    setError(null);

    try {
      const result = await apiClient.calibrateVessel(daysSinceDrydock);
      setResult(result);
      await loadCalibration();
      onCalibrationChange?.();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Calibration failed');
    } finally {
      setCalibrating(false);
    }
  };

  // Format factor as percentage change
  const formatFactor = (factor: number): string => {
    const pct = (factor - 1) * 100;
    if (pct === 0) return '0%';
    return pct > 0 ? `+${pct.toFixed(1)}%` : `${pct.toFixed(1)}%`;
  };

  return (
    <div className="bg-maritime-medium rounded-lg border border-white/10 overflow-hidden">
      {/* Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-white/5 transition-colors"
      >
        <div className="flex items-center gap-3">
          <Gauge className="w-5 h-5 text-primary-400" />
          <div>
            <h3 className="text-white font-medium">Vessel Calibration</h3>
            <p className="text-xs text-gray-400">
              {calibration?.calibrated
                ? `Calibrated with ${calibration.num_reports_used} reports`
                : 'Using theoretical model'}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {calibration?.calibrated ? (
            <CheckCircle className="w-4 h-4 text-green-400" />
          ) : (
            <Info className="w-4 h-4 text-gray-400" />
          )}
          {expanded ? (
            <ChevronUp className="w-4 h-4 text-gray-400" />
          ) : (
            <ChevronDown className="w-4 h-4 text-gray-400" />
          )}
        </div>
      </button>

      {/* Expanded content */}
      {expanded && (
        <div className="px-4 pb-4 space-y-4 border-t border-white/10 pt-4">
          {/* Current calibration factors */}
          {calibration && (
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="bg-maritime-dark rounded p-2">
                <div className="text-gray-400 text-xs">Hull Fouling</div>
                <div className={`font-medium ${
                  calibration.factors.calm_water > 1.1 ? 'text-amber-400' : 'text-white'
                }`}>
                  {formatFactor(calibration.factors.calm_water)}
                </div>
              </div>
              <div className="bg-maritime-dark rounded p-2">
                <div className="text-gray-400 text-xs">Wind Response</div>
                <div className="text-white font-medium">
                  {formatFactor(calibration.factors.wind)}
                </div>
              </div>
              <div className="bg-maritime-dark rounded p-2">
                <div className="text-gray-400 text-xs">Wave Response</div>
                <div className="text-white font-medium">
                  {formatFactor(calibration.factors.waves)}
                </div>
              </div>
              <div className="bg-maritime-dark rounded p-2">
                <div className="text-gray-400 text-xs">SFOC Factor</div>
                <div className="text-white font-medium">
                  {formatFactor(calibration.factors.sfoc_factor)}
                </div>
              </div>
            </div>
          )}

          {/* Noon Reports section */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-300">
                Noon Reports: <span className="text-white font-medium">{reportsCount}</span>
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

            {/* Upload button */}
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={loading}
              className="w-full py-2 px-3 bg-maritime-dark border border-white/10 rounded text-sm text-gray-300 hover:text-white hover:border-primary-500/50 transition-colors flex items-center justify-center gap-2"
            >
              <FileSpreadsheet className="w-4 h-4" />
              {loading ? 'Uploading...' : 'Upload Noon Reports CSV'}
            </button>

            <p className="text-xs text-gray-500 mt-1">
              CSV with: timestamp, latitude, longitude, speed_over_ground_kts, fuel_consumption_mt
            </p>
          </div>

          {/* Days since drydock */}
          <div>
            <label className="block text-sm text-gray-300 mb-1">
              <div className="flex items-center gap-1">
                <Anchor className="w-3 h-3" />
                Days Since Drydock: {daysSinceDrydock}
              </div>
            </label>
            <input
              type="range"
              min="0"
              max="730"
              step="30"
              value={daysSinceDrydock}
              onChange={(e) => setDaysSinceDrydock(parseInt(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-500">
              <span>0</span>
              <span>1 year</span>
              <span>2 years</span>
            </div>
          </div>

          {/* Calibrate button */}
          <button
            onClick={handleCalibrate}
            disabled={reportsCount < 5 || calibrating}
            className="w-full py-2 px-3 bg-primary-500 text-white rounded text-sm font-medium hover:bg-primary-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {calibrating ? (
              <>
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Calibrating...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Run Calibration
              </>
            )}
          </button>

          {reportsCount < 5 && (
            <p className="text-xs text-amber-400 flex items-center gap-1">
              <AlertTriangle className="w-3 h-3" />
              Need at least 5 noon reports to calibrate
            </p>
          )}

          {/* Error display */}
          {error && (
            <div className="p-2 bg-red-500/20 border border-red-500/30 rounded text-sm text-red-400">
              {error}
            </div>
          )}

          {/* Calibration result */}
          {result && (
            <div className="p-3 bg-green-500/10 border border-green-500/30 rounded">
              <div className="flex items-center gap-2 text-green-400 font-medium mb-2">
                <CheckCircle className="w-4 h-4" />
                Calibration Complete
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                  <span className="text-gray-400">Reports used:</span>{' '}
                  <span className="text-white">{result.reports_used}</span>
                </div>
                <div>
                  <span className="text-gray-400">Skipped:</span>{' '}
                  <span className="text-white">{result.reports_skipped}</span>
                </div>
                <div>
                  <span className="text-gray-400">Error before:</span>{' '}
                  <span className="text-white">{result.mean_error_before_mt.toFixed(2)} MT</span>
                </div>
                <div>
                  <span className="text-gray-400">Error after:</span>{' '}
                  <span className="text-green-400">{result.mean_error_after_mt.toFixed(2)} MT</span>
                </div>
              </div>
              <div className="mt-2 text-sm text-green-400">
                Improvement: {result.improvement_pct.toFixed(1)}%
              </div>
            </div>
          )}

          {/* Info about calibration */}
          <div className="text-xs text-gray-500">
            <p>
              Calibration adjusts the theoretical Holtrop-Mennen model to match your vessel&apos;s
              actual performance. Upload noon reports with actual fuel consumption to derive
              calibration factors for hull fouling, wind, and wave response.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

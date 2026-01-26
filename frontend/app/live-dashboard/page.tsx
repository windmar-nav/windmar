'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import Header from '@/components/Header';
import Card from '@/components/Card';
import {
  Activity,
  Waves,
  Wind,
  Navigation,
  Gauge,
  RefreshCw,
  Play,
  Square,
  Wifi,
  WifiOff,
  Compass,
  Anchor,
} from 'lucide-react';

interface FusedState {
  timestamp: string;
  position: {
    latitude: number;
    longitude: number;
  };
  motion: {
    speed_kts: number;
    heading_deg: number;
    course_deg: number;
  };
  attitude: {
    roll_deg: number;
    pitch_deg: number;
    heave_m: number;
  };
  waves_measured: {
    hs_m: number;
    tp_s: number;
    tm_s: number;
    confidence: number;
  };
  waves_forecast: {
    hs_m: number;
    tp_s: number;
    direction_deg: number;
  };
  current: {
    speed_ms: number;
    direction_deg: number;
  };
  wind: {
    speed_ms: number;
    direction_deg: number;
  };
  deltas: {
    hs_delta_m: number;
    tp_delta_s: number;
  };
  quality: {
    sbg_valid: boolean;
    wave_estimate_valid: boolean;
    forecast_valid: boolean;
    forecast_age_minutes: number;
  };
}

interface CalibrationCoeffs {
  C1_calm_water: number;
  C2_wind: number;
  C3_waves: number;
  C4_current: number;
  C5_fouling: number;
  C6_trim: number;
}

interface LiveUpdate {
  type: string;
  timestamp: string;
  state: FusedState;
  calibration: CalibrationCoeffs;
  simulation: {
    time_s: number;
    wave_buffer_fill: number;
  };
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const WS_URL = API_URL.replace('http', 'ws');

export default function LiveDashboardPage() {
  const [isConnected, setIsConnected] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [state, setState] = useState<FusedState | null>(null);
  const [calibration, setCalibration] = useState<CalibrationCoeffs | null>(null);
  const [simulation, setSimulation] = useState({ time_s: 0, wave_buffer_fill: 0 });
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Connect to WebSocket
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      const ws = new WebSocket(`${WS_URL}/ws/live`);

      ws.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setError(null);
      };

      ws.onmessage = (event) => {
        try {
          const data: LiveUpdate = JSON.parse(event.data);
          if (data.type === 'state_update') {
            setState(data.state);
            setCalibration(data.calibration);
            setSimulation(data.simulation);
          }
        } catch (e) {
          console.error('Failed to parse message:', e);
        }
      };

      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        // Try to reconnect after 2 seconds
        reconnectTimeoutRef.current = setTimeout(() => {
          if (isRunning) connectWebSocket();
        }, 2000);
      };

      ws.onerror = (e) => {
        console.error('WebSocket error:', e);
        setError('Connection error');
      };

      wsRef.current = ws;
    } catch (e) {
      console.error('Failed to create WebSocket:', e);
      setError('Failed to connect');
    }
  }, [isRunning]);

  // Start simulation
  const startSimulation = async () => {
    try {
      const response = await fetch(`${API_URL}/api/live/start`, {
        method: 'POST',
      });
      const data = await response.json();
      if (data.status === 'started' || data.status === 'already_running') {
        setIsRunning(true);
        connectWebSocket();
      }
    } catch (e) {
      console.error('Failed to start simulation:', e);
      setError('Failed to start simulation');
    }
  };

  // Stop simulation
  const stopSimulation = async () => {
    try {
      await fetch(`${API_URL}/api/live/stop`, { method: 'POST' });
      setIsRunning(false);
      if (wsRef.current) {
        wsRef.current.close();
      }
    } catch (e) {
      console.error('Failed to stop simulation:', e);
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
    };
  }, []);

  // Check initial status
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const response = await fetch(`${API_URL}/api/live/status`);
        const data = await response.json();
        if (data.simulation_running) {
          setIsRunning(true);
          connectWebSocket();
        }
      } catch (e) {
        console.error('Failed to check status:', e);
      }
    };
    checkStatus();
  }, [connectWebSocket]);

  return (
    <div className="min-h-screen bg-gradient-maritime">
      <Header />

      <main className="container mx-auto px-4 pt-24 pb-12">
        {/* Hero Section */}
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h2 className="text-4xl font-bold text-white mb-3">Live Dashboard</h2>
            <p className="text-gray-300 text-lg">
              Real-time vessel monitoring and model calibration
            </p>
          </div>

          {/* Controls */}
          <div className="flex items-center space-x-4">
            {/* Connection Status */}
            <div className="flex items-center space-x-2">
              {isConnected ? (
                <Wifi className="w-5 h-5 text-green-400" />
              ) : (
                <WifiOff className="w-5 h-5 text-red-400" />
              )}
              <span className={`text-sm ${isConnected ? 'text-green-400' : 'text-red-400'}`}>
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>

            {/* Start/Stop Button */}
            <button
              onClick={isRunning ? stopSimulation : startSimulation}
              className={`flex items-center space-x-2 px-6 py-3 rounded-lg font-semibold transition-all ${
                isRunning
                  ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30 border border-red-500/50'
                  : 'bg-green-500/20 text-green-400 hover:bg-green-500/30 border border-green-500/50'
              }`}
            >
              {isRunning ? (
                <>
                  <Square className="w-5 h-5" />
                  <span>Stop</span>
                </>
              ) : (
                <>
                  <Play className="w-5 h-5" />
                  <span>Start Simulation</span>
                </>
              )}
            </button>
          </div>
        </div>

        {error && (
          <div className="mb-6 p-4 bg-red-500/20 border border-red-500/50 rounded-lg text-red-400">
            {error}
          </div>
        )}

        {/* Main Content */}
        {state ? (
          <div className="space-y-6">
            {/* Position & Navigation */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <MetricCard
                title="Position"
                icon={<Navigation className="w-5 h-5" />}
                value={`${state.position.latitude.toFixed(4)}°N`}
                subtitle={`${state.position.longitude.toFixed(4)}°E`}
              />
              <MetricCard
                title="Speed"
                icon={<Gauge className="w-5 h-5" />}
                value={`${state.motion.speed_kts.toFixed(1)} kts`}
                subtitle={`Heading: ${state.motion.heading_deg.toFixed(0)}°`}
              />
              <MetricCard
                title="Simulation Time"
                icon={<RefreshCw className="w-5 h-5" />}
                value={formatTime(simulation.time_s)}
                subtitle={`Buffer: ${(simulation.wave_buffer_fill * 100).toFixed(0)}%`}
              />
              <MetricCard
                title="Data Quality"
                icon={<Activity className="w-5 h-5" />}
                value={state.quality.sbg_valid ? 'Valid' : 'Invalid'}
                subtitle={`Forecast age: ${state.quality.forecast_age_minutes.toFixed(0)} min`}
                valueColor={state.quality.sbg_valid ? 'text-green-400' : 'text-red-400'}
              />
            </div>

            {/* Attitude & Motion */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card title="Ship Attitude" icon={<Anchor className="w-5 h-5" />}>
                <div className="grid grid-cols-3 gap-4">
                  <AttitudeGauge label="Roll" value={state.attitude.roll_deg} unit="°" max={15} />
                  <AttitudeGauge label="Pitch" value={state.attitude.pitch_deg} unit="°" max={10} />
                  <AttitudeGauge label="Heave" value={state.attitude.heave_m} unit="m" max={3} />
                </div>
              </Card>

              <Card title="Wave Comparison" icon={<Waves className="w-5 h-5" />}>
                <div className="space-y-4">
                  <ComparisonRow
                    label="Significant Height (Hs)"
                    measured={state.waves_measured.hs_m}
                    forecast={state.waves_forecast.hs_m}
                    unit="m"
                  />
                  <ComparisonRow
                    label="Peak Period (Tp)"
                    measured={state.waves_measured.tp_s}
                    forecast={state.waves_forecast.tp_s}
                    unit="s"
                  />
                  <div className="pt-2 border-t border-gray-700">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Wave Confidence</span>
                      <span className="text-white font-semibold">
                        {(state.waves_measured.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="mt-2 h-2 bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-ocean-500 to-primary-400"
                        style={{ width: `${state.waves_measured.confidence * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              </Card>
            </div>

            {/* Environment */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card title="Wind Conditions" icon={<Wind className="w-5 h-5" />}>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-3xl font-bold text-white">
                      {(state.wind.speed_ms * 1.944).toFixed(1)} kts
                    </p>
                    <p className="text-gray-400">{state.wind.speed_ms.toFixed(1)} m/s</p>
                  </div>
                  <div className="text-right">
                    <CompassIndicator direction={state.wind.direction_deg} label="From" />
                  </div>
                </div>
              </Card>

              <Card title="Ocean Current" icon={<Compass className="w-5 h-5" />}>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-3xl font-bold text-white">
                      {(state.current.speed_ms * 1.944).toFixed(2)} kts
                    </p>
                    <p className="text-gray-400">{state.current.speed_ms.toFixed(2)} m/s</p>
                  </div>
                  <div className="text-right">
                    <CompassIndicator direction={state.current.direction_deg} label="Towards" />
                  </div>
                </div>
              </Card>
            </div>

            {/* Calibration Coefficients */}
            {calibration && (
              <Card title="Model Calibration Coefficients" icon={<RefreshCw className="w-5 h-5" />}>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                  <CoefficientBadge label="C1 Calm" value={calibration.C1_calm_water} />
                  <CoefficientBadge label="C2 Wind" value={calibration.C2_wind} />
                  <CoefficientBadge label="C3 Waves" value={calibration.C3_waves} highlight />
                  <CoefficientBadge label="C4 Current" value={calibration.C4_current} />
                  <CoefficientBadge label="C5 Fouling" value={calibration.C5_fouling} />
                  <CoefficientBadge label="C6 Trim" value={calibration.C6_trim} />
                </div>
              </Card>
            )}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-96">
            <Activity className="w-16 h-16 text-gray-500 mb-4" />
            <p className="text-xl text-gray-400">No data available</p>
            <p className="text-gray-500 mt-2">Start the simulation to see live data</p>
          </div>
        )}
      </main>
    </div>
  );
}

// Helper Components

function MetricCard({
  title,
  icon,
  value,
  subtitle,
  valueColor = 'text-white',
}: {
  title: string;
  icon: React.ReactNode;
  value: string;
  subtitle: string;
  valueColor?: string;
}) {
  return (
    <div className="glass rounded-xl p-6">
      <div className="flex items-center space-x-2 text-gray-400 mb-2">
        {icon}
        <span className="text-sm font-medium">{title}</span>
      </div>
      <p className={`text-2xl font-bold ${valueColor}`}>{value}</p>
      <p className="text-sm text-gray-500 mt-1">{subtitle}</p>
    </div>
  );
}

function AttitudeGauge({
  label,
  value,
  unit,
  max,
}: {
  label: string;
  value: number;
  unit: string;
  max: number;
}) {
  const percentage = Math.min(Math.abs(value) / max, 1) * 100;
  const isNegative = value < 0;

  return (
    <div className="text-center">
      <p className="text-sm text-gray-400 mb-2">{label}</p>
      <p className={`text-2xl font-bold ${Math.abs(value) > max * 0.7 ? 'text-amber-400' : 'text-white'}`}>
        {value > 0 ? '+' : ''}{value.toFixed(1)}{unit}
      </p>
      <div className="mt-2 h-2 bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full ${Math.abs(value) > max * 0.7 ? 'bg-amber-500' : 'bg-primary-500'} transition-all`}
          style={{
            width: `${percentage}%`,
            marginLeft: isNegative ? `${50 - percentage / 2}%` : '50%',
            marginRight: isNegative ? '50%' : `${50 - percentage / 2}%`,
          }}
        />
      </div>
    </div>
  );
}

function ComparisonRow({
  label,
  measured,
  forecast,
  unit,
}: {
  label: string;
  measured: number;
  forecast: number;
  unit: string;
}) {
  const delta = measured - forecast;
  const deltaPercent = forecast > 0 ? (delta / forecast) * 100 : 0;

  return (
    <div>
      <p className="text-sm text-gray-400 mb-2">{label}</p>
      <div className="flex items-center justify-between">
        <div>
          <span className="text-lg font-semibold text-white">{measured.toFixed(2)}</span>
          <span className="text-gray-400 ml-1">{unit}</span>
          <span className="text-xs text-gray-500 ml-2">(measured)</span>
        </div>
        <div>
          <span className="text-lg text-gray-400">{forecast.toFixed(2)}</span>
          <span className="text-gray-500 ml-1">{unit}</span>
          <span className="text-xs text-gray-600 ml-2">(forecast)</span>
        </div>
        <div className={`text-sm ${delta > 0 ? 'text-amber-400' : 'text-green-400'}`}>
          {delta > 0 ? '+' : ''}{deltaPercent.toFixed(0)}%
        </div>
      </div>
    </div>
  );
}

function CompassIndicator({ direction, label }: { direction: number; label: string }) {
  const cardinals = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
  const index = Math.round(direction / 45) % 8;

  return (
    <div>
      <p className="text-sm text-gray-500">{label}</p>
      <p className="text-2xl font-bold text-white">{direction.toFixed(0)}°</p>
      <p className="text-sm text-primary-400">{cardinals[index]}</p>
    </div>
  );
}

function CoefficientBadge({
  label,
  value,
  highlight = false,
}: {
  label: string;
  value: number;
  highlight?: boolean;
}) {
  const deviation = Math.abs(value - 1.0) * 100;
  const color =
    deviation > 20 ? 'text-red-400' : deviation > 10 ? 'text-amber-400' : 'text-green-400';

  return (
    <div
      className={`p-4 rounded-lg ${
        highlight ? 'bg-primary-500/20 border border-primary-500/50' : 'bg-gray-800/50'
      }`}
    >
      <p className="text-xs text-gray-400 mb-1">{label}</p>
      <p className={`text-xl font-bold ${color}`}>{value.toFixed(3)}</p>
      <p className="text-xs text-gray-500">{deviation > 0 ? `${deviation.toFixed(0)}% adj` : 'baseline'}</p>
    </div>
  );
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

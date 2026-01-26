"""
Sensor Fusion Engine.

Combines multiple data streams into a unified vessel state:
- SBG IMU: Real-time ship motion (roll, pitch, heave, position, speed)
- Wave Estimator: Derived wave spectrum from heave FFT
- Copernicus: Forecast ocean conditions (waves, currents, wind)

The fusion engine provides:
1. Current vessel state (FusedState)
2. Measured vs Forecast comparison (for calibration)
3. Data quality metrics
4. Time-synchronized data streams

Architecture:
    SBG IMU (1 Hz)  ─┐
                     ├──> Fusion Engine ──> FusedState
    Wave FFT (0.1 Hz)─┤                  ──> CalibrationSignal
                     │
    Copernicus (1/hr)─┘
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Callable, Any
from collections import deque
from threading import Lock
import numpy as np

from ..sensors.sbg_nmea import ShipMotionData
from ..sensors.wave_estimator import WaveEstimate, WaveEstimator
from ..data.copernicus_client import OceanConditions, WindConditions, CopernicusClient
from ..metrics import metrics, timed

logger = logging.getLogger(__name__)


@dataclass
class FusedState:
    """
    Unified vessel state combining all sensor data.

    This is the primary output of the fusion engine, providing
    a complete picture of vessel and environmental conditions.
    """
    timestamp: datetime

    # Vessel position and motion (from SBG)
    latitude: float = 0.0
    longitude: float = 0.0
    speed_kts: float = 0.0
    heading_deg: float = 0.0
    course_deg: float = 0.0

    # Vessel attitude (from SBG)
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    heave_m: float = 0.0
    heave_rate_ms: float = 0.0

    # Measured waves (from FFT of heave)
    measured_hs_m: float = 0.0      # Significant wave height
    measured_tp_s: float = 0.0      # Peak period
    measured_tm_s: float = 0.0      # Mean period
    wave_confidence: float = 0.0

    # Forecast waves (from Copernicus)
    forecast_hs_m: float = 0.0
    forecast_tp_s: float = 0.0
    forecast_wave_dir_deg: float = 0.0

    # Forecast currents (from Copernicus)
    forecast_current_ms: float = 0.0
    forecast_current_dir_deg: float = 0.0

    # Forecast wind (from Copernicus or derived)
    forecast_wind_ms: float = 0.0
    forecast_wind_dir_deg: float = 0.0

    # Sea state
    sea_surface_temp_c: float = 15.0

    # Data quality flags
    sbg_valid: bool = False
    wave_estimate_valid: bool = False
    forecast_valid: bool = False
    forecast_age_minutes: float = 0.0

    # Deltas (measured - forecast) for calibration
    hs_delta_m: float = 0.0         # Wave height difference
    tp_delta_s: float = 0.0         # Period difference

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'position': {
                'latitude': self.latitude,
                'longitude': self.longitude,
            },
            'motion': {
                'speed_kts': self.speed_kts,
                'heading_deg': self.heading_deg,
                'course_deg': self.course_deg,
            },
            'attitude': {
                'roll_deg': self.roll_deg,
                'pitch_deg': self.pitch_deg,
                'heave_m': self.heave_m,
            },
            'waves_measured': {
                'hs_m': self.measured_hs_m,
                'tp_s': self.measured_tp_s,
                'tm_s': self.measured_tm_s,
                'confidence': self.wave_confidence,
            },
            'waves_forecast': {
                'hs_m': self.forecast_hs_m,
                'tp_s': self.forecast_tp_s,
                'direction_deg': self.forecast_wave_dir_deg,
            },
            'current': {
                'speed_ms': self.forecast_current_ms,
                'direction_deg': self.forecast_current_dir_deg,
            },
            'wind': {
                'speed_ms': self.forecast_wind_ms,
                'direction_deg': self.forecast_wind_dir_deg,
            },
            'deltas': {
                'hs_delta_m': self.hs_delta_m,
                'tp_delta_s': self.tp_delta_s,
            },
            'quality': {
                'sbg_valid': self.sbg_valid,
                'wave_estimate_valid': self.wave_estimate_valid,
                'forecast_valid': self.forecast_valid,
                'forecast_age_minutes': self.forecast_age_minutes,
            },
        }


@dataclass
class CalibrationSignal:
    """
    Signal for calibration loop.

    Contains the measured-vs-forecast deltas that can be used
    to update model coefficients.
    """
    timestamp: datetime

    # Wave calibration
    wave_hs_error: float = 0.0      # (measured - forecast) / forecast
    wave_tp_error: float = 0.0

    # Position/speed for fuel tracking
    distance_traveled_nm: float = 0.0
    average_speed_kts: float = 0.0

    # Motion intensity (RMS of roll, pitch)
    roll_rms_deg: float = 0.0
    pitch_rms_deg: float = 0.0

    # Environmental conditions
    relative_wave_dir_deg: float = 0.0  # Wave direction relative to heading
    relative_wind_dir_deg: float = 0.0  # Wind direction relative to heading

    # Confidence in signal
    confidence: float = 0.0


class FusionEngine:
    """
    Combines multiple sensor streams into unified vessel state.

    Usage:
        engine = FusionEngine()

        # Start engine (initializes Copernicus client)
        engine.start()

        # Feed SBG data as it arrives
        engine.update_sbg(ship_motion_data)

        # Get current fused state
        state = engine.get_state()

        # Get calibration signal
        cal_signal = engine.get_calibration_signal()
    """

    # How long before forecast data is considered stale
    FORECAST_STALE_MINUTES = 120

    # How long to accumulate heave for wave estimate
    WAVE_ESTIMATE_WINDOW_S = 600  # 10 minutes

    def __init__(
        self,
        copernicus_mock: bool = True,
        sample_rate: float = 1.0,
    ):
        """
        Initialize fusion engine.

        Args:
            copernicus_mock: Use mock Copernicus data
            sample_rate: Expected SBG sample rate (Hz)
        """
        self.sample_rate = sample_rate

        # Initialize sub-components
        self._wave_estimator = WaveEstimator(
            sample_rate=sample_rate,
            window_seconds=self.WAVE_ESTIMATE_WINDOW_S,
        )
        self._copernicus = CopernicusClient(mock_mode=copernicus_mock)

        # Latest data from each source
        self._latest_sbg: Optional[ShipMotionData] = None
        self._latest_wave_estimate: Optional[WaveEstimate] = None
        self._latest_ocean: Optional[OceanConditions] = None
        self._latest_wind: Optional[WindConditions] = None
        self._ocean_fetch_time: Optional[datetime] = None

        # History buffers for statistics
        self._sbg_history: deque = deque(maxlen=600)  # 10 min at 1 Hz
        self._position_history: deque = deque(maxlen=60)  # For distance calc

        # Thread safety
        self._lock = Lock()

        # State
        self._running = False
        self._state_callbacks: List[Callable[[FusedState], None]] = []

        logger.info("Fusion engine initialized")

    def start(self):
        """Start the fusion engine."""
        self._running = True
        logger.info("Fusion engine started")

    def stop(self):
        """Stop the fusion engine."""
        self._running = False
        logger.info("Fusion engine stopped")

    def update_sbg(self, motion: ShipMotionData):
        """
        Update with new SBG motion data.

        This is the primary input method, called at SBG sample rate.
        """
        metrics.increment("fusion_sbg_samples_received")

        with metrics.timer("fusion_sbg_update"):
            with self._lock:
                self._latest_sbg = motion
                self._sbg_history.append(motion)

            # Feed heave to wave estimator
            self._wave_estimator.add_sample(
                heave=motion.heave_m,
                roll=motion.roll_deg,
                pitch=motion.pitch_deg,
            )

            # Track position for distance calculation
            if motion.latitude != 0 and motion.longitude != 0:
                self._position_history.append({
                    'time': motion.timestamp,
                    'lat': motion.latitude,
                    'lon': motion.longitude,
                })

        # Check if we should update forecast
        self._maybe_update_forecast(motion.latitude, motion.longitude)

        # Trigger wave estimate periodically
        if self._wave_estimator.sample_count >= 60:  # At least 1 minute
            self._update_wave_estimate()

    def _maybe_update_forecast(self, lat: float, lon: float):
        """Fetch new forecast if needed."""
        if lat == 0 and lon == 0:
            return  # No valid position

        now = datetime.utcnow()

        # Fetch if no forecast or stale
        should_fetch = (
            self._ocean_fetch_time is None or
            (now - self._ocean_fetch_time).total_seconds() > 3600  # 1 hour
        )

        if should_fetch:
            try:
                self._latest_ocean = self._copernicus.get_ocean_conditions(lat, lon)
                self._latest_wind = self._copernicus.get_wind_conditions(lat, lon)
                self._ocean_fetch_time = now
                logger.debug(f"Updated forecast for {lat:.2f}, {lon:.2f}")
            except Exception as e:
                logger.error(f"Failed to fetch forecast: {e}")

    def _update_wave_estimate(self):
        """Run wave spectrum estimation."""
        with metrics.timer("fusion_wave_estimation"):
            estimate = self._wave_estimator.estimate()

        if estimate:
            with self._lock:
                self._latest_wave_estimate = estimate
            metrics.increment("fusion_wave_estimates_computed")
            metrics.set_gauge("fusion_wave_confidence", estimate.confidence)
            logger.debug(
                f"Wave estimate: Hs={estimate.significant_height_m:.2f}m, "
                f"Tp={estimate.peak_period_s:.1f}s, "
                f"confidence={estimate.confidence:.0%}"
            )

    def get_state(self) -> FusedState:
        """
        Get current fused state.

        Returns:
            FusedState combining all available data
        """
        with self._lock:
            return self._build_state()

    def _build_state(self) -> FusedState:
        """Build fused state from current data."""
        now = datetime.utcnow()
        state = FusedState(timestamp=now)

        # SBG data
        if self._latest_sbg:
            sbg = self._latest_sbg
            state.latitude = sbg.latitude
            state.longitude = sbg.longitude
            state.speed_kts = sbg.speed_kts
            state.heading_deg = sbg.heading_deg
            state.roll_deg = sbg.roll_deg
            state.pitch_deg = sbg.pitch_deg
            state.heave_m = sbg.heave_m
            state.course_deg = sbg.course_deg if sbg.course_deg else sbg.heading_deg
            state.sbg_valid = True

            # Calculate heave rate from recent samples
            if len(self._sbg_history) >= 2:
                recent = list(self._sbg_history)[-5:]
                heaves = [s.heave_m for s in recent]
                state.heave_rate_ms = (heaves[-1] - heaves[0]) / (len(heaves) / self.sample_rate)

        # Wave estimate
        if self._latest_wave_estimate:
            wave = self._latest_wave_estimate
            state.measured_hs_m = wave.significant_height_m
            state.measured_tp_s = wave.peak_period_s
            state.measured_tm_s = wave.mean_period_s
            state.wave_confidence = wave.confidence
            state.wave_estimate_valid = wave.confidence > 0.3

        # Forecast data
        if self._latest_ocean:
            ocean = self._latest_ocean
            state.forecast_hs_m = ocean.significant_wave_height_m
            state.forecast_tp_s = ocean.peak_wave_period_s
            state.forecast_wave_dir_deg = ocean.wave_direction_deg
            state.forecast_current_ms = ocean.current_speed_ms
            state.forecast_current_dir_deg = ocean.current_direction_deg
            state.sea_surface_temp_c = ocean.sea_surface_temp_c

            # Calculate forecast age
            if self._ocean_fetch_time:
                age = (now - self._ocean_fetch_time).total_seconds() / 60
                state.forecast_age_minutes = age
                state.forecast_valid = age < self.FORECAST_STALE_MINUTES

        if self._latest_wind:
            state.forecast_wind_ms = self._latest_wind.wind_speed_ms
            state.forecast_wind_dir_deg = self._latest_wind.wind_direction_deg

        # Calculate deltas
        if state.wave_estimate_valid and state.forecast_valid:
            state.hs_delta_m = state.measured_hs_m - state.forecast_hs_m
            state.tp_delta_s = state.measured_tp_s - state.forecast_tp_s

        return state

    def get_calibration_signal(self) -> CalibrationSignal:
        """
        Get calibration signal for model tuning.

        Returns:
            CalibrationSignal with error metrics
        """
        state = self.get_state()
        signal = CalibrationSignal(timestamp=state.timestamp)

        # Wave errors (normalized)
        if state.forecast_hs_m > 0:
            signal.wave_hs_error = state.hs_delta_m / state.forecast_hs_m
        if state.forecast_tp_s > 0:
            signal.wave_tp_error = state.tp_delta_s / state.forecast_tp_s

        # Motion statistics from history
        with self._lock:
            if len(self._sbg_history) >= 10:
                rolls = [s.roll_deg for s in self._sbg_history]
                pitches = [s.pitch_deg for s in self._sbg_history]
                signal.roll_rms_deg = np.sqrt(np.mean(np.array(rolls) ** 2))
                signal.pitch_rms_deg = np.sqrt(np.mean(np.array(pitches) ** 2))

            # Distance traveled
            if len(self._position_history) >= 2:
                signal.distance_traveled_nm = self._calc_distance()

        # Relative directions
        if state.heading_deg:
            signal.relative_wave_dir_deg = (state.forecast_wave_dir_deg - state.heading_deg) % 360
            signal.relative_wind_dir_deg = (state.forecast_wind_dir_deg - state.heading_deg) % 360

        # Average speed
        signal.average_speed_kts = state.speed_kts

        # Overall confidence
        if state.sbg_valid and state.wave_estimate_valid and state.forecast_valid:
            signal.confidence = min(state.wave_confidence, 0.8)
        elif state.sbg_valid and state.forecast_valid:
            signal.confidence = 0.5
        else:
            signal.confidence = 0.2

        return signal

    def _calc_distance(self) -> float:
        """Calculate distance traveled in nautical miles."""
        if len(self._position_history) < 2:
            return 0.0

        total_nm = 0.0
        positions = list(self._position_history)

        for i in range(1, len(positions)):
            p1 = positions[i - 1]
            p2 = positions[i]

            # Haversine distance
            lat1, lon1 = np.radians(p1['lat']), np.radians(p1['lon'])
            lat2, lon2 = np.radians(p2['lat']), np.radians(p2['lon'])

            dlat = lat2 - lat1
            dlon = lon2 - lon1

            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))

            # Earth radius in nm
            r_nm = 3440.065
            total_nm += r_nm * c

        return total_nm

    def register_callback(self, callback: Callable[[FusedState], None]):
        """Register callback for state updates."""
        self._state_callbacks.append(callback)

    def get_wave_spectrum(self):
        """Get the wave power spectrum for visualization."""
        return self._wave_estimator.get_spectrum()

    @property
    def sbg_count(self) -> int:
        """Number of SBG samples received."""
        return len(self._sbg_history)

    @property
    def wave_buffer_fill(self) -> float:
        """Wave estimator buffer fill ratio (0-1)."""
        return self._wave_estimator.buffer_fill_ratio

    @property
    def has_valid_state(self) -> bool:
        """Check if we have enough data for valid state."""
        return self._latest_sbg is not None


# Test function
def test_fusion_engine():
    """Test the fusion engine with simulated data."""
    import math

    print("Testing Fusion Engine")
    print("=" * 50)

    # Create fusion engine
    engine = FusionEngine(copernicus_mock=True, sample_rate=1.0)
    engine.start()

    # Simulation parameters
    wave_height = 2.5  # meters
    wave_period = 8.0  # seconds
    omega = 2 * math.pi / wave_period

    # Starting position (Mediterranean)
    base_lat = 43.5
    base_lon = 7.0
    speed_kts = 12.0
    heading = 270.0  # West

    # Simulate 2 minutes of data
    print("\nSimulating 2 minutes of ship motion...")

    for i in range(120):
        t = i  # seconds

        # Simulate wave-induced motion
        heave = (wave_height / 2) * math.sin(omega * t)
        roll = 5.0 * math.sin(omega * t * 0.8 + 0.5)
        pitch = 2.0 * math.sin(omega * t * 1.2 + 0.3)

        # Update position (simple linear movement)
        # 1 knot = 1 nm/hour = 1/3600 nm/s
        distance_nm = speed_kts * t / 3600
        # Moving west: lon decreases
        lat = base_lat
        lon = base_lon - distance_nm / 60  # Approximate

        # Create motion data
        motion = ShipMotionData(
            timestamp=datetime.utcnow(),
            roll_deg=roll,
            pitch_deg=pitch,
            heading_deg=heading,
            heave_m=heave,
            latitude=lat,
            longitude=lon,
            speed_kts=speed_kts,
            course_deg=heading,
        )

        engine.update_sbg(motion)

        if (i + 1) % 30 == 0:
            state = engine.get_state()
            print(f"\n[{i+1:3d}s] State update:")
            print(f"  Position: {state.latitude:.4f}°N, {state.longitude:.4f}°E")
            print(f"  Speed: {state.speed_kts:.1f} kts, Heading: {state.heading_deg:.0f}°")
            print(f"  Roll: {state.roll_deg:+.1f}°, Pitch: {state.pitch_deg:+.1f}°")
            print(f"  Measured Hs: {state.measured_hs_m:.2f}m (conf: {state.wave_confidence:.0%})")
            print(f"  Forecast Hs: {state.forecast_hs_m:.2f}m")
            print(f"  Delta Hs: {state.hs_delta_m:+.2f}m")

    # Final calibration signal
    cal = engine.get_calibration_signal()
    print(f"\nCalibration Signal:")
    print(f"  Wave Hs error: {cal.wave_hs_error:+.1%}")
    print(f"  Roll RMS: {cal.roll_rms_deg:.2f}°")
    print(f"  Pitch RMS: {cal.pitch_rms_deg:.2f}°")
    print(f"  Distance: {cal.distance_traveled_nm:.2f} nm")
    print(f"  Confidence: {cal.confidence:.0%}")

    engine.stop()
    return engine


if __name__ == "__main__":
    test_fusion_engine()

"""
Real-time Model Calibration Loop.

Adjusts hydrodynamic model coefficients based on measured vs predicted
discrepancies from the fusion engine.

Calibration Parameters (C1-C6):
- C1: Calm water resistance coefficient
- C2: Wind resistance coefficient
- C3: Wave added resistance coefficient
- C4: Current effect coefficient
- C5: Hull fouling factor
- C6: Trim effect coefficient

The loop uses exponential moving average (EMA) to smooth updates
and prevent oscillations from noisy sensor data.

Reference: ISO 19030 - Ships and marine technology - Measurement of
changes in hull and propeller performance
"""

import logging
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Callable
from collections import deque
from pathlib import Path
import numpy as np

from ..fusion.fusion_engine import CalibrationSignal, FusedState
from ..metrics import metrics, timed

logger = logging.getLogger(__name__)


@dataclass
class CalibrationCoefficients:
    """
    Model calibration coefficients.

    All coefficients are multiplicative factors around 1.0.
    E.g., C3=1.1 means wave resistance is 10% higher than model predicts.
    """
    C1_calm_water: float = 1.0      # Calm water resistance factor
    C2_wind: float = 1.0            # Wind resistance factor
    C3_waves: float = 1.0           # Wave added resistance factor
    C4_current: float = 1.0         # Current effect factor
    C5_fouling: float = 1.0         # Hull fouling factor
    C6_trim: float = 1.0            # Trim effect factor

    # Uncertainty estimates (standard deviation)
    C1_std: float = 0.1
    C2_std: float = 0.15
    C3_std: float = 0.2
    C4_std: float = 0.15
    C5_std: float = 0.1
    C6_std: float = 0.1

    # Last update times
    last_update: Optional[str] = None
    total_samples: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'CalibrationCoefficients':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'CalibrationCoefficients':
        """Deserialize from JSON."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class CalibrationState:
    """Current state of the calibration loop."""
    is_running: bool = False
    samples_processed: int = 0
    last_signal_time: Optional[datetime] = None

    # Recent history for diagnostics
    wave_errors: List[float] = field(default_factory=list)
    coefficients_history: List[Dict[str, float]] = field(default_factory=list)

    # Convergence metrics
    is_converged: bool = False
    convergence_metric: float = 1.0  # Lower is better


class CalibrationLoop:
    """
    Real-time calibration loop for the hydrodynamic model.

    Usage:
        loop = CalibrationLoop()
        loop.start()

        # Feed signals from fusion engine
        loop.process_signal(calibration_signal)

        # Get current coefficients for model
        coeffs = loop.get_coefficients()
        print(f"Wave factor: {coeffs.C3_waves}")

        # Save state
        loop.save("calibration.json")
    """

    # Learning rate for coefficient updates
    DEFAULT_LEARNING_RATE = 0.01

    # Minimum confidence to process signal
    MIN_CONFIDENCE = 0.3

    # Bounds for coefficients (prevent runaway)
    COEFF_MIN = 0.5
    COEFF_MAX = 2.0

    # History length for convergence detection
    HISTORY_LENGTH = 100

    def __init__(
        self,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        persistence_path: Optional[str] = None,
    ):
        """
        Initialize calibration loop.

        Args:
            learning_rate: Rate of coefficient adjustment (0-1)
            persistence_path: Path to save/load calibration state
        """
        self.learning_rate = learning_rate
        self.persistence_path = Path(persistence_path) if persistence_path else None

        # Current coefficients
        self._coefficients = CalibrationCoefficients()

        # Try to load existing calibration
        if self.persistence_path and self.persistence_path.exists():
            self._load()

        # State tracking
        self._state = CalibrationState()
        self._signal_history: deque = deque(maxlen=self.HISTORY_LENGTH)

        # Callbacks for updates
        self._callbacks: List[Callable[[CalibrationCoefficients], None]] = []

        logger.info(f"Calibration loop initialized (lr={learning_rate})")

    def start(self):
        """Start the calibration loop."""
        self._state.is_running = True
        logger.info("Calibration loop started")

    def stop(self):
        """Stop the calibration loop."""
        self._state.is_running = False
        if self.persistence_path:
            self._save()
        logger.info("Calibration loop stopped")

    def process_signal(self, signal: CalibrationSignal) -> bool:
        """
        Process a calibration signal and update coefficients.

        Args:
            signal: CalibrationSignal from fusion engine

        Returns:
            True if coefficients were updated
        """
        metrics.increment("calibration_signals_received")

        if not self._state.is_running:
            return False

        # Check confidence threshold
        if signal.confidence < self.MIN_CONFIDENCE:
            metrics.increment("calibration_signals_rejected_low_confidence")
            logger.debug(f"Signal confidence too low: {signal.confidence:.0%}")
            return False

        with metrics.timer("calibration_signal_processing"):
            # Store signal
            self._signal_history.append(signal)
            self._state.samples_processed += 1
            self._state.last_signal_time = signal.timestamp

            # Track wave error history
            self._state.wave_errors.append(signal.wave_hs_error)
            if len(self._state.wave_errors) > self.HISTORY_LENGTH:
                self._state.wave_errors = self._state.wave_errors[-self.HISTORY_LENGTH:]

            # Update coefficients based on errors
            updated = self._update_coefficients(signal)

            if updated:
                metrics.increment("calibration_coefficients_updated")
                metrics.set_gauge("calibration_c3_waves", self._coefficients.C3_waves)
                metrics.set_gauge("calibration_c2_wind", self._coefficients.C2_wind)

                # Update timestamp
                self._coefficients.last_update = datetime.utcnow().isoformat()
                self._coefficients.total_samples += 1

                # Track history
                self._state.coefficients_history.append({
                    'C3_waves': self._coefficients.C3_waves,
                    'C2_wind': self._coefficients.C2_wind,
                })
                if len(self._state.coefficients_history) > self.HISTORY_LENGTH:
                    self._state.coefficients_history = self._state.coefficients_history[-self.HISTORY_LENGTH:]

                # Check convergence
                self._check_convergence()

                # Notify callbacks
                for cb in self._callbacks:
                    cb(self._coefficients)

        return updated

    def _update_coefficients(self, signal: CalibrationSignal) -> bool:
        """Update coefficients based on signal."""
        updated = False

        # Scale learning rate by confidence
        lr = self.learning_rate * signal.confidence

        # C3: Wave resistance - adjust based on wave height error
        # If measured waves > forecast, increase C3
        if abs(signal.wave_hs_error) > 0.05:  # >5% error
            delta = signal.wave_hs_error * lr
            new_c3 = self._coefficients.C3_waves * (1 + delta)
            self._coefficients.C3_waves = np.clip(new_c3, self.COEFF_MIN, self.COEFF_MAX)
            updated = True
            logger.debug(f"C3_waves adjusted to {self._coefficients.C3_waves:.4f}")

        # C2: Wind resistance - adjust based on roll/pitch
        # Higher motion with same wind = underestimate wind effect
        if signal.roll_rms_deg > 3.0:  # Significant motion
            # Use relative wind direction to scale effect
            # Head wind (0°) = max effect, beam wind (90°) = less
            wind_factor = abs(np.cos(np.radians(signal.relative_wind_dir_deg)))
            if wind_factor > 0.5:  # Roughly head/following wind
                motion_error = (signal.roll_rms_deg - 3.0) / 10.0  # Normalize
                delta = motion_error * lr * 0.5  # Smaller adjustment for wind
                new_c2 = self._coefficients.C2_wind * (1 + delta)
                self._coefficients.C2_wind = np.clip(new_c2, self.COEFF_MIN, self.COEFF_MAX)
                updated = True

        # C4: Current effect - would need speed-over-ground vs speed-through-water
        # For now, skip this as we don't have STW

        # Update uncertainty estimates
        if updated:
            self._update_uncertainties(signal)

        return updated

    def _update_uncertainties(self, signal: CalibrationSignal):
        """Update coefficient uncertainty estimates."""
        # Reduce uncertainty as we get more samples
        decay = 0.99  # Slow decay

        self._coefficients.C3_std *= decay
        self._coefficients.C2_std *= decay

        # But increase if we see large errors
        if abs(signal.wave_hs_error) > 0.3:
            self._coefficients.C3_std = min(0.3, self._coefficients.C3_std * 1.1)

    def _check_convergence(self):
        """Check if calibration has converged."""
        if len(self._state.coefficients_history) < 20:
            self._state.is_converged = False
            return

        # Check stability of C3 over recent history
        recent_c3 = [h['C3_waves'] for h in self._state.coefficients_history[-20:]]
        c3_std = np.std(recent_c3)
        c3_mean = np.mean(recent_c3)

        # Coefficient of variation
        cv = c3_std / c3_mean if c3_mean > 0 else 1.0

        self._state.convergence_metric = cv
        self._state.is_converged = cv < 0.02  # <2% variation

        if self._state.is_converged:
            logger.info(f"Calibration converged (CV={cv:.1%})")

    def get_coefficients(self) -> CalibrationCoefficients:
        """Get current calibration coefficients."""
        return self._coefficients

    def set_coefficients(self, coeffs: CalibrationCoefficients):
        """Manually set coefficients."""
        self._coefficients = coeffs
        logger.info("Coefficients manually updated")

    def get_state(self) -> CalibrationState:
        """Get current calibration state."""
        return self._state

    def register_callback(self, callback: Callable[[CalibrationCoefficients], None]):
        """Register callback for coefficient updates."""
        self._callbacks.append(callback)

    def _save(self):
        """Save calibration to file."""
        if not self.persistence_path:
            return

        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            self.persistence_path.write_text(self._coefficients.to_json())
            logger.info(f"Calibration saved to {self.persistence_path}")
        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")

    def _load(self):
        """Load calibration from file."""
        if not self.persistence_path or not self.persistence_path.exists():
            return

        try:
            data = self.persistence_path.read_text()
            self._coefficients = CalibrationCoefficients.from_json(data)
            logger.info(f"Calibration loaded from {self.persistence_path}")
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")

    def save(self, path: Optional[str] = None):
        """Save calibration to specified path."""
        if path:
            self.persistence_path = Path(path)
        self._save()

    def reset(self):
        """Reset calibration to defaults."""
        self._coefficients = CalibrationCoefficients()
        self._state = CalibrationState()
        self._signal_history.clear()
        logger.info("Calibration reset to defaults")

    def get_diagnostics(self) -> dict:
        """Get diagnostic information."""
        return {
            'coefficients': self._coefficients.to_dict(),
            'state': {
                'is_running': self._state.is_running,
                'samples_processed': self._state.samples_processed,
                'is_converged': self._state.is_converged,
                'convergence_metric': round(self._state.convergence_metric, 4),
            },
            'recent_wave_errors': self._state.wave_errors[-10:] if self._state.wave_errors else [],
        }


# Test function
def test_calibration_loop():
    """Test the calibration loop with simulated signals."""
    print("Testing Calibration Loop")
    print("=" * 50)

    # Create loop
    loop = CalibrationLoop(learning_rate=0.05)
    loop.start()

    # Simulate signals with systematic error
    # Waves are 20% higher than forecast
    print("\nSimulating calibration with 20% wave error...")

    for i in range(100):
        signal = CalibrationSignal(
            timestamp=datetime.utcnow(),
            wave_hs_error=0.20 + np.random.normal(0, 0.05),  # 20% ± 5%
            wave_tp_error=0.10 + np.random.normal(0, 0.03),
            roll_rms_deg=4.0 + np.random.normal(0, 1.0),
            pitch_rms_deg=2.0 + np.random.normal(0, 0.5),
            relative_wave_dir_deg=30 + np.random.normal(0, 10),
            relative_wind_dir_deg=45 + np.random.normal(0, 15),
            distance_traveled_nm=0.1,
            average_speed_kts=12.0,
            confidence=0.7 + np.random.random() * 0.2,
        )

        loop.process_signal(signal)

        if (i + 1) % 25 == 0:
            coeffs = loop.get_coefficients()
            state = loop.get_state()
            print(f"\n[{i+1:3d}] Calibration update:")
            print(f"  C3 (waves): {coeffs.C3_waves:.4f} (std: {coeffs.C3_std:.3f})")
            print(f"  C2 (wind):  {coeffs.C2_wind:.4f}")
            print(f"  Converged:  {state.is_converged} (metric: {state.convergence_metric:.4f})")

    # Final state
    print("\n" + "=" * 50)
    print("Final Diagnostics:")
    diag = loop.get_diagnostics()
    print(f"  Samples processed: {diag['state']['samples_processed']}")
    print(f"  Converged: {diag['state']['is_converged']}")
    print(f"  Final C3: {diag['coefficients']['C3_waves']:.4f}")

    loop.stop()
    return loop


if __name__ == "__main__":
    test_calibration_loop()

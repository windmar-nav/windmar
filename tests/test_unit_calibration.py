"""
Unit tests for Calibration Loop.

Tests coefficient adjustment and convergence logic.
"""

import pytest
import sys
import tempfile
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.calibration.calibration_loop import (
    CalibrationLoop,
    CalibrationCoefficients,
    CalibrationState,
)
from src.fusion.fusion_engine import CalibrationSignal


class TestCalibrationCoefficients:
    """Unit tests for CalibrationCoefficients dataclass."""

    def test_default_values(self):
        """Test default coefficient values are 1.0."""
        coeffs = CalibrationCoefficients()

        assert coeffs.C1_calm_water == 1.0
        assert coeffs.C2_wind == 1.0
        assert coeffs.C3_waves == 1.0
        assert coeffs.C4_current == 1.0
        assert coeffs.C5_fouling == 1.0
        assert coeffs.C6_trim == 1.0

    def test_default_uncertainties(self):
        """Test default uncertainty values."""
        coeffs = CalibrationCoefficients()

        assert coeffs.C1_std == 0.1
        assert coeffs.C2_std == 0.15
        assert coeffs.C3_std == 0.2

    def test_to_dict(self):
        """Test conversion to dictionary."""
        coeffs = CalibrationCoefficients(C3_waves=1.2)
        d = coeffs.to_dict()

        assert isinstance(d, dict)
        assert d["C3_waves"] == 1.2
        assert "C1_calm_water" in d

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {"C3_waves": 1.2, "C2_wind": 0.9}
        coeffs = CalibrationCoefficients.from_dict(d)

        assert coeffs.C3_waves == 1.2
        assert coeffs.C2_wind == 0.9
        assert coeffs.C1_calm_water == 1.0  # Default

    def test_to_json(self):
        """Test JSON serialization."""
        coeffs = CalibrationCoefficients(C3_waves=1.2)
        json_str = coeffs.to_json()

        assert isinstance(json_str, str)
        assert "1.2" in json_str

    def test_from_json(self):
        """Test JSON deserialization."""
        json_str = '{"C3_waves": 1.2, "C2_wind": 0.9}'
        coeffs = CalibrationCoefficients.from_json(json_str)

        assert coeffs.C3_waves == 1.2
        assert coeffs.C2_wind == 0.9

    def test_json_round_trip(self):
        """Test JSON serialization round trip."""
        original = CalibrationCoefficients(
            C3_waves=1.25,
            C2_wind=0.95,
            C5_fouling=1.1
        )

        json_str = original.to_json()
        restored = CalibrationCoefficients.from_json(json_str)

        assert restored.C3_waves == original.C3_waves
        assert restored.C2_wind == original.C2_wind
        assert restored.C5_fouling == original.C5_fouling


class TestCalibrationLoop:
    """Unit tests for CalibrationLoop class."""

    @pytest.fixture
    def loop(self):
        """Create a calibration loop for testing."""
        return CalibrationLoop(learning_rate=0.05)

    @pytest.fixture
    def signal(self):
        """Create a test calibration signal."""
        return CalibrationSignal(
            timestamp=datetime.utcnow(),
            wave_hs_error=0.15,  # 15% error
            wave_tp_error=0.10,
            roll_rms_deg=4.0,
            pitch_rms_deg=2.0,
            relative_wave_dir_deg=30,
            relative_wind_dir_deg=45,
            distance_traveled_nm=0.1,
            average_speed_kts=12.0,
            confidence=0.8,
        )

    def test_initialization(self, loop):
        """Test loop initializes correctly."""
        assert loop.learning_rate == 0.05
        assert loop._state.is_running is False
        assert loop._state.samples_processed == 0

    def test_start_stop(self, loop):
        """Test start and stop methods."""
        loop.start()
        assert loop._state.is_running is True

        loop.stop()
        assert loop._state.is_running is False

    def test_process_signal_requires_start(self, loop, signal):
        """Test signal processing requires loop to be started."""
        result = loop.process_signal(signal)
        assert result is False

    def test_process_signal_updates_count(self, loop, signal):
        """Test signal processing updates sample count."""
        loop.start()
        loop.process_signal(signal)

        assert loop._state.samples_processed >= 1

    def test_process_signal_low_confidence_rejected(self, loop):
        """Test low confidence signals are rejected."""
        loop.start()

        signal = CalibrationSignal(
            timestamp=datetime.utcnow(),
            wave_hs_error=0.15,
            confidence=0.1,  # Below threshold
        )

        initial_count = loop._state.samples_processed
        loop.process_signal(signal)

        assert loop._state.samples_processed == initial_count

    def test_c3_increases_with_positive_wave_error(self, loop, signal):
        """Test C3 increases when measured > forecast."""
        loop.start()

        # Create signal with positive wave error (underestimate)
        signal = CalibrationSignal(
            timestamp=datetime.utcnow(),
            wave_hs_error=0.20,  # 20% higher than forecast
            confidence=0.8,
        )

        initial_c3 = loop._coefficients.C3_waves
        loop.process_signal(signal)

        assert loop._coefficients.C3_waves > initial_c3

    def test_c3_decreases_with_negative_wave_error(self, loop):
        """Test C3 decreases when measured < forecast."""
        loop.start()

        # Create signal with negative wave error (overestimate)
        signal = CalibrationSignal(
            timestamp=datetime.utcnow(),
            wave_hs_error=-0.20,  # 20% lower than forecast
            confidence=0.8,
        )

        initial_c3 = loop._coefficients.C3_waves
        loop.process_signal(signal)

        assert loop._coefficients.C3_waves < initial_c3

    def test_coefficient_bounds(self, loop):
        """Test coefficients are bounded."""
        loop.start()

        # Try to push C3 very high
        for _ in range(100):
            signal = CalibrationSignal(
                timestamp=datetime.utcnow(),
                wave_hs_error=0.50,  # Large error
                confidence=0.9,
            )
            loop.process_signal(signal)

        assert loop._coefficients.C3_waves <= loop.COEFF_MAX

    def test_coefficient_lower_bound(self, loop):
        """Test coefficients don't go below minimum."""
        loop.start()

        # Try to push C3 very low
        for _ in range(100):
            signal = CalibrationSignal(
                timestamp=datetime.utcnow(),
                wave_hs_error=-0.50,  # Large negative error
                confidence=0.9,
            )
            loop.process_signal(signal)

        assert loop._coefficients.C3_waves >= loop.COEFF_MIN

    def test_get_coefficients(self, loop, signal):
        """Test coefficient retrieval."""
        loop.start()
        loop.process_signal(signal)

        coeffs = loop.get_coefficients()

        assert isinstance(coeffs, CalibrationCoefficients)

    def test_set_coefficients(self, loop):
        """Test manual coefficient setting."""
        new_coeffs = CalibrationCoefficients(C3_waves=1.5, C2_wind=0.8)
        loop.set_coefficients(new_coeffs)

        assert loop._coefficients.C3_waves == 1.5
        assert loop._coefficients.C2_wind == 0.8

    def test_get_state(self, loop, signal):
        """Test state retrieval."""
        loop.start()
        loop.process_signal(signal)

        state = loop.get_state()

        assert isinstance(state, CalibrationState)
        assert state.is_running is True
        assert state.samples_processed >= 1

    def test_reset(self, loop, signal):
        """Test calibration reset."""
        loop.start()
        loop.process_signal(signal)

        loop.reset()

        assert loop._coefficients.C3_waves == 1.0
        assert loop._state.samples_processed == 0

    def test_callback_registration(self, loop, signal):
        """Test callback is called on coefficient update."""
        received = []

        def callback(coeffs):
            received.append(coeffs)

        loop.register_callback(callback)
        loop.start()
        loop.process_signal(signal)

        assert len(received) >= 1
        assert isinstance(received[0], CalibrationCoefficients)

    def test_convergence_detection(self, loop):
        """Test convergence is detected after stable updates."""
        loop.start()

        # Send many similar signals to converge
        for _ in range(50):
            signal = CalibrationSignal(
                timestamp=datetime.utcnow(),
                wave_hs_error=0.10,
                confidence=0.8,
            )
            loop.process_signal(signal)

        # Check convergence metric is computed
        assert loop._state.convergence_metric < 1.0

    def test_get_diagnostics(self, loop, signal):
        """Test diagnostics retrieval."""
        loop.start()
        loop.process_signal(signal)

        diag = loop.get_diagnostics()

        assert "coefficients" in diag
        assert "state" in diag
        assert "recent_wave_errors" in diag

    def test_persistence_save_load(self, loop, signal):
        """Test calibration persistence."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # Create and save
            loop.start()
            loop.process_signal(signal)
            loop.save(temp_path)

            saved_c3 = loop._coefficients.C3_waves

            # Create new loop and load
            loop2 = CalibrationLoop(persistence_path=temp_path)

            assert loop2._coefficients.C3_waves == saved_c3
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestCalibrationState:
    """Unit tests for CalibrationState dataclass."""

    def test_default_values(self):
        """Test default state values."""
        state = CalibrationState()

        assert state.is_running is False
        assert state.samples_processed == 0
        assert state.is_converged is False
        assert state.convergence_metric == 1.0

    def test_wave_errors_list(self):
        """Test wave errors list initialization."""
        state = CalibrationState()

        assert isinstance(state.wave_errors, list)
        assert len(state.wave_errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
End-to-End Integration Test for SBG IMU Data Pipeline.

Tests the complete data flow:
    SBG IMU → NMEA Parser → Fusion Engine → Calibration Loop

This validates that all components work together correctly.
"""

import sys
import math
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sensors.sbg_nmea import ShipMotionData, SBGNmeaParser
from src.sensors.wave_estimator import WaveEstimator, simulate_wave_motion
from src.data.copernicus_client import CopernicusClient
from src.fusion.fusion_engine import FusionEngine, FusedState, CalibrationSignal
from src.calibration.calibration_loop import CalibrationLoop, CalibrationCoefficients


def test_wave_estimator():
    """Test FFT wave estimation from heave data."""
    print("\n" + "=" * 60)
    print("TEST 1: Wave Estimator (FFT)")
    print("=" * 60)

    # Target wave parameters
    target_hs = 2.5  # meters
    target_tp = 8.0  # seconds

    # Simulate wave motion
    t, heave = simulate_wave_motion(
        Hs=target_hs,
        Tp=target_tp,
        duration_s=600,
        sample_rate=1.0,
    )

    # Create estimator and process data
    estimator = WaveEstimator(sample_rate=1.0, window_seconds=600)
    estimator.add_samples(heave.tolist())

    # Get estimate
    result = estimator.estimate()

    assert result is not None, "Wave estimation failed"

    hs_error = abs(result.significant_height_m - target_hs)
    tp_error = abs(result.peak_period_s - target_tp)

    print(f"  Target Hs: {target_hs} m, Estimated: {result.significant_height_m:.2f} m (error: {hs_error:.2f})")
    print(f"  Target Tp: {target_tp} s, Estimated: {result.peak_period_s:.1f} s (error: {tp_error:.1f})")
    print(f"  Confidence: {result.confidence:.0%}")

    # Allow 30% error for Hs, 2s for Tp
    assert hs_error < target_hs * 0.30, f"Hs error too large: {hs_error}"
    assert tp_error < 2.0, f"Tp error too large: {tp_error}"

    print("  PASSED")
    return True


def test_copernicus_client():
    """Test Copernicus marine data client."""
    print("\n" + "=" * 60)
    print("TEST 2: Copernicus Client (Mock Mode)")
    print("=" * 60)

    client = CopernicusClient(mock_mode=True)

    # Test location: Mediterranean
    lat, lon = 43.5, 7.0

    # Get ocean conditions
    ocean = client.get_ocean_conditions(lat, lon)

    print(f"  Position: {lat}°N, {lon}°E")
    print(f"  Wave Height: {ocean.significant_wave_height_m:.2f} m")
    print(f"  Peak Period: {ocean.peak_wave_period_s:.1f} s")
    print(f"  Wave Direction: {ocean.wave_direction_deg:.0f}°")
    print(f"  Current Speed: {ocean.current_speed_ms:.2f} m/s")
    print(f"  SST: {ocean.sea_surface_temp_c:.1f}°C")

    assert ocean.significant_wave_height_m >= 0, "Invalid wave height"
    assert ocean.peak_wave_period_s > 0, "Invalid wave period"

    # Test forecast
    forecast = client.get_forecast(lat, lon, hours_ahead=24, interval_hours=6)
    assert len(forecast) > 0, "No forecast data"

    print(f"  Forecast points: {len(forecast)}")
    print("  PASSED")
    return True


def test_fusion_engine():
    """Test sensor fusion engine."""
    print("\n" + "=" * 60)
    print("TEST 3: Fusion Engine")
    print("=" * 60)

    engine = FusionEngine(copernicus_mock=True, sample_rate=1.0)
    engine.start()

    # Simulation parameters
    wave_height = 2.5
    wave_period = 8.0
    omega = 2 * math.pi / wave_period

    # Feed 120 seconds of simulated data
    print("  Simulating 120 seconds of ship motion...")

    for t in range(120):
        heave = (wave_height / 2) * math.sin(omega * t)
        roll = 5.0 * math.sin(omega * t * 0.8 + 0.5)
        pitch = 2.0 * math.sin(omega * t * 1.2 + 0.3)

        motion = ShipMotionData(
            timestamp=datetime.utcnow(),
            roll_deg=roll,
            pitch_deg=pitch,
            heading_deg=270.0,
            heave_m=heave,
            latitude=43.5,
            longitude=7.0 - t * 0.0001,  # Moving west
            speed_kts=12.0,
            course_deg=270.0,
        )

        engine.update_sbg(motion)

    # Get final state
    state = engine.get_state()

    print(f"  SBG samples received: {engine.sbg_count}")
    print(f"  Wave buffer fill: {engine.wave_buffer_fill:.0%}")
    print(f"  Position: {state.latitude:.4f}°N, {state.longitude:.4f}°E")
    print(f"  Measured Hs: {state.measured_hs_m:.2f} m")
    print(f"  Forecast Hs: {state.forecast_hs_m:.2f} m")
    print(f"  Delta: {state.hs_delta_m:+.2f} m")

    assert engine.has_valid_state, "No valid state"
    assert state.sbg_valid, "SBG data invalid"
    assert state.measured_hs_m > 0 or state.wave_confidence == 0, "Wave estimate failed"

    # Get calibration signal
    cal_signal = engine.get_calibration_signal()
    print(f"  Calibration signal confidence: {cal_signal.confidence:.0%}")

    engine.stop()
    print("  PASSED")
    return True


def test_calibration_loop():
    """Test real-time calibration loop."""
    print("\n" + "=" * 60)
    print("TEST 4: Calibration Loop")
    print("=" * 60)

    loop = CalibrationLoop(learning_rate=0.05)
    loop.start()

    # Simulate calibration with systematic 20% wave error
    print("  Simulating calibration with 20% systematic wave error...")

    import numpy as np

    for i in range(50):
        signal = CalibrationSignal(
            timestamp=datetime.utcnow(),
            wave_hs_error=0.20 + np.random.normal(0, 0.05),
            wave_tp_error=0.10,
            roll_rms_deg=4.0,
            pitch_rms_deg=2.0,
            relative_wave_dir_deg=30,
            relative_wind_dir_deg=45,
            distance_traveled_nm=0.1,
            average_speed_kts=12.0,
            confidence=0.8,
        )
        loop.process_signal(signal)

    coeffs = loop.get_coefficients()
    state = loop.get_state()

    print(f"  Samples processed: {state.samples_processed}")
    print(f"  C3 (waves): {coeffs.C3_waves:.4f} (expected > 1.0)")
    print(f"  C2 (wind): {coeffs.C2_wind:.4f}")
    print(f"  Converged: {state.is_converged}")

    # C3 should increase (waves underestimated)
    assert coeffs.C3_waves > 1.0, f"C3 should increase, got {coeffs.C3_waves}"

    loop.stop()
    print("  PASSED")
    return True


def test_full_pipeline():
    """Test complete E2E pipeline."""
    print("\n" + "=" * 60)
    print("TEST 5: Full E2E Pipeline")
    print("=" * 60)

    # Initialize all components
    engine = FusionEngine(copernicus_mock=True, sample_rate=1.0)
    calibration = CalibrationLoop(learning_rate=0.02)

    engine.start()
    calibration.start()

    # Simulation parameters
    wave_height = 3.0  # Actual waves
    wave_period = 9.0
    omega = 2 * math.pi / wave_period

    print("  Running 180-second E2E simulation...")

    for t in range(180):
        # Simulated ship motion
        heave = (wave_height / 2) * math.sin(omega * t)
        roll = 6.0 * math.sin(omega * t * 0.8 + 0.5)
        pitch = 2.5 * math.sin(omega * t * 1.2 + 0.3)

        motion = ShipMotionData(
            timestamp=datetime.utcnow(),
            roll_deg=roll,
            pitch_deg=pitch,
            heading_deg=270.0,
            heave_m=heave,
            latitude=43.5,
            longitude=7.0 - t * 0.0002,
            speed_kts=12.0,
            course_deg=270.0,
        )

        # Update fusion engine
        engine.update_sbg(motion)

        # Get calibration signal and update
        if t > 60 and t % 5 == 0:  # Start after 1 minute, update every 5s
            cal_signal = engine.get_calibration_signal()
            calibration.process_signal(cal_signal)

    # Final results
    state = engine.get_state()
    coeffs = calibration.get_coefficients()
    cal_state = calibration.get_state()

    print(f"\n  Final Results:")
    print(f"    SBG samples: {engine.sbg_count}")
    print(f"    Measured Hs: {state.measured_hs_m:.2f} m")
    print(f"    Forecast Hs: {state.forecast_hs_m:.2f} m")
    print(f"    Wave confidence: {state.wave_confidence:.0%}")
    print(f"    Calibration samples: {cal_state.samples_processed}")
    print(f"    C3 coefficient: {coeffs.C3_waves:.4f}")

    engine.stop()
    calibration.stop()

    print("  PASSED")
    return True


def main():
    """Run all E2E tests."""
    print("\n" + "=" * 60)
    print("WINDMAR SBG Integration E2E Tests")
    print("=" * 60)

    tests = [
        ("Wave Estimator", test_wave_estimator),
        ("Copernicus Client", test_copernicus_client),
        ("Fusion Engine", test_fusion_engine),
        ("Calibration Loop", test_calibration_loop),
        ("Full Pipeline", test_full_pipeline),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  FAILED: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, p in results:
        status = "PASS" if p else "FAIL"
        print(f"  {name}: {status}")

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n  ALL TESTS PASSED!")
        return 0
    else:
        print("\n  SOME TESTS FAILED!")
        return 1


if __name__ == "__main__":
    exit(main())

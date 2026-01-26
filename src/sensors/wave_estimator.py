"""
Wave Spectrum Estimator from Ship Motion Data.

Implements the "Wave Buoy Analogy" (WBA) to derive wave parameters
from heave, roll, and pitch measurements.

Based on:
- Nielsen (2006): Estimating directional wave spectrum from ship motions
- IMO MSC.1/Circ.1228: Guidelines for voluntary use of the ship
  speed-dependant sea keeping performance

References:
- Significant Wave Height: Hs = 4 * sqrt(m0)
- Peak Period: Tp = 1 / f_peak
- Mean Period: Tm = m0 / m1
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
from collections import deque
import numpy as np
from scipy import signal
from scipy.integrate import trapezoid

logger = logging.getLogger(__name__)


@dataclass
class WaveEstimate:
    """Estimated wave parameters from ship motion."""
    significant_height_m: float  # Hs - significant wave height
    peak_period_s: float         # Tp - peak period
    mean_period_s: float         # Tm - mean period
    dominant_frequency_hz: float # Peak frequency
    spectral_energy: float       # Total energy (m0)
    confidence: float            # 0-1 confidence in estimate
    sample_count: int            # Number of samples used
    duration_s: float            # Duration of analysis window


class WaveEstimator:
    """
    Estimates wave spectrum from ship heave measurements.

    Uses FFT-based spectral analysis to derive:
    - Significant wave height (Hs)
    - Peak period (Tp)
    - Mean period (Tm)

    Usage:
        estimator = WaveEstimator(sample_rate=1.0, window_seconds=600)

        # Add samples as they arrive
        for heave in heave_stream:
            estimator.add_sample(heave)

        # Get wave estimate
        waves = estimator.estimate()
        print(f"Hs: {waves.significant_height_m:.2f} m")
    """

    # Minimum samples needed for reliable estimate
    MIN_SAMPLES = 60  # At least 1 minute at 1 Hz

    # Wave frequency bounds (typical ocean waves)
    MIN_WAVE_FREQ = 0.03  # ~33s period (long swell)
    MAX_WAVE_FREQ = 0.30  # ~3.3s period (wind waves)

    def __init__(
        self,
        sample_rate: float = 1.0,
        window_seconds: float = 600,  # 10 minutes default
    ):
        """
        Initialize wave estimator.

        Args:
            sample_rate: Samples per second (Hz)
            window_seconds: Analysis window duration
        """
        self.sample_rate = sample_rate
        self.window_seconds = window_seconds
        self.max_samples = int(window_seconds * sample_rate)

        # Circular buffer for heave samples
        self._heave_buffer: deque = deque(maxlen=self.max_samples)

        # Optional roll/pitch buffers for future multi-DOF analysis
        self._roll_buffer: deque = deque(maxlen=self.max_samples)
        self._pitch_buffer: deque = deque(maxlen=self.max_samples)

        # Last estimate (cached)
        self._last_estimate: Optional[WaveEstimate] = None

    def add_sample(
        self,
        heave: float,
        roll: Optional[float] = None,
        pitch: Optional[float] = None,
    ):
        """
        Add a motion sample to the buffer.

        Args:
            heave: Vertical displacement (m)
            roll: Roll angle (deg) - optional
            pitch: Pitch angle (deg) - optional
        """
        self._heave_buffer.append(heave)

        if roll is not None:
            self._roll_buffer.append(roll)
        if pitch is not None:
            self._pitch_buffer.append(pitch)

    def add_samples(self, heave_array: List[float]):
        """Add multiple heave samples at once."""
        for h in heave_array:
            self._heave_buffer.append(h)

    def clear(self):
        """Clear all buffers."""
        self._heave_buffer.clear()
        self._roll_buffer.clear()
        self._pitch_buffer.clear()
        self._last_estimate = None

    def estimate(self, force: bool = False) -> Optional[WaveEstimate]:
        """
        Estimate wave parameters from buffered data.

        Args:
            force: Force estimate even with insufficient samples

        Returns:
            WaveEstimate or None if insufficient data
        """
        n_samples = len(self._heave_buffer)

        if n_samples < self.MIN_SAMPLES and not force:
            logger.warning(
                f"Insufficient samples for wave estimate: {n_samples} < {self.MIN_SAMPLES}"
            )
            return None

        # Convert buffer to numpy array
        heave = np.array(self._heave_buffer)

        # Remove mean (detrend)
        heave = heave - np.mean(heave)

        # Compute power spectral density using Welch method
        # nperseg: length of each segment for averaging
        nperseg = min(256, n_samples // 4)
        if nperseg < 16:
            nperseg = n_samples  # Use whole signal if too short

        try:
            frequencies, psd = signal.welch(
                heave,
                fs=self.sample_rate,
                nperseg=nperseg,
                noverlap=nperseg // 2,
                window='hann',
                detrend='linear',
            )
        except Exception as e:
            logger.error(f"FFT failed: {e}")
            return None

        # Filter to wave frequency band
        wave_mask = (frequencies >= self.MIN_WAVE_FREQ) & (frequencies <= self.MAX_WAVE_FREQ)

        if not np.any(wave_mask):
            logger.warning("No energy in wave frequency band")
            return self._create_zero_estimate(n_samples)

        wave_freqs = frequencies[wave_mask]
        wave_psd = psd[wave_mask]

        # Spectral moments
        # m0 = integral of S(f) df - total variance/energy
        # m1 = integral of f * S(f) df
        # m2 = integral of f^2 * S(f) df
        df = wave_freqs[1] - wave_freqs[0] if len(wave_freqs) > 1 else 0.01

        m0 = trapezoid(wave_psd, wave_freqs)
        m1 = trapezoid(wave_psd * wave_freqs, wave_freqs)
        m2 = trapezoid(wave_psd * wave_freqs**2, wave_freqs)

        # Significant wave height: Hs = 4 * sqrt(m0)
        # This is the standard definition used in oceanography
        Hs = 4.0 * np.sqrt(m0) if m0 > 0 else 0.0

        # Peak frequency and period
        peak_idx = np.argmax(wave_psd)
        f_peak = wave_freqs[peak_idx] if len(wave_freqs) > 0 else 0.1
        Tp = 1.0 / f_peak if f_peak > 0 else 0.0

        # Mean period: Tm = m0 / m1
        Tm = m0 / m1 if m1 > 0 else Tp

        # Zero-crossing period: Tz = sqrt(m0 / m2)
        # Tz = np.sqrt(m0 / m2) if m2 > 0 else Tm

        # Confidence based on sample count and signal quality
        confidence = self._calculate_confidence(n_samples, m0, heave)

        duration = n_samples / self.sample_rate

        estimate = WaveEstimate(
            significant_height_m=round(Hs, 3),
            peak_period_s=round(Tp, 2),
            mean_period_s=round(Tm, 2),
            dominant_frequency_hz=round(f_peak, 4),
            spectral_energy=round(m0, 6),
            confidence=round(confidence, 2),
            sample_count=n_samples,
            duration_s=round(duration, 1),
        )

        self._last_estimate = estimate
        return estimate

    def get_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the full power spectrum for plotting.

        Returns:
            (frequencies, power_spectral_density)
        """
        if len(self._heave_buffer) < 16:
            return np.array([]), np.array([])

        heave = np.array(self._heave_buffer)
        heave = heave - np.mean(heave)

        nperseg = min(256, len(heave) // 4)
        if nperseg < 16:
            nperseg = len(heave)

        frequencies, psd = signal.welch(
            heave,
            fs=self.sample_rate,
            nperseg=nperseg,
            window='hann',
        )

        return frequencies, psd

    def _calculate_confidence(
        self,
        n_samples: int,
        m0: float,
        heave: np.ndarray,
    ) -> float:
        """Calculate confidence score for the estimate."""
        confidence = 0.0

        # Factor 1: Sample count (0-0.4)
        # Full confidence at 600 samples (10 min at 1 Hz)
        sample_factor = min(n_samples / 600, 1.0) * 0.4
        confidence += sample_factor

        # Factor 2: Signal energy (0-0.3)
        # Very low energy = low confidence
        if m0 > 0.01:  # At least ~0.4m Hs
            confidence += 0.3
        elif m0 > 0.001:  # At least ~0.13m Hs
            confidence += 0.15

        # Factor 3: Signal stability (0-0.3)
        # Check variance of variance in chunks
        if len(heave) >= 60:
            chunks = np.array_split(heave, min(6, len(heave) // 10))
            variances = [np.var(c) for c in chunks if len(c) > 5]
            if len(variances) > 1:
                cv = np.std(variances) / (np.mean(variances) + 1e-9)
                stability = max(0, 1 - cv) * 0.3
                confidence += stability

        return min(confidence, 1.0)

    def _create_zero_estimate(self, n_samples: int) -> WaveEstimate:
        """Create a zero estimate for calm conditions."""
        return WaveEstimate(
            significant_height_m=0.0,
            peak_period_s=0.0,
            mean_period_s=0.0,
            dominant_frequency_hz=0.0,
            spectral_energy=0.0,
            confidence=0.5,  # Medium confidence for "calm"
            sample_count=n_samples,
            duration_s=n_samples / self.sample_rate,
        )

    @property
    def buffer_fill_ratio(self) -> float:
        """Get buffer fill percentage (0-1)."""
        return len(self._heave_buffer) / self.max_samples

    @property
    def sample_count(self) -> int:
        """Get current sample count."""
        return len(self._heave_buffer)

    @property
    def last_estimate(self) -> Optional[WaveEstimate]:
        """Get last computed estimate."""
        return self._last_estimate


def simulate_wave_motion(
    Hs: float = 2.5,
    Tp: float = 8.0,
    duration_s: float = 600,
    sample_rate: float = 1.0,
    noise_level: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate ship heave motion for testing.

    Uses JONSWAP-like spectrum to generate realistic wave motion.

    Args:
        Hs: Significant wave height (m)
        Tp: Peak period (s)
        duration_s: Duration of simulation
        sample_rate: Samples per second
        noise_level: Noise as fraction of Hs

    Returns:
        (time_array, heave_array)
    """
    n_samples = int(duration_s * sample_rate)
    t = np.arange(n_samples) / sample_rate

    # Generate wave as sum of sinusoids (simplified JONSWAP)
    fp = 1.0 / Tp  # Peak frequency

    # Frequency components
    n_components = 20
    frequencies = np.linspace(0.5 * fp, 2.0 * fp, n_components)

    # JONSWAP-like amplitude distribution
    gamma = 3.3  # Peak enhancement factor
    sigma = np.where(frequencies <= fp, 0.07, 0.09)

    # Spectrum
    alpha = 0.0081  # Phillips constant (adjusted for Hs)
    S = (alpha * 9.81**2 / (2 * np.pi)**4 / frequencies**5 *
         np.exp(-1.25 * (fp / frequencies)**4) *
         gamma ** np.exp(-0.5 * ((frequencies - fp) / (sigma * fp))**2))

    # Scale to match desired Hs
    m0_target = (Hs / 4.0)**2
    m0_actual = trapezoid(S, frequencies)
    S = S * (m0_target / m0_actual) if m0_actual > 0 else S

    # Generate time series from spectrum
    heave = np.zeros(n_samples)
    for i, (f, s) in enumerate(zip(frequencies, S)):
        amplitude = np.sqrt(2 * s * (frequencies[1] - frequencies[0]) if len(frequencies) > 1 else s)
        phase = np.random.uniform(0, 2 * np.pi)
        heave += amplitude * np.sin(2 * np.pi * f * t + phase)

    # Add noise
    heave += np.random.normal(0, Hs * noise_level, n_samples)

    return t, heave


# Test function
def test_wave_estimator():
    """Test wave estimation with simulated data."""
    print("Testing Wave Estimator")
    print("=" * 50)

    # Simulate waves: Hs=2.5m, Tp=8s
    target_Hs = 2.5
    target_Tp = 8.0

    print(f"Target: Hs={target_Hs}m, Tp={target_Tp}s")

    t, heave = simulate_wave_motion(
        Hs=target_Hs,
        Tp=target_Tp,
        duration_s=600,
        sample_rate=1.0,
    )

    # Create estimator and add samples
    estimator = WaveEstimator(sample_rate=1.0, window_seconds=600)
    estimator.add_samples(heave.tolist())

    # Get estimate
    result = estimator.estimate()

    if result:
        print(f"\nEstimated:")
        print(f"  Hs: {result.significant_height_m:.2f} m (error: {abs(result.significant_height_m - target_Hs):.2f}m)")
        print(f"  Tp: {result.peak_period_s:.1f} s (error: {abs(result.peak_period_s - target_Tp):.1f}s)")
        print(f"  Tm: {result.mean_period_s:.1f} s")
        print(f"  Confidence: {result.confidence:.0%}")
        print(f"  Samples: {result.sample_count}")
    else:
        print("Estimation failed!")

    return result


if __name__ == "__main__":
    test_wave_estimator()

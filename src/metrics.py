"""
Performance Metrics Module.

Provides production-grade metrics collection for monitoring:
- Timing measurements for critical operations
- Throughput tracking (samples/second)
- Error rate monitoring
- Resource utilization estimates

Usage:
    from src.metrics import metrics, timed

    # Decorator for timing functions
    @timed("fusion_update")
    def update_state():
        ...

    # Manual timing
    with metrics.timer("wave_estimation"):
        estimate = estimator.compute()

    # Counter increments
    metrics.increment("sbg_samples_processed")

    # Get metrics summary
    summary = metrics.get_summary()
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from threading import Lock
from typing import Callable, Dict, List, Optional, Any
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class TimingStats:
    """Statistics for a timed operation."""
    name: str
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float('inf')
    max_ms: float = 0.0
    recent_ms: deque = field(default_factory=lambda: deque(maxlen=100))

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count > 0 else 0.0

    @property
    def recent_avg_ms(self) -> float:
        if not self.recent_ms:
            return 0.0
        return sum(self.recent_ms) / len(self.recent_ms)

    def record(self, duration_ms: float):
        """Record a timing measurement."""
        self.count += 1
        self.total_ms += duration_ms
        self.min_ms = min(self.min_ms, duration_ms)
        self.max_ms = max(self.max_ms, duration_ms)
        self.recent_ms.append(duration_ms)

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "avg_ms": round(self.avg_ms, 3),
            "min_ms": round(self.min_ms, 3) if self.min_ms != float('inf') else 0,
            "max_ms": round(self.max_ms, 3),
            "recent_avg_ms": round(self.recent_avg_ms, 3),
        }


class PerformanceMetrics:
    """
    Production metrics collector for monitoring system performance.

    Thread-safe metrics collection with support for:
    - Timing measurements (with context manager and decorator)
    - Counters for event tracking
    - Gauges for current values
    - Throughput calculation
    """

    # Threshold for warning on slow operations (ms)
    SLOW_THRESHOLD_MS = 100.0

    # Interval for periodic metrics logging
    LOG_INTERVAL_SECONDS = 60.0

    def __init__(self, enable_logging: bool = True, log_interval: float = 60.0):
        """
        Initialize metrics collector.

        Args:
            enable_logging: Whether to periodically log metrics summary
            log_interval: Interval between log outputs in seconds
        """
        self._timings: Dict[str, TimingStats] = {}
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._lock = Lock()

        self._start_time = datetime.now()
        self._last_log_time = datetime.now()
        self._enable_logging = enable_logging
        self._log_interval = log_interval

    @contextmanager
    def timer(self, name: str):
        """
        Context manager for timing a block of code.

        Usage:
            with metrics.timer("operation_name"):
                do_something()
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._record_timing(name, elapsed_ms)

    def _record_timing(self, name: str, elapsed_ms: float):
        """Record a timing measurement."""
        with self._lock:
            if name not in self._timings:
                self._timings[name] = TimingStats(name=name)
            self._timings[name].record(elapsed_ms)

        # Log warning for slow operations
        if elapsed_ms > self.SLOW_THRESHOLD_MS:
            logger.warning(
                f"Slow operation: {name} took {elapsed_ms:.1f}ms "
                f"(threshold: {self.SLOW_THRESHOLD_MS}ms)"
            )

        # Periodic metrics logging
        self._maybe_log_summary()

    def increment(self, name: str, amount: int = 1):
        """Increment a counter."""
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    def set_gauge(self, name: str, value: float):
        """Set a gauge value."""
        with self._lock:
            self._gauges[name] = value

    def get_counter(self, name: str) -> int:
        """Get counter value."""
        with self._lock:
            return self._counters.get(name, 0)

    def get_gauge(self, name: str) -> float:
        """Get gauge value."""
        with self._lock:
            return self._gauges.get(name, 0.0)

    def get_timing(self, name: str) -> Optional[TimingStats]:
        """Get timing statistics for an operation."""
        with self._lock:
            return self._timings.get(name)

    def get_summary(self) -> dict:
        """Get complete metrics summary."""
        with self._lock:
            uptime = (datetime.now() - self._start_time).total_seconds()

            return {
                "uptime_seconds": round(uptime, 1),
                "timings": {
                    name: stats.to_dict()
                    for name, stats in self._timings.items()
                },
                "counters": self._counters.copy(),
                "gauges": {k: round(v, 4) for k, v in self._gauges.items()},
                "throughput": self._calculate_throughput(),
            }

    def _calculate_throughput(self) -> dict:
        """Calculate throughput metrics."""
        uptime = (datetime.now() - self._start_time).total_seconds()
        if uptime <= 0:
            return {}

        throughput = {}
        for name, count in self._counters.items():
            if name.endswith("_processed") or name.endswith("_received"):
                rate = count / uptime
                throughput[f"{name}_per_sec"] = round(rate, 2)

        return throughput

    def _maybe_log_summary(self):
        """Log summary if interval has passed."""
        if not self._enable_logging:
            return

        now = datetime.now()
        if (now - self._last_log_time).total_seconds() >= self._log_interval:
            self._last_log_time = now
            self._log_summary()

    def _log_summary(self):
        """Log current metrics summary."""
        summary = self.get_summary()

        # Format timings
        timing_str = ", ".join(
            f"{name}: {stats['avg_ms']:.1f}ms avg ({stats['count']} calls)"
            for name, stats in summary["timings"].items()
        )

        # Format counters
        counter_str = ", ".join(
            f"{name}={value}"
            for name, value in summary["counters"].items()
        )

        # Format throughput
        throughput_str = ", ".join(
            f"{name}={value:.1f}"
            for name, value in summary["throughput"].items()
        )

        logger.info(
            f"Performance metrics - "
            f"Uptime: {summary['uptime_seconds']:.0f}s | "
            f"Timings: [{timing_str or 'none'}] | "
            f"Counters: [{counter_str or 'none'}] | "
            f"Throughput: [{throughput_str or 'none'}]"
        )

    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._timings.clear()
            self._counters.clear()
            self._gauges.clear()
            self._start_time = datetime.now()


# Global metrics instance
metrics = PerformanceMetrics(enable_logging=True, log_interval=60.0)


def timed(name: str):
    """
    Decorator to time a function.

    Usage:
        @timed("my_function")
        def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with metrics.timer(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def get_metrics() -> PerformanceMetrics:
    """Get the global metrics instance."""
    return metrics

"""
Unit tests for Performance Metrics Module.

Tests timing, counters, gauges, and throughput tracking.
"""

import pytest
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import (
    PerformanceMetrics,
    TimingStats,
    metrics,
    timed,
    get_metrics,
)


class TestTimingStats:
    """Unit tests for TimingStats class."""

    def test_initial_values(self):
        """Test initial stats values."""
        stats = TimingStats(name="test")

        assert stats.name == "test"
        assert stats.count == 0
        assert stats.total_ms == 0.0
        assert stats.min_ms == float('inf')
        assert stats.max_ms == 0.0

    def test_record_single(self):
        """Test recording a single timing."""
        stats = TimingStats(name="test")
        stats.record(10.0)

        assert stats.count == 1
        assert stats.total_ms == 10.0
        assert stats.min_ms == 10.0
        assert stats.max_ms == 10.0

    def test_record_multiple(self):
        """Test recording multiple timings."""
        stats = TimingStats(name="test")
        stats.record(5.0)
        stats.record(15.0)
        stats.record(10.0)

        assert stats.count == 3
        assert stats.total_ms == 30.0
        assert stats.min_ms == 5.0
        assert stats.max_ms == 15.0

    def test_avg_ms(self):
        """Test average calculation."""
        stats = TimingStats(name="test")
        stats.record(5.0)
        stats.record(15.0)

        assert stats.avg_ms == 10.0

    def test_avg_ms_empty(self):
        """Test average with no recordings."""
        stats = TimingStats(name="test")

        assert stats.avg_ms == 0.0

    def test_recent_avg_ms(self):
        """Test recent average calculation."""
        stats = TimingStats(name="test")
        for i in range(10):
            stats.record(float(i + 1))

        # Average of 1-10 = 5.5
        assert stats.recent_avg_ms == 5.5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = TimingStats(name="test")
        stats.record(5.0)
        stats.record(15.0)

        d = stats.to_dict()

        assert d["count"] == 2
        assert d["avg_ms"] == 10.0
        assert d["min_ms"] == 5.0
        assert d["max_ms"] == 15.0


class TestPerformanceMetrics:
    """Unit tests for PerformanceMetrics class."""

    @pytest.fixture
    def metrics_instance(self):
        """Create a fresh metrics instance for testing."""
        return PerformanceMetrics(enable_logging=False)

    def test_initialization(self, metrics_instance):
        """Test metrics initialization."""
        assert len(metrics_instance._timings) == 0
        assert len(metrics_instance._counters) == 0
        assert len(metrics_instance._gauges) == 0

    def test_timer_context_manager(self, metrics_instance):
        """Test timer context manager."""
        with metrics_instance.timer("test_operation"):
            time.sleep(0.01)  # 10ms

        timing = metrics_instance.get_timing("test_operation")
        assert timing is not None
        assert timing.count == 1
        assert timing.avg_ms >= 10  # At least 10ms

    def test_timer_multiple_calls(self, metrics_instance):
        """Test timer with multiple calls."""
        for _ in range(3):
            with metrics_instance.timer("test_operation"):
                pass

        timing = metrics_instance.get_timing("test_operation")
        assert timing.count == 3

    def test_increment_counter(self, metrics_instance):
        """Test counter increment."""
        metrics_instance.increment("test_counter")
        metrics_instance.increment("test_counter")

        assert metrics_instance.get_counter("test_counter") == 2

    def test_increment_by_amount(self, metrics_instance):
        """Test counter increment by specific amount."""
        metrics_instance.increment("test_counter", amount=5)

        assert metrics_instance.get_counter("test_counter") == 5

    def test_get_counter_nonexistent(self, metrics_instance):
        """Test getting nonexistent counter returns 0."""
        assert metrics_instance.get_counter("nonexistent") == 0

    def test_set_gauge(self, metrics_instance):
        """Test gauge setting."""
        metrics_instance.set_gauge("test_gauge", 42.5)

        assert metrics_instance.get_gauge("test_gauge") == 42.5

    def test_gauge_overwrite(self, metrics_instance):
        """Test gauge overwrites previous value."""
        metrics_instance.set_gauge("test_gauge", 10.0)
        metrics_instance.set_gauge("test_gauge", 20.0)

        assert metrics_instance.get_gauge("test_gauge") == 20.0

    def test_get_gauge_nonexistent(self, metrics_instance):
        """Test getting nonexistent gauge returns 0."""
        assert metrics_instance.get_gauge("nonexistent") == 0.0

    def test_get_summary(self, metrics_instance):
        """Test summary retrieval."""
        metrics_instance.increment("counter1")
        metrics_instance.set_gauge("gauge1", 42.0)

        with metrics_instance.timer("operation1"):
            pass

        summary = metrics_instance.get_summary()

        assert "uptime_seconds" in summary
        assert "timings" in summary
        assert "counters" in summary
        assert "gauges" in summary
        assert "throughput" in summary

    def test_throughput_calculation(self, metrics_instance):
        """Test throughput calculation."""
        # Counters with _processed suffix are tracked for throughput
        for _ in range(10):
            metrics_instance.increment("samples_processed")

        time.sleep(0.1)  # Wait a bit for uptime

        summary = metrics_instance.get_summary()

        # Check throughput is calculated
        if summary["throughput"]:
            assert "samples_processed_per_sec" in summary["throughput"]

    def test_reset(self, metrics_instance):
        """Test metrics reset."""
        metrics_instance.increment("counter")
        metrics_instance.set_gauge("gauge", 42.0)

        with metrics_instance.timer("operation"):
            pass

        metrics_instance.reset()

        assert len(metrics_instance._timings) == 0
        assert len(metrics_instance._counters) == 0
        assert len(metrics_instance._gauges) == 0


class TestTimedDecorator:
    """Test the @timed decorator."""

    def test_timed_decorator(self):
        """Test @timed decorator records timing."""
        # Reset global metrics
        metrics.reset()

        @timed("decorated_function")
        def sample_function():
            return 42

        result = sample_function()

        assert result == 42

        timing = metrics.get_timing("decorated_function")
        assert timing is not None
        assert timing.count == 1

    def test_timed_decorator_preserves_return(self):
        """Test decorator preserves function return value."""
        @timed("test_func")
        def returns_value():
            return {"key": "value"}

        result = returns_value()
        assert result == {"key": "value"}

    def test_timed_decorator_multiple_calls(self):
        """Test decorator with multiple function calls."""
        metrics.reset()

        @timed("multi_call")
        def multi_function():
            pass

        for _ in range(5):
            multi_function()

        timing = metrics.get_timing("multi_call")
        assert timing.count == 5


class TestGlobalMetrics:
    """Test global metrics instance."""

    def test_global_instance_exists(self):
        """Test global metrics instance exists."""
        assert metrics is not None
        assert isinstance(metrics, PerformanceMetrics)

    def test_get_metrics_returns_same_instance(self):
        """Test get_metrics returns global instance."""
        instance = get_metrics()
        assert instance is metrics


class TestThreadSafety:
    """Test thread safety of metrics."""

    def test_concurrent_increments(self, ):
        """Test concurrent counter increments."""
        import threading

        m = PerformanceMetrics(enable_logging=False)
        iterations = 100
        threads = 4

        def increment_counter():
            for _ in range(iterations):
                m.increment("concurrent_counter")

        thread_list = [
            threading.Thread(target=increment_counter)
            for _ in range(threads)
        ]

        for t in thread_list:
            t.start()
        for t in thread_list:
            t.join()

        assert m.get_counter("concurrent_counter") == iterations * threads

    def test_concurrent_timers(self):
        """Test concurrent timer usage."""
        import threading

        m = PerformanceMetrics(enable_logging=False)

        def use_timer():
            for _ in range(10):
                with m.timer("concurrent_timer"):
                    time.sleep(0.001)

        threads = [
            threading.Thread(target=use_timer)
            for _ in range(4)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        timing = m.get_timing("concurrent_timer")
        assert timing.count == 40


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

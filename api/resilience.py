"""
Resilience patterns for WINDMAR API.

Provides circuit breakers, retry logic, and fallback mechanisms
for external service calls.
"""
import logging
import functools
import asyncio
from typing import TypeVar, Callable, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import threading

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    RetryError,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """
    Thread-safe circuit breaker implementation.

    Prevents cascading failures by stopping calls to failing services
    and allowing them time to recover.

    Usage:
        breaker = CircuitBreaker(name="copernicus_api")

        @breaker
        def call_external_service():
            ...
    """
    name: str
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    half_open_max_calls: int = 3

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _last_failure_time: Optional[datetime] = field(default=None, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    _half_open_calls: int = field(default=0, init=False)

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._check_state() == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting calls)."""
        return self._check_state() == CircuitState.OPEN

    def _check_state(self) -> CircuitState:
        """Check and potentially transition circuit state."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has elapsed
                if self._last_failure_time:
                    elapsed = (datetime.utcnow() - self._last_failure_time).total_seconds()
                    if elapsed >= self.recovery_timeout:
                        self._transition_to_half_open()

            return self._state

    def _transition_to_half_open(self):
        """Transition to half-open state for testing."""
        self._state = CircuitState.HALF_OPEN
        self._half_open_calls = 0
        logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")

    def _transition_to_open(self):
        """Transition to open state."""
        self._state = CircuitState.OPEN
        self._last_failure_time = datetime.utcnow()
        logger.warning(f"Circuit breaker '{self.name}' OPENED after {self._failure_count} failures")

    def _transition_to_closed(self):
        """Transition to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        logger.info(f"Circuit breaker '{self.name}' CLOSED - service recovered")

    def record_success(self):
        """Record a successful call."""
        with self._lock:
            self._success_count += 1

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1
                if self._half_open_calls >= self.half_open_max_calls:
                    self._transition_to_closed()
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self, error: Exception):
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.utcnow()

            logger.warning(f"Circuit breaker '{self.name}' recorded failure: {error}")

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self._transition_to_open()
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._transition_to_open()

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to wrap function with circuit breaker."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            state = self._check_state()

            if state == CircuitState.OPEN:
                raise CircuitOpenError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Service unavailable, try again in {self.recovery_timeout}s"
                )

            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure(e)
                raise

        return wrapper

    def call_async(self, func: Callable[..., T]) -> Callable[..., T]:
        """Async decorator to wrap function with circuit breaker."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            state = self._check_state()

            if state == CircuitState.OPEN:
                raise CircuitOpenError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Service unavailable, try again in {self.recovery_timeout}s"
                )

            try:
                result = await func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure(e)
                raise

        return wrapper

    def get_status(self) -> dict:
        """Get circuit breaker status."""
        with self._lock:
            return {
                'name': self.name,
                'state': self._state.value,
                'failure_count': self._failure_count,
                'success_count': self._success_count,
                'last_failure': self._last_failure_time.isoformat() if self._last_failure_time else None,
                'failure_threshold': self.failure_threshold,
                'recovery_timeout_seconds': self.recovery_timeout,
            }


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


# Pre-configured circuit breakers for common services
copernicus_breaker = CircuitBreaker(
    name="copernicus_api",
    failure_threshold=3,
    recovery_timeout=120,  # 2 minutes
)

external_api_breaker = CircuitBreaker(
    name="external_api",
    failure_threshold=5,
    recovery_timeout=60,
)


def with_retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 30.0,
    exceptions: tuple = (Exception,),
):
    """
    Decorator for adding retry logic with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
        exceptions: Tuple of exception types to retry on

    Usage:
        @with_retry(max_attempts=3, min_wait=1.0)
        def call_external_service():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
            retry=retry_if_exception_type(exceptions),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return func(*args, **kwargs)

        return wrapper

    return decorator


def with_retry_async(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 30.0,
    exceptions: tuple = (Exception,),
):
    """
    Async decorator for adding retry logic with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
        exceptions: Tuple of exception types to retry on
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
            retry=retry_if_exception_type(exceptions),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def with_fallback(fallback_value: T = None, fallback_func: Callable = None):
    """
    Decorator to provide fallback value or function on failure.

    Args:
        fallback_value: Static value to return on failure
        fallback_func: Function to call for fallback value (receives original args)

    Usage:
        @with_fallback(fallback_value={"status": "unavailable"})
        def call_external_service():
            ...

        @with_fallback(fallback_func=get_cached_value)
        def fetch_data():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Function {func.__name__} failed, using fallback: {e}")

                if fallback_func is not None:
                    return fallback_func(*args, **kwargs)
                return fallback_value

        return wrapper

    return decorator


def with_timeout(seconds: float):
    """
    Decorator to add timeout to synchronous functions.

    Note: Uses threading for timeout, may not interrupt all operations.
    For async functions, use asyncio.timeout instead.

    Args:
        seconds: Timeout in seconds
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=seconds)
                except concurrent.futures.TimeoutError:
                    raise TimeoutError(f"Function {func.__name__} timed out after {seconds}s")

        return wrapper

    return decorator


# Registry for all circuit breakers (for health monitoring)
_circuit_breaker_registry: dict[str, CircuitBreaker] = {}


def register_circuit_breaker(breaker: CircuitBreaker):
    """Register a circuit breaker for monitoring."""
    _circuit_breaker_registry[breaker.name] = breaker


def get_all_circuit_breaker_status() -> dict:
    """Get status of all registered circuit breakers."""
    return {
        name: breaker.get_status()
        for name, breaker in _circuit_breaker_registry.items()
    }


# Register default breakers
register_circuit_breaker(copernicus_breaker)
register_circuit_breaker(external_api_breaker)

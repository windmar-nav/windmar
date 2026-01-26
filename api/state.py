"""
Thread-safe state management for WINDMAR API.

Provides proper locking and isolation for shared state in concurrent environments.
This replaces the unsafe global state pattern with a singleton that ensures
thread safety and proper initialization.
"""
import threading
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class VesselState:
    """
    Thread-safe container for vessel-related state.

    Uses a lock to ensure atomic updates across all related objects
    (specs, model, calculators).
    """
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # Lazy imports to avoid circular dependencies
    _specs: Any = None
    _model: Any = None
    _voyage_calculator: Any = None
    _route_optimizer: Any = None
    _calibrator: Any = None
    _calibration: Any = None

    def __post_init__(self):
        """Initialize with default vessel specs."""
        self._initialize_defaults()

    def _initialize_defaults(self):
        """Initialize default vessel components."""
        from src.optimization.vessel_model import VesselModel, VesselSpecs
        from src.optimization.voyage import VoyageCalculator
        from src.optimization.route_optimizer import RouteOptimizer
        from src.optimization.vessel_calibration import VesselCalibrator

        self._specs = VesselSpecs()
        self._model = VesselModel(specs=self._specs)
        self._voyage_calculator = VoyageCalculator(vessel_model=self._model)
        self._route_optimizer = RouteOptimizer(vessel_model=self._model)
        self._calibrator = VesselCalibrator(vessel_specs=self._specs)
        self._calibration = None

    @property
    def specs(self):
        """Get vessel specs (thread-safe read)."""
        with self._lock:
            return self._specs

    @property
    def model(self):
        """Get vessel model (thread-safe read)."""
        with self._lock:
            return self._model

    @property
    def voyage_calculator(self):
        """Get voyage calculator (thread-safe read)."""
        with self._lock:
            return self._voyage_calculator

    @property
    def route_optimizer(self):
        """Get route optimizer (thread-safe read)."""
        with self._lock:
            return self._route_optimizer

    @property
    def calibrator(self):
        """Get calibrator (thread-safe read)."""
        with self._lock:
            return self._calibrator

    @property
    def calibration(self):
        """Get current calibration (thread-safe read)."""
        with self._lock:
            return self._calibration

    @contextmanager
    def update_lock(self):
        """
        Context manager for updating vessel state.

        Usage:
            with vessel_state.update_lock():
                vessel_state.update_specs(new_specs)
        """
        with self._lock:
            yield self

    def update_specs(self, specs_dict: Dict[str, Any]) -> None:
        """
        Update vessel specifications atomically.

        Args:
            specs_dict: Dictionary of vessel specification parameters
        """
        from src.optimization.vessel_model import VesselModel, VesselSpecs
        from src.optimization.voyage import VoyageCalculator
        from src.optimization.route_optimizer import RouteOptimizer

        with self._lock:
            self._specs = VesselSpecs(**specs_dict)
            self._model = VesselModel(specs=self._specs)
            self._voyage_calculator = VoyageCalculator(vessel_model=self._model)
            self._route_optimizer = RouteOptimizer(vessel_model=self._model)

            logger.info(f"Vessel specs updated: DWT={self._specs.dwt}")

    def update_calibration(self, calibration_factors: Any) -> None:
        """
        Update calibration factors atomically.

        Args:
            calibration_factors: CalibrationFactors instance
        """
        from src.optimization.vessel_model import VesselModel
        from src.optimization.voyage import VoyageCalculator
        from src.optimization.route_optimizer import RouteOptimizer

        with self._lock:
            self._calibration = calibration_factors

            # Rebuild model with calibration
            self._model = VesselModel(
                specs=self._specs,
                calibration_factors={
                    'calm_water': calibration_factors.calm_water,
                    'wind': calibration_factors.wind,
                    'waves': calibration_factors.waves,
                }
            )
            self._voyage_calculator = VoyageCalculator(vessel_model=self._model)
            self._route_optimizer = RouteOptimizer(vessel_model=self._model)

            logger.info("Vessel calibration updated")

    def get_snapshot(self) -> Dict[str, Any]:
        """
        Get a snapshot of current state for read operations.

        Returns a copy that can be used without holding the lock.
        """
        with self._lock:
            return {
                'specs': self._specs,
                'model': self._model,
                'voyage_calculator': self._voyage_calculator,
                'route_optimizer': self._route_optimizer,
                'calibrator': self._calibrator,
                'calibration': self._calibration,
            }


class ApplicationState:
    """
    Singleton application state manager.

    Centralizes all shared state with proper thread safety.
    Use get_app_state() to access the singleton instance.
    """

    _instance: Optional['ApplicationState'] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                # Double-check locking
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize application state (only once)."""
        if self._initialized:
            return

        self._initialized = True
        self._vessel_state = VesselState()
        self._weather_providers = None
        self._startup_time = datetime.utcnow()

        logger.info("Application state initialized")

    @property
    def vessel(self) -> VesselState:
        """Get vessel state manager."""
        return self._vessel_state

    @property
    def weather_providers(self):
        """
        Get weather providers (lazy initialization).

        Returns tuple of (copernicus, climatology, unified, synthetic)
        """
        if self._weather_providers is None:
            self._initialize_weather_providers()
        return self._weather_providers

    def _initialize_weather_providers(self):
        """Initialize weather data providers."""
        from src.data.copernicus import (
            CopernicusDataProvider,
            SyntheticDataProvider,
            ClimatologyProvider,
            UnifiedWeatherProvider,
        )

        copernicus = CopernicusDataProvider(cache_dir="data/copernicus_cache")
        climatology = ClimatologyProvider(cache_dir="data/climatology_cache")
        unified = UnifiedWeatherProvider(
            copernicus=copernicus,
            climatology=climatology,
            cache_dir="data/weather_cache",
        )
        synthetic = SyntheticDataProvider()

        self._weather_providers = {
            'copernicus': copernicus,
            'climatology': climatology,
            'unified': unified,
            'synthetic': synthetic,
        }

        logger.info("Weather providers initialized")

    @property
    def uptime_seconds(self) -> float:
        """Get application uptime in seconds."""
        return (datetime.utcnow() - self._startup_time).total_seconds()

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all components.

        Returns:
            Dict with health status of each component
        """
        return {
            'vessel_state': 'healthy' if self._vessel_state.specs is not None else 'unhealthy',
            'weather_providers': 'healthy' if self._weather_providers is not None else 'not_initialized',
            'uptime_seconds': self.uptime_seconds,
        }


def get_app_state() -> ApplicationState:
    """
    Get the application state singleton.

    This is the preferred way to access shared state throughout the application.

    Returns:
        ApplicationState: The singleton application state instance
    """
    return ApplicationState()


# Convenience aliases for backward compatibility
def get_vessel_state() -> VesselState:
    """Get the vessel state manager."""
    return get_app_state().vessel

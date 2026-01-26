"""
WINDMAR Configuration Module.

Centralized configuration management using environment variables.
Supports .env files for local development.

Usage:
    from src.config import settings

    print(settings.api_host)
    print(settings.copernicus_mock_mode)
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import logging

# Load .env file if present
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, use system env vars


def get_bool(key: str, default: bool = False) -> bool:
    """Get boolean from environment variable."""
    value = os.getenv(key, str(default)).lower()
    return value in ("true", "1", "yes", "on")


def get_float(key: str, default: float) -> float:
    """Get float from environment variable."""
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def get_int(key: str, default: int) -> int:
    """Get int from environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def get_list(key: str, default: str = "") -> List[str]:
    """Get list from comma-separated environment variable."""
    value = os.getenv(key, default)
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class Settings:
    """Application settings loaded from environment."""

    # API Configuration
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: get_int("API_PORT", 8000))
    api_reload: bool = field(default_factory=lambda: get_bool("API_RELOAD", False))
    api_log_level: str = field(default_factory=lambda: os.getenv("API_LOG_LEVEL", "info"))
    cors_origins: List[str] = field(
        default_factory=lambda: get_list("CORS_ORIGINS", "http://localhost:3000,http://localhost:3001")
    )

    # Copernicus Configuration
    copernicus_mock_mode: bool = field(default_factory=lambda: get_bool("COPERNICUS_MOCK_MODE", True))
    copernicus_username: Optional[str] = field(default_factory=lambda: os.getenv("COPERNICUS_USERNAME"))
    copernicus_password: Optional[str] = field(default_factory=lambda: os.getenv("COPERNICUS_PASSWORD"))

    # Calibration Configuration
    calibration_learning_rate: float = field(
        default_factory=lambda: get_float("CALIBRATION_LEARNING_RATE", 0.01)
    )
    calibration_persistence_path: Optional[str] = field(
        default_factory=lambda: os.getenv("CALIBRATION_PERSISTENCE_PATH")
    )

    # Simulation Defaults
    sim_wave_height_m: float = field(default_factory=lambda: get_float("SIM_WAVE_HEIGHT_M", 2.5))
    sim_wave_period_s: float = field(default_factory=lambda: get_float("SIM_WAVE_PERIOD_S", 8.0))
    sim_start_lat: float = field(default_factory=lambda: get_float("SIM_START_LAT", 43.5))
    sim_start_lon: float = field(default_factory=lambda: get_float("SIM_START_LON", 7.0))
    sim_speed_kts: float = field(default_factory=lambda: get_float("SIM_SPEED_KTS", 12.0))
    sim_heading_deg: float = field(default_factory=lambda: get_float("SIM_HEADING_DEG", 270.0))

    # Data Storage
    grib_cache_dir: str = field(default_factory=lambda: os.getenv("GRIB_CACHE_DIR", "data/grib_cache"))
    data_dir: str = field(default_factory=lambda: os.getenv("DATA_DIR", "data"))

    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_format: str = field(
        default_factory=lambda: os.getenv(
            "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )

    def __post_init__(self):
        """Validate settings after initialization."""
        # Validate learning rate bounds
        if not 0.001 <= self.calibration_learning_rate <= 0.5:
            logging.warning(
                f"Calibration learning rate {self.calibration_learning_rate} outside "
                f"recommended range [0.001, 0.5], using 0.01"
            )
            self.calibration_learning_rate = 0.01

        # Warn if Copernicus credentials missing in live mode
        if not self.copernicus_mock_mode:
            if not self.copernicus_username or not self.copernicus_password:
                logging.warning(
                    "Copernicus credentials not set, falling back to mock mode. "
                    "Set COPERNICUS_USERNAME and COPERNICUS_PASSWORD for live data."
                )
                self.copernicus_mock_mode = True

    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.api_reload and self.api_log_level != "debug"

    def configure_logging(self):
        """Configure logging based on settings."""
        level = getattr(logging, self.log_level.upper(), logging.INFO)
        logging.basicConfig(level=level, format=self.log_format)


# Singleton instance
settings = Settings()


# Convenience function for testing
def get_settings() -> Settings:
    """Get the settings instance (useful for dependency injection)."""
    return settings

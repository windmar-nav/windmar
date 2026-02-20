"""
Configuration management for WINDMAR API.
Loads environment variables and provides typed configuration.
"""
import os
from typing import List, Optional
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ========================================================================
    # Database Configuration
    # ========================================================================
    database_url: str = "sqlite:///./windmar.db"
    db_echo: bool = False
    demo_mode: bool = False
    demo_api_key_hash: Optional[str] = None

    # ========================================================================
    # Redis Configuration
    # ========================================================================
    redis_url: str = "redis://localhost:6379/0"
    redis_enabled: bool = True

    # ========================================================================
    # API Configuration
    # ========================================================================
    api_secret_key: str = "dev_secret_key_change_in_production"
    api_key_header: str = "X-API-Key"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4

    # ========================================================================
    # CORS Configuration
    # ========================================================================
    cors_origins: str = "http://localhost:3000,http://localhost:3001"
    cors_credentials: bool = True
    cors_methods: str = "GET,POST,PUT,DELETE,OPTIONS"
    cors_headers: str = "*"

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    # ========================================================================
    # Security Configuration
    # ========================================================================
    auth_enabled: bool = True
    session_timeout: int = 3600
    bcrypt_rounds: int = 12

    # ========================================================================
    # Rate Limiting
    # ========================================================================
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000

    # ========================================================================
    # Application Configuration
    # ========================================================================
    environment: str = "development"
    log_level: str = "info"
    debug: bool = False

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment.lower() == "development"

    # ========================================================================
    # Copernicus Weather Data
    # ========================================================================
    # CDS API (ERA5 wind data) — register at https://cds.climate.copernicus.eu
    cdsapi_url: str = "https://cds.climate.copernicus.eu/api"
    cdsapi_key: Optional[str] = None

    # CMEMS (wave/current data) — register at https://marine.copernicus.eu
    copernicusmarine_service_username: Optional[str] = None
    copernicusmarine_service_password: Optional[str] = None

    @property
    def has_cds_credentials(self) -> bool:
        return self.cdsapi_key is not None

    @property
    def has_cmems_credentials(self) -> bool:
        return (self.copernicusmarine_service_username is not None
                and self.copernicusmarine_service_password is not None)

    # ========================================================================
    # Performance Configuration
    # ========================================================================
    max_calculation_time: int = 300
    cache_enabled: bool = True
    cache_ttl: int = 3600

    # ========================================================================
    # Monitoring Configuration
    # ========================================================================
    sentry_dsn: Optional[str] = None
    metrics_enabled: bool = False

    # ========================================================================
    # Pydantic Settings Configuration
    # ========================================================================
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings: Application settings
    """
    return Settings()


# Convenience exports
settings = get_settings()

# Validate critical settings in production (skip for demo deployments)
if settings.is_production and not settings.demo_mode:
    if settings.api_secret_key == "dev_secret_key_change_in_production":
        raise ValueError(
            "API_SECRET_KEY must be changed in production! "
            "Generate one with: openssl rand -hex 32"
        )

    if not settings.auth_enabled:
        raise ValueError("AUTH_ENABLED must be true in production!")

    if "localhost" in settings.cors_origins.lower():
        raise ValueError(
            "CORS_ORIGINS must not include localhost in production!"
        )

"""
WINDMAR API Pydantic schemas.

Re-exports all schema classes for backward-compatible imports:
    from api.schemas import Position, VoyageRequest, ...
"""

# Common
from .common import Position, WaypointModel, RouteModel  # noqa: F401

# Weather
from .weather import WindDataPoint, WeatherGridResponse, VelocityDataResponse  # noqa: F401

# Voyage
from .voyage import (  # noqa: F401
    VoyageRequest,
    LegResultModel,
    DataSourceSummary,
    VoyageResponse,
    MonteCarloRequest,
    PercentileFloat,
    PercentileString,
    MonteCarloResponse,
    # Voyage History
    SaveVoyageLeg,
    SaveVoyageRequest,
    VoyageLegResponse,
    VoyageSummaryResponse,
    VoyageDetailResponse,
    VoyageListResponse,
    NoonReportEntry,
    NoonReportsResponse,
    DepartureReportData,
    ArrivalReportData,
    VoyageReportsResponse,
)

# Optimization
from .optimization import (  # noqa: F401
    WeatherProvenanceModel,
    OptimizationRequest,
    OptimizationLegModel,
    SafetySummary,
    SpeedScenarioModel,
    ParetoSolutionModel,
    OptimizationResponse,
    BenchmarkRequest,
    BenchmarkEngineResult,
    BenchmarkResponse,
)

# Vessel
from .vessel import (  # noqa: F401
    VesselConfig,
    NoonReportModel,
    CalibrationFactorsModel,
    CalibrationResponse,
    PerformancePredictionRequest,
)

# Zones
from .zones import ZoneCoordinate, CreateZoneRequest, ZoneResponse  # noqa: F401

# CII
from .cii import (  # noqa: F401
    CIIFuelConsumption,
    CIICalculateRequest,
    CIIProjectRequest,
    CIIReductionRequest,
    CIISpeedSweepRequest,
    CIISpeedSweepPoint,
    CIISpeedSweepResponse,
    CIIThresholdYear,
    CIIThresholdsResponse,
    CIIFleetVessel,
    CIIFleetRequest,
    CIIFleetResult,
    CIIFleetResponse,
)

# FuelEU Maritime
from .fueleu import (  # noqa: F401
    FuelEUFuelConsumption,
    FuelEUCalculateRequest,
    FuelEUCalculateResponse,
    FuelEUComplianceRequest,
    FuelEUComplianceResponse,
    FuelEUPenaltyRequest,
    FuelEUPenaltyResponse,
    FuelEUPoolingVessel,
    FuelEUPoolingRequest,
    FuelEUPoolingResponse,
    FuelEUProjectRequest,
    FuelEUProjectResponse,
    FuelEUProjectionYear,
    FuelEULimitYear,
    FuelEULimitsResponse,
    FuelEUFuelInfo,
    FuelEUFuelTypesResponse,
)

# Engine Log
from .engine_log import (  # noqa: F401
    EngineLogUploadResponse,
    EngineLogEntryResponse,
    EngineLogSummaryResponse,
    EngineLogCalibrateResponse,
)

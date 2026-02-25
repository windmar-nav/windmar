"""
Weather field registry — single source of truth for all weather layers.

Each field defines: source, DB parameters, component type, resolution,
forecast hours, default bounding box, and display metadata (colorscale, units).

The router, ingestion service, and frontend all derive their behaviour
from this registry instead of maintaining parallel per-field code paths.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Bump this when cache envelope shape/semantics change.
# Stale caches with older versions are discarded on read.
CACHE_SCHEMA_VERSION = 5


@dataclass(frozen=True)
class FieldConfig:
    """Immutable configuration for one weather field."""

    # Identity
    name: str                       # e.g. "wind", "waves", "sst"
    source: str                     # DB source key: "gfs", "cmems_wave", etc.

    # DB parameter names stored in weather_grid_data
    parameters: Tuple[str, ...]

    # How to interpret the parameters
    # "vector"      → (u, v) → speed + direction
    # "scalar"      → single data grid
    # "wave_decomp" → wave_hs/dir + swell + windwave decomposition
    components: str                 # "vector" | "scalar" | "wave_decomp"

    # Grid properties
    native_resolution: float        # degrees (0.5 for GFS, 0.083 for CMEMS)

    # Forecast timeline
    forecast_hours: Tuple[int, ...]  # e.g. (0, 3, 6, ..., 120)
    expected_frames: int             # minimum frames to consider "complete"

    # Default bounding box (when no viewport is supplied)
    default_bbox: Tuple[float, float, float, float]  # (lat_min, lat_max, lon_min, lon_max)

    # Display
    unit: str
    needs_ocean_mask: bool = True
    nan_fill: float = -999.0        # sentinel for NaN in JSON output

    # Colorscale for frontend rendering
    colorscale_min: float = 0.0
    colorscale_max: float = 1.0
    colorscale_colors: Tuple[str, ...] = ()

    # Subsample target: max grid points per axis in API responses
    subsample_target: int = 250

    # Rounding precision for JSON output
    decimals: int = 2

    # Cache subdirectory name (for ForecastLayerManager)
    cache_subdir: str = ""

    # Provider fetch method name on copernicus/gfs provider
    fetch_method: str = ""

    # Ingestion method name on WeatherIngestionService
    ingest_method: str = ""

    # Use Redis for distributed lock (False for GFS wind which uses file-based)
    use_redis: bool = True


# ---------------------------------------------------------------------------
# Standard forecast hours
# ---------------------------------------------------------------------------
_GFS_HOURS = tuple(range(0, 121, 3))       # 0, 3, 6, ..., 120 → 41 frames
_ICE_HOURS = tuple(range(0, 217, 24))       # 0, 24, 48, ..., 216 → 10 frames

# ---------------------------------------------------------------------------
# Default bounding boxes
# ---------------------------------------------------------------------------
_GLOBAL_BBOX = (-85.0, 85.0, -179.75, 179.75)
_CMEMS_BBOX = (20.0, 65.0, -35.0, 45.0)   # NE Atlantic + Med + Nordic
_SST_BBOX = (20.0, 65.0, -35.0, 45.0)     # Same as CMEMS
_ICE_BBOX = (55.0, 80.0, -35.0, 45.0)     # High latitude — wider for Arctic

# ---------------------------------------------------------------------------
# Field definitions
# ---------------------------------------------------------------------------

WEATHER_FIELDS: Dict[str, FieldConfig] = {

    "wind": FieldConfig(
        name="wind",
        source="gfs",
        parameters=("wind_u", "wind_v"),
        components="vector",
        native_resolution=0.5,
        forecast_hours=_GFS_HOURS,
        expected_frames=41,
        default_bbox=_GLOBAL_BBOX,
        unit="m/s",
        needs_ocean_mask=True,
        colorscale_min=0, colorscale_max=30,
        colorscale_colors=("#3b82f6", "#22d3ee", "#a3e635", "#facc15", "#ef4444"),
        subsample_target=500,
        decimals=2,
        cache_subdir="wind",
        fetch_method="fetch_wind_data",
        ingest_method="ingest_wind",
        use_redis=False,
    ),

    "waves": FieldConfig(
        name="waves",
        source="cmems_wave",
        parameters=("wave_hs", "wave_dir", "swell_hs", "swell_tp", "swell_dir",
                     "windwave_hs", "windwave_tp", "windwave_dir"),
        components="wave_decomp",
        native_resolution=0.083,
        forecast_hours=_GFS_HOURS,
        expected_frames=41,
        default_bbox=_CMEMS_BBOX,
        unit="m",
        needs_ocean_mask=True,
        colorscale_min=0, colorscale_max=6,
        colorscale_colors=("#00ff00", "#ffff00", "#ff8800", "#ff0000", "#800000"),
        subsample_target=250,
        decimals=2,
        cache_subdir="wave",
        fetch_method="fetch_wave_forecast",
        ingest_method="ingest_waves",
    ),

    "swell": FieldConfig(
        name="swell",
        source="cmems_wave",  # shares wave data
        parameters=("wave_hs", "wave_dir", "swell_hs", "swell_tp", "swell_dir",
                     "windwave_hs", "windwave_tp", "windwave_dir"),
        components="wave_decomp",
        native_resolution=0.083,
        forecast_hours=_GFS_HOURS,
        expected_frames=41,
        default_bbox=_CMEMS_BBOX,
        unit="m",
        needs_ocean_mask=True,
        colorscale_min=0, colorscale_max=6,
        colorscale_colors=("#00ff00", "#ffff00", "#ff8800", "#ff0000", "#800000"),
        subsample_target=250,
        decimals=2,
        cache_subdir="wave",  # shares wave cache
        fetch_method="fetch_wave_forecast",
        ingest_method="ingest_waves",
    ),

    "currents": FieldConfig(
        name="currents",
        source="cmems_current",
        parameters=("current_u", "current_v"),
        components="vector",
        native_resolution=0.083,
        forecast_hours=_GFS_HOURS,
        expected_frames=41,
        default_bbox=_CMEMS_BBOX,
        unit="m/s",
        needs_ocean_mask=True,
        colorscale_min=0, colorscale_max=2,
        colorscale_colors=("#22d3ee", "#3b82f6", "#8b5cf6", "#d946ef"),
        subsample_target=250,
        decimals=2,
        cache_subdir="current",
        fetch_method="fetch_current_forecast",
        ingest_method="ingest_currents",
    ),

    "sst": FieldConfig(
        name="sst",
        source="cmems_sst",
        parameters=("sst",),
        components="scalar",
        native_resolution=0.083,
        forecast_hours=_GFS_HOURS,
        expected_frames=41,
        default_bbox=_SST_BBOX,
        unit="\u00b0C",
        needs_ocean_mask=True,
        nan_fill=-999.0,
        colorscale_min=-2, colorscale_max=32,
        colorscale_colors=("#0000ff", "#00ccff", "#00ff88", "#ffff00", "#ff8800", "#ff0000"),
        subsample_target=250,
        decimals=2,
        cache_subdir="sst",
        fetch_method="fetch_sst_forecast",
        ingest_method="ingest_sst",
    ),

    "visibility": FieldConfig(
        name="visibility",
        source="gfs_visibility",
        parameters=("visibility",),
        components="scalar",
        native_resolution=0.5,
        forecast_hours=_GFS_HOURS,
        expected_frames=41,
        default_bbox=_GLOBAL_BBOX,
        unit="km",
        needs_ocean_mask=True,
        nan_fill=-999.0,
        colorscale_min=0, colorscale_max=50,
        colorscale_colors=("#ff0000", "#ff8800", "#ffff00", "#88ff00", "#00ff00"),
        subsample_target=250,
        decimals=1,
        cache_subdir="vis",
        fetch_method="fetch_visibility_forecast",
        ingest_method="ingest_visibility",
    ),

    "ice": FieldConfig(
        name="ice",
        source="cmems_ice",
        parameters=("ice_siconc",),
        components="scalar",
        native_resolution=0.083,
        forecast_hours=_ICE_HOURS,
        expected_frames=9,
        default_bbox=_ICE_BBOX,
        unit="fraction",
        needs_ocean_mask=True,
        nan_fill=-999.0,
        colorscale_min=0, colorscale_max=1,
        colorscale_colors=("#ffffff", "#ccddff", "#6688ff", "#0033cc", "#001166"),
        subsample_target=250,
        decimals=4,
        cache_subdir="ice",
        fetch_method="fetch_ice_forecast",
        ingest_method="ingest_ice",
    ),
}

# All valid field names
FIELD_NAMES = tuple(WEATHER_FIELDS.keys())

# Reverse lookup: source → field name (first match; swell shares wave source)
SOURCE_TO_FIELD = {cfg.source: cfg.name for cfg in WEATHER_FIELDS.values()}

# Layer → source mapping (backwards compat for resync endpoint)
LAYER_TO_SOURCE = {name: cfg.source for name, cfg in WEATHER_FIELDS.items()}

# Layer → ingest method name
LAYER_TO_INGEST = {name: cfg.ingest_method for name, cfg in WEATHER_FIELDS.items()}


def get_field(name: str) -> FieldConfig:
    """Look up a field by name. Raises KeyError if not found."""
    return WEATHER_FIELDS[name]


def validate_field_name(name: str) -> str:
    """Validate and return a field name. Raises ValueError with available names."""
    if name not in WEATHER_FIELDS:
        raise ValueError(
            f"Unknown weather field: {name!r}. "
            f"Valid fields: {', '.join(FIELD_NAMES)}"
        )
    return name

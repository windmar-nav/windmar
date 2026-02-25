# Weather Data Pipeline — Architecture & Limitations

Technical reference for the WindMar weather ingestion, storage, caching, visualization, and downsampling pipeline.

Last updated: 2026-02-25 (v0.1.2 — data-derived ocean mask, higher resolution grids, wider bboxes, cache versioning)

---

## 1. Data Sources

| Source Key | Label | Provider | Native Resolution | Forecast Range | Frames |
|---|---|---|---|---|---|
| `gfs` | Wind (U/V) | NOAA GFS via NOMADS | 0.50° | 0–120 h, 3 h step | 41 |
| `gfs_visibility` | Visibility | NOAA GFS via NOMADS | 0.50° | 0–120 h, 3 h step | 41 |
| `cmems_wave` | Waves + Swell | Copernicus Marine | 0.083° | 0–120 h, 3 h step | 41 |
| `cmems_current` | Currents (U/V) | Copernicus Marine | 0.083° | 0–120 h, 3 h step | 41 |
| `cmems_sst` | Sea Surface Temp | Copernicus Marine | 0.083° | 0–120 h, 3 h step | 41 |
| `cmems_ice` | Ice concentration | Copernicus Marine | ~1.3° lat × ~0.29° lon | 0–216 h, 24 h step | 9 |

**Note on ice resolution:** CMEMS advertises the ice product as 0.083° but delivers a coarser native grid (~19×280 for the Arctic region 55–80°N). This is the actual data resolution from the provider, not a bug in Windmar.

### Field Registry

All weather layers are defined in `api/weather_fields.py` as `FieldConfig` dataclass instances. This single file is the source of truth for: source key, DB parameters, component type, resolution, forecast hours, default bounding box, colorscale, NaN sentinel, subsampling target, and cache/fetch/ingest method names.

```python
WEATHER_FIELDS = {
    "wind":       FieldConfig(source="gfs",           components="vector",     ...),
    "waves":      FieldConfig(source="cmems_wave",     components="wave_decomp",...),
    "swell":      FieldConfig(source="cmems_wave",     components="wave_decomp",...),
    "currents":   FieldConfig(source="cmems_current",  components="vector",     ...),
    "sst":        FieldConfig(source="cmems_sst",      components="scalar",     ...),
    "visibility": FieldConfig(source="gfs_visibility", components="scalar",     ...),
    "ice":        FieldConfig(source="cmems_ice",      components="scalar",     ...),
}
```

### Cache Schema Versioning

`CACHE_SCHEMA_VERSION` (currently **5**) is stamped into every cached frame envelope. When the API reads a cache file, it compares the stamped version to the current code version. Stale caches are auto-discarded and rebuilt from PostgreSQL. Bump this version whenever:
- Grid subsampling parameters change
- NaN sentinel values change
- Bounding box defaults change
- Ocean mask derivation method changes

---

## 2. Architecture Overview

The pipeline uses a **user-triggered overlay model**: no background ingestion loops, no startup health gates. Weather data loads on demand when the user activates a layer.

**Data flow:**
1. User clicks a weather layer button (Wind, Waves, etc.)
2. Backend checks PostgreSQL for existing data
3. If data exists in DB → return it with `ingested_at` timestamp
4. If DB is empty → fetch from external API (GFS/CMEMS), ingest into DB, return with `ingested_at = now()`
5. Staleness indicator shows data age in the overlay controls
6. User clicks Resync → per-layer re-fetch with `force=True`, fresh data returned

---

## 3. Source Independence — Architectural Decision

The 7 weather sources are **logically independent within a shared database**.

All sources share two PostgreSQL tables (`weather_forecast_runs` + `weather_grid_data`), discriminated by a `source` column. Every DB query is scoped to a single source via `WHERE source = %s`. No cross-source JOINs exist anywhere in the codebase.

| Operation | Scope | Cross-source impact |
|---|---|---|
| `ingest_wind(force=True)` | Supersedes only `gfs` runs | None |
| `POST /api/weather/wind/resync` | Supersedes + cleans only `gfs` | None |
| `_supersede_old_runs(source)` | Marks old runs for given source | None |
| `cleanup_orphaned_grid_data(source)` | Deletes dead data for given source | None |

---

## 4. Storage Tiers

### Tier 1 — PostgreSQL (persistent, authoritative)

- `weather_forecast_runs` — one row per source per ingestion cycle
- `weather_grid_data` — one row per (run, forecast_hour, parameter), zlib-compressed float32 arrays

### Tier 2 — File Cache (ephemeral, fast reads)

- **Location:** `/tmp/windmar_cache/{wind,wave,current,ice,sst,vis}/`
- **Format:** JSON keyed by padded viewport: `{layer}_{lat_min}_{lat_max}_{lon_min}_{lon_max}.json`
- **TTL:** 12 hours + auto-invalidated by `CACHE_SCHEMA_VERSION`

### Tier 3 — Redis (shared in-memory cache)

- Single-timestep snapshots for map overlay rendering
- TTL: 60 minutes

### Tier 4 — Raw Download Cache

| Cache | Location | TTL |
|---|---|---|
| CMEMS NetCDF | `data/copernicus_cache/*.nc` | 24 hours |
| GFS GRIB2 | `data/gfs_cache/*.grib2` | 48 hours |

---

## 5. Bounding Boxes & OOM Protection

### Default Bounding Boxes (ingestion when no viewport provided)

| Source | Default Bbox | Rationale |
|---|---|---|
| `gfs` (wind) | Global (−85 to 85°N, −180 to 180°E) | GFS at 0.5° is lightweight |
| `gfs_visibility` | Global | Same as wind |
| `cmems_wave` | 20–65°N, −35 to 45°E | NE Atlantic + Med + Nordic |
| `cmems_current` | 20–65°N, −35 to 45°E | Same as wave |
| `cmems_sst` | 20–65°N, −35 to 45°E | Same as wave |
| `cmems_ice` | 55–80°N, −35 to 45°E | Arctic — wider for high latitudes |

### Bbox Clamping (`_clamp_bbox`)

Both the backend and frontend enforce maximum bbox spans to prevent OOM:

| Parameter | Value | Location |
|---|---|---|
| `_MAX_LAT_SPAN` | **50°** | `api/routers/weather.py` |
| `_MAX_LON_SPAN` | **80°** | `api/routers/weather.py` |
| `MAX_LAT_SPAN` | **50** | `frontend/components/ForecastTimeline.tsx` |
| `MAX_LON_SPAN` | **80** | `frontend/components/ForecastTimeline.tsx` |

The clamping is centered on the viewport midpoint. If the viewport exceeds the span limit, the bbox is symmetrically shrunk.

### Coordinate Clamping (3-layer defense)

1. **Frontend `paddedBounds()`** — clamps lat to ±89.9°, lon to ±180° after 10° grid-snapping
2. **Backend `_clamp_bbox()`** — clamps after centering/capping
3. **`build_ocean_mask()`** — last-resort clamp before any mask computation

---

## 6. Timeline Frame Subsampling

### Subsampling Caps

Timeline frames are subsampled server-side to keep payloads within browser limits. Three caps handle different data densities:

| Cap Constant | Value | Used By | Rationale |
|---|---|---|---|
| `_FRAMES_MAX_DIM` | **200** | Wind, Currents (2 arrays/frame) | ~5 MB per layer |
| `_WAVE_DECOMP_MAX_DIM` | **60** | Waves, Swell (8 arrays/frame) | ~4.5 MB per layer |
| `_SCALAR_FRAMES_MAX_DIM` | **350** | SST, Visibility, Ice (1 array/frame) | ~17 MB per layer |
| `_OVERLAY_MAX_DIM` | **500** | Single-frame overlays | ~5 MB per request |

The subsample step is: `STEP = ceil(max(len(lats), len(lons)) / MAX_DIM)`

### Actual Layer Sizes (for default bbox 20–65°N, −35 to 45°E)

| Layer | Raw Grid (from DB) | Step | Subsampled Grid | API Response | Frames |
|---|---|---|---|---|---|
| **Wind** | 101 × 161 (GFS 0.5°) | 1 | 101 × 161 | ~14 MB | 41 |
| **Waves** | ~542 × 964 (CMEMS 0.083°) | 17 | 32 × 57 | ~3.3 MB | 41 |
| **Swell** | (shares wave data) | 17 | 32 × 57 | ~3.3 MB | 41 |
| **Currents** | ~542 × 964 | 5 | 109 × 193 | ~9.5 MB | 41 |
| **SST** | ~542 × 964 | 3 | 181 × 321 | ~17 MB | 41 |
| **Visibility** | ~91 × 161 (GFS 0.5°) | 1 | 91 × 161 | ~14 MB | 41 |
| **Ice** | 19 × 280 (native coarse) | 1 | 19 × 280 | ~0.3 MB | 9 |

**Total if all layers loaded:** ~61 MB. In practice, only 1–2 layers are active at a time.

### Memory Management on Layer Switch

When the user switches weather layers, the frontend:
1. Nulls `extendedWeatherData` (single-frame overlay data) immediately
2. Clears all 6 frame data stores and their refs in ForecastTimeline
3. Then loads the new layer's data

This ensures only **one layer's data** exists in browser memory at any time, preventing the cumulative ~61 MB worst case.

---

## 7. Ocean Mask

### Data-Derived Mask (current approach, since `CACHE_SCHEMA_VERSION=5`)

The ocean mask is derived from the **actual weather data**, not from an external land/sea database. For each grid cell, if ANY forecast frame has a valid (non-NaN, finite) value, the cell is marked as ocean.

```python
data_ocean = np.zeros((len(lats_full), len(lons_full)), dtype=bool)
for fh in sorted(hours):
    _, _, d = grids[primary_param][fh]
    data_ocean |= np.isfinite(d)
ocean_mask_data = data_ocean[::mask_step, ::mask_step].tolist()
```

**Why not `global_land_mask`?** The previous approach used the `global_land_mask` Python library, which has a fixed 1km raster. At coastlines, this library incorrectly marks valid CMEMS ocean grid cells as land, creating black holes in SST, wave, and current visualizations — particularly visible in the Mediterranean, English Channel, and North Sea. The data-derived mask eliminates these artifacts because CMEMS itself knows exactly where its ocean grid points are.

**Exception — wind:** Wind uses `_apply_ocean_mask_velocity()` with `global_land_mask` to zero out U/V over land (GFS provides wind data globally including land; zeroing land prevents arrows over continents). This is kept because wind data has no NaN over land to derive a mask from.

### Frontend Mask Usage

In `WeatherGridLayer.tsx`, for each pixel the renderer checks `oceanMask[latIdx][lonIdx]`. If `false` (land), the pixel is set to transparent. The mask grid coordinates are interpolated to the pixel position using linear index mapping.

---

## 8. NaN Sentinel Strategy

All scalar layers use a consistent NaN fill value of **`-999.0`**. This is applied server-side when building frames:

```python
clean = np.nan_to_num(vals[::S_STEP, ::S_STEP], nan=cfg.nan_fill)  # nan_fill = -999.0
```

The frontend filters these with a generic check: `if (val < -100) continue;` — this catches all sentinel values without per-layer special cases.

| Layer | `nan_fill` | Frontend filter |
|---|---|---|
| SST | −999.0 | `val < -100` |
| Visibility | −999.0 | `val < -100` |
| Ice | −999.0 | `val < -100` |

---

## 9. Debug Endpoint

`GET /api/weather/{field}/debug?lat_min=...&lat_max=...&lon_min=...&lon_max=...`

Returns lightweight diagnostics for any layer's cached data WITHOUT the full frame payload:

```json
{
  "field": "sst",
  "source": "cmems",
  "run_time": "2026-02-25T19:57:25+00:00",
  "schema_version": 5,
  "frame_count": 41,
  "lats_len": 181, "lats_first": 20.0, "lats_last": 65.0,
  "lons_len": 321, "lons_first": -35.0, "lons_last": 45.0,
  "ny": 181, "nx": 321,
  "sample_frame_data_rows": 181, "sample_frame_data_cols": 321,
  "has_ocean_mask": true, "ocean_mask_shape": "181x321",
  "checks": [
    {"check": "ny == len(lats)", "pass": true},
    {"check": "nx == len(lons)", "pass": true},
    {"check": "frame_data_rows == ny", "pass": true},
    {"check": "frame_data_cols == nx", "pass": true},
    {"check": "ocean_mask rows == mask_lats", "pass": true},
    {"check": "ocean_mask cols == mask_lons", "pass": true},
    {"check": "schema_version current", "pass": true}
  ],
  "all_checks_pass": true
}
```

Use to verify data integrity after any backend change: `curl -s localhost:8003/api/weather/sst/debug | python3 -m json.tool`

---

## 10. API Endpoints (Weather Pipeline)

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/weather/{field}` | GET | Single-frame grid data (DB-first, API fallback) |
| `/api/weather/{field}/debug` | GET | Lightweight diagnostics (shapes, checks) |
| `/api/weather/{field}/resync` | POST | Per-layer re-ingest (accepts viewport bbox) |
| `/api/weather/{field}/frames` | GET | Generic: all forecast frames for any field |
| `/api/weather/forecast/frames` | GET | Compat: wind timeline frames |
| `/api/weather/forecast/wave/frames` | GET | Compat: wave timeline frames |
| `/api/weather/forecast/current/frames` | GET | Compat: current timeline frames |
| `/api/weather/forecast/ice/frames` | GET | Compat: ice timeline frames |
| `/api/weather/forecast/sst/frames` | GET | Compat: SST timeline frames |
| `/api/weather/forecast/visibility/frames` | GET | Compat: visibility timeline frames |
| `/api/weather/wind/velocity` | GET | Wind velocity for particle animation |
| `/api/weather/currents/velocity` | GET | Current velocity for particle animation |
| `/api/weather/forecast/prefetch` | POST | Trigger wind forecast file cache build |
| `/api/weather/forecast/status` | GET | Wind prefetch progress |
| `/api/weather/health` | GET | Per-source health check |

---

## 11. Frontend Data Flow

### Layer Activation
1. Frontend calls the layer's grid endpoint
2. Backend: DB first → if empty, fetch from API → return JSON with `ingested_at`
3. Frontend renders grid on Leaflet map via `WeatherGridLayer` (Canvas 2D)

### Timeline Activation
1. Frontend calls the layer's `/frames` endpoint with `paddedBounds()`
2. Backend serves from file cache (schema version validated) or rebuilds from DB
3. All frames loaded into browser memory — scrubbing is instant (client-side)
4. Slider auto-positions to nearest-to-now forecast hour

### Race Condition Guard
When the forecast timeline is active (`forecastEnabled`), single-frame auto-reload on layer change is **skipped**. Without this guard, the async single-frame response arrives ~300ms later and overwrites the timeline's frame data.

### This is NOT Windy.com

WindMar is a **local-first** application:
- All data fetched on demand from NOAA GFS / Copernicus Marine into local PostgreSQL
- Full forecast dataset lives in browser memory (no tile streaming)
- Heatmap + particles rendered client-side via Canvas 2D
- Data stops at grid edges when panning — by design (no global pre-rendered tiles)

---

## 12. Limitations Summary

| Limitation | Impact | Severity |
|---|---|---|
| **First-time CMEMS layer load takes 30–120s** | User waits on first activation if DB empty | Medium |
| **CMEMS bbox capped at 50° × 80°** | Data centered on viewport, may not cover full screen for very wide views | Medium |
| **Ice resolution coarse (1.3° lat)** | Only ~19 latitude points for Arctic region — sparse visualization | Low |
| **Wave/swell grid coarse (32×57)** | 8 arrays/frame limits subsampling target to 60 | Medium |
| **SST/visibility payloads ~17 MB** | Noticeable load time on slow connections | Low |
| **Pan/zoom does not auto-refetch timeline** | Overlay truncated at grid edges — user must Resync + reopen | By design |
| **Wind arrows over narrow straits** | `global_land_mask` zeroing may hide valid coastal wind | Low |
| **GFS publishes progressively** | Wind may have <41 frames for recent model runs | Informational |

---

## 13. File Reference

| File | Role |
|---|---|
| `api/weather_fields.py` | Field registry: all layer configs, CACHE_SCHEMA_VERSION |
| `api/routers/weather.py` | All weather API endpoints, frame builders, debug endpoint, resync |
| `api/forecast_layer_manager.py` | File cache management (read/write/serve) |
| `api/weather_service.py` | Ocean mask builder, velocity masking |
| `src/data/copernicus.py` | CMEMS data provider (waves, currents, SST, ice) |
| `src/data/gfs_provider.py` | GFS GRIB data provider (wind, visibility) |
| `src/data/weather_ingestion.py` | DB ingestion (all sources) |
| `src/data/db_weather_provider.py` | DB query layer (grids, timeline, latest run) |
| `frontend/components/WeatherGridLayer.tsx` | Canvas tile renderer for all weather layers |
| `frontend/components/ForecastTimeline.tsx` | Timeline component, paddedBounds, frame loading |
| `frontend/hooks/useWeatherDisplay.ts` | Layer state, auto-reload, forecast handlers |

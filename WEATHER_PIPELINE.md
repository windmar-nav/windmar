# Weather Data Pipeline — Architecture & Limitations

Technical reference for the WindMar weather ingestion, storage, caching, and visualization pipeline.

Last updated: 2026-02-17 (user-triggered overlay refactor)

---

## 1. Data Sources

| Source Key | Label | Provider | Resolution | Forecast Range | Frames |
|---|---|---|---|---|---|
| `gfs` | Wind (U/V) | NOAA GFS via NOMADS | 0.5 deg | 0-120 h, 3 h step | 41 |
| `cmems_wave` | Waves + Swell | Copernicus Marine | 0.083 deg | 0-120 h, 3 h step | 41 |
| `cmems_current` | Currents (U/V) | Copernicus Marine | 0.083 deg | 0-120 h, 3 h step | 41 |
| `cmems_ice` | Ice concentration | Copernicus Marine | 0.083 deg | 0-216 h, 24 h step | 10 |
| ~~`cmems_sst`~~ | ~~Sea Surface Temp~~ | ~~Copernicus Marine~~ | ~~0.083 deg~~ | ~~0-120 h, 3 h step~~ | ~~41~~ |
| `gfs_visibility` | Visibility | NOAA GFS via NOMADS | 0.5 deg | 0-120 h, 3 h step | 41 |

**Note:** SST (`cmems_sst`) is **disabled** — global 0.083 deg download is 4+ GB and requires `copernicusmarine.subset()` + h5py, which proved unreliable. Code is preserved but commented out. The active pipeline ingests 5 sources.

---

## 2. Architecture Overview

The pipeline uses a **user-triggered overlay model**: no background ingestion loops, no startup health gates. Weather data loads on demand when the user activates a layer, and fresh data is fetched only when the user explicitly requests it via the Resync button.

**Data flow:**
1. User clicks a weather layer button (Wind, Waves, etc.)
2. Backend checks PostgreSQL for existing data
3. If data exists in DB → return it with `ingested_at` timestamp
4. If DB is empty → fetch from external API (GFS/CMEMS), ingest into DB, return with `ingested_at = now()`
5. Staleness indicator shows data age in the overlay controls
6. User clicks Resync → per-layer re-fetch with `force=True`, fresh data returned

---

## 3. Storage Tiers

### Tier 1 — PostgreSQL (persistent, authoritative)

**Tables:**
- `weather_forecast_runs` — one row per source per ingestion cycle (metadata: source, run_time, status, bounds, forecast_hours array, `ingested_at`)
- `weather_grid_data` — one row per (run, forecast_hour, parameter), storing compressed grid arrays

**Compression:** All grid data stored as zlib-compressed float32 numpy arrays in `bytea` columns. Coordinate arrays (lats, lons) stored separately per row. Data lives in PostgreSQL TOAST storage (out-of-line).

### Tier 2 — File Cache (ephemeral, fast reads)

**Location:** `/tmp/windmar_cache/{wind,wave,current,ice,sst,vis}/`

**Format:** JSON files keyed by padded viewport bounds: `{layer}_{lat_min}_{lat_max}_{lon_min}_{lon_max}.json`

**Contents:** Pre-assembled forecast frame responses (all timesteps for a given layer and viewport). These are what the frontend timeline directly consumes.

**TTL:** 12 hours (cleaned by `_cleanup_stale_caches()` during per-layer resync)

### Tier 3 — Redis (shared in-memory cache, hot path)

**Scope:** Individual weather snapshots (single timestep, single layer) for map overlay rendering.

**TTL:** 60 minutes

**Key pattern:** `weather:{layer}:{bounds_hash}:{hour}`

### Tier 4 — Raw Download Cache

| Cache | Location | TTL | Contents |
|---|---|---|---|
| CMEMS NetCDF | `data/copernicus_cache/*.nc` | 24 hours | Raw downloads before ingestion |
| GFS GRIB2 | `data/gfs_cache/*.grib2` | 48 hours | Raw downloads before ingestion |

---

## 4. Ingestion Model

### On-Demand (DB-first with API fallback)

When a user activates a weather layer:
1. Backend queries `weather_forecast_runs` for the latest `complete` run for that source
2. If found → decompress grid data, return with `ingested_at` from the run
3. If not found → call external API (GFS/CMEMS), ingest into DB, return data with `ingested_at = now()`

### Per-Layer Resync (user-triggered)

**Endpoint:** `POST /api/weather/{layer}/resync` where layer is one of: `wind`, `waves`, `currents`, `ice`, `visibility`, `swell`

**Behavior:**
1. Call the layer's ingest function with `force=True` (bypasses freshness checks)
2. Supersede old runs, clean up orphaned grid data
3. Clear stale cache files
4. Return `{ "status": "complete", "ingested_at": "<ISO>" }`

**Timeout:** Can take 30-120s for CMEMS layers (network + processing).

### Supersede Logic

When a new run is ingested for a source:
1. All previous `complete` runs for that source are marked `status = 'superseded'`
2. The new run is marked `complete` after all frames are inserted
3. After resync, `_supersede_old_runs()` + `cleanup_orphaned_grid_data()` run to remove stale data

### What Gets Cleaned and When

| Event | DB Runs | DB Grid Data | File Cache | Redis | Raw Downloads |
|---|---|---|---|---|---|
| **Per-layer resync** | Old runs superseded | Orphans cleaned | Stale files (>12h) removed | Untouched | Stale files removed |
| **Container restart** | Persisted | Persisted | **Lost** (`/tmp/`) | Persisted (Redis volume) | Persisted (`data/` volume) |

---

## 5. Frontend Data Flow

### App Startup

No startup gate. The app loads immediately with a clean map — no background health checks, no ensure-all polling, no readiness state.

### Layer Activation (user clicks a weather button)

1. Frontend calls the layer's grid endpoint (e.g., `GET /api/weather/wind?lat_min=...`)
2. Backend: try DB first → if empty, fetch from API → return JSON with `ingested_at`
3. Frontend renders the grid on the Leaflet map
4. Staleness indicator updates from the `ingested_at` field in the response

### Staleness Indicator

Displayed in the map overlay controls when a layer is active:
- **Green** (`<1h` / `<4h ago`): Data is current
- **Yellow** (`<12h ago`): Data is aging
- **Red** (`>=12h ago` / `Xd ago`): Data is stale

Computed client-side from the `ingested_at` timestamp in each weather response.

### Resync (user clicks Resync button)

1. Frontend calls `POST /api/weather/{activeLayer}/resync`
2. Shows spinning indicator while waiting (can take 30-120s)
3. On success, updates `layerIngestedAt` and reloads the layer data
4. Staleness indicator resets to "< 1h ago"

Only available when a layer is active (button hidden when `weatherLayer === 'none'`).

### Timeline Activation (user clicks Timeline button)

1. Frontend calls the layer's `/frames` endpoint (e.g., `GET /api/weather/forecast/frames?lat_min=...`)
2. Backend reads ALL forecast frames from file cache (Tier 2) or assembles from DB
3. Entire multi-frame response loaded into browser memory
4. User scrubs the timeline slider — frame switching is instant (client-side)

---

## 6. `ingested_at` Propagation

Every weather endpoint now returns an `ingested_at` ISO timestamp alongside the grid data.

**Backend path:**
1. `db_weather_provider._find_latest_run()` returns `(run_id, ingested_at)` tuple
2. Each `get_*_from_db()` method returns `(data, ingested_at)` or `(None, None)`
3. Endpoints include `ingested_at` in JSON response

**Frontend path:**
1. `loadWeatherData()` extracts `ingested_at` from each response
2. Stored in `layerIngestedAt` state
3. Passed to `MapOverlayControls` as a prop
4. `computeStaleness()` calculates age and color

---

## 7. Memory & Performance Constraints

### API Container Memory

Grids are **fully decompressed in process memory** for each request. They are NOT streamed.

**Per-request memory estimates (single frame):**

| Layer | Grid Size | Params | Memory per Frame |
|---|---|---|---|
| Wind (GFS 0.5 deg) | 341 x 720 | 2 (u, v) | ~2 MB |
| Wave (CMEMS 0.083 deg) | 2041 x 4320 | 9 | ~315 MB |
| Current (CMEMS 0.083 deg) | 2041 x 4320 | 2 | ~70 MB |
| Ice (CMEMS 0.083 deg) | 2041 x 4320 | 1 | ~35 MB |

**Timeline load (all 41 frames):** Wind ~80 MB, Wave ~12.9 GB (theoretical full global; in practice, viewport-cropped).

### Browser Memory

The frontend loads ALL timeline frames into JavaScript heap at once. For wave data (9 parameters, 41 frames), this can consume several hundred MB of browser memory for large viewports.

### First-Time Layer Load

If the DB is empty for a given source, the first layer activation triggers an external API fetch. This can take 30-120s for CMEMS layers. A loading spinner is shown during this time.

---

## 8. API Endpoints (Weather Pipeline)

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/weather/health` | GET | Per-source health check (present/complete/fresh) |
| `/api/weather/{layer}/resync` | POST | Per-layer re-ingest with force=True |
| `/api/weather/wind` | GET | Wind grid data (DB-first, API fallback) |
| `/api/weather/wind/velocity` | GET | Wind velocity data for particle animation |
| `/api/weather/waves` | GET | Wave grid data |
| `/api/weather/currents` | GET | Current grid data |
| `/api/weather/currents/velocity` | GET | Current velocity for particle animation |
| `/api/weather/ice` | GET | Ice concentration grid |
| `/api/weather/visibility` | GET | Visibility grid |
| `/api/weather/sst` | GET | SST grid (disabled) |
| `/api/weather/swell` | GET | Swell decomposition grid |
| `/api/weather/forecast/prefetch` | POST | Trigger wind forecast file cache build |
| `/api/weather/forecast/status` | GET | Wind prefetch progress |
| `/api/weather/forecast/frames` | GET | All wind timeline frames |
| `/api/weather/forecast/wave/frames` | GET | All wave timeline frames |
| `/api/weather/forecast/current/frames` | GET | All current timeline frames |
| `/api/weather/forecast/ice/frames` | GET | All ice timeline frames |
| `/api/weather/forecast/sst/frames` | GET | All SST timeline frames |
| `/api/weather/forecast/visibility/frames` | GET | All visibility timeline frames |

### Removed Endpoints

The following endpoints were removed in the user-triggered overlay refactor:

| Endpoint | Reason |
|---|---|
| `POST /api/weather/ensure-all` | No startup gate — layers load on demand |
| `GET /api/weather/sync-status` | No viewport sync tracking |
| `POST /api/weather/resync` | Replaced by per-layer `/api/weather/{layer}/resync` |
| `GET /api/weather/resync/status` | Resync is now synchronous (blocks until complete) |
| `POST /api/weather/ingest` | No manual ingestion trigger — use per-layer resync |
| `GET /api/weather/ingest/status` | No ingestion status polling |

---

## 9. Limitations Summary

| Limitation | Impact | Severity |
|---|---|---|
| **First-time CMEMS layer load takes 30-120s** | User waits on first activation if DB empty | Medium |
| **No SST in DB** (cmems_sst disabled) | SST layer unavailable | Medium |
| **Full grid decompression in memory** | Memory spikes on concurrent large requests | Medium |
| **All timeline frames loaded to browser** | Browser memory pressure on large viewports | Medium |
| **No per-request memory limit** | Theoretical OOM risk under concurrent load | Low |
| **Redis not invalidated on resync** | Up to 60 min stale data after resync | Low |
| **GFS publishes progressively** | Wind may have <41 frames for recent model runs | Informational |

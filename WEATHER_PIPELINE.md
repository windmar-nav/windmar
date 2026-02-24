# Weather Data Pipeline — Architecture & Limitations

Technical reference for the WindMar weather ingestion, storage, caching, visualization, and downsampling pipeline.

Last updated: 2026-02-24 (v0.1.2 — CMEMS bbox cap 55×130, coordinate clamping, stale-data guard)

---

## 1. Data Sources

| Source Key | Label | Provider | Native Resolution | Grid Size (global) | Forecast Range | Frames |
|---|---|---|---|---|---|---|
| `gfs` | Wind (U/V) | NOAA GFS via NOMADS | 0.50° | 361 × 720 (~260K pts) | 0–120 h, 3 h step | 41 |
| `gfs_visibility` | Visibility | NOAA GFS via NOMADS | 0.25° | 681 × 1439 (~980K pts) | 0–120 h, 3 h step | 41 |
| `cmems_wave` | Waves + Swell | Copernicus Marine | 0.083° | 2041 × 4320 (~8.8M pts) | 0–120 h, 3 h step | 41 |
| `cmems_current` | Currents (U/V) | Copernicus Marine | 0.083° | 2041 × 4320 (~8.8M pts) | 0–120 h, 3 h step | 41 |
| `cmems_ice` | Ice concentration | Copernicus Marine | 0.083° | ~360 × 720 (polar only) | 0–216 h, 24 h step | 10 |
| ~~`cmems_sst`~~ | ~~Sea Surface Temp~~ | ~~Copernicus Marine~~ | ~~0.083°~~ | ~~2041 × 4320~~ | ~~Disabled~~ | ~~—~~ |

**SST (`cmems_sst`) is disabled** — global 0.083° download exceeds 4 GB and requires `copernicusmarine.subset()` + h5py, which proved unreliable in production. Code is preserved but commented out.

### Dataset Size Estimates (single timestep)

| Source | Params | Bytes/Point | Per-Frame (global) | All Frames (global) |
|---|---|---|---|---|
| GFS Wind | 2 (u, v) | 8 B (float32 × 2) | ~2 MB | ~82 MB (41 frames) |
| GFS Visibility | 1 | 4 B | ~3.9 MB | ~160 MB (41 frames) |
| CMEMS Wave | 9 (Hs, Tp, dir, swell Hs/Tp/dir, windsea Hs/Tp/dir) | 36 B | ~317 MB | ~13 GB (41 frames) |
| CMEMS Current | 2 (u, v) | 8 B | ~70 MB | ~2.9 GB (41 frames) |
| CMEMS Ice | 1 | 4 B | ~1 MB (polar) | ~10 MB (10 frames) |

These are theoretical maximums at full global resolution. In practice, all CMEMS data is fetched for a **viewport-bounded region** (see Section 5), reducing sizes by 10–50×.

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

## 3. Source Independence — Architectural Decision

The 6 weather sources (gfs, cmems_wave, cmems_current, cmems_ice, gfs_visibility, cmems_sst) are **logically independent within a shared database**.

### Design Choice: Shared Tables, Scoped Operations

All sources share two PostgreSQL tables (`weather_forecast_runs` + `weather_grid_data`), discriminated by a `source` column. This was chosen over 6 separate databases because:

1. **No cross-source JOINs exist.** Every DB query is scoped to a single source via `WHERE source = %s`. The `_find_latest_run(source)` method resolves each source independently.

2. **Routing merges in-memory, not in SQL.** `RouteWeatherAssessment.provision()` issues 3 separate `get_grids_for_timeline()` calls (wind, wave, current), then combines results into a `TemporalGridWeatherProvider`. No query ever joins wind rows with wave rows.

3. **Partial availability is handled gracefully.** The provisioner succeeds if at least wind or wave is available. Missing wind triggers a live GFS supplement. Missing current means currents are ignored. Each source failing does not block the others.

4. **6 separate databases would add complexity for no functional gain.** Connection pooling ×6, health checks ×6, VACUUM scheduling ×6 — all for data that's already logically isolated by a column value.

### Isolation Guarantees

| Operation | Scope | Cross-source impact |
|---|---|---|
| `ingest_wind(force=True)` | Supersedes only `gfs` runs | None — other sources untouched |
| `POST /api/weather/wind/resync` | Supersedes + cleans only `gfs` | None — scoped `source` param |
| `_supersede_old_runs(source)` | Marks old runs for given source | None — `WHERE source = %s` |
| `cleanup_orphaned_grid_data(source)` | Deletes dead data for given source | None — `WHERE source = %s` |
| Layer toggle (frontend) | Queries only relevant source | None — each endpoint hits one source |
| Voyage calculation | Queries 3 sources independently | None — separate DB calls merged in-memory |

### What Is NOT Independent

- **VACUUM FULL** requires a table-wide lock — cannot vacuum one source's TOAST data without locking all sources' data. Autovacuum handles this incrementally without user impact.
- **File cache cleanup** (`_cleanup_stale_caches()`) operates globally on `/tmp/windmar_cache/` — but only removes files older than 12h, which is safe regardless of source.

---

## 4. Storage Tiers

### Tier 1 — PostgreSQL (persistent, authoritative)

**Tables:**
- `weather_forecast_runs` — one row per source per ingestion cycle (metadata: source, run_time, status, bounds, forecast_hours array, `ingested_at`)
- `weather_grid_data` — one row per (run, forecast_hour, parameter), storing compressed grid arrays

**Source column:** All queries include `WHERE source = %s` to ensure per-source isolation. No cross-source JOINs exist anywhere in the codebase.

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

## 5. Ingestion Model

### On-Demand (DB-first with API fallback)

When a user activates a weather layer:
1. Backend queries `weather_forecast_runs` for the latest `complete` run for that source
2. If found → decompress grid data, return with `ingested_at` from the run
3. If not found → call external API (GFS/CMEMS), ingest into DB, return with `ingested_at = now()`

### Per-Layer Resync (user-triggered, viewport-aware)

**Endpoint:** `POST /api/weather/{layer}/resync?lat_min=...&lat_max=...&lon_min=...&lon_max=...`

Layers: `wind`, `waves`, `currents`, `ice`, `visibility`, `swell`

**Viewport-aware bbox:** The frontend passes its current map viewport bounds as query parameters. For CMEMS layers (waves, currents, swell, ice), these bounds determine which geographic region is downloaded from Copernicus. GFS layers (wind, visibility) are global and ignore the bbox.

**CMEMS bbox cap (OOM protection):** The viewport bbox is capped to a maximum of **55° latitude × 130° longitude**, centered on the viewport midpoint. This prevents OOM kills in the API container — the `copernicusmarine` Python client downloads data at **full native 0.083° resolution regardless of any post-download subsampling** (`isel`, stride, etc.), so the bbox is the only mechanism to control download size and memory usage.

**Why 55° × 130°:** This covers the demo viewport (NE Atlantic + Mediterranean: lat ~12.5°–67.5°, lon ~−55°–75°) with margin. It produces a ~660 × 1560 grid per timestep at 0.083° resolution, which results in approximately 700 MB download and ~2 GB peak RAM for wave data (9 parameters × 41 timesteps).

**Critical constraint — `copernicusmarine` native resolution:** The Copernicus Marine client (`copernicusmarine.open_dataset()`) always downloads the full native-resolution data within the requested bbox. Post-download operations like `ds.isel(latitude=slice(None, None, 3))` only discard data *after* it has been fetched and loaded into memory. This means subsampling does NOT reduce download time, network bandwidth, or peak memory during ingestion. The bbox cap is the **sole** protection against OOM.

| Viewport Size | Grid Points (0.083°) | Estimated RAM (waves) | Status |
|---|---|---|---|
| 35° × 60° (default) | ~420 × 720 | ~1.5 GB | Safe |
| 55° × 130° (max cap) | ~660 × 1560 | ~2 GB | Safe |
| 65° × 150° (uncapped) | ~780 × 1800 | ~5 GB | Risky |
| 78° × 207° (full viewport) | ~940 × 2490 | ~10+ GB | **OOM kill** |

**Behavior:**
1. Cap bbox to 55° × 130° centered on viewport midpoint
2. Clamp all coordinates to valid geographic range (lat ≤ 89.9°, lon ≤ 180°)
3. Call the layer's ingest function with `force=True` (bypasses freshness checks)
4. Supersede old runs **for this source only** (`_supersede_old_runs(source)`)
5. Clean up orphaned grid data **for this source only** (`cleanup_orphaned_grid_data(source)`)
6. Clear the layer's frame cache + stale file caches
7. Return `{ "status": "complete", "ingested_at": "<ISO>" }`

**Source isolation:** Resyncing Wind never touches Wave/Current/Ice data. The `source` parameter scopes both supersede and cleanup operations to the specific DB source (`gfs`, `cmems_wave`, etc.).

**Coordinate clamping:** Both the backend bbox cap and the frontend `paddedBounds()` clamp coordinates to safe ranges. See Section 13 (Coordinate Clamping) for the full defense-in-depth strategy.

**Timeout:** 30–120s for CMEMS layers depending on bbox size and Copernicus server load.

### Default Ingestion Bounds

When no viewport bbox is provided (e.g., first-time ingestion from DB-empty state):

| Source | Default Bounds | Rationale |
|---|---|---|
| `gfs` (wind) | Global | GFS at 0.5° is lightweight (~2 MB/frame) |
| `gfs_visibility` | Global | GFS at 0.25° is manageable (~4 MB/frame) |
| `cmems_wave` | lat [25, 60] lon [−20, 40] | North Atlantic + Med — safe download size |
| `cmems_current` | lat [25, 60] lon [−20, 40] | Same as wave |
| `cmems_ice` | lat [55, 75] lon [−20, 40] | High-latitude North Atlantic |

### Deferred Supersede Logic

When a new run is ingested for a source:
1. New frames are inserted with `status = 'ingesting'`
2. The count of new forecast hours is compared against existing best run
3. Old runs are only superseded **if the new run has ≥ hours** — this prevents data loss when NOMADS/CMEMS is still publishing a cycle
4. After resync, `_supersede_old_runs(source)` + `cleanup_orphaned_grid_data(source)` clean up

### What Gets Cleaned and When

| Event | DB Runs | DB Grid Data | File Cache | Redis | Raw Downloads |
|---|---|---|---|---|---|
| **Per-layer resync** | Old runs superseded (scoped) | Orphans cleaned (scoped) | Layer cache wiped + stale removed | Untouched | Stale files removed |
| **Container restart** | Persisted | Persisted | **Lost** (`/tmp/`) | Persisted (Redis volume) | Persisted (`data/` volume) |

---

## 6. Overlay Grid Subsampling

### Problem

CMEMS native resolution is 0.083° (~9 km). For a 40° × 60° viewport, that produces grids of ~480 × 720 = **346K points**. Sending this as JSON to the browser for a single overlay frame is workable, but larger viewports or multiple parameters push into multi-MB responses that cause:

1. **Browser memory pressure** — the JSON response is parsed into JavaScript objects, each grid point becoming a heap allocation
2. **Canvas rendering load** — the heatmap renderer must iterate every grid point per frame
3. **Network transfer time** — uncompressed JSON for 500K+ points exceeds 5 MB per overlay request

### Solution: Server-Side Subsampling

Three helper functions in `api/main.py` cap overlay grid dimensions:

```
_OVERLAY_MAX_DIM = 500  # max grid points per axis

_overlay_step(lats, lons)     → math.ceil(max(len(lats), len(lons)) / 500)
_sub2d(arr, step, decimals)   → arr[::step, ::step] rounded to N decimals
_dynamic_mask_step(bbox)      → ocean mask step scaled to viewport span
```

### Which Endpoints Are Subsampled (and Why)

| Endpoint | Subsampled | Reason |
|---|---|---|
| `GET /api/weather/waves` | Yes | CMEMS 0.083° — up to 660 × 1560 grid at 55° × 130° |
| `GET /api/weather/swell` | Yes | Same CMEMS wave data (swell decomposition) |
| `GET /api/weather/currents` | Yes | CMEMS 0.083° — same grid dimensions |
| `GET /api/weather/currents/velocity` | Yes | CMEMS 0.083° — velocity format for particle animation |
| `GET /api/weather/sst` | Yes | CMEMS 0.083° (disabled but code preserved) |
| `GET /api/weather/visibility` | Yes | GFS 0.25° — 681 × 1439 global grid |
| `GET /api/weather/ice` | **No** | Ice grids are polar-only (~360 × 720) — always under 500/axis for typical viewports |
| `GET /api/weather/wind` | **No** | GFS 0.5° — 361 × 720 global grid, already under browser limits |
| `GET /api/weather/wind/velocity` | **No** | Same GFS 0.5° data in velocity format |

### Subsampling in Practice

| Viewport | Layer | Native Grid | Step | Subsampled Grid | Reduction |
|---|---|---|---|---|---|
| 55° × 130° (NE Atlantic + Med) | Waves | 660 × 1560 | 4 | 165 × 390 | 16× fewer points |
| 55° × 130° | Currents | 660 × 1560 | 4 | 165 × 390 | 16× fewer points |
| 40° × 60° (North Atlantic) | Waves | 480 × 720 | 2 | 240 × 360 | 4× fewer points |
| 30° × 40° (Indian Ocean) | Ice | 360 × 480 | 1 | 360 × 480 | None (under cap) |
| Global (uncapped) | Visibility | 681 × 1439 | 3 | 227 × 480 | 9× fewer points |

### Ocean Mask Subsampling

The ocean mask (land/sea bitmap) uses a separate dynamic step:

```
_dynamic_mask_step(lat_min, lat_max, lon_min, lon_max)
    → max(0.05, viewport_span / 500) degrees
```

For small viewports (< 25°), the mask stays at 0.05° (~5.5 km). For large viewports, it coarsens proportionally. This keeps the ocean mask grid under ~500 × 500 regardless of zoom level.

### Rounding

Subsampled values are rounded to reduce JSON payload size:

| Data Type | Decimals | Rationale |
|---|---|---|
| Wave height, current speed | 2 | 0.01 m precision is sufficient for visualization |
| Wave period, direction | 1 | 0.1 s / 0.1° is sufficient |
| Ice concentration | 4 | Fraction 0.0000–1.0000 |
| SST | 2 | 0.01 °C precision |
| Visibility | 1 | 0.1 km precision |

### What Is NOT Subsampled

- **Timeline frames** (`/api/weather/forecast/*/frames`) — these have their own subsampling logic with `STEP = max(1, round(max_dim / 250))`, applied at cache-build time. Timeline frames are pre-assembled and served from file cache.
- **Wind overlay** — GFS 0.5° grids are already coarse enough for browser rendering.
- **Wind velocity** — Same as wind overlay.
- **Ice overlay** — Polar-only extent means grids stay small.

---

## 7. Frontend Data Flow

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

1. Frontend calls `POST /api/weather/{activeLayer}/resync` **with current viewport bounds**
2. Backend caps bbox to 55° × 130° (CMEMS layers), clamps coordinates to valid range, then re-ingests
3. Shows spinning indicator while waiting (can take 30–120s)
4. On success, updates `layerIngestedAt` and reloads the layer data
5. Staleness indicator resets to "< 1h ago"

Only available when a layer is active (button hidden when `weatherLayer === 'none'`).

### Timeline Activation (user clicks Timeline button)

1. Frontend calls the layer's `/frames` endpoint (e.g., `GET /api/weather/forecast/frames?lat_min=...`)
2. Backend reads ALL forecast frames from file cache (Tier 2) or rebuilds from DB (Tier 1)
3. Entire multi-frame response loaded into browser memory
4. User scrubs the timeline slider — frame switching is instant (client-side)
5. Slider auto-positions to the forecast hour nearest to "now" based on the model run time (see stale-data guard below)

**All 7 layers use the same direct-load pattern:**
- Check client-side cache (React ref) first — if frames exist, restore UI instantly with no network call
- If cache miss, call `loadXxxFrames()` which hits the `/frames` endpoint
- Backend serves from file cache if available, otherwise rebuilds from PostgreSQL (~3s for wind, up to ~50s for large viewports)
- No prefetch/polling — a single request-response cycle

**Stale-data guard (`nearestHourToNow`):**
- When the timeline opens, the slider positions to the frame closest to "now" relative to the forecast model run time
- If the forecast is fully expired (elapsed hours > max available hour), the slider starts at T+0 instead of pinning to the last frame — this prevents a confusing state where the timeline displays a date far in the past
- An amber "Forecast expired — resync for latest data" warning appears below the timestamp when data is stale
- The `isStale` flag is computed client-side: `(Date.now() - runTime) / 3600000 > maxAvailableHour`

**Viewport bounds and the MAX_SPAN cap:**
- When the timeline opens, the current viewport bounds are padded to 10-degree grid cell edges
- A MAX_SPAN cap (120 degrees per axis) prevents requests for near-global grids that would produce multi-hundred-MB responses
- Pan/zoom after timeline open does NOT trigger re-fetch — data stays fixed to the bounds captured at load time
- To get data for a different region: click Resync with the desired region visible, then re-open the timeline

**refTime consistency:**
- All frames in a single response share the same `refTime` (the GFS/CMEMS model run time), not the per-frame valid time
- This prevents the frontend's wind field cache from thrashing (rebuilding 180K-point 2D arrays on every frame change)

### This is NOT Windy.com

WindMar is a **local-first** application. Unlike Windy.com which serves pre-rendered tiles from a global CDN:

- **All data is fetched on demand** from NOAA GFS / Copernicus Marine, ingested into a local PostgreSQL, and served from a local API container
- **The full forecast dataset lives in the browser** — all 41 frames (wind) or 10 frames (ice) are loaded into JavaScript heap memory at once. There is no streaming or tile-based progressive loading.
- **The heatmap overlay and wind particles are rendered client-side** using Canvas 2D. The browser must decompress, parse, and render the full grid for each frame.
- **Viewport-based fetching means data stops at the grid edges.** If the user pans beyond the loaded bounds, the overlay is truncated. This is by design — fetching global data would exceed browser memory.

---

## 8. Browser Memory Limits

The frontend loads ALL timeline frames into JavaScript heap at once. For wave data (9 parameters, 41 frames), this can consume several hundred MB of browser memory for large viewports.

**Practical browser limits observed during testing (Chrome 141, 16 GB RAM system):**

| Viewport Span | Grid Points (0.25°) | JSON Response | Browser Behavior |
|---|---|---|---|
| 30° × 55° (Europe) | ~26K per component | ~24 MB | Loads in ~3s, smooth animation |
| 80° × 80° (Indian Ocean) | ~103K per component | ~129 MB | Loads in ~50s, smooth animation |
| 120° × 120° (max cap) | ~231K per component | ~290 MB (est.) | At browser memory limit |
| 160° × 260° (uncapped) | ~601K per component | ~627 MB | **Browser OOM crash** |

The MAX_SPAN cap at 120 degrees is a pragmatic limit. Users needing global coverage should zoom to the region of interest before opening the timeline.

### Overlay vs. Timeline Memory

| Concern | Overlay (single frame) | Timeline (all frames) |
|---|---|---|
| Data in browser | 1 grid per active layer | 41 grids × all params |
| Typical memory | 1–5 MB | 50–300 MB |
| OOM risk | Low (subsampled to ≤500/axis) | Medium–High (capped at 120° span) |
| Mitigation | Server-side subsampling | MAX_SPAN cap, viewport padding |

---

## 9. `ingested_at` Propagation

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

## 10. API Container Memory

Grids are **fully decompressed in process memory** for each request. They are NOT streamed.

**Per-request memory estimates (single frame, viewport-cropped to 55° × 130°):**

| Layer | Cropped Grid Size | Params | Memory per Frame |
|---|---|---|---|
| Wind (GFS 0.5°) | 111 × 261 | 2 (u, v) | ~0.23 MB |
| Wave (CMEMS 0.083°) | 660 × 1560 | 9 | ~37 MB |
| Current (CMEMS 0.083°) | 660 × 1560 | 2 | ~8.2 MB |
| Ice (CMEMS 0.083°) | 240 × 1560 | 1 | ~1.5 MB |
| Visibility (GFS 0.25°) | 221 × 521 | 1 | ~0.46 MB |

**Timeline rebuild (all frames from DB):**

| Layer | Frames | Estimated RAM (55° × 130° cap) |
|---|---|---|
| Wind | 41 | ~10 MB |
| Wave | 41 | ~1.5 GB |
| Current | 41 | ~340 MB |
| Ice | 10 | ~15 MB |
| Visibility | 41 | ~19 MB |

Wave timeline rebuild is the most expensive operation. At the 55° × 130° cap, it requires ~1.5 GB. The API container should have at least **3 GB of available RAM** (wave rebuild + overhead + concurrent requests).

**CMEMS ingestion (download + xarray load):**

| Viewport | Estimated RAM (wave ingestion) | Status |
|---|---|---|
| 35° × 60° | ~1.5 GB | Safe |
| 55° × 130° (cap) | ~2 GB | Safe |
| 65° × 150° | ~5 GB | Risky |
| 78° × 207° (uncapped) | ~10+ GB | **OOM kill** |

The 55° × 130° cap in the resync endpoint prevents the most dangerous case. See Section 5 for why `isel` subsampling cannot replace the bbox cap.

---

## 11. API Endpoints (Weather Pipeline)

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/weather/health` | GET | Per-source health check (present/complete/fresh) |
| `/api/weather/{layer}/resync` | POST | Per-layer re-ingest (accepts viewport bbox) |
| `/api/weather/wind` | GET | Wind grid data (DB-first, API fallback) |
| `/api/weather/wind/velocity` | GET | Wind velocity data for particle animation |
| `/api/weather/waves` | GET | Wave grid data (subsampled) |
| `/api/weather/currents` | GET | Current grid data (subsampled) |
| `/api/weather/currents/velocity` | GET | Current velocity for particle animation (subsampled) |
| `/api/weather/ice` | GET | Ice concentration grid (full resolution) |
| `/api/weather/visibility` | GET | Visibility grid (subsampled) |
| `/api/weather/sst` | GET | SST grid (disabled, subsampled) |
| `/api/weather/swell` | GET | Swell decomposition grid (subsampled) |
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

## 12. Limitations Summary

| Limitation | Impact | Severity |
|---|---|---|
| **First-time CMEMS layer load takes 30–120s** | User waits on first activation if DB empty | Medium |
| **No SST in DB** (cmems_sst disabled) | SST layer unavailable | Medium |
| **CMEMS resync capped at 55° × 130°** | Users viewing regions beyond this span get data centered on viewport, not full extent. Wave coverage may not match wind (GFS is global) for very wide viewports. | Medium |
| **`copernicusmarine` downloads at full resolution** | `isel` / stride subsampling does NOT reduce download size or peak memory — the bbox cap is the only OOM protection | High |
| **Full grid decompression in memory** | Memory spikes on concurrent large requests — wave rebuild at cap size requires ~1.5 GB | Medium |
| **All timeline frames loaded to browser** | Browser memory pressure on large viewports — capped at 120° per axis | Medium |
| **Overlay subsampled to ≤500 pts/axis** | Slight resolution loss on CMEMS layers at wide zoom — not applied to ice or wind | Low |
| **Pan/zoom does not auto-refetch timeline** | Heatmap truncated at grid edges when panning beyond loaded bounds — user must Resync + reopen timeline | By design |
| **Wind particles may extend beyond heatmap** | leaflet-velocity extrapolates particles past grid edges — cosmetic only | Low |
| **No per-request memory limit** | Theoretical OOM risk under concurrent load | Low |
| **Redis not invalidated on resync** | Up to 60 min stale data after resync | Low |
| **GFS publishes progressively** | Wind may have <41 frames for recent model runs | Informational |
| **SST/Visibility refTime uses valid_time** | Initial slider position may be off by one frame for SST and Visibility layers | Low |
| **Stale data on hard refresh** | If data in DB is >48h old, timeline slider starts at T+0 with amber warning — user must Resync | Low |

---

## 13. Coordinate Clamping — Defense in Depth

The `global_land_mask.globe.is_ocean()` function used for ocean masks rejects latitude values at exactly ±90° with `ValueError: latitude must be <= 90`. The `paddedBounds()` function on the frontend rounds viewport edges to 10° grid cells, which can produce exactly 90° (e.g., `ceil(80.85 / 10) * 10 = 90`). To prevent 500 errors, coordinates are clamped at three independent layers:

### Layer 1 — Frontend: `paddedBounds()` (`ForecastTimeline.tsx`)

```
lat_min = Math.max(-89.9, lat_min);
lat_max = Math.min(89.9, lat_max);
lon_min = Math.max(-180, lon_min);
lon_max = Math.min(180, lon_max);
```

Applied after 10° grid-snapping and MAX_SPAN capping. Prevents the frontend from ever sending lat=±90 to the API.

### Layer 2 — Backend: CMEMS bbox cap (`api/routers/weather.py`)

```python
lat_min = max(-89.9, lat_center - lat_half)
lat_max = min(89.9, lat_center + lat_half)
lon_min = max(-180.0, lon_center - lon_half)
lon_max = min(180.0, lon_center + lon_half)
```

Applied after centering and capping the bbox to 55° × 130°. Prevents any API request from producing extreme coordinates after arithmetic.

### Layer 3 — Backend: `build_ocean_mask()` (`api/weather_service.py`)

```python
lat_min = max(-89.99, lat_min)
lat_max = min(89.99, lat_max)
lon_min = max(-180.0, lon_min)
lon_max = min(180.0, lon_max)
```

Applied at the point of use, immediately before `globe.is_ocean()` is called. This is the last-resort clamp — even if both upstream layers are bypassed (e.g., by a direct API call), the ocean mask builder will not crash.

### Why Three Layers

Each layer operates independently. If a future code change removes or breaks one clamping point, the others still protect against the `ValueError`. The cost is negligible (4 comparisons per call).

---

## 14. Stability Guarantees & Known Failure Modes

### What Makes the Wave Pipeline Stable

1. **Bbox cap is mandatory.** The 55° × 130° cap in the resync endpoint (`_CMEMS_MAX_LAT_SPAN`, `_CMEMS_MAX_LON_SPAN`) ensures that `copernicusmarine` never downloads more than ~2 GB of wave data. This is the single most important stability mechanism.

2. **Coordinate clamping is three-deep.** See Section 13. No combination of viewport geometry can produce invalid coordinates that crash `global_land_mask`.

3. **Source isolation prevents cascade failures.** A wave resync failure does not affect wind, currents, or ice. Each source is scoped by the `source` column in PostgreSQL. See Section 3.

4. **DB-first architecture provides resilience.** Once wave data is ingested, it survives container restarts (PostgreSQL volume). Only the file cache (`/tmp/windmar_cache/`) is lost on restart, and it rebuilds lazily from DB.

5. **Stale-data guard prevents confusing UI.** When forecast data is fully expired (elapsed > max hour), the timeline slider starts at T+0 with an amber warning instead of pinning to a date far in the past.

### Known Failure Modes

| Failure | Cause | Symptom | Recovery |
|---|---|---|---|
| **Wave resync OOM** | Bbox cap removed or increased beyond 55° × 130° | API container killed by kernel OOM killer | Restore bbox cap, restart API container |
| **`latitude must be <= 90`** | Coordinate clamp removed from all three layers | 500 error on wave/overlay endpoints | Re-add clamp to `build_ocean_mask()` at minimum |
| **Partial wave coverage** | Bbox cap too small for the visible viewport | Waves stop at a geographic boundary while wind covers the full view | Increase `_CMEMS_MAX_LAT_SPAN` / `_CMEMS_MAX_LON_SPAN` (test memory first) |
| **Copernicus server timeout** | CMEMS API slow or unavailable | Resync hangs for 120s then fails | Retry later; DB retains last successful ingestion |
| **Stale data after hard refresh** | No resync triggered, DB contains old forecast run | Amber "Forecast expired" warning, timeline stuck at T+0 | Click Resync to fetch fresh data |
| **Frame cache mismatch** | Container rebuilt without volume clear | Timeline shows data for wrong region | Delete file cache files or resync the layer |

### Adjusting the Bbox Cap

If the demo viewport is changed (e.g., to cover the Indian Ocean or Pacific), the bbox cap must be adjusted:

1. Calculate the required lat/lon span from the new viewport bounds
2. Estimate memory: `grid_lat × grid_lon × 9_params × 4_bytes × 41_frames` for waves
3. Ensure the API container has 2× the estimated memory available
4. Update `_CMEMS_MAX_LAT_SPAN` and `_CMEMS_MAX_LON_SPAN` in `api/routers/weather.py`
5. Test with a full resync cycle — monitor container memory via `docker stats`

**Rule of thumb:** Every 10° of additional longitude at 0.083° adds ~120 columns to the grid. Every 10° of additional latitude adds ~120 rows. Memory scales quadratically with both dimensions.

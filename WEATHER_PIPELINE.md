# Weather Data Pipeline — Architecture & Limitations

Technical reference for the WindMar weather ingestion, storage, caching, and visualization pipeline.

Last updated: 2026-02-16 (v0.0.7 + pipeline overhaul)

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

**Total parameters stored per cycle:** 15 (wind_u, wind_v, wave_hs, wave_tp, wave_dir, swell_hs, swell_tp, swell_dir, windwave_hs, windwave_tp, windwave_dir, current_u, current_v, ice_siconc, visibility).

---

## 2. Storage Tiers

### Tier 1 — PostgreSQL (persistent, authoritative)

**Tables:**
- `weather_forecast_runs` — one row per source per ingestion cycle (metadata: source, run_time, status, bounds, forecast_hours array)
- `weather_grid_data` — one row per (run, forecast_hour, parameter), storing compressed grid arrays

**Compression:** All grid data stored as zlib-compressed float32 numpy arrays in `bytea` columns. Coordinate arrays (lats, lons) stored separately per row. Data lives in PostgreSQL TOAST storage (out-of-line).

**Current measured size (2026-02-16):**

| Metric | Value |
|---|---|
| Total DB size | **16 GB** |
| TOAST (compressed grids) | 16 GB (99.9%) |
| Heap (row metadata) | 12 MB |
| Indexes | 784 KB |
| Active "complete" runs | 5 (one per source) |
| Superseded/failed/ingesting runs | 379 |
| Grid data rows | 9,043 |

**Per-source data volume (raw decompressed):**

| Source | Parameters | Rows | Raw Data |
|---|---|---|---|
| cmems_current | current_u, current_v | 1,430 | ~5.6 GB |
| cmems_wave | 9 params (wave/swell/windwave h/t/dir) | 5,931 | ~7.4 GB |
| cmems_ice | ice_siconc | 427 | 127 MB |
| gfs | wind_u, wind_v | 1,172 | ~2.1 GB |
| gfs_visibility | visibility | 41 | 4.5 MB |
| cmems_sst | sst | 0 | 0 (not yet ingested) |

### Tier 2 — File Cache (ephemeral, fast reads)

**Location:** `/tmp/windmar_cache/{wind,wave,current,ice,sst,vis}/`

**Format:** JSON files keyed by padded viewport bounds: `{layer}_{lat_min}_{lat_max}_{lon_min}_{lon_max}.json`

**Contents:** Pre-assembled forecast frame responses (all timesteps for a given layer and viewport). These are what the frontend timeline directly consumes.

**Current measured size:** 1.2 GB

**TTL:** 12 hours (cleaned by `_cleanup_stale_caches()`)

### Tier 3 — Redis (shared in-memory cache, hot path)

**Scope:** Individual weather snapshots (single timestep, single layer) for map overlay rendering.

**TTL:** 60 minutes

**Key pattern:** `weather:{layer}:{bounds_hash}:{hour}`

**Purpose:** Avoids DB decompression for repeated requests from multiple browser tabs or API workers.

### Tier 4 — Raw Download Cache

| Cache | Location | TTL | Contents |
|---|---|---|---|
| CMEMS NetCDF | `data/copernicus_cache/*.nc` | 24 hours | Raw downloads before ingestion |
| GFS GRIB2 | `data/gfs_cache/*.grib2` | 48 hours | Raw downloads before ingestion |

---

## 3. Ingestion Lifecycle

### Automatic Ingestion Loop

**Frequency:** Every 6 hours (configured in `_ingestion_loop()`)

**Lock:** Redis distributed lock (NX, 7200s TTL). Only one worker runs ingestion across all Uvicorn processes.

**Decision logic (health-check-based):**
1. Call `db_weather.get_health(max_age_hours=12)`
2. For each of the 6 sources, evaluate:
   - **Present** = at least one `complete` run exists
   - **Complete** = frame count >= 75% of expected frames
   - **Fresh** = age < 12 hours
   - **Healthy** = present AND complete AND fresh
3. Only ingest unhealthy sources (selective fetch)
4. If all 6 healthy, skip ingestion entirely

**Per-source skip logic (if `force=False`):**
Each source checks `_has_multistep_run(source, max_age_hours=12)`. If a complete run with >1 frame exists within 12 hours, that source is skipped.

**After ingestion:** `_auto_prefetch_all()` runs to populate the file cache (Tier 2) for all 6 layers at global bounds.

### Supersede Logic

When a new run is ingested for a source:
1. All previous `complete` runs for that source are marked `status = 'superseded'`
2. The new run is marked `complete` after all frames are inserted
3. After `ingest_all()`, `_supersede_old_runs()` marks any complete run older than 24h as superseded

### What Gets Cleaned and When

| Event | DB Runs | DB Grid Data | File Cache | Redis | Raw Downloads |
|---|---|---|---|---|---|
| **New ingestion** | Old runs marked `superseded` | **NOT deleted** | Stale files (>12h) removed | Untouched | Stale files removed |
| **Resync button** | Non-superseded runs deleted | **Deleted (DELETE FROM)** | **Fully wiped** | Untouched | **Fully wiped** |
| **Container restart** | Persisted | Persisted | **Lost** (`/tmp/`) | Persisted (Redis volume) | Persisted (`data/` volume) |
| **Cache cleanup (6h)** | Untouched | Untouched | Files >12h removed | Untouched | NC >24h, GRIB >48h removed |

---

## 4. Known Limitation: Orphaned Grid Data

**Superseded runs' grid data rows are never deleted.** Only the `weather_forecast_runs.status` column changes to `'superseded'`. The corresponding `weather_grid_data` rows remain in TOAST storage indefinitely.

**Impact:** DB grows by ~16 GB per week of continuous operation (assuming ~2-3 ingestion cycles per day with full global grids).

**Current state:** 384 forecast runs exist, but only 5 are `complete`. The other 379 (superseded, failed, ingesting) still have grid data occupying TOAST space.

**Mitigation (not yet implemented):** A periodic `DELETE FROM weather_grid_data WHERE run_id IN (SELECT id FROM weather_forecast_runs WHERE status = 'superseded')` followed by `VACUUM FULL weather_grid_data` would reclaim space. This is disruptive (table lock during VACUUM FULL) and should run during low-traffic windows.

---

## 5. Frontend Data Flow

### Startup Sequence

1. Frontend calls `GET /api/weather/health`
2. If all 6 sources healthy: `weatherReady = true` immediately (zero latency startup)
3. If any source unhealthy: calls `GET /api/weather/ensure-all` which triggers selective ingestion for missing sources, then sets ready

### Layer Activation (user clicks a weather button)

1. Frontend calls the layer's grid endpoint with `db_only=true` (e.g., `GET /api/weather/wind?db_only=true&lat_min=...`)
2. Backend: query DB for the latest complete run, decompress grid, return JSON
3. If DB returns nothing (source not ingested): return `204 No Content`
4. Frontend renders the grid on the Leaflet map

### Timeline Activation (user clicks Timeline button)

1. Frontend calls the layer's `/frames` endpoint (e.g., `GET /api/weather/forecast/frames?lat_min=...`)
2. Backend reads ALL forecast frames for that layer from file cache (Tier 2) or assembles from DB
3. **Entire multi-frame response is loaded into browser memory**
4. User scrubs the timeline slider — frame switching is instant (client-side, no network)

### Viewport Change (pan/zoom)

- **No auto-fetch on pan** (removed in pipeline overhaul)
- Frontend calls `GET /api/weather/sync-status?lat_min=...&lon_min=...` (debounced 600ms)
- If viewport exceeds DB bounds: "Out of sync" orange badge appears
- User can click "Resync" to re-ingest for the new viewport

---

## 6. Resync Flow

Triggered by the user clicking the "Resync" button in the map overlay.

**Endpoint:** `POST /api/weather/resync?lat_min=X&lat_max=Y&lon_min=Z&lon_max=W`

**Phases:**

1. **Truncate** — `DELETE FROM weather_grid_data` + delete all non-superseded runs
2. **Clear caches** — wipe `/tmp/windmar_cache/`, `data/copernicus_cache/`, `data/gfs_cache/`
3. **Set bounds** — temporarily override ingestion bounds to match the requested viewport
4. **Re-ingest** — fetch all 6 sources with `force=True` for the scoped viewport
5. **Restore bounds** — reset to global defaults (-85 to 85 lat)
6. **Prefetch** — run `_auto_prefetch_all()` to rebuild file caches

**Lock:** Redis (NX, 7200s). Only one resync at a time. Concurrent requests get `{"status": "already_running"}`.

**Progress polling:** `GET /api/weather/resync/status` returns `{running, progress: {phase, current_source, completed[], total}}`.

---

## 7. Memory & Performance Constraints

### API Container Memory

**Measured:** 4.7 GB RSS (15% of 31 GB host)

Grids are **fully decompressed in process memory** for each request. They are NOT streamed.

**Per-request memory estimates (single frame):**

| Layer | Grid Size | Params | Memory per Frame |
|---|---|---|---|
| Wind (GFS 0.5 deg) | 341 x 720 | 2 (u, v) | ~2 MB |
| Wave (CMEMS 0.083 deg) | 2041 x 4320 | 9 | ~315 MB |
| Current (CMEMS 0.083 deg) | 2041 x 4320 | 2 | ~70 MB |
| Ice (CMEMS 0.083 deg) | 2041 x 4320 | 1 | ~35 MB |

**Timeline load (all 41 frames):** Wind ~80 MB, Wave ~12.9 GB (theoretical full global; in practice, viewport-cropped to ~10-30% of this).

**Concurrency risk:** Multiple simultaneous timeline loads for high-resolution layers (wave, current) can spike memory significantly. No per-request memory limit is enforced.

### Browser Memory

The frontend loads ALL timeline frames into JavaScript heap at once. For wave data (9 parameters, 41 frames), this can consume several hundred MB of browser memory for large viewports.

### Database I/O

- Grid data lives in TOAST — reading a single frame requires TOAST decompression (sequential I/O)
- No columnar compression or partitioning — full table scan for frame lookups indexed only by `(run_id, forecast_hour, parameter)`
- VACUUM FULL requires exclusive table lock and rewrites the entire TOAST relation

---

## 8. Sync Status & "Out of Sync" Logic

**Endpoint:** `GET /api/weather/sync-status?lat_min=&lat_max=&lon_min=&lon_max=`

Compares the user's viewport bounds against the union bounding box of all complete DB runs (`get_db_bounds()`).

**Response:**
```json
{
  "in_sync": true|false,
  "coverage": "full"|"partial"|"none",
  "db_bounds": {"lat_min": -85, "lat_max": 85, "lon_min": -179.75, "lon_max": 179.75}
}
```

- `"full"`: viewport is entirely within DB bounds
- `"partial"`: viewport partially overlaps DB bounds
- `"none"`: no complete runs exist, or viewport is entirely outside

The frontend shows an orange "Out of sync" badge when `in_sync === false` and hides it during active resync.

---

## 9. Cache Invalidation Rules

| Cache Tier | Invalidated When | Behavior |
|---|---|---|
| **Redis (60 min TTL)** | TTL expires naturally | Lazy invalidation — no active purge |
| **File cache (12h TTL)** | Cleanup runs every 6h in ingestion loop | Files older than 12h deleted |
| **File cache (bounds change)** | Frontend detects 10-degree grid region change | Client-side frame caches cleared; new frames loaded |
| **DB runs** | New ingestion supersedes old | Old runs marked `superseded` but data not deleted |
| **All caches** | User clicks Resync | File cache wiped, DB truncated, Redis untouched |

**Important:** Redis is NOT invalidated on resync. Stale Redis entries (up to 60 min old) may persist after a resync. This is acceptable because the next request after TTL expiry will read fresh data from the newly ingested DB.

---

## 10. Limitations Summary

| Limitation | Impact | Severity |
|---|---|---|
| **Orphaned grid data never deleted** | DB grows ~16 GB/week | High |
| **No SST in DB yet** (cmems_sst missing from runs) | SST timeline may show empty frames | Medium |
| **Full grid decompression in memory** | Memory spikes on concurrent large requests | Medium |
| **All timeline frames loaded to browser** | Browser memory pressure on large viewports | Medium |
| **No per-request memory limit** | Theoretical OOM risk under concurrent load | Low |
| **Redis not invalidated on resync** | Up to 60 min stale data after resync | Low |
| **VACUUM FULL requires table lock** | Cannot reclaim TOAST space without downtime | Low |
| **GFS publishes progressively** | Wind may have <41 frames for recent model runs | Informational |
| **Global bounds by default** | Ingestion downloads global grids even if user only views one region | Informational |

---

## 11. API Endpoints (Weather Pipeline)

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/weather/health` | GET | Per-source health check (present/complete/fresh) |
| `/api/weather/sync-status` | GET | Compare viewport vs DB bounds |
| `/api/weather/resync` | POST | Full truncate + re-ingest for viewport |
| `/api/weather/resync/status` | GET | Resync progress polling |
| `/api/weather/ensure-all` | POST | Selective ensure: ingest only missing sources |
| `/api/weather/ingest` | POST | Trigger full ingestion cycle |
| `/api/weather/ingest/status` | GET | Latest ingestion run info |
| `/api/weather/forecast/prefetch` | POST | Trigger wind forecast file cache build |
| `/api/weather/forecast/status` | GET | Wind prefetch progress |
| `/api/weather/forecast/frames` | GET | All wind timeline frames |
| `/api/weather/forecast/wave/frames` | GET | All wave timeline frames |
| `/api/weather/forecast/current/frames` | GET | All current timeline frames |
| `/api/weather/forecast/ice/frames` | GET | All ice timeline frames |
| `/api/weather/forecast/sst/frames` | GET | All SST timeline frames |
| `/api/weather/forecast/vis/frames` | GET | All visibility timeline frames |

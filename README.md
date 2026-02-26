# WINDMAR - Weather Routing & Performance Analytics

> **Warning**: This project is under active development and is **not production-ready**. It is being built in public as a learning and portfolio project. APIs, data models, and features may change without notice. Do not use for actual voyage planning or navigation.

A weather routing and performance analytics platform for merchant ships. Optimizes fuel consumption through weather-aware A\* and Dijkstra routing, physics-based vessel modeling, engine log analytics, and real-time sensor fusion. Ships with a default MR Product Tanker configuration; all vessel parameters are fully configurable.

**Live demo**: [demo-windmar.slmar.co](https://demo-windmar.slmar.co) | **Documentation**: [windmar-nav.github.io](https://windmar-nav.github.io)

## Features

### Vessel Performance Modeling
- **Holtrop-Mennen resistance prediction** — calm water resistance with frictional, wave-making, and appendage components
- **Two wave resistance methods** — STAWAVE-1 (ISO 15016, default) and Kwon's method (speed-loss percentage, per TN001)
- **SFOC curves** at variable engine loads with calibration factor support
- **Performance predictor** — dual-mode inverse solver: find achievable speed at a given engine load, or find required power for a target calm-water speed in weather
- Hull fouling calibration from operational noon reports and engine log data
- Laden and ballast condition support with 20 configurable vessel parameters
- Vessel specifications persisted in PostgreSQL across restarts

### Engine Log Analytics
- **Engine log ingestion** — upload CSV/Excel operational data with automatic column mapping
- **Performance dashboard** — 5 KPIs (avg speed, avg fuel, efficiency, distance, operating hours) + 6 interactive engine charts
- **Analytics tab** — fuel consumption distributions, speed-power scatter, voyage statistics
- **Engine-log calibration bridge** — calibrate vessel model directly from engine log data
- **Batch management** — upload, browse, filter, and delete engine log batches

### Route Optimization
- **A\* grid search** (primary engine) with weather-aware cost function and 9 pre-validated strait shortcuts
- **Dijkstra** (time-expanded graph) — voluntary speed reduction in heavy weather, node-budget capped
- 3 safety-weight variants per engine (fuel-optimal / balanced / safety-first) with progressive UI updates
- A\* grid at 0.2 deg (~12nm) resolution; Dijkstra at 0.25 deg (~15nm) — aligned with professional routing software
- **GSHHS coastline polygons** — sub-km vector land boundaries with cached shapefile loading
- **Strait visibility graph** — 9 pre-validated commercial straits (Gibraltar EB/WB, Dover, Malacca, Hormuz, Bab el-Mandeb, Bosporus, Suez approach, Messina) with direct vertex-to-vertex edges
- **Multi-objective Pareto front** — fuel vs. time tradeoff curve with interactive chart and smart default selection
- **Course-change penalty** — graduated heading penalty (0-20% per edge) discourages zigzag paths from grid artifacts
- **Safety fallback** — when severe weather blocks departure, automatic retry with relaxed hard limits (10x cost penalty instead of rejection); structured error diagnostics on complete failure
- **Hard avoidance limits** — Hs >= 6m and wind >= 70 kts are instant rejection (no motion calculation)
- **Seakeeping safety constraints** — graduated roll, pitch, acceleration limits with motion-based cost multipliers
- **Variable speed voyage calculation** — per-leg speed optimization (fuel ~ speed³) with configurable time penalty
- Variable speed optimization (10-16 knots per leg, 0.5 kt steps)
- Turn-angle path smoothing to eliminate grid staircase artifacts
- SOG profile analysis — estimated speed-over-ground per waypoint accounting for weather and current
- RTZ file import/export (IEC 61174 ECDIS standard)

### Weather Integration
- NOAA GFS (0.25 deg) for near-real-time wind fields via NOMADS GRIB filter
- 5-day wind forecast timeline (f000-f120, 3-hourly steps) with Windy-style animation
- Copernicus Marine Service (CMEMS) for wave and ocean current data
- ERA5 reanalysis as secondary wind fallback (~5-day lag)
- Climatology fallback for beyond-forecast-horizon voyages
- Unified provider that blends forecast and climatology with smooth transitions
- **Pre-ingested weather database** — grids compressed (zlib/float32) in PostgreSQL, served in milliseconds
- **Redis shared cache** across all API workers (replaces per-worker in-memory dict)
- User-triggered overlay model — no background loops; per-layer resync with viewport-aware CMEMS downloads
- Server-side grid subsampling (<=500 pts/axis) prevents browser OOM on large viewports
- Synthetic data generator for testing and demos

### Monte Carlo Simulation
- Parametric Monte Carlo with temporally correlated perturbations
- Divides voyage into up to 100 time slices (~1 per 1.2 hours)
- Cholesky decomposition of exponential temporal correlation matrix
- Log-normal perturbation model: wind sigma=0.35, wave sigma=0.20 (70% correlated with wind), current sigma=0.15
- P10/P50/P90 confidence intervals for ETA, fuel consumption, and voyage time
- Pre-fetches multi-timestep wave forecast grids from database (0-120h)
- 100 simulations complete in <500ms

### Regulatory Compliance
- IMO CII (Carbon Intensity Indicator) calculations with annual tightening
- **FuelEU Maritime** — GHG intensity (Well-to-Wake), compliance balance, pooling scenarios, penalty estimator
- Emission Control Areas (ECA/SECA) with fuel switching requirements
- High Risk Areas (HRA), Traffic Separation Schemes (TSS)
- **Charter Party weather clause tools** — good weather day counter, warranted speed/consumption verification, off-hire detection
- Custom zone creation with penalty/exclusion/mandatory interactions
- GeoJSON export for frontend visualization

### Live Operations
- SBG Electronics IMU sensor integration (roll, pitch, heave)
- FFT-based wave spectrum estimation from ship motion
- Multi-source sensor fusion engine
- Continuous model recalibration from live data

### Web Interface
- ECDIS-style map-centric layout with full-width chart and header dropdowns
- Interactive Leaflet maps with weather overlays, coastline rendering, and route visualization
- Wind particle animation layer (leaflet-velocity)
- Windy-style wave crest rendering with click-to-inspect polar diagram popup
- Forecast timeline with play/pause, speed control, and 5-day scrubbing
- Optimized routes displayed simultaneously with per-route color coding and toggleable visibility
- Interactive Pareto front chart (fuel vs. time tradeoff) in analysis panel
- Unified comparison table with fuel, distance, time, and waypoint counts for every route variant
- Sequential optimization with progressive map updates (routes appear one by one)
- **Session persistence** — waypoints, optimization results, viewport, and settings survive page navigation and full reloads (sessionStorage-backed React Context)
- **Settings page** — dedicated optimization engine configuration with educational content on Pareto analysis, variable speed, and startup procedure
- Voyage calculation with per-leg fuel, speed, and ETA breakdown
- Consolidated vessel configuration, calibration, fuel analysis, and performance prediction page
- Engine log upload, entries browser, and analytics dashboard
- CII compliance tracking and projections
- FuelEU Maritime compliance page (4 tabs)
- Charter party weather clause analysis
- Dark maritime theme, responsive design

## Screenshots

<table>
  <tr>
    <td align="center"><img src="docs/screenshots/weather-wind.png" width="400"><br><strong>Wind Forecast</strong><br>GFS wind particle animation with 5-day forecast timeline</td>
    <td align="center"><img src="docs/screenshots/weather-ice.png" width="400"><br><strong>Sea Ice</strong><br>Arctic ice concentration overlay from CMEMS</td>
  </tr>
  <tr>
    <td align="center"><img src="docs/screenshots/route-portugal-casquets.png" width="400"><br><strong>Route Optimization</strong><br>Weather-optimal routing with Pareto front and strait shortcuts</td>
    <td align="center"><img src="docs/screenshots/vessel-model.png" width="400"><br><strong>Vessel Model</strong><br>Resistance, power, SFOC, and fuel curves with calibration</td>
  </tr>
  <tr>
    <td align="center"><img src="docs/screenshots/engine-log-analytics.png" width="400"><br><strong>Engine Log Analytics</strong><br>Speed-power scatter, fuel distributions, voyage statistics</td>
    <td align="center"><img src="docs/screenshots/fuel-analysis.png" width="400"><br><strong>Fuel Analysis</strong><br>Physics-based fuel scenarios across loading conditions</td>
  </tr>
  <tr>
    <td align="center"><img src="docs/screenshots/cii-compliance.png" width="400"><br><strong>CII Compliance</strong><br>IMO Carbon Intensity rating with multi-year projection</td>
    <td align="center"><img src="docs/screenshots/regulations.png" width="400"><br><strong>Regulatory Zones</strong><br>ECA/SECA, TSS, HRA zones with GeoJSON visualization</td>
  </tr>
</table>

## Architecture

```
windmar/
├── api/                           # FastAPI backend
│   ├── main.py                    # Application factory + startup
│   ├── routers/                   # Domain routers (12 modules)
│   │   ├── weather.py             # 31 weather endpoints, forecast layers, cache mgmt
│   │   ├── vessel.py              # Vessel specs, calibration, noon reports, prediction
│   │   ├── voyage.py              # Voyage calculation, Monte Carlo, weather-along-route
│   │   ├── voyage_history.py      # Voyage history and reporting
│   │   ├── optimization.py        # A* / Dijkstra route optimization with safety fallback
│   │   ├── engine_log.py          # Engine log upload, entries, summary, calibration
│   │   ├── zones.py               # Regulatory zone CRUD and spatial queries
│   │   ├── cii.py                 # CII compliance calculations and projections
│   │   ├── fueleu.py              # FuelEU Maritime GHG intensity and compliance
│   │   ├── charter_party.py       # Weather clause tools (good weather days, warranted speed)
│   │   ├── routes.py              # RTZ parsing, waypoint route creation
│   │   └── system.py              # Health, metrics, status, data sources
│   ├── schemas/                   # Pydantic request/response models (11 modules)
│   │   ├── common.py, weather.py, vessel.py, voyage.py, optimization.py
│   │   ├── engine_log.py, zones.py, cii.py, fueleu.py, charter_party.py
│   │   └── ...
│   ├── reports/                   # PDF report generation (noon, departure, arrival)
│   ├── state.py                   # Thread-safe application state (singleton)
│   ├── weather_service.py         # Weather field accessors (wind, wave, current, SST, ice)
│   ├── forecast_layer_manager.py  # Forecast dedup, progress tracking, frame serving
│   ├── auth.py                    # API key authentication (bcrypt, demo licensing)
│   ├── config.py                  # API configuration (pydantic-settings)
│   ├── middleware.py              # Security headers, structured logging, metrics
│   ├── rate_limit.py              # Token bucket rate limiter (Redis-backed)
│   ├── database.py                # SQLAlchemy ORM setup
│   ├── models.py                  # Database models (weather, engine log, vessel specs)
│   ├── health.py, cache.py, resilience.py, demo.py, cli.py, live.py
│   └── ...
├── src/
│   ├── optimization/
│   │   ├── vessel_model.py        # Holtrop-Mennen + Kwon resistance, SFOC, performance predictor
│   │   ├── base_optimizer.py      # Abstract base class for route optimizers
│   │   ├── route_optimizer.py     # A* grid search with strait shortcuts + course-change penalty
│   │   ├── dijkstra_optimizer.py   # Dijkstra time-expanded graph with node budget cap
│   │   ├── routing_graph.py       # Shared routing graph utilities
│   │   ├── router.py              # Engine dispatcher (A*/VISIR selection)
│   │   ├── voyage.py              # Per-leg voyage calculator (LegWeather, VoyageResult)
│   │   ├── monte_carlo.py         # Temporal MC simulation with Cholesky correlation
│   │   ├── seakeeping.py          # Ship motion safety assessment + safety fallback
│   │   ├── grid_weather_provider.py     # Bilinear interpolation from pre-fetched grids
│   │   ├── temporal_weather_provider.py # Trilinear interpolation (lat, lon, time)
│   │   ├── weather_assessment.py  # Route weather assessment + DB provisioning
│   │   └── vessel_calibration.py  # Noon report + engine log calibration (scipy)
│   ├── data/
│   │   ├── copernicus.py          # GFS, ERA5, CMEMS providers + forecast prefetch
│   │   ├── copernicus_client.py   # CMEMS client wrapper
│   │   ├── db_weather_provider.py # DB-backed weather (compressed grids from PostgreSQL)
│   │   ├── weather_ingestion.py   # Scheduled weather grid ingestion service
│   │   ├── strait_waypoints.py    # 9 commercial strait definitions with waypoint coordinates
│   │   ├── tss_zones.py           # Traffic Separation Scheme zone definitions
│   │   ├── regulatory_zones.py    # Zone management and point-in-polygon (Shapely)
│   │   ├── eca_zones.py           # ECA zone definitions
│   │   └── land_mask.py           # GSHHS coastline + ocean/land detection
│   ├── compliance/
│   │   ├── cii.py                 # IMO CII rating calculations
│   │   ├── fueleu.py              # FuelEU Maritime GHG intensity
│   │   └── charter_party.py       # Weather clause analysis
│   ├── calibration/               # Calibration loop and utilities
│   ├── database/                  # Engine log parser, Excel parser
│   ├── sensors/                   # SBG IMU, wave estimator, timeseries
│   ├── fusion/                    # Multi-source data fusion
│   ├── routes/                    # RTZ XML route file parser
│   ├── visualization/             # Plotting utilities
│   ├── validation.py, config.py, metrics.py
│   └── ...
├── frontend/                      # Next.js 15 + TypeScript
│   ├── app/                       # Pages (route planner, vessel, CII, FuelEU, charter party, live)
│   ├── components/                # 42 React components (maps, charts, weather, Pareto, analysis)
│   └── lib/                       # API client, utilities
├── tests/                         # 713 tests (531 unit + 182 integration)
│   ├── unit/                      # Vessel model, routing, safety, straits, zones, CII, FuelEU...
│   ├── integration/               # API endpoints, optimization flow
│   └── test_e2e_*.py              # End-to-end sensor integration
├── examples/                      # Demo scripts (simple, ARA-MED, calibration)
├── docker/                        # init-db.sql, migrations/
├── docker-compose.yml             # Full stack (API + frontend + PostgreSQL + Redis)
├── Dockerfile                     # Multi-stage production build
└── pyproject.toml                 # Poetry project definition
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Uvicorn, Python 3.10+ |
| Frontend | Next.js 15, TypeScript, React, Tailwind CSS |
| Maps | React Leaflet |
| Charts | Recharts |
| Database | PostgreSQL 16, SQLAlchemy |
| Cache | Redis 7 |
| Scientific | NumPy, SciPy, Pandas |
| Auth | API keys, bcrypt |
| Containerization | Docker, Docker Compose |

## Quick Start

### Docker Compose (recommended)

```bash
git clone https://github.com/windmar-nav/windmar.git
cd windmar
cp .env.example .env    # Edit with your settings
docker compose up -d --build
```

Services start on:

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3003 |
| API | http://localhost:8003 |
| API Docs (Swagger) | http://localhost:8003/api/docs |
| PostgreSQL | localhost:5434 |
| Redis | localhost:6380 |

### Manual Setup

> **Important**: The frontend requires the backend API to be running. Start the backend first, then the frontend in a separate terminal.

```bash
# Terminal 1 — Backend API (must be running for the frontend to work)
pip install -r requirements.txt
python api/main.py
# API starts on http://localhost:8000

# Terminal 2 — Frontend
cd frontend
cp .env.example .env.local   # Sets API URL to http://localhost:8000
npm install --legacy-peer-deps
npm run dev
# Frontend starts on http://localhost:3000
```

### Python Examples

```bash
python examples/demo_simple.py          # Synthetic weather demo
python examples/example_ara_med.py      # Rotterdam to Augusta optimization
python examples/example_calibration.py  # Noon report calibration
```

## Configuration

Copy `.env.example` to `.env` and configure:

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | development / staging / production | development |
| `DATABASE_URL` | PostgreSQL connection string | postgresql://windmar:...@db:5432/windmar |
| `REDIS_URL` | Redis connection string | redis://:...@redis:6379/0 |
| `API_SECRET_KEY` | API key hashing secret | (generate with `openssl rand -hex 32`) |
| `CORS_ORIGINS` | Allowed frontend origins | http://localhost:3000 |
| `COPERNICUS_MOCK_MODE` | Use synthetic weather data | true |
| `AUTH_ENABLED` | Require API key authentication | true |
| `RATE_LIMIT_PER_MINUTE` | API rate limit | 60 |

### Weather Data Sources

Windmar uses a three-tier provider chain that automatically falls back when a source is unavailable:

| Data Type | Primary Source | Fallback | Credentials Required |
|-----------|---------------|----------|---------------------|
| **Wind** | NOAA GFS (0.25 deg, ~3.5h lag) | ERA5 reanalysis, Synthetic | None (GFS is free) |
| **Waves** | CMEMS global wave model | Synthetic | CMEMS account |
| **Currents** | CMEMS global physics model | Synthetic | CMEMS account |
| **Forecast** | GFS f000-f120 (5-day, 3h steps) | — | None |

**Wind data works out of the box** — GFS is fetched from NOAA NOMADS without authentication. For wave and current data, you need Copernicus Marine credentials.

### Obtaining Weather Credentials

**CMEMS (waves and currents):**
1. Register at [marine.copernicus.eu](https://marine.copernicus.eu/)
2. Set in `.env`:
   ```
   COPERNICUSMARINE_SERVICE_USERNAME=your_username
   COPERNICUSMARINE_SERVICE_PASSWORD=your_password
   ```

**CDS ERA5 (wind fallback):**
1. Register at [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu/)
2. Copy your Personal Access Token from your profile page
3. Set in `.env`:
   ```
   CDSAPI_KEY=your_personal_access_token
   ```

Without these credentials, the system falls back to synthetic data automatically for waves and currents. Wind visualization always works via GFS.

See `WEATHER_PIPELINE.md` for full technical details on data acquisition, GRIB processing, and the forecast timeline.

## API Endpoints

### Weather
- `GET /api/weather/health` - Weather subsystem health
- `GET /api/weather/wind` - Wind field grid (U/V components)
- `GET /api/weather/wind/velocity` - Wind in leaflet-velocity format (GFS)
- `GET /api/weather/waves` - Wave height field (CMEMS)
- `GET /api/weather/swell` - Swell field
- `GET /api/weather/currents` - Ocean current field (CMEMS)
- `GET /api/weather/currents/velocity` - Currents in leaflet-velocity format
- `GET /api/weather/sst` - Sea surface temperature
- `GET /api/weather/visibility` - Visibility field
- `GET /api/weather/ice` - Sea ice concentration
- `GET /api/weather/point` - Weather at specific coordinates
- `GET /api/weather/freshness` - Weather data age indicator
- `POST /api/weather/{layer}/resync` - Per-layer viewport-aware resync

### Forecast (Wind)
- `GET /api/weather/forecast/status` - GFS prefetch progress and run info
- `POST /api/weather/forecast/prefetch` - Trigger 5-day forecast download (f000-f120)
- `GET /api/weather/forecast/frames` - Bulk download all forecast frames

### Forecast (Wave / Current / Ice / SST / Visibility)
- `GET /api/weather/forecast/{layer}/status` - Prefetch progress
- `POST /api/weather/forecast/{layer}/prefetch` - Trigger forecast download
- `GET /api/weather/forecast/{layer}/frames` - Bulk download forecast frames

### Routes
- `POST /api/routes/parse-rtz` - Parse RTZ route file
- `POST /api/routes/from-waypoints` - Create route from coordinates

### Voyage
- `POST /api/voyage/calculate` - Full voyage calculation with weather
- `GET /api/voyage/weather-along-route` - Weather conditions per waypoint
- `POST /api/voyage/monte-carlo` - Parametric MC simulation (P10/P50/P90)

### Optimization
- `POST /api/optimize/route` - Weather-optimal route finding (A\* or Dijkstra engine)
- `POST /api/optimize/pareto` - Multi-objective Pareto front (fuel vs. time)
- `GET /api/optimize/status` - Optimizer configuration and available targets

### Vessel
- `GET /api/vessel/specs` - Current vessel specifications
- `POST /api/vessel/specs` - Update vessel specifications (persisted to DB)
- `GET /api/vessel/calibration` - Current calibration factors
- `POST /api/vessel/calibration/set` - Manually set calibration factors
- `POST /api/vessel/calibrate` - Run calibration from noon reports
- `POST /api/vessel/calibration/estimate-fouling` - Estimate hull fouling factor
- `GET /api/vessel/model-status` - Full model parameters, calibration state, computed values
- `GET /api/vessel/fuel-scenarios` - Physics-based fuel scenarios (4 conditions)
- `POST /api/vessel/predict` - Performance predictor (engine load or target speed mode)
- `GET /api/vessel/noon-reports` - List uploaded noon reports
- `POST /api/vessel/noon-reports` - Add a single noon report
- `POST /api/vessel/noon-reports/upload-csv` - Upload operational data (CSV)
- `POST /api/vessel/noon-reports/upload-excel` - Upload operational data (Excel .xlsx/.xls)
- `DELETE /api/vessel/noon-reports` - Clear all noon reports

### Engine Log
- `POST /api/engine-log/upload` - Upload engine log CSV/Excel
- `GET /api/engine-log/entries` - Browse entries with pagination and filters
- `GET /api/engine-log/summary` - Aggregate statistics per batch
- `POST /api/engine-log/calibrate` - Calibrate vessel model from engine log data
- `DELETE /api/engine-log/batch/{batch_id}` - Delete a batch

### Zones
- `GET /api/zones` - All regulatory zones (GeoJSON)
- `GET /api/zones/list` - Zone summary list
- `GET /api/zones/{zone_id}` - Single zone details
- `POST /api/zones` - Create custom zone
- `DELETE /api/zones/{zone_id}` - Delete a custom zone
- `GET /api/zones/at-point` - Zones at specific coordinates
- `GET /api/zones/check-path` - Check zone intersections along a route

### CII Compliance
- `GET /api/cii/vessel-types` - IMO vessel type categories
- `GET /api/cii/fuel-types` - Fuel types and CO2 emission factors
- `POST /api/cii/calculate` - Calculate CII rating
- `POST /api/cii/project` - Multi-year CII projection
- `POST /api/cii/reduction` - Required fuel reduction for target rating

### Live Sensor Data
- `GET /api/live/status` - Sensor connection status
- `POST /api/live/connect` - Connect to SBG IMU sensor
- `POST /api/live/disconnect` - Disconnect sensor
- `GET /api/live/data` - Current fused sensor data
- `GET /api/live/timeseries/{channel}` - Time series for a specific channel
- `GET /api/live/timeseries` - All time series data
- `GET /api/live/motion/statistics` - Motion statistics (roll, pitch, heave)
- `GET /api/live/channels` - Available data channels
- `POST /api/live/export` - Export recorded data

### System
- `GET /api/health` - Health check
- `GET /api/health/live` - Liveness probe
- `GET /api/health/ready` - Readiness probe
- `GET /api/status` - Application status summary
- `GET /api/metrics` - Prometheus metrics
- `GET /api/metrics/json` - Metrics in JSON format
- `GET /api/data-sources` - Weather data source configuration

Full interactive documentation at `/api/docs` when the server is running.

## Testing

713 tests (531 unit, 182 integration).

```bash
pytest tests/ -v                             # All tests
pytest tests/unit/ -v                        # Unit tests only
pytest tests/integration/ -v                 # Integration tests
pytest tests/unit/test_vessel_model.py -v    # Specific test file
```

## Default Vessel

The system ships with a default MR Product Tanker configuration (all values configurable via API and persisted in DB):

| Parameter | Value |
|-----------|-------|
| DWT | 49,000 MT |
| LOA / LPP | 183m / 176m |
| Beam | 32m |
| Draft (laden / ballast) | 11.8m / 6.5m |
| Displacement (laden / ballast) | 65,000 / 20,000 MT |
| Block coefficient (laden / ballast) | 0.82 / 0.75 |
| Wetted surface (laden / ballast) | 7,500 / 5,200 m2 |
| Main Engine MCR | 6,600 kW |
| SFOC at MCR | 171 g/kWh |
| Service Speed (laden / ballast) | 13.0 / 13.0 knots |
| Frontal area (laden / ballast) | 450 / 850 m2 |
| Lateral area (laden / ballast) | 2,100 / 2,800 m2 |

## Changelog

### v0.1.2 — Settings Page, Variable Speed & Session Persistence

**Settings & Documentation**

- **Dedicated Settings page** (`/settings`) — optimization engine configuration (grid resolution, variable resolution, Pareto analysis, variable speed) with educational content explaining each feature, the optimization process, safety weights, and the local-first startup procedure
- **VISIR renamed to Dijkstra** — all references across backend, frontend, API schemas, and documentation updated to reflect the actual algorithm
- Settings link added to header navigation

**Variable Speed Voyage Calculation**

- **Per-leg speed optimization** — tests 13 candidate speeds from 60%-100% of calm speed per leg, selects minimum fuel+time score using fuel ~ speed³ relationship
- **Time penalty tuning** — LAMBDA_TIME=1.5 MT-equivalent per hour prevents extreme slow-steaming in calm conditions
- Speed profile bar chart in voyage summary shows per-leg speed variations
- 8 new unit tests covering calm weather convergence, heavy weather savings, and speed profile bounds

**Session Persistence**

- All VoyageContext state backed by sessionStorage — waypoints, optimization results, route visibility, viewport, view mode, and settings survive full page reloads and hard navigations
- Map viewport restored from persisted state on remount (no more reset to default view)
- `displayedAnalysisId` moved from local page state to shared context for cross-navigation persistence

**Dijkstra Performance**

- Frontend optimization timeout increased from 180s to 600s for long routes on time-expanded graphs
- Default node budget reduced from 350K to 150K to cap memory usage

**Demo Mode Hardening**

- File uploads (RTZ import, Load from File) hidden for demo users
- Map viewport fully locked in demo mode — dragging, zoom, scroll, keyboard, and touch interactions disabled
- Weather data: frozen snapshot (Feb 2025), forecast capped at 48h, Visibility/SST layers hidden, resync blocked

### v0.1.0 — Commercial Compliance & Production Optimizer

Phase 2 (commercial credibility) and Phase 3 (optimizer upgrade), plus security hardening and dark theme.

**Phase 2 — Regulatory & Commercial Features**

- **2a: Voyage Reporting** — noon, departure, and arrival reports in IMO format; PDF export with branding placeholder; voyage history with search and filters
- **2b: CII Simulator** — what-if CII projection with speed/fuel/route adjustments; fleet-level dashboard; A-E band threshold visualization with tightening schedule
- **2c: FuelEU Maritime** — GHG intensity calculation (Well-to-Wake), compliance balance tracking, pooling scenario modeling, penalty exposure estimator (4-tab page, 7 endpoints, 35 tests)
- **2d: Charter Party Weather Clause Tools** — good weather day counter (Beaufort thresholds), warranted speed/consumption verification, off-hire event detection from engine logs

**Phase 3 — Optimizer Upgrade (ALGO-OPT-001)**

- **GSHHS coastline polygons** — sub-km vector land boundaries with cached shapefile loading
- **Variable resolution corridor grid** — 0.1 deg nearshore, 0.5 deg open ocean, auto-refined around obstacles; UI toggle for variable/uniform resolution
- **Strait visibility graph** — 9 pre-validated commercial straits (Gibraltar EB/WB, Dover, Malacca, Hormuz, Bab el-Mandeb, Bosporus, Suez approach, Messina) with direct vertex-to-vertex edges injected into the A\* search graph
- **Multi-objective Pareto front** — fuel vs. time tradeoff curve with widened lambda sweep and smart default selection; interactive Pareto chart in analysis panel
- **Course-change penalty** — graduated heading change cost (0-20% per edge) prevents zigzag paths from grid discretization artifacts
- **Safety fallback routing** — automatic retry with relaxed hard limits when severe weather blocks departure; structured error diagnostics (422 with explored-node count and failure reason)
- **Speed optimization** — optimizer selects speed from discrete set (10-16 kts in 0.5 kt steps) per leg
- **SOG profile analysis** — estimated speed-over-ground per waypoint with weather and current effects
- **Dijkstra made optional** — A\* is the primary engine; Dijkstra time-expanded graph available via UI toggle (off by default) with node budget cap
- **Smart grid bbox** — strait waypoint expansion checks both lat AND lon proximity to avoid pulling in distant straits
- **Smart retry logic** — skip safety-fallback retry when >10K nodes explored (topology issue, not weather)

**Infrastructure & Security**

- **Security hardening** — fail-fast secrets validation, pinned dependency versions, container image scanning
- **Tiered demo auth** — bcrypt license keys with frame-limited demo mode
- **Dark navy theme** — unified dark palette across all pages

### v0.0.9 — Modular Architecture & Calibration Improvements

Complete structural refactoring of the API layer plus calibration accuracy improvements. Zero endpoint changes, zero test regressions (426 tests passing).

**Monolith → Modular**

- **main.py reduced from 6,922 to 281 lines** — now an application factory with startup/shutdown lifecycle only
- **9 domain routers** extracted into `api/routers/`: weather, vessel, voyage, optimization, engine_log, zones, cii, routes, system
- **37 Pydantic schemas** extracted into `api/schemas/` (9 schema modules) — request/response models no longer embedded in endpoint code
- **Thread-safe VesselState** — vessel model, specs, and calibration state managed by a singleton with `threading.Lock` guards
- **WeatherService module** — weather field accessors (wind, wave, current, SST, ice, visibility) extracted from inline endpoint logic
- **ForecastLayerManager** — deduplicates concurrent prefetch requests, tracks progress per layer, serves cached forecast frames
- All existing endpoints, URL paths, and response schemas unchanged

**Calibration**

- **ME-specific fuel** — calibration now uses main engine fuel (HFO ME + MGO ME) when available, falling back to total fuel only when ME columns are not reported; previous approach compared predicted ME fuel against total fuel (including auxiliary), biasing the calm water factor
- **Laden/ballast detection from ME load %** — entries classified by ME load percentage (>55% = laden), enabling correct displacement for resistance calculations; previously all entries were treated as laden
- **Widened calibration bounds** — calm water factor range expanded from (0.85, 1.5) to (0.6, 1.5), SFOC factor from (0.9, 1.2) to (0.85, 1.2), accommodating vessels where Holtrop-Mennen overpredicts
- **Engine log deduplication** — duplicate entries (same timestamp) are automatically skipped during upload

### v0.0.8 — Vessel Model Upgrade, Engine Log Analytics, Optimizer Convergence

Engine log ingestion and analytics, physics model upgrade (SFOC fix, Kwon's wave resistance, performance predictor), weather pipeline refactoring, and dual-engine optimizer convergence with corrected cost formulas and MR safety limits.

**Optimizer Fixes**

- **Dijkstra cost formula corrected** — safety and zone multipliers now apply to fuel cost only, not the time penalty; previous formula `(fuel + lambda*hours) * safety * zone` inflated detour costs; fixed to `fuel * safety * zone + lambda*hours`
- **Hard avoidance limits** — Hs >= 6m and wind >= 70 kts trigger instant rejection (`inf` cost) before computing vessel motions, matching MR Product Tanker operational limits (Beaufort 9+)
- **Wind speed plumbed to safety checks** — `get_safety_cost_factor()` now receives wind speed in both A\* and Dijkstra engines
- **Dijkstra converges on 901nm route** (Portugal to Casquets): 377 cells, 1.5s compute, -0.7% fuel savings
- **A\* converges on 901nm route**: -2.6% fuel savings

**Model Curves & Calibration**

- **Model curves endpoint** — `GET /api/vessel/model-curves` returns speed-indexed arrays for resistance, power, SFOC, and fuel consumption
- **Auto-load calibration** — saved calibration factors restored on startup (survives container restarts)
- **AnalysisPanel** — calibration indicator and smart optimization route display

**Layout Harmonization**

- Standardized container layout (`container mx-auto px-6 pt-20 pb-12`) across all dashboard pages
- Vessel Model tab: 2x2 chart grid (was stacked), all charts at consistent height
- Engine Log: wider tab bar and upload section, consistent chart heights
- Analysis, CII, Live dashboard: consistent padding and offsets

**Vessel Model Physics**

- **SFOC calibration factor fix** — `sfoc_factor` was calibrated but never applied to the SFOC curve; now propagated through `VesselModel`, `state.py`, and all model rebuild paths in the API
- **Kwon's wave resistance method** — alternative to STAWAVE-1, selectable via `wave_method` parameter (`'stawave1'` | `'kwon'`); uses speed-loss percentage from Hs, Cb, Lpp, and directional factor (per TN001)
- **Performance predictor** — bisection solver for the inverse problem: given engine load + weather, what speed is achievable? Returns STW, SOG, fuel/day, fuel/nm, resistance breakdown, weather speed loss
- **Dual-mode prediction** — `POST /api/vessel/predict` accepts either `engine_load_pct` (find speed at power) or `calm_speed_kts` (find power for target speed); MCR capping with automatic fallback
- **Relative direction convention** — all predictor directions are relative to bow (0 deg = ahead, 90 deg = beam, 180 deg = astern) instead of absolute compass headings
- **Full model status endpoint** — `GET /api/vessel/model-status` exposes all 20 VesselSpecs fields, calibration state, and computed optimal speeds
- **Physics-based fuel scenarios** — `GET /api/vessel/fuel-scenarios` replaces hardcoded frontend scenarios with real `VesselModel.calculate_fuel_consumption()` results

**Engine Log Analytics**

- **Engine log ingestion** — upload CSV/Excel with automatic column mapping, parser handles multiple date formats and unit conversions
- **Engine log DB model** — batch + entries tables with indexes on timestamp, batch_id
- **Entries browser** — paginated entries with shared filters across Entries, Analytics, and Performance tabs
- **Analytics dashboard** — 5 KPIs (avg speed, avg fuel, efficiency, total distance, operating hours) + 6 interactive Recharts charts (speed-power scatter, fuel distribution, SFOC profile, voyage timeline)
- **Performance tab** — engine performance KPIs derived from operational data
- **Engine-log calibration bridge** — `POST /api/engine-log/calibrate` runs vessel calibration against engine log entries
- **Vessel specs persistence** — specs saved to PostgreSQL, restored on startup

**Weather Pipeline**

- **User-triggered overlay model** — removed all background ingestion loops, startup health gates, and ensure-all polling; weather data loads on demand when the user activates a layer
- **Viewport-aware resync** — per-layer `POST /api/weather/{layer}/resync` accepts the frontend's current viewport bounds, so CMEMS data is downloaded for the region the user is viewing (not a hardcoded North Atlantic bbox)
- **CMEMS bbox cap** — resync viewport capped at 40 deg lat x 60 deg lon to prevent API container OOM
- **Overlay grid subsampling** — all CMEMS overlay endpoints server-side subsampled to <=500 grid points per axis before JSON serialization
- **Per-source isolation** — resyncing one layer never touches another; supersede and orphan cleanup scoped by `source` column
- **Deferred supersede** — new ingestion runs only replace old ones if they have >= forecast hours, preventing data loss when NOMADS/CMEMS is still publishing
- **Wind DB fallback** — when no GRIB file cache exists, wind frames are rebuilt from PostgreSQL
- **Comprehensive pipeline documentation** — `WEATHER_PIPELINE.md` rewritten with dataset sizes, memory estimates, subsampling rationale

### v0.0.7 — Two-Mode Architecture & 7-Layer Forecast Timeline

Two-mode UI (Weather Viewer + Route Analysis), 7 weather overlay layers with forecast timeline, analysis panel with passage plan detail, route management, and production infrastructure.

### v0.0.6 — ECDIS UI Redesign & Dual Speed-Strategy Optimization

Major UI overhaul to an ECDIS-style map-centric layout, enhanced weather visualization, and a formalized route optimization workflow with two speed strategies.

- **ECDIS-style UI redesign** — remove left sidebar, full-width map with header icon dropdowns for voyage parameters and regulation zones; consolidated vessel config, calibration, and fuel analysis into single `/vessel` page; ECDIS-style route indicator panel (bottom-left overlay) and right slide-out analysis panel
- **Wave crest rendering** — Windy-style curved arc crest marks perpendicular to wave propagation direction, opacity scaled by wave height; click-to-inspect popup with SVG polar diagram showing wind, swell, and windwave components on compass rose
- **Dual speed-strategy display** — after A\* path optimization, present two scenarios: **Same Speed** (constant speed, arrive earlier, moderate fuel savings) and **Match ETA** (slow-steam to match baseline arrival time, maximum fuel savings); strategy selector tabs in route comparison panel
- **Voyage baseline gating** — Optimize A\* button disabled until a voyage calculation baseline is computed, ensuring meaningful fuel/time comparisons
- **Dual-route visualization** — display original (blue) and optimized (green dashed) routes simultaneously on map with comparison table (distance, fuel, time, waypoints) and Dismiss/Apply buttons
- **GFS wind DB ingestion** — add wind grids to the 6-hourly ingestion cycle (41 GFS forecast hours, 3h steps)
- **Turn-angle path smoothing** — post-filter removes waypoints with <15 deg course change to eliminate grid staircase artifacts from A\* output
- **A\* optimizer tuning** — increase time penalty to prevent long zigzag detours; scale smoothing tolerance to grid resolution

### v0.0.5 — Weather Database Architecture

Pre-ingested weather grids in PostgreSQL, eliminating live download latency during route calculations.

- **Weather ingestion service** — background task downloads CMEMS wave/current and GFS wind grids every 6 hours, compresses with zlib (float32), stores in PostgreSQL
- **DB weather provider** — reads compressed grids, crops to route bounding box, returns `WeatherData` objects compatible with `GridWeatherProvider`
- **Multi-tier fallback chain** — Redis shared cache, DB pre-ingested, live CMEMS/GFS, synthetic
- **Redis shared cache** — replaces per-worker in-memory dict, all 4 Uvicorn workers share weather data
- **Frontend freshness indicator** — shows weather data age (green/yellow/red) in map overlay controls
- **Performance**: route optimization from ~90-180s to 2-5s; voyage calculation from minutes to sub-second

### v0.0.4 — Frontend Refactor, Monte Carlo, Grid Weather

Component architecture refactor and Monte Carlo simulation engine.

- **Frontend component split** — monolithic `page.tsx` refactored into reusable components
- **Monte Carlo simulation** — N=100 parametric simulation engine with P10/P50/P90 confidence intervals for ETA, fuel, and voyage time
- **GridWeatherProvider** — bilinear interpolation from pre-fetched weather grids, enabling 1000x faster A\* routing
- **Analysis tab** — persistent storage of voyage results for comparison across routes

### v0.0.3 — Real Weather Data Integration

Live connectivity to Copernicus and NOAA weather services.

- **CMEMS wave and current data** — Copernicus Marine Service API integration with swell/wind-wave decomposition for accurate seakeeping
- **GFS 5-day wind forecast timeline** — f000-f120 (3-hourly steps) with Windy-style particle animation on the map
- **ERA5 wind fallback** — Climate Data Store reanalysis as secondary wind source (~5-day lag)
- **Data sources documentation** — credential setup guide, provider chain documentation

## Codebase

~67,000 lines of code across 197 files: 36K Python backend, 22K TypeScript frontend, 9K tests.

## Branch Strategy

- `main` — stable release branch (pushes trigger demo deployment via CI/CD)
- `dev` — active development branch

## Documentation

Full technical documentation, safety criteria, algorithm details, and changelog available at [windmar-nav.github.io](https://windmar-nav.github.io).

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

## Author

**SL Mar**

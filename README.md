# WINDMAR - Maritime Route Optimization System

> **Warning**: This project is under active development and is **not production-ready**. It is being built in public as a learning and portfolio project. APIs, data models, and features may change without notice. Do not use for actual voyage planning or navigation.

A maritime route optimization platform for Medium Range (MR) Product Tankers. Minimizes fuel consumption through weather-aware A\* routing, physics-based vessel modeling, and real-time sensor fusion.

## Features

### Vessel Performance Modeling
- Holtrop-Mennen resistance prediction (calm water, wind, waves)
- SFOC curves at variable engine loads
- Hull fouling calibration from operational noon reports
- Laden and ballast condition support
- Configurable vessel specifications (default: 49,000 DWT MR tanker)

### Route Optimization
- **Dual-engine optimization**: A\* grid search + VISIR-style Dijkstra with time-expanded graph
- All 6 route variants computed per request (2 engines × 3 safety weights: fuel / balanced / safety)
- Sequential execution with progressive UI updates as each route completes
- A\* grid at 0.2° (~12nm) resolution; VISIR at 0.25° (~15nm) — aligned with professional routing software (VISIR-2, StormGeo)
- Distance-adaptive land mask sampling (~1 check per 2nm, auto-scaled per segment length)
- Per-edge land crossing checks on both engines using `global-land-mask` (1km resolution)
- Voluntary speed reduction (VSR) in heavy weather (VISIR engine)
- Variable speed optimization (10-18 knots per leg)
- Turn-angle path smoothing to eliminate grid staircase artifacts
- Seakeeping safety constraints (roll, pitch, acceleration limits)
- RTZ file import/export (IEC 61174 ECDIS standard)

### Weather Integration
- NOAA GFS (0.25°) for near-real-time wind fields via NOMADS GRIB filter
- 5-day wind forecast timeline (f000–f120, 3-hourly steps) with Windy-style animation
- Copernicus Marine Service (CMEMS) for wave and ocean current data
- ERA5 reanalysis as secondary wind fallback (~5-day lag)
- Climatology fallback for beyond-forecast-horizon voyages
- Unified provider that blends forecast and climatology with smooth transitions
- **Pre-ingested weather database** — grids compressed (zlib/float32) in PostgreSQL, served in milliseconds
- **Redis shared cache** across all API workers (replaces per-worker in-memory dict)
- 6-hourly background ingestion cycle (waves, currents, and wind) with DB → live → synthetic fallback chain
- Synthetic data generator for testing and demos

### Monte Carlo Simulation
- Parametric Monte Carlo with temporally correlated perturbations
- Divides voyage into up to 100 time slices (~1 per 1.2 hours)
- Cholesky decomposition of exponential temporal correlation matrix
- Log-normal perturbation model: wind σ=0.35, wave σ=0.20 (70% correlated with wind), current σ=0.15
- P10/P50/P90 confidence intervals for ETA, fuel consumption, and voyage time
- Pre-fetches multi-timestep wave forecast grids from database (0–120h)
- 100 simulations complete in <500ms

### Regulatory Compliance
- IMO CII (Carbon Intensity Indicator) calculations with annual tightening
- Emission Control Areas (ECA/SECA) with fuel switching requirements
- High Risk Areas (HRA), Traffic Separation Schemes (TSS)
- Custom zone creation with penalty/exclusion/mandatory interactions
- GeoJSON export for frontend visualization

### Live Operations
- SBG Electronics IMU sensor integration (roll, pitch, heave)
- FFT-based wave spectrum estimation from ship motion
- Multi-source sensor fusion engine
- Continuous model recalibration from live data

### Web Interface
- ECDIS-style map-centric layout with full-width chart and header dropdowns
- Interactive Leaflet maps with weather overlays and route visualization
- Wind particle animation layer (leaflet-velocity)
- Windy-style wave crest rendering with click-to-inspect polar diagram popup
- Forecast timeline with play/pause, speed control, and 5-day scrubbing
- All 6 optimized routes displayed simultaneously with per-route color coding and toggleable visibility
- Unified comparison table with fuel, distance, time, and waypoint counts for every route variant
- Sequential optimization with progressive map updates (routes appear one by one)
- Navigation persistence — waypoints, route name, and optimization results survive page navigation via React Context
- Voyage calculation with per-leg fuel, speed, and ETA breakdown
- Consolidated vessel configuration, calibration, and fuel analysis page (CSV + Excel upload)
- CII compliance tracking and projections
- Dark maritime theme, responsive design

## Architecture

```
windmar/
├── api/                        # FastAPI backend
│   ├── main.py                 # API endpoints, weather ingestion loop, DB provider chain
│   ├── auth.py                 # API key authentication (bcrypt)
│   ├── config.py               # API configuration (pydantic-settings)
│   ├── middleware.py            # Security headers, structured logging, metrics
│   ├── rate_limit.py           # Token bucket rate limiter (Redis-backed)
│   ├── database.py             # SQLAlchemy ORM setup
│   ├── models.py               # Database models
│   ├── health.py               # Health check logic
│   ├── state.py                # Thread-safe application state
│   ├── cache.py                # Weather data caching (Redis shared cache)
│   ├── resilience.py           # Circuit breakers
│   ├── cli.py                  # CLI utilities
│   └── live.py                 # Live sensor data API router
├── src/
│   ├── optimization/
│   │   ├── vessel_model.py     # Holtrop-Mennen fuel consumption model
│   │   ├── base_optimizer.py   # Abstract base class for route optimizers
│   │   ├── route_optimizer.py  # A* grid search with weather costs (0.2°)
│   │   ├── visir_optimizer.py  # VISIR-style Dijkstra time-expanded graph (0.25°)
│   │   ├── router.py           # Engine dispatcher (A*/VISIR selection)
│   │   ├── voyage.py           # Per-leg voyage calculator (LegWeather, VoyageResult)
│   │   ├── monte_carlo.py      # Temporal MC simulation with Cholesky correlation
│   │   ├── grid_weather_provider.py  # Bilinear interpolation from pre-fetched grids
│   │   ├── temporal_weather_provider.py  # Trilinear interpolation (lat, lon, time)
│   │   ├── weather_assessment.py  # Route weather assessment + DB provisioning
│   │   ├── vessel_calibration.py  # Noon report calibration (scipy)
│   │   └── seakeeping.py       # Ship motion safety assessment
│   ├── data/
│   │   ├── copernicus.py       # GFS, ERA5, CMEMS providers + forecast prefetch
│   │   ├── db_weather_provider.py  # DB-backed weather (compressed grids from PostgreSQL)
│   │   ├── weather_ingestion.py    # Scheduled weather grid ingestion service
│   │   ├── regulatory_zones.py # Zone management and point-in-polygon
│   │   ├── eca_zones.py        # ECA zone definitions
│   │   └── land_mask.py        # Ocean/land detection
│   ├── sensors/
│   │   ├── sbg_nmea.py         # SBG IMU NMEA parsing
│   │   ├── sbg_ellipse.py      # SBG Ellipse sensor driver
│   │   └── wave_estimator.py   # FFT wave spectrum from heave data
│   ├── fusion/
│   │   └── fusion_engine.py    # Multi-source data fusion
│   ├── compliance/
│   │   └── cii.py              # IMO CII rating calculations
│   ├── routes/
│   │   └── rtz_parser.py       # RTZ XML route file parser
│   ├── validation.py           # Input validation
│   ├── config.py               # Application configuration
│   └── metrics.py              # Performance metrics collection
├── frontend/                   # Next.js 15 + TypeScript
│   ├── app/                    # Pages (route planner, fuel analysis, vessel config, CII, live dashboard)
│   ├── components/             # React components (maps, charts, weather layers, forecast timeline)
│   └── lib/                    # API client, utilities
├── tests/
│   ├── unit/                   # Vessel model, router, validation, ECA zones, Excel parser, CII, calibration, SBG NMEA, metrics
│   ├── integration/            # API endpoints, optimization flow
│   └── test_e2e_*.py           # End-to-end sensor integration
├── examples/                   # Demo scripts (simple, ARA-MED, calibration)
├── docker/                     # init-db.sql, migrations/ (weather tables)
├── data/                       # Runtime data (GRIB cache, calibration, climatology)
├── docker-compose.yml          # Full stack (API + frontend + PostgreSQL + Redis)
├── Dockerfile                  # Multi-stage production build
└── pyproject.toml              # Poetry project definition
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
git clone https://github.com/SL-Mar/Windmar.git
cd Windmar
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
| **Wind** | NOAA GFS (0.25°, ~3.5h lag) | ERA5 reanalysis → Synthetic | None (GFS is free) |
| **Waves** | CMEMS global wave model | Synthetic | CMEMS account |
| **Currents** | CMEMS global physics model | Synthetic | CMEMS account |
| **Forecast** | GFS f000–f120 (5-day, 3h steps) | — | None |

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

See the [Weather Data Documentation](https://quantcoder-fs.com/windmar/weather-data.html) for full technical details on data acquisition, GRIB processing, and the forecast timeline.

See the [Monte Carlo Simulation](https://quantcoder-fs.com/windmar/monte-carlo.html) article for the mathematical framework behind the temporal perturbation model.

## API Endpoints

### Weather
- `GET /api/weather/wind` - Wind field grid (U/V components)
- `GET /api/weather/wind/velocity` - Wind in leaflet-velocity format (GFS)
- `GET /api/weather/waves` - Wave height field (CMEMS)
- `GET /api/weather/currents` - Ocean current field (CMEMS)
- `GET /api/weather/currents/velocity` - Currents in leaflet-velocity format
- `GET /api/weather/point` - Weather at specific coordinates

### Forecast (Wind)
- `GET /api/weather/forecast/status` - GFS prefetch progress and run info
- `POST /api/weather/forecast/prefetch` - Trigger 5-day forecast download (f000–f120)
- `GET /api/weather/forecast/frames` - Bulk download all forecast frames

### Forecast (Wave)
- `GET /api/weather/forecast/wave/status` - Wave forecast prefetch progress
- `POST /api/weather/forecast/wave/prefetch` - Trigger wave forecast download
- `GET /api/weather/forecast/wave/frames` - Bulk download wave forecast frames

### Routes
- `POST /api/routes/parse-rtz` - Parse RTZ route file
- `POST /api/routes/from-waypoints` - Create route from coordinates

### Voyage
- `POST /api/voyage/calculate` - Full voyage calculation with weather
- `GET /api/voyage/weather-along-route` - Weather conditions per waypoint

### Monte Carlo
- `POST /api/voyage/monte-carlo` - Parametric MC simulation (P10/P50/P90)

### Optimization
- `POST /api/optimize/route` - Weather-optimal route finding (A\* or VISIR engine, selectable via `engine` param)
- `GET /api/optimize/status` - Optimizer configuration and available targets

### Weather Ingestion
- `POST /api/weather/ingest` - Trigger immediate weather ingestion cycle
- `GET /api/weather/ingest/status` - Latest ingestion run info and grid counts
- `GET /api/weather/freshness` - Weather data age indicator

### Vessel
- `GET /api/vessel/specs` - Current vessel specifications
- `POST /api/vessel/specs` - Update vessel specifications
- `GET /api/vessel/calibration` - Current calibration factors
- `POST /api/vessel/calibration/set` - Manually set calibration factors
- `POST /api/vessel/calibrate` - Run calibration from noon reports
- `POST /api/vessel/calibration/estimate-fouling` - Estimate hull fouling factor
- `GET /api/vessel/noon-reports` - List uploaded noon reports
- `POST /api/vessel/noon-reports` - Add a single noon report
- `POST /api/vessel/noon-reports/upload-csv` - Upload operational data (CSV)
- `POST /api/vessel/noon-reports/upload-excel` - Upload operational data (Excel .xlsx/.xls)
- `DELETE /api/vessel/noon-reports` - Clear all noon reports

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

```bash
pytest tests/ -v                             # All tests
pytest tests/unit/ -v                        # Unit tests only
pytest tests/integration/ -v                 # Integration tests
pytest tests/unit/test_vessel_model.py -v    # Specific test file
```

## Default Vessel

The system ships with a default MR Product Tanker configuration:

| Parameter | Value |
|-----------|-------|
| DWT | 49,000 MT |
| LOA / Beam | 183m / 32m |
| Draft (laden / ballast) | 11.8m / 6.5m |
| Main Engine | 8,840 kW |
| SFOC at MCR | 171 g/kWh |
| Service Speed (laden / ballast) | 14.5 / 15.0 knots |

## Changelog

### v0.0.6 — ECDIS UI Redesign & Dual Speed-Strategy Optimization

Major UI overhaul to an ECDIS-style map-centric layout, enhanced weather visualization, and a formalized route optimization workflow with two speed strategies.

- **ECDIS-style UI redesign** — remove left sidebar, full-width map with header icon dropdowns for voyage parameters and regulation zones; consolidated vessel config, calibration, and fuel analysis into single `/vessel` page; ECDIS-style route indicator panel (bottom-left overlay) and right slide-out analysis panel
- **Wave crest rendering** — Windy-style curved arc crest marks perpendicular to wave propagation direction, opacity scaled by wave height; click-to-inspect popup with SVG polar diagram showing wind, swell, and windwave components on compass rose
- **Dual speed-strategy display** — after A\* path optimization, present two scenarios: **Same Speed** (constant speed, arrive earlier, moderate fuel savings) and **Match ETA** (slow-steam to match baseline arrival time, maximum fuel savings); strategy selector tabs in route comparison panel
- **Voyage baseline gating** — Optimize A\* button disabled until a voyage calculation baseline is computed, ensuring meaningful fuel/time comparisons
- **Dual-route visualization** — display original (blue) and optimized (green dashed) routes simultaneously on map with comparison table (distance, fuel, time, waypoints) and Dismiss/Apply buttons
- **GFS wind DB ingestion** — add wind grids to the 6-hourly ingestion cycle (41 GFS forecast hours, 3h steps); supplement temporal weather with live GFS wind when DB grids are unavailable
- **Forecast data indicator** — timeline scrubber shows data source field for forecast frames
- **Weather forecast coverage fix** — remove CMEMS bounding-box cap so downloads cover full viewport; dynamic subsampling keeps payload size manageable; cache staleness detection triggers fresh downloads (#21)
- **Turn-angle path smoothing** — post-filter removes waypoints with <15° course change to eliminate grid staircase artifacts from A\* output
- **A\* optimizer tuning** — increase time penalty (λ\_time from 0.5× to 1.0× service fuel) to prevent long zigzag detours; scale smoothing tolerance to grid resolution
- **Data consistency fixes** — fuel savings display shows amber for increases; constant-speed baseline for fair comparison; SOG-based speed loss; temporal weather wired into voyage calculator; enforce max\_time\_factor on variable speed

### v0.0.5 — Weather Database Architecture

Pre-ingested weather grids in PostgreSQL, eliminating live download latency during route calculations.

- **Weather ingestion service** — background task downloads CMEMS wave/current and GFS wind grids every 6 hours, compresses with zlib (float32), stores in PostgreSQL (`weather_forecast_runs` + `weather_grid_data` tables)
- **DB weather provider** — `DbWeatherProvider` reads compressed grids, crops to route bounding box, returns `WeatherData` objects compatible with `GridWeatherProvider`
- **Multi-tier fallback chain** — Redis shared cache → DB pre-ingested → live CMEMS/GFS → synthetic
- **Redis shared cache** — replaces per-worker in-memory dict, all 4 Uvicorn workers share weather data
- **Frontend freshness indicator** — shows weather data age (green/yellow/red) in map overlay controls
- **Performance**: route optimization from ~90-180s → 2-5s; voyage calculation from minutes → sub-second

### v0.0.4 — Frontend Refactor, Monte Carlo, Grid Weather

Component architecture refactor and Monte Carlo simulation engine.

- **Frontend component split** — monolithic `page.tsx` refactored into reusable components (`MapOverlayControls`, `VoyagePanel`, `AnalysisTab`, etc.)
- **Monte Carlo simulation** — N=100 parametric simulation engine with P10/P50/P90 confidence intervals for ETA, fuel, and voyage time
- **GridWeatherProvider** — bilinear interpolation from pre-fetched weather grids (microsecond-level lookups vs. per-leg API calls), enabling 1000x faster A\* routing
- **Analysis tab** — persistent storage of voyage results for comparison across routes

### v0.0.3 — Real Weather Data Integration

Live connectivity to Copernicus and NOAA weather services.

- **CMEMS wave and current data** — Copernicus Marine Service API integration with swell/wind-wave decomposition for accurate seakeeping
- **GFS 5-day wind forecast timeline** — f000–f120 (3-hourly steps) with Windy-style particle animation on the map
- **ERA5 wind fallback** — Climate Data Store reanalysis as secondary wind source (~5-day lag)
- **Data sources documentation** — credential setup guide, provider chain documentation
- **Weather data page** — dedicated documentation for acquisition, GRIB processing, forecast timeline

## Branch Strategy

- `main` - Stable release branch
- `development` - Integration branch for features in progress
- `feature/*` - Feature branches for experimental work

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

## Author

**SL Mar**

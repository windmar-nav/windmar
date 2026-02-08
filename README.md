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
- A\* grid-based pathfinding with configurable resolution (0.25-2.0 degrees)
- Variable speed optimization (6-18 knots per leg)
- Seakeeping safety constraints (roll, pitch, acceleration limits)
- Land avoidance via vectorized ocean mask (global-land-mask)
- RTZ file import/export (IEC 61174 ECDIS standard)

### Weather Integration
- NOAA GFS (0.25°) for near-real-time wind fields via NOMADS GRIB filter
- 5-day wind forecast timeline (f000–f120, 3-hourly steps) with Windy-style animation
- Copernicus Marine Service (CMEMS) for wave and ocean current data
- ERA5 reanalysis as secondary wind fallback (~5-day lag)
- Climatology fallback for beyond-forecast-horizon voyages
- Unified provider that blends forecast and climatology with smooth transitions
- Synthetic data generator for testing and demos

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
- Interactive Leaflet maps with weather overlays and route visualization
- Wind particle animation layer (leaflet-velocity)
- Forecast timeline with play/pause, speed control, and 5-day scrubbing
- Voyage calculation with per-leg fuel, speed, and ETA breakdown
- Vessel configuration and calibration panels
- CII compliance tracking and projections
- Dark maritime theme, responsive design

## Architecture

```
windmar/
├── api/                        # FastAPI backend
│   ├── main.py                 # API endpoints (weather, routes, voyage, zones, vessel, calibration)
│   ├── auth.py                 # JWT / API key authentication
│   ├── config.py               # API configuration (pydantic-settings)
│   ├── middleware.py            # Security headers, structured logging, metrics
│   ├── rate_limit.py           # Token bucket rate limiter (Redis-backed)
│   ├── database.py             # SQLAlchemy ORM setup
│   ├── models.py               # Database models
│   ├── health.py               # Health check logic
│   ├── state.py                # Thread-safe application state
│   ├── cache.py                # Weather data caching
│   └── resilience.py           # Circuit breakers
├── src/
│   ├── optimization/
│   │   ├── vessel_model.py     # Holtrop-Mennen fuel consumption model
│   │   ├── route_optimizer.py  # A* pathfinding with weather costs
│   │   ├── voyage.py           # Per-leg voyage calculator
│   │   ├── vessel_calibration.py  # Noon report calibration (scipy)
│   │   └── seakeeping.py       # Ship motion safety assessment
│   ├── data/
│   │   ├── copernicus.py       # GFS, ERA5, CMEMS providers + forecast prefetch
│   │   ├── regulatory_zones.py # Zone management and point-in-polygon
│   │   ├── eca_zones.py        # ECA zone definitions
│   │   └── land_mask.py        # Ocean/land detection
│   ├── sensors/
│   │   ├── sbg_nmea.py         # SBG IMU NMEA parsing
│   │   └── sbg_ellipse.py      # SBG Ellipse sensor driver
│   ├── fusion/
│   │   ├── fusion_engine.py    # Multi-source data fusion
│   │   └── wave_estimator.py   # FFT wave spectrum from heave data
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
│   ├── unit/                   # Vessel model, router, validation, ECA zones, Excel parser
│   ├── integration/            # API endpoints, optimization flow
│   └── test_e2e_*.py           # End-to-end sensor integration
├── examples/                   # Demo scripts (simple, ARA-MED, calibration)
├── docker/                     # init-db.sql (PostgreSQL schema)
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
| Auth | JWT, API keys, bcrypt |
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
| Frontend | http://localhost:3000 |
| API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/api/docs |
| PostgreSQL | localhost:5432 |
| Redis | localhost:6379 |

### Manual Setup

```bash
# Backend
pip install -r requirements.txt
python api/main.py

# Frontend (separate terminal)
cd frontend
npm install --legacy-peer-deps
npm run dev
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
| `API_SECRET_KEY` | JWT signing key | (generate with `openssl rand -hex 32`) |
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

## API Endpoints

### Weather
- `GET /api/weather/wind` - Wind field grid (U/V components)
- `GET /api/weather/wind/velocity` - Wind in leaflet-velocity format (GFS)
- `GET /api/weather/waves` - Wave height field (CMEMS)
- `GET /api/weather/currents` - Ocean current field (CMEMS)
- `GET /api/weather/currents/velocity` - Currents in leaflet-velocity format
- `GET /api/weather/point` - Weather at specific coordinates

### Forecast
- `GET /api/weather/forecast/status` - GFS prefetch progress and run info
- `POST /api/weather/forecast/prefetch` - Trigger 5-day forecast download (f000–f120)
- `GET /api/weather/forecast/frames` - Bulk download all forecast frames

### Routes
- `POST /api/routes/parse-rtz` - Parse RTZ route file
- `POST /api/routes/from-waypoints` - Create route from coordinates

### Voyage
- `POST /api/voyage/calculate` - Full voyage calculation with weather
- `GET /api/voyage/weather-along-route` - Weather conditions per waypoint

### Optimization
- `POST /api/optimize/route` - A\* weather-optimal route finding

### Vessel
- `GET /api/vessel/specs` - Current vessel specifications
- `POST /api/vessel/specs` - Update vessel specifications
- `POST /api/vessel/calibrate` - Run calibration from noon reports
- `POST /api/vessel/noon-reports/upload-csv` - Upload operational data

### Zones
- `GET /api/zones` - All regulatory zones (GeoJSON)
- `POST /api/zones` - Create custom zone
- `GET /api/zones/check-path` - Check zone intersections

### System
- `GET /api/health` - Health check
- `GET /api/metrics` - Prometheus metrics

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

## Branch Strategy

- `main` - Stable release branch
- `development` - Integration branch for features in progress

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

## Author

**SL Mar**

# WINDMAR Roadmap - Community Validation Path

## Vision

Build an open-source maritime weather visualization and route optimization platform,
validated step-by-step by the community. Starting with what we can verify visually
(weather data on a map), then layering in the physics engine once the foundation is solid.

## Current State (Honest Assessment)

| Component | Status | Notes |
|-----------|--------|-------|
| Frontend (Next.js + Leaflet) | Working | Map, route planning, vessel config pages render |
| Backend API (FastAPI) | Working | 20+ endpoints, auth, rate limiting |
| Weather data pipeline | Partial | Copernicus integration exists but defaults to synthetic data |
| leaflet-velocity endpoint | Ready | `/api/weather/wind/velocity` already serves grib2json format |
| Vessel fuel model | Broken | MCR cap bug + inverted wind resistance (6/16 physics tests fail) |
| A* route optimizer | Functional | Algorithm works, but cost function is distorted by vessel model bugs |
| Seakeeping model | Good | IMO-aligned safety constraints, most complete module |
| Docker deployment | Working | Multi-stage builds, health checks, docker-compose |
| CI/CD | Working | 7-job GitHub Actions pipeline |

---

## Phase 0: Open Source Preparation

**Goal**: Make the repo ready for community contribution.

- [ ] Switch LICENSE from commercial to open source (Apache 2.0 or MIT recommended)
- [ ] Update README with honest project status and "help wanted" areas
- [ ] Add CONTRIBUTING.md with setup instructions and contribution guidelines
- [ ] Add issue templates (bug report, feature request, weather data source)
- [ ] Add GitHub Discussions for architecture decisions
- [ ] Remove hardcoded dev API key from `docker/init-db.sql`
- [ ] Tag current state as `v0.1.0-alpha`

---

## Phase 1: Windy-Like Weather Visualization (First Community Milestone)

**Goal**: A beautiful, interactive weather map that anyone can verify visually.
This is the "proof of life" — no physics model needed, just real data rendered well.

### 1.1 - Animated Wind Particles

- [ ] Install `leaflet-velocity` (or `leaflet-velocity-ts`)
- [ ] Create `WindParticleLayer.tsx` wrapping leaflet-velocity in react-leaflet
- [ ] Wire to existing `/api/weather/wind/velocity` endpoint
- [ ] Color-coded particles by wind speed (blue calm -> red storm)
- [ ] Configurable particle density and trail length
- [ ] Dynamic import with SSR disabled (Next.js requirement)

### 1.2 - Wave Height Heatmap

- [ ] Create `WaveHeatmapLayer.tsx` using Canvas overlay on Leaflet
- [ ] Bilinear interpolation between grid points for smooth rendering
- [ ] Color ramp: green (< 1m) -> yellow (2m) -> orange (3m) -> red (5m+)
- [ ] Semi-transparent overlay blending with base map
- [ ] Wire to existing `/api/weather/waves` endpoint

### 1.3 - Ocean Current Visualization

- [ ] Create `CurrentLayer.tsx` with animated arrows or streamlines
- [ ] Wire to existing `/api/weather/currents` endpoint
- [ ] Show current speed and direction
- [ ] Different visual style from wind (dashed lines or thinner arrows)

### 1.4 - Time Slider

- [ ] Create `TimeSlider.tsx` component (horizontal bar at bottom of map)
- [ ] Add backend endpoint for forecast time range (`/api/weather/forecast-times`)
- [ ] Pre-fetch adjacent time steps for smooth scrubbing
- [ ] Play/pause animation through forecast hours
- [ ] Display current forecast time prominently

### 1.5 - Interactive Controls

- [ ] Layer toggle panel (Wind / Waves / Currents / Pressure)
- [ ] Color legend with auto-scaling min/max values
- [ ] Click-to-inspect: show exact values at any point on the map
- [ ] Overlay opacity slider per layer

### 1.6 - Real Weather Data Connection

- [ ] Add NOAA GFS data pipeline (free, no API key, 0.25 deg resolution)
  - Download GRIB2 from NOMADS filter (UGRD + VGRD at 10m)
  - Convert to grib2json format server-side
  - Cache with 6-hour TTL matching GFS update cycle
- [ ] Add Open-Meteo as alternative source (JSON API, no key needed)
- [ ] Data source indicator on map (showing: GFS / Copernicus / Synthetic)
- [ ] Fallback chain: GFS -> Copernicus -> Open-Meteo -> Synthetic

### Validation Criteria for Phase 1
- [ ] Wind patterns visually match windy.com for the same region and time
- [ ] Wave heights match published buoy data (NDBC) within +/- 0.5m
- [ ] Community members can run locally and confirm visual correctness
- [ ] Performance: smooth 30fps animation with 5000+ particles

---

## Phase 2: Fix the Physics Engine

**Goal**: Make the vessel model produce correct fuel predictions.
Community can validate against published noon report datasets.

### 2.1 - Fix Critical Vessel Model Bugs

- [ ] Fix MCR cap: recalibrate resistance so service speed = ~75% MCR (not 100%)
- [ ] Fix wind resistance: following wind should produce thrust, not drag
- [ ] Fix wave resistance: don't zero it out above Froude number 0.4
- [ ] Fix form factor: use full Holtrop-Mennen formulation with lcb_fraction
- [ ] Get all 16 vessel model tests passing

### 2.2 - Model Validation Framework

- [ ] Create benchmark dataset from public noon report sources
- [ ] Comparison tool: model prediction vs actual consumption
- [ ] Statistical metrics: MAPE, RMSE, bias for fuel predictions
- [ ] Automated regression tests against benchmark data
- [ ] Visual comparison plots (predicted vs actual)

### 2.3 - Enable MyPy in CI

- [ ] Uncomment mypy check in `.github/workflows/ci.yml`
- [ ] Fix type errors across codebase
- [ ] Enforce minimum test coverage threshold (80%)

### Validation Criteria for Phase 2
- [ ] All 16 vessel model unit tests pass
- [ ] MAPE < 15% on benchmark noon report dataset
- [ ] Laden fuel > Ballast fuel (for same voyage)
- [ ] Head wind fuel > Following wind fuel
- [ ] Fuel increases monotonically with speed (within operational range)

---

## Phase 3: Route Optimization Validation

**Goal**: Demonstrate that weather routing actually saves fuel.

- [ ] Compare optimized vs great circle routes for historical voyages
- [ ] Show fuel savings as percentage with confidence intervals
- [ ] Validate against published weather routing case studies
- [ ] Add Dijkstra as alternative algorithm for comparison
- [ ] Performance profiling: optimize A* grid lookup (add spatial indexing)
- [ ] Test with real Copernicus forecast data over known routes

### Validation Criteria for Phase 3
- [ ] Optimized routes avoid known storm systems (visual check)
- [ ] Fuel savings of 3-15% vs great circle (consistent with industry literature)
- [ ] Route optimization completes in < 30 seconds for typical voyages
- [ ] No routes cross land

---

## Phase 4: Production Hardening

**Goal**: Make it reliable enough for real operational use.

- [ ] E2E smoke tests with Playwright
- [ ] Load testing with k6 or locust
- [ ] Database backup/restore procedures
- [ ] Monitoring: Sentry integration, Prometheus alerting rules
- [ ] Rate limiting per API key with tiered plans
- [ ] Structured logging with ELK or Loki
- [ ] SSL/TLS via reverse proxy (nginx/caddy)
- [ ] Pagination on all list endpoints

---

## Phase 5: Fleet & Community Features

**Goal**: Multi-vessel support and community-driven improvements.

- [ ] Multi-vessel tracking dashboard
- [ ] Fleet-wide CII compliance reporting
- [ ] Community-contributed vessel profiles (different ship types)
- [ ] Plugin architecture for custom data sources
- [ ] Mobile-responsive design for bridge tablet use
- [ ] Offline mode with cached weather data

---

## How to Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions and guidelines.

### High-Impact First Contributions

| Area | Difficulty | Impact | Issue Label |
|------|-----------|--------|-------------|
| Animated wind particles (Phase 1.1) | Medium | High | `good-first-issue` |
| Fix wind resistance formula (Phase 2.1) | Medium | Critical | `bug` |
| NOAA GFS data pipeline (Phase 1.6) | Medium | High | `data-pipeline` |
| Wave heatmap overlay (Phase 1.2) | Medium | High | `visualization` |
| Add Open-Meteo integration (Phase 1.6) | Easy | Medium | `good-first-issue` |

### Tech Stack

- **Backend**: Python 3.10+ / FastAPI
- **Frontend**: Next.js 15 / TypeScript / React 19 / Tailwind CSS
- **Maps**: Leaflet 1.9 / react-leaflet 4.2
- **Database**: PostgreSQL 16 / Redis 7
- **CI**: GitHub Actions
- **Containers**: Docker with multi-stage builds

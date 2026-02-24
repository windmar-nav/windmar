# WINDMAR Roadmap

> Last updated: 2026-02-24

## Product Vision

**Local-first onboard decision support** for weather routing and vessel performance analytics. Replaces black-box dependency with transparent, crew-accessible analysis running on dedicated shipboard hardware.

### Distribution & Business Model

| Aspect | Decision |
|--------|----------|
| **Build format** | Docker Compose bundle, downloadable from slmar.co |
| **Licensing** | Offline Ed25519-signed license file with expiry. Revocable at renewal. |
| **Pricing** | Application is **free**. Revenue from paid services. |
| **Paid services** | Noon report ingestion adaptation, AIS feed integration, hull fouling modeling, custom support |
| **Onboarding** | BYOK — user provides own Copernicus Marine credentials |
| **User data** | No user base on server. All data stays onboard. |
| **Waitlist** | Email capture form on slmar.co |
| **Target hardware** | Dedicated Linux box alongside other shipboard ops automation tools |

### Demo (demo-windmar.slmar.co)

Same codebase as full application, with restrictions:
- **Fixed coverage**: NE Atlantic + Mediterranean basin only
- **Ship performance**: Pre-loaded sample Excel, persistent on server
- **No upload / no download**
- **No user accounts**

### Platform Layer (slmar.co/windmar)

Research-oriented: engineers and researchers can upload specs, models, and optimization engines. The platform is for exchange and collaboration — the product is local-first.

---

## Phase 1 — Stabilize & Refactor (COMPLETE)

Delivered in v0.0.8 → v0.1.0:

- **TN002 Physics Audit** — 69 stress tests (Holtrop-Mennen, SFOC, Kwon, seakeeping, predictor)
- **Demo Alignment** — Unified codebase behind `DEMO_MODE` flag, CI/CD pipeline, bcrypt demo auth
- **Modular Refactoring** — `main.py` 6,922 → 281 LOC, 9 domain routers, thread-safe VesselState
- **Calibration Fixes** — ME-specific fuel, laden/ballast detection, engine log deduplication

## Phase 2 — Commercial Credibility (COMPLETE)

Delivered in v0.1.0. Reporting and compliance features ship operators evaluate when shortlisting routing tools.

- **2a. Voyage Reporting** — Noon reports (IMO format), departure/arrival templates, PDF export, voyage history
- **2b. CII Simulator** — What-if projections, fleet dashboard, regulatory threshold visualization (A-E bands)
- **2c. FuelEU Maritime** — GHG intensity, compliance balance, pooling scenarios, penalty estimator
- **2d. Charter Party Weather Clauses** — Good weather day counter, warranted speed verification, off-hire detection

## Phase 3 — Optimizer Upgrade (COMPLETE)

Delivered in v0.1.0. Production-grade routing graph replacing uniform-grid engines.

- **GSHHS coastline polygons** — high-resolution land boundaries replacing 1km grid
- **Variable resolution grid** — 0.1° nearshore, 0.5° open ocean
- **Strait shortcuts** — 8 commercial straits, 36 bidirectional edges
- **Variable speed per leg** — 13 steps (10–16 kts, 0.5 kt increments)
- **Pareto front** — fuel vs. time tradeoff curve (7 lambda sweep)

## Phase 4 — Local-First & Distribution

Prepare the application for onboard deployment. This is the next major phase.

### 4a. Licensing & Onboarding
- **Ed25519 license system** — offline-signed license file with vessel ID, expiry date, feature tier
- **License check** — validated on container start, blocks app if invalid/expired
- **Onboarding wizard** — guided vessel setup, Copernicus BYOK credential entry, first route calculation
- **Demo data cleanup** — separate demo seed from production startup path

### 4b. Weather Pipeline Optimization
- **Latency profiling** — measure end-to-end fetch time for CMEMS wind/wave/current/SST
- **Aggressive caching** — local NetCDF cache with TTL, prefetch next 5-day window
- **Offline graceful degradation** — app functions with stale/cached weather when connectivity drops
- **Bandwidth optimization** — subsample at fetch time, compress transfers, delta updates

### 4c. Hardware Requirements Assessment
- **Profile minimum specs** — CPU, RAM, disk for typical 7-day route optimization
- **GPU optional** — confirm app runs on CPU-only (no CUDA dependency in core)
- **Storage sizing** — weather cache, engine logs, voyage history retention policy
- **Network** — minimum bandwidth for CMEMS data refresh (VSAT-friendly)

### 4d. Demo Hardening
- **Fix coordinate boundaries** — lock to NE Atlantic + Mediterranean basin
- **Sample vessel data** — tailor Excel file with realistic MR tanker performance data, permanently on server
- **Disable all upload/download endpoints** in demo mode
- **Waiting list form** — email capture on slmar.co/windmar, stores to local JSON or Notion

### 4e. Packaging
- **Docker Compose distribution bundle** — single `docker compose up` deployment
- **Install script** — checks Docker, pulls images, validates license, runs onboarding
- **Auto-update mechanism** — versioned image tags, update check on startup
- **User documentation** — deployment guide, API reference, user manual

## Phase 5 — Performance Feedback Loop

Close the loop between predicted and actual vessel performance.

- **Hull degradation model** — fouling rate estimation from engine log trends
- **Trim optimization** — ballast/cargo distribution recommendations
- **Auto-calibration** — continuous model update from noon reports
- **Digital twin dashboard** — real-time predicted vs. actual comparison with drift alerts

## Phase 6 — Research Platform

The slmar.co/windmar exchange layer for engineers and researchers.

- **Spec upload** — vessel specifications in structured format
- **Model upload** — custom vessel performance models (Python modules, validated schema)
- **Optimization engine upload** — pluggable routing engines (interface contract, sandboxed execution)
- **Benchmarking** — compare uploaded engines against reference routes
- **Community showcase** — published results, leaderboards

## Phase 7 — Probabilistic Engine (Deferred)

Advanced ensemble weather uncertainty modeling. Deferred until commercial traction validates the investment.

- ERA5 archive ingestion + EOF decomposition
- Wasserstein clustering + analogue library
- GFS-analogue splicing with daily score/prune/recruit
- Retrospective validation (CRPS, rank histogram, Brier skill score)
- Strip chart voyage report (P10/P50/P90 envelopes)

## Paid Services (Revenue)

Delivered as custom engagements per client:

| Service | Description |
|---------|-------------|
| **Noon report ingestion** | Adapt pipeline to client's specific noon report format (Excel, PDF, API) |
| **AIS feed integration** | Connect live AIS data for fleet tracking and route monitoring |
| **Hull fouling modeling** | Custom fouling rate model calibrated to client's hull coating and trading pattern |
| **Custom support** | Vessel-specific calibration, onboarding assistance, training |

## Dropped

- ~~Multi-tenant auth~~ — no user base on server, local-first
- ~~Stripe billing~~ — app is free, services invoiced directly
- ~~Fleet-level SaaS dashboard~~ — out of scope for local-first model

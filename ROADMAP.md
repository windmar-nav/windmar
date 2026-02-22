# WINDMAR Roadmap

> Last updated: 2026-02-22

## Phase 1 — Stabilize & Refactor (COMPLETE)

Delivered in v0.0.8 and v0.0.9 (consolidated into v0.1.0):

- **TN002 Physics Audit** — 69 stress tests covering Holtrop-Mennen, SFOC, Kwon, seakeeping, and performance predictor. All passing.
- **Demo Alignment** — Unified codebase behind `DEMO_MODE` env flag. 344-entry engine log seed, CI/CD pipeline, bcrypt demo auth.
- **Modular Refactoring** — `main.py` reduced from 6,922 to 281 lines. 9 domain routers, 37 Pydantic schemas, thread-safe VesselState singleton.
- **Calibration Fixes** — ME-specific fuel, laden/ballast detection from load %, widened bounds, engine log deduplication.

## Phase 2 — Commercial Credibility (COMPLETE)

Delivered in v0.1.0. Reporting and compliance features that ship operators evaluate when shortlisting routing tools.

### 2a. Voyage Reporting Module
- Noon report generation (IMO format) from completed route calculations
- Departure / arrival report templates
- PDF export with company branding placeholder
- Voyage history list with search and filters

### 2b. CII Simulator
- What-if CII projection: adjust speed, fuel type, or route and see rating impact
- Fleet-level CII dashboard (multiple vessels)
- Regulatory threshold visualization (A-E bands with tightening schedule)

### 2c. FuelEU Maritime Compliance
- GHG intensity calculation per voyage (Well-to-Wake)
- Compliance balance tracking (surplus/deficit)
- Pooling scenario modeling
- Penalty exposure estimator

### 2d. Charter Party Weather Clause Tools
- Good weather day counter along route (Beaufort scale thresholds)
- Warranted speed / consumption verification against model predictions
- Off-hire event detection from engine log data

## Phase 3 — Optimizer Upgrade (COMPLETE)

Delivered in v0.1.0. Production-grade routing graph replacing uniform-grid engines.

- **GSHHS coastline polygons** — high-resolution land boundaries replacing `global-land-mask` (1km grid)
- **Corridor grid with variable resolution** — 0.1 deg nearshore, 0.5 deg open ocean, auto-refined around obstacles
- **16-neighbor connectivity** — reduces course-change artifacts from 8-neighbor grid
- **Visibility graph merge** — optimal strait and channel transit via vertex-to-vertex edges
- **Variable speed per leg** — optimizer selects speed from a discrete set (e.g., 10-16 kts in 0.5 kt steps)
- **Multi-objective Pareto front** — fuel vs. time tradeoff curve instead of single lambda parameter

## Phase 4 — Performance Feedback Loop

Close the loop between predicted and actual vessel performance.

- **Hull degradation model** — fouling rate estimation from engine log trends, periodic recalibration
- **Trim optimization** — ballast/cargo distribution recommendations for minimum resistance
- **Auto-calibration** — continuous model update from incoming noon reports without manual intervention
- **Digital twin dashboard** — real-time predicted vs. actual comparison with drift alerts

## Phase 5 — Auth & Commercialization

- **Ed25519 license system** — per-vessel license keys with tier-based feature gating (standard / professional)
- **Multi-tenant auth** — customer accounts, fleet grouping, role-based access
- **Onboarding wizard** — guided vessel setup, first route calculation, demo data cleanup
- **User documentation** — deployment guide, API reference, user manual
- **Stripe billing integration** — subscription management, usage metering

## Phase 6 — Probabilistic Engine (Deferred)

Advanced ensemble weather uncertainty modeling. Deferred until commercial traction validates the investment.

- **P2a** — ERA5 archive ingestion + EOF decomposition (6 corridors, 50 modes)
- **P2b** — Wasserstein clustering + analogue library (k=500 reduced to 150)
- **P2c** — GFS-analogue splicing with daily score/prune/recruit
- **P2d** — Retrospective validation (CRPS, rank histogram, Brier skill score)
- **P3** — Strip chart voyage report (Jinja2 + Plotly.js, P10/P50/P90 envelopes)

## Dropped

- **AIS / SPEC-P2** — Not required for commercial launch. May revisit for fleet tracking in Phase 5+.

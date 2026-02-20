# Development Guide

## Prerequisites

| Tool | Version | Check |
|------|---------|-------|
| Docker & Compose | 24+ | `docker compose version` |
| Python | 3.10+ | `python3 --version` |
| Node.js | 20+ | `node --version` |
| npm | 9+ | `npm --version` |

System libraries (Ubuntu/Debian, for running outside Docker):
```bash
sudo apt install libeccodes0 libgeos-dev libproj-dev
```

## Quick Start (Docker)

```bash
# 1. Clone and configure
git clone https://github.com/SL-Mar/windmar.git
cd windmar
cp .env.example .env
```

Edit `.env` for local development:
```bash
# Ports (avoid conflicts with other services)
API_PORT=8003
FRONTEND_PORT=3003
DB_PORT=5434
REDIS_PORT=6380

# Dev mode — disable auth and rate limiting
AUTH_ENABLED=false
RATE_LIMIT_ENABLED=false
ENVIRONMENT=development

# Database
DB_USER=windmar
DB_PASSWORD=windmar_dev_password
DB_NAME=windmar

# Redis
REDIS_PASSWORD=windmar_redis_password

# CORS — must match FRONTEND_PORT
CORS_ORIGINS=http://localhost:3003

# Weather data (optional — falls back to synthetic data without credentials)
COPERNICUS_MOCK_MODE=false
COPERNICUSMARINE_SERVICE_USERNAME=
COPERNICUSMARINE_SERVICE_PASSWORD=
```

```bash
# 2. Build and start
docker compose up -d --build

# 3. Verify
docker compose ps          # All 4 services should be healthy
curl localhost:8003/api/health
```

**Services:**

| Service | Container | Internal Port | Default Host Port |
|---------|-----------|---------------|-------------------|
| PostgreSQL 16 | windmar-db | 5432 | 5434 |
| Redis 7 | windmar-redis | 6379 | 6380 |
| FastAPI API | windmar-api | 8000 | 8003 |
| Next.js Frontend | windmar-frontend | 3000 | 3003 |

Startup order: DB → Redis → API → Frontend (health-checked dependencies).

## Quick Start (Manual)

```bash
# Backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 api/main.py          # Starts on port 8000

# Frontend (separate terminal)
cd frontend
npm install --legacy-peer-deps
npm run dev                   # Starts on port 3000
```

Requires PostgreSQL and Redis running locally (or via Docker):
```bash
docker compose up -d db redis
```

Set `NEXT_PUBLIC_API_URL=http://localhost:8000` in `frontend/.env.local` when running outside Docker.

## Running Tests

```bash
source .venv/bin/activate

# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests (requires DB)
pytest tests/integration/ -v

# Specific file
pytest tests/unit/test_vessel_model.py -v

# With coverage
pytest tests/ --cov=src --cov=api --cov-report=term-missing
```

Test markers: `unit`, `integration`, `slow`.

## Project Structure

```
windmar/
├── api/                        # FastAPI backend
│   ├── main.py                 # App factory + startup (281 lines)
│   ├── routers/                # Domain routers (9 modules)
│   │   ├── weather.py          # Weather endpoints, forecast layers
│   │   ├── vessel.py           # Vessel specs, calibration, prediction
│   │   ├── voyage.py           # Voyage calc, Monte Carlo
│   │   ├── optimization.py     # A*/Dijkstra route optimization
│   │   ├── engine_log.py       # Engine log upload and analytics
│   │   ├── zones.py            # Regulatory zone CRUD
│   │   ├── cii.py              # CII compliance
│   │   ├── routes.py           # RTZ parsing, waypoint routes
│   │   └── system.py           # Health, metrics, status
│   ├── schemas/                # Pydantic models (37 schemas)
│   ├── state.py                # Thread-safe app state (singleton)
│   ├── weather_service.py      # Weather field accessors
│   ├── forecast_layer_manager.py  # Forecast dedup + progress
│   ├── config.py               # Pydantic settings
│   ├── models.py               # SQLAlchemy ORM models
│   ├── auth.py                 # API key authentication
│   ├── cache.py                # Redis weather cache
│   └── rate_limit.py           # slowapi rate limiting
├── src/                        # Core domain logic
│   ├── optimization/           # Route optimizers (A*, Dijkstra, vessel model)
│   ├── data/                   # Weather providers (GFS, CMEMS, Copernicus)
│   ├── sensors/                # SBG IMU drivers
│   ├── fusion/                 # Data fusion
│   └── compliance/             # IMO CII calculator
├── frontend/                   # Next.js 15 + React 19
│   ├── app/                    # Pages (map, analysis, vessel, engine-log)
│   ├── components/             # React components
│   └── lib/                    # API client, utilities
├── tests/                      # pytest suite
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration tests
├── docker/                     # DB init scripts, migrations
├── examples/                   # Demo scripts
├── data/                       # Runtime data (gitignored)
├── docker-compose.yml          # Full stack
├── Dockerfile                  # Backend multi-stage build
├── requirements.txt            # Python dependencies
└── pytest.ini                  # Test configuration
```

## Common Commands

```bash
# Rebuild API after code changes
docker compose up -d --build api

# Watch API logs
docker compose logs -f api

# Restart a single service
docker compose restart api

# Access PostgreSQL
docker exec -it windmar-db psql -U windmar -d windmar

# Access Redis
docker exec -it windmar-redis redis-cli -a windmar_redis_password

# Run a Python script inside the API container
docker exec -it windmar-api python3 -c "from src.optimization.vessel_model import VesselSpecs; print(VesselSpecs())"
```

## Environment Variables Reference

### Dev Mode Flags

| Variable | Dev Value | Prod Value | Effect |
|----------|-----------|------------|--------|
| `AUTH_ENABLED` | `false` | `true` | API key required on mutation endpoints |
| `RATE_LIMIT_ENABLED` | `false` | `true` | slowapi rate limits enforced |
| `COPERNICUS_MOCK_MODE` | `false` | `false` | `true` uses synthetic weather data |
| `ENVIRONMENT` | `development` | `production` | Logging verbosity, error detail |
| `LOG_LEVEL` | `debug` | `info` | Log verbosity |

### Weather Data

Live weather requires free Copernicus Marine credentials. Without them, the app falls back to synthetic data automatically.

Register at: https://marine.copernicus.eu/

```bash
COPERNICUSMARINE_SERVICE_USERNAME=your_username
COPERNICUSMARINE_SERVICE_PASSWORD=your_password
```

## Code Quality

```bash
# Format
black --line-length 88 src/ api/ tests/

# Lint
ruff check src/ api/ tests/

# Type check
mypy src/ api/
```

## Architecture Notes

- **Router-based API** — `main.py` is an application factory (281 lines). All endpoint code lives in `api/routers/` (9 domain routers). Shared state is accessed via `api/state.py` singletons, weather logic via `api/weather_service.py`, and request/response schemas via `api/schemas/`.
- **Source code is baked into the Docker image** (not bind-mounted). Run `docker compose up -d --build api` after backend changes.
- **Frontend is also baked**. Run `docker compose up -d --build frontend` after frontend changes.
- **Data directory** (`./data`) IS bind-mounted — weather caches, calibration files, and vessel DB persist across rebuilds.
- **Forecast cache** uses a Docker volume (`forecast_cache`) at `/tmp/windmar_cache` inside the container.
- The API runs behind **gunicorn** with 4 workers (600s timeout) for production resilience.
- The frontend uses Next.js **standalone** output mode for minimal container size.

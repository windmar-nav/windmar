# Contributing to WINDMAR

Thank you for your interest in contributing to WINDMAR. This guide will help you
get set up and find meaningful work to do.

## Project Status

WINDMAR is an early-stage open-source maritime route optimization platform.
The current priority is **Phase 1: Weather Visualization** (see [ROADMAP.md](ROADMAP.md)).
We welcome contributions at all skill levels.

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- Docker (optional, for full-stack deployment)

### Local Development Setup

```bash
# Clone the repo
git clone https://github.com/SL-Mar/Windmar.git
cd Windmar

# Backend
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Frontend
cd frontend
npm install
cd ..

# Start backend (terminal 1)
python api/main.py

# Start frontend (terminal 2)
cd frontend && npm run dev
```

The app will be available at:
- Frontend: http://localhost:3000
- API docs: http://localhost:8000/api/docs

### Docker Setup (Full Stack)

```bash
docker-compose up --build
```

### Running Tests

```bash
# Backend unit tests
pytest tests/unit/ -v

# Backend integration tests
pytest tests/integration/ -v

# Frontend tests
cd frontend && npm test

# Type checking
cd frontend && npx tsc --noEmit
```

## Where to Contribute

### Phase 1: Weather Visualization (Current Priority)

These are the highest-impact contributions right now:

| Task | Skills Needed | Difficulty |
|------|--------------|------------|
| Animated wind particles with leaflet-velocity | React, Leaflet, Canvas | Medium |
| Wave height heatmap overlay | React, Canvas, color theory | Medium |
| Time slider for forecast navigation | React, UI/UX | Easy-Medium |
| NOAA GFS data pipeline | Python, GRIB2, data engineering | Medium |
| Open-Meteo weather integration | Python, REST APIs | Easy |
| Color legend component | React, Tailwind CSS | Easy |

### Bug Fixes (Critical)

| Bug | Location | Skills |
|-----|----------|--------|
| Wind resistance formula inverted | `src/optimization/vessel_model.py` | Physics, Python |
| MCR cap flattens fuel predictions | `src/optimization/vessel_model.py` | Naval architecture, Python |
| Wave resistance zero at high Froude | `src/optimization/vessel_model.py` | Physics, Python |

### Always Welcome

- Improving test coverage
- Documentation improvements
- Performance optimizations
- Accessibility improvements
- Bug reports with reproducible steps

## Development Guidelines

### Code Style

**Python (backend)**:
- Formatted with Black (line length 88)
- Linted with Ruff
- Type hints required on all functions (mypy strict mode)
- Run before committing: `black api/ src/ && ruff check api/ src/`

**TypeScript (frontend)**:
- Strict TypeScript (no `any` types)
- ESLint with Next.js rules
- Run before committing: `cd frontend && npm run lint && npx tsc --noEmit`

### Commit Messages

Use conventional commits:
```
feat: add animated wind particle layer
fix: correct wind resistance coefficient direction
docs: update setup instructions for macOS
test: add benchmark for vessel model accuracy
```

### Pull Request Process

1. Fork the repo and create a branch from `main`
2. Name your branch: `feat/description`, `fix/description`, or `docs/description`
3. Make your changes with tests where applicable
4. Ensure all existing tests pass: `pytest tests/unit/ -v`
5. Ensure frontend builds: `cd frontend && npm run build`
6. Open a PR with a clear description of what changed and why
7. Reference the relevant ROADMAP phase in your PR description

### Architecture Overview

```
api/            FastAPI backend (REST API, auth, rate limiting)
  main.py       All API endpoints (~1800 lines)
  config.py     Environment-based configuration
  auth.py       API key authentication
  middleware.py Security headers, logging, metrics

src/            Core library (no web framework dependency)
  optimization/
    vessel_model.py      Holtrop-Mennen resistance + SFOC
    route_optimizer.py   A* pathfinding with weather costs
    voyage.py            Per-leg fuel/time calculator
    seakeeping.py        IMO safety constraints
    vessel_calibration.py  Noon report calibration
  data/
    copernicus.py        Weather data from Copernicus CDS/CMEMS
    land_mask.py         Land avoidance
    regulatory_zones.py  ECA/TSS/HRA zones

frontend/       Next.js 15 application
  app/          Pages (route planning, fuel analysis, vessel config, CII)
  components/   React components (map layers, charts, forms)
  lib/          API client, utilities

tests/          pytest test suite
  unit/         Unit tests for core library
  integration/  API integration tests
```

### Key Data Flow

```
Frontend (React)  -->  API (FastAPI)  -->  Weather Provider  -->  Copernicus / GFS / Synthetic
                                      -->  Vessel Model      -->  Fuel prediction
                                      -->  Route Optimizer   -->  A* pathfinding
                                      -->  Voyage Calculator -->  Per-leg results
```

## Community

- **Issues**: Use GitHub Issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions and architecture proposals
- **PRs**: All contributions via pull request, reviewed by maintainers

## License

By contributing, you agree that your contributions will be licensed under the
same license as the project (see [LICENSE](LICENSE)).

Note: The project is transitioning from a commercial license to an open-source
license. Check the current LICENSE file for the applicable terms.

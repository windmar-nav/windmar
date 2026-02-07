#!/bin/bash
# Create GitHub issues for WINDMAR open-source launch
# Run: gh auth login && bash scripts/create-github-issues.sh

set -e

echo "Creating labels..."

gh label create "good-first-issue" --color "7057ff" --description "Good for newcomers" --force
gh label create "visualization" --color "0075ca" --description "Weather visualization features" --force
gh label create "bug" --color "d73a4a" --description "Something isn't working" --force
gh label create "critical" --color "b60205" --description "Critical priority" --force
gh label create "data-pipeline" --color "0e8a16" --description "Weather data sources and pipelines" --force
gh label create "help-wanted" --color "008672" --description "Extra attention is needed" --force
gh label create "physics" --color "e4e669" --description "Vessel model and naval architecture" --force
gh label create "phase-1" --color "c5def5" --description "Phase 1: Weather Visualization" --force
gh label create "phase-2" --color "bfdadc" --description "Phase 2: Fix Physics Engine" --force

echo "Creating issues..."

# Issue 1: Wind particles
gh issue create \
  --title "Add animated wind particles with leaflet-velocity" \
  --label "good-first-issue,visualization,phase-1" \
  --body "$(cat <<'EOF'
## Summary

The backend already serves wind data in leaflet-velocity format. We need a React component that renders animated wind particles on the map.

## What exists

- `frontend/lib/api.ts:389` — `getWindVelocity()` already calls the endpoint
- `api/main.py:772` — `GET /api/weather/wind/velocity` returns grib2json format (U + V wind components)
- The endpoint is functional and returns properly formatted data

## What to build

1. Install `leaflet-velocity` (or `leaflet-velocity-ts` for TypeScript support)
2. Create `frontend/components/map/WindParticleLayer.tsx`
3. Wrap as a react-leaflet component using the `useMap()` hook
4. Color-code particles by wind speed (blue calm → red storm)
5. Dynamic import with `ssr: false` (Next.js + Leaflet requirement)

## Key integration pattern

```tsx
import { useEffect } from 'react';
import { useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet-velocity';

export function WindParticleLayer({ data }) {
  const map = useMap();
  useEffect(() => {
    if (!data) return;
    const layer = L.velocityLayer({
      displayValues: true,
      data: data,
      maxVelocity: 15,
      velocityScale: 0.01,
    });
    layer.addTo(map);
    return () => { map.removeLayer(layer); };
  }, [map, data]);
  return null;
}
```

## References

- [leaflet-velocity tutorial](https://wlog.viltstigen.se/articles/2021/11/08/visualizing-wind-using-leaflet/)
- [react-leaflet integration pattern](https://kulkarniprem.hashnode.dev/how-to-create-custom-overlay-component-in-react-leaflet-using-leaflet-velocity)

## Acceptance criteria

- [ ] Animated particles render on the map
- [ ] Particles move according to wind direction
- [ ] Color reflects wind speed
- [ ] No SSR errors in Next.js
EOF
)"

echo "  ✓ Issue 1: Wind particles"

# Issue 2: Wind resistance bug
gh issue create \
  --title "Bug: Wind resistance formula is inverted — following wind worse than head wind" \
  --label "bug,critical,physics,phase-2" \
  --body "$(cat <<'EOF'
## Bug description

The vessel model calculates **7x more resistance** for following wind than head wind. This is physically backwards — following wind should produce thrust (negative resistance), not drag.

## Failing test

```
tests/unit/test_vessel_model.py::test_head_wind_worse_than_following
```

Run it yourself:
```bash
pytest tests/unit/test_vessel_model.py::TestVesselModel::test_head_wind_worse_than_following -v
```

## Bug location

`src/optimization/vessel_model.py` — `_wind_resistance()` method

The `cx` aerodynamic coefficient yields:
- **0.2** at 0° relative angle (head wind) — should be the HIGHEST resistance
- **1.4** at 180° relative angle (following wind) — should produce THRUST

Then `abs(cx)` is applied, treating both directions as pure drag.

## Expected behavior

A proper Blendermann-style wind coefficient should:
- Produce **positive drag** for head winds (0° relative)
- Produce **near-zero or negative (thrust)** for following winds (180° relative)
- Peak resistance around 30-60° relative angle (beam/quarter wind)

## References

- Blendermann, W. (1994) "Parameter identification of wind loads on ships"
- IMO MSC.1/Circ.1228 — Wind resistance coefficients

## Impact

This bug affects **every route optimization** — the A* cost function penalizes routes with favorable wind, which is the opposite of what weather routing should do.
EOF
)"

echo "  ✓ Issue 2: Wind resistance bug"

# Issue 3: MCR cap bug
gh issue create \
  --title "Bug: MCR cap makes all fuel predictions identical regardless of conditions" \
  --label "bug,critical,physics,phase-2" \
  --body "$(cat <<'EOF'
## Bug description

At service speed (~14.5 kts), the resistance model overestimates power so much that it hits the engine's maximum continuous rating (MCR) ceiling. Once capped, **every scenario produces identical fuel: 36.73 MT** — whether laden or ballast, calm or storm, 12 kts or 16 kts.

## Failing tests (4 tests)

```bash
pytest tests/unit/test_vessel_model.py -v -k "fuel_increases_with_speed or laden_uses_more or weather_impact or calibration_factors"
```

All produce the same output:
```
fuel at 12 kts = 36.732852 MT
fuel at 14 kts = 36.732852 MT
fuel at 16 kts = 36.732852 MT
laden fuel    = 36.732852 MT
ballast fuel  = 36.732852 MT
calm fuel     = 36.732852 MT
storm fuel    = 36.732852 MT
```

## Root cause

In `src/optimization/vessel_model.py`:
```python
brake_power_kw = min(brake_power_kw, self.specs.mcr_kw)  # Clips to 8840 kW
```

At service speed, the resistance model produces a power demand that exceeds MCR. Once clipped, `load_fraction = 1.0` for all conditions, so SFOC and fuel are identical.

## Expected behavior

At service speed, the engine should operate at approximately **75% MCR load** — this is standard for commercial shipping. The resistance coefficients need recalibration:

- Frictional resistance (ITTC 1957) — likely correct
- Form factor `k1` — simplified beyond recognition, `lcb_fraction` defined but unused
- Wave-making resistance — set to zero above Fn > 0.4 (should increase)

## Fix approach

Recalibrate the Holtrop-Mennen coefficients so that:
1. Service speed (14.5 kts, laden) → ~75% MCR
2. Slow steaming (10 kts) → ~30% MCR
3. Full speed (16 kts) → ~90% MCR

These are typical values for an MR tanker with 8,840 kW MCR.

## Impact

This is the **most critical bug** in the codebase. Until fixed:
- Route optimization cannot distinguish fuel-efficient paths from wasteful ones
- Weather has zero effect on fuel predictions
- Calibration factors have no effect
- The entire optimization engine is effectively a shortest-distance pathfinder
EOF
)"

echo "  ✓ Issue 3: MCR cap bug"

# Issue 4: Open-Meteo integration
gh issue create \
  --title "Add Open-Meteo as weather data source (no API key needed)" \
  --label "good-first-issue,data-pipeline,phase-1" \
  --body "$(cat <<'EOF'
## Summary

We currently fall back to synthetic (fake) weather data because Copernicus requires package installation and credentials. Open-Meteo provides free, real weather data via a simple JSON API with no authentication.

## What to build

Create a new weather provider class in `src/data/` following the existing pattern:

```python
class OpenMeteoProvider:
    """Weather data from Open-Meteo (free, no API key)."""

    def get_wind(self, lat: float, lon: float, time: datetime) -> dict:
        # GET https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=wind_speed_10m,wind_direction_10m
        ...

    def get_waves(self, lat: float, lon: float, time: datetime) -> dict:
        # GET https://marine-api.open-meteo.com/v1/marine?latitude={lat}&longitude={lon}&hourly=wave_height,wave_period,wave_direction
        ...
```

## Integration point

- Look at `src/data/copernicus.py` — the `SyntheticDataProvider` class
- Create `src/data/open_meteo.py` with the same interface
- Add it to the fallback chain in `api/main.py`: Copernicus → **Open-Meteo** → Synthetic

## API documentation

- Wind/weather: https://open-meteo.com/en/docs
- Marine/waves: https://open-meteo.com/en/docs/marine-weather-api
- No API key required for non-commercial use
- Rate limit: fair use (~10,000 requests/day)

## Acceptance criteria

- [ ] Real wind speed/direction returned for any lat/lon
- [ ] Real wave height/period returned for ocean coordinates
- [ ] Graceful fallback to synthetic if Open-Meteo is unreachable
- [ ] Unit tests with mocked HTTP responses
EOF
)"

echo "  ✓ Issue 4: Open-Meteo integration"

# Issue 5: Wave heatmap
gh issue create \
  --title "Add wave height heatmap overlay on map" \
  --label "good-first-issue,visualization,phase-1" \
  --body "$(cat <<'EOF'
## Summary

The backend already serves wave height data. We need a color-coded heatmap overlay on the Leaflet map showing wave conditions.

## What exists

- `frontend/lib/api.ts:401` — `getWaveField()` calls `GET /api/weather/waves`
- Response includes a grid of wave heights with lat/lon bounds and resolution

## What to build

1. Create `frontend/components/map/WaveHeatmapLayer.tsx`
2. Render a semi-transparent Canvas overlay on the Leaflet map
3. Use bilinear interpolation between grid points for smooth rendering
4. Color ramp by wave height:
   - Green: < 1m (calm)
   - Yellow: 1-2m (moderate)
   - Orange: 2-3m (rough)
   - Red: 3-5m (very rough)
   - Dark red: > 5m (high)
5. Opacity slider to blend with base map

## Implementation approach

Use Leaflet's `L.ImageOverlay` with a dynamically generated canvas:

```tsx
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');
// For each grid cell, interpolate color from wave height
// Draw as rectangles on canvas
// Create image URL from canvas
const overlay = L.imageOverlay(canvas.toDataURL(), bounds);
```

## Acceptance criteria

- [ ] Wave heights render as colored overlay on the map
- [ ] Colors correctly represent wave height ranges
- [ ] Overlay updates when time slider changes (future issue)
- [ ] Opacity is adjustable
- [ ] No SSR errors
EOF
)"

echo "  ✓ Issue 5: Wave heatmap"

# Issue 6: Time slider
gh issue create \
  --title "Add time slider for forecast navigation" \
  --label "visualization,phase-1" \
  --body "$(cat <<'EOF'
## Summary

Add a horizontal time slider at the bottom of the map that lets users scrub through weather forecast hours. This is a core feature of any Windy-like interface.

## What to build

1. Create `frontend/components/map/TimeSlider.tsx`
2. Horizontal slider spanning the forecast range (e.g., 0-120 hours)
3. Step through in 3h or 6h increments (matching GFS/Copernicus data)
4. Display current forecast time prominently (e.g., "Wed Feb 6, 18:00 UTC")
5. Play/pause button to animate through time steps
6. When slider changes, re-fetch wind/wave data with the new time parameter

## Design reference

Similar to Windy.com's bottom timeline bar — minimal, always visible, with a play button.

## Backend support

The velocity endpoint already accepts a \`time\` parameter:
```
GET /api/weather/wind/velocity?time=2026-02-06T18:00:00
```

## Acceptance criteria

- [ ] Slider renders at bottom of map
- [ ] Moving slider updates the weather visualization
- [ ] Play button animates through time steps
- [ ] Current time is clearly displayed
- [ ] Pre-fetches adjacent time steps for smooth scrubbing
EOF
)"

echo "  ✓ Issue 6: Time slider"

# Issue 7: NOAA GFS pipeline
gh issue create \
  --title "Add NOAA GFS wind data pipeline (free, no API key)" \
  --label "data-pipeline,phase-1" \
  --body "$(cat <<'EOF'
## Summary

Connect to real global wind forecast data from NOAA's GFS model. This is the same data source that Windy.com uses. Free, updated every 6 hours, no API key needed.

## Data source

NOAA NOMADS GFS filter: https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl

Select:
- Variables: \`UGRD\` (U-wind), \`VGRD\` (V-wind)
- Level: \`10 m above ground\`
- Resolution: 0.25° or 0.5°

## What to build

1. Create \`src/data/gfs_provider.py\`
2. Download GRIB2 files from NOMADS (filter URL to get only wind at 10m)
3. Convert to the grib2json format (same as \`/api/weather/wind/velocity\` expects)
4. Cache downloaded data with 6-hour TTL
5. Add as provider in the fallback chain

## Conversion options

- **pygrib** (Python): parse GRIB2 natively, extract U/V arrays
- **cfgrib + xarray** (Python): higher-level, handles coordinates automatically
- **grib2json** (Java CLI): used by leaflet-velocity ecosystem

## File size

- 0.25° global wind at one time step: ~5 MB GRIB2
- 0.5° global: ~1.5 MB GRIB2
- 1.0° global: ~400 KB GRIB2

## Acceptance criteria

- [ ] Downloads latest GFS wind data automatically
- [ ] Converts to grib2json format consumed by leaflet-velocity
- [ ] Caches data to avoid redundant downloads
- [ ] Falls back gracefully if NOMADS is unreachable
- [ ] Unit tests with sample GRIB2 fixture
EOF
)"

echo "  ✓ Issue 7: GFS pipeline"

# Issue 8: License change
gh issue create \
  --title "Switch LICENSE from commercial to open source (MIT or Apache 2.0)" \
  --label "critical,help-wanted" \
  --body "$(cat <<'EOF'
## Summary

The current LICENSE file is a commercial proprietary license that prohibits modification, derivative works, and redistribution. This must be replaced before the project can accept community contributions.

## Current state

- \`LICENSE\` — full commercial license for "SL Mar"
- \`pyproject.toml:7\` — says \`license = "MIT"\` (contradicts LICENSE file)
- \`api/main.py:110\` — references "Commercial License"
- \`Dockerfile:47\` — label says \`licenses="Commercial"\`
- \`README.md:187\` — says "Private - SL Mar"

## What to do

1. Replace \`LICENSE\` with MIT or Apache 2.0 text
2. Update \`pyproject.toml\` license field to match
3. Update \`api/main.py\` license references (lines 11, 110-111)
4. Update \`Dockerfile\` label (line 47)
5. Update \`README.md\` (lines 187, 191)
6. Update \`frontend/README.md\` (line 222)
7. Update \`src/__init__.py\` (line 4)

## Decision needed

**MIT** — simpler, more permissive, widely used
**Apache 2.0** — includes patent grant, better for enterprise adoption

This is a decision for the project maintainer (@SL-Mar).
EOF
)"

echo "  ✓ Issue 8: License change"

# Issue 9: .gitignore hardening
gh issue create \
  --title "Harden .gitignore before public release" \
  --label "good-first-issue,critical" \
  --body "$(cat <<'EOF'
## Summary

The root \`.gitignore\` is missing entries for sensitive files. A \`git add .\` could accidentally commit secrets.

## Missing entries to add

\`\`\`gitignore
# Environment files (CRITICAL — currently missing!)
.env
.env.local
.env.*.local

# Private keys and certificates
*.pem
*.key
*.p12
*.pfx
*.cert
*.crt

# SSL directory
docker/nginx/ssl/

# Logs
logs/

# Calibration state
data/calibration.json
\`\`\`

## Context

- The \`frontend/.gitignore\` correctly excludes \`.env.local\` but the **root** \`.gitignore\` does not exclude \`.env\`
- \`.dockerignore\` excludes \`.env\` but that only affects Docker builds, not git
- No secrets have been committed in git history (verified), but this gap should be closed

## Acceptance criteria

- [ ] \`.env\` added to root \`.gitignore\`
- [ ] All patterns above added
- [ ] Verified no \`.env\` files currently tracked: \`git ls-files | grep env\`
EOF
)"

echo "  ✓ Issue 9: .gitignore hardening"

echo ""
echo "✅ All 9 issues created successfully!"
echo ""
echo "Summary:"
echo "  Phase 1 (Weather Viz): Issues 1, 4, 5, 6, 7"
echo "  Phase 2 (Physics):     Issues 2, 3"
echo "  Release Blockers:      Issues 8, 9"

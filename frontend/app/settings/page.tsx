'use client';

import Header from '@/components/Header';
import { useVoyage } from '@/components/VoyageContext';
import { Settings, Grid3X3, TrendingUp, Compass, Gauge, Zap } from 'lucide-react';

export default function SettingsPage() {
  const {
    gridResolution, setGridResolution,
    variableResolution, setVariableResolution,
    paretoEnabled, setParetoEnabled,
    variableSpeed, setVariableSpeed,
  } = useVoyage();

  return (
    <div className="min-h-screen bg-maritime-darker text-white">
      <Header />
      <div className="container mx-auto px-6 pt-20 pb-12 max-w-7xl">

        {/* Title */}
        <div className="flex items-center space-x-3 mb-8">
          <Settings className="w-8 h-8 text-primary-400" />
          <div>
            <h1 className="text-3xl font-bold maritime-gradient-text">Settings</h1>
            <p className="text-sm text-gray-400">Optimization engine configuration</p>
          </div>
        </div>

        {/* 2-column grid of cards */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">

          {/* ── A. Optimization Engine ── */}
          <section className="bg-white/5 rounded-lg p-8">
            <div className="flex items-center space-x-2 mb-4">
              <Grid3X3 className="w-5 h-5 text-primary-400" />
              <h2 className="text-xl font-semibold">Optimization Engine</h2>
            </div>

            <div className="space-y-5 mb-4">
              {/* Grid Resolution */}
              <div>
                <label className="flex items-center justify-between text-sm text-gray-300 mb-2">
                  <span>Grid Resolution</span>
                  <span className="text-primary-400 font-mono">{gridResolution.toFixed(2)}&deg;</span>
                </label>
                <input
                  type="range"
                  min="0.05"
                  max="1.0"
                  step="0.05"
                  value={gridResolution}
                  onChange={(e) => setGridResolution(parseFloat(e.target.value))}
                  className="w-full h-1.5 accent-ocean-500 cursor-pointer"
                />
                <div className="flex justify-between text-xs text-gray-600 mt-1">
                  <span>0.05&deg; (fine)</span>
                  <span>1.0&deg; (coarse)</span>
                </div>
              </div>

              {/* Variable Resolution */}
              <label className="flex items-center gap-3 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={variableResolution}
                  onChange={(e) => setVariableResolution(e.target.checked)}
                  className="accent-ocean-500 w-4 h-4"
                />
                <div>
                  <div className="text-sm text-gray-300">Variable resolution grid</div>
                  <div className="text-xs text-gray-500">0.05&deg; nearshore + 0.5&deg; open ocean</div>
                </div>
              </label>
            </div>

            <div className="text-sm text-gray-400 leading-relaxed space-y-2">
              <p>
                The grid resolution controls the cell size of the search graph. Smaller
                cells produce more accurate routes but take longer to compute. For most
                ocean voyages, 0.2&deg; (approximately 12 nm) is a good balance.
              </p>
              <p>
                The variable resolution option uses a two-tier grid: a fine 0.05&deg; grid
                within 50 nm of the coast (where precise navigation around headlands and
                islands matters) and a coarser 0.5&deg; grid in the open ocean (where small
                deviations have minimal impact). This significantly improves coastal routing
                accuracy without the computational cost of a uniformly fine grid.
              </p>
            </div>
          </section>

          {/* ── B. Variable Speed ── */}
          <section className="bg-white/5 rounded-lg p-8">
            <div className="flex items-center space-x-2 mb-4">
              <Gauge className="w-5 h-5 text-primary-400" />
              <h2 className="text-xl font-semibold">Variable Speed</h2>
            </div>

            <div className="mb-4">
              <label className="flex items-center gap-3 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={variableSpeed}
                  onChange={(e) => setVariableSpeed(e.target.checked)}
                  className="accent-ocean-500 w-4 h-4"
                />
                <div>
                  <div className="text-sm text-gray-300">Optimize speed per leg</div>
                  <div className="text-xs text-gray-500">Allow the vessel to adjust speed on each leg to minimize fuel</div>
                </div>
              </label>
            </div>

            <div className="text-sm text-gray-400 leading-relaxed space-y-3">
              <p>
                <strong className="text-gray-300">What it does:</strong> Instead of maintaining
                a constant commanded speed for the entire voyage, the calculator tests multiple
                speeds on each leg (from 50% to 100% of the set speed) and selects the one that
                minimizes a combined fuel + time score. In heavy weather legs, the vessel slows
                down significantly &mdash; saving fuel because power scales with the cube of speed.
                In calm legs, the vessel maintains or slightly increases speed.
              </p>
              <p>
                <strong className="text-gray-300">When to use it:</strong> Enable variable speed
                when you want to see how much fuel could be saved by adapting speed to weather
                conditions along the route. This is most effective on routes with mixed weather
                &mdash; a few legs with strong headwinds where slowing down saves significant fuel.
              </p>
              <p>
                <strong className="text-gray-300">How it works:</strong> Fuel consumption is
                roughly proportional to speed&sup3;. A 20% speed reduction on a heavy-weather leg
                saves approximately 50% of fuel for that leg, at the cost of taking 25% longer.
                The optimizer balances fuel savings against transit time to avoid extreme
                slow-steaming.
              </p>
            </div>
          </section>

          {/* ── C. Pareto Analysis ── */}
          <section className="bg-white/5 rounded-lg p-8">
            <div className="flex items-center space-x-2 mb-4">
              <TrendingUp className="w-5 h-5 text-primary-400" />
              <h2 className="text-xl font-semibold">Pareto Analysis</h2>
            </div>

            <div className="mb-4">
              <label className="flex items-center gap-3 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={paretoEnabled}
                  onChange={(e) => setParetoEnabled(e.target.checked)}
                  className="accent-ocean-500 w-4 h-4"
                />
                <div>
                  <div className="text-sm text-gray-300">Run Pareto analysis after optimization</div>
                  <div className="text-xs text-gray-500">Automatically explores the fuel-vs-time trade-off frontier</div>
                </div>
              </label>
            </div>

            <div className="text-sm text-gray-400 leading-relaxed space-y-3">
              <p>
                <strong className="text-gray-300">What it is:</strong> Pareto analysis
                explores the trade-off between fuel consumption and transit time. Instead
                of finding a single &quot;best&quot; route, it generates a set of routes that
                represent the best possible compromises &mdash; no route on the Pareto front
                is both cheaper <em>and</em> faster than any other.
              </p>
              <p>
                <strong className="text-gray-300">How to read the chart:</strong> Each dot
                represents one candidate route. The X-axis is fuel consumption, the Y-axis
                is transit time. The bottom-left corner is the ideal (least fuel, shortest
                time) but usually unreachable. The Pareto front is the curve of
                non-dominated solutions &mdash; moving along it, you can only reduce fuel by
                accepting more time, or vice versa. The yellow dot marks the recommended
                balanced solution.
              </p>
              <p>
                <strong className="text-gray-300">When to use it:</strong> Enable Pareto
                analysis when you want to understand whether a small increase in fuel
                consumption would yield a significantly shorter transit time, or whether
                slowing down by a few hours would save meaningful fuel. It is most useful
                for routes with strong weather gradients where the trade-off is non-obvious.
              </p>
            </div>
          </section>

          {/* ── D. How Route Optimization Works ── */}
          <section className="bg-white/5 rounded-lg p-8">
            <div className="flex items-center space-x-2 mb-4">
              <Compass className="w-5 h-5 text-primary-400" />
              <h2 className="text-xl font-semibold">How Route Optimization Works</h2>
            </div>

            <div className="text-sm text-gray-400 leading-relaxed space-y-3">
              <p>
                When you press <strong className="text-gray-300">Optimize Route</strong>,
                Windmar runs two independent optimization engines at three safety weights
                each, producing up to six candidate routes:
              </p>

              <div className="bg-white/10 rounded-lg p-4 space-y-3">
                <div>
                  <div className="text-gray-300 font-medium mb-1">A* (heuristic grid search)</div>
                  <p>
                    A fast graph-search algorithm that expands cells from origin to
                    destination, guided by a heuristic that estimates remaining cost. Good
                    for most routes and produces results quickly.
                  </p>
                </div>
                <div>
                  <div className="text-gray-300 font-medium mb-1">Dijkstra (time-expanded graph)</div>
                  <p>
                    A more sophisticated engine that uses a time-expanded graph where the
                    vessel can choose to reduce speed voluntarily in heavy weather, waiting
                    for conditions to improve. This produces better results in severe
                    weather but is slower to compute.
                  </p>
                </div>
              </div>

              <p>
                The optimizer finds alternative <strong className="text-gray-300">geographic
                paths</strong> that avoid adverse weather or exploit favourable currents. It
                does not optimize the speed profile along the original route &mdash; that is
                a separate capability planned for a future release.
              </p>

              <div className="bg-white/10 rounded-lg p-4">
                <div className="text-gray-300 font-medium mb-2">Safety Weights</div>
                <div className="space-y-1.5">
                  <div className="flex items-center gap-3">
                    <span className="text-xs font-mono text-ocean-400 w-6">0.0</span>
                    <span><strong className="text-gray-300">Fuel</strong> &mdash; pure fuel minimization, may route through heavier weather</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-xs font-mono text-ocean-400 w-6">0.5</span>
                    <span><strong className="text-gray-300">Balanced</strong> &mdash; equal weight to fuel cost and weather severity</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-xs font-mono text-ocean-400 w-6">1.0</span>
                    <span><strong className="text-gray-300">Safety</strong> &mdash; full safety-first, avoids heavy weather even at higher fuel cost</span>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* ── E. Startup Procedure — spans full width ── */}
          <section className="bg-white/5 rounded-lg p-8 lg:col-span-2">
            <div className="flex items-center space-x-2 mb-4">
              <Zap className="w-5 h-5 text-primary-400" />
              <h2 className="text-xl font-semibold">Startup Procedure</h2>
            </div>

            <div className="text-sm text-gray-400 leading-relaxed space-y-3">
              <p>
                Windmar is designed as a <strong className="text-gray-300">local-first
                application</strong>. All data processing, weather interpolation, and route
                optimization run on your machine. This means the first launch after a fresh
                install or container rebuild involves several one-time initialization steps
                that add latency before the app is fully responsive.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-white/10 rounded-lg p-4">
                  <div className="text-gray-300 text-xs font-medium mb-1">1. Shoreline geometry download</div>
                  <p>
                    The API server uses Cartopy for land-avoidance checks. On first boot,
                    Cartopy downloads Natural Earth shoreline shapefiles (~30&ndash;60 s).
                    Until this completes, route calculation requests will queue. Subsequent
                    boots use the cached files and skip this step entirely.
                  </p>
                </div>

                <div className="bg-white/10 rounded-lg p-4">
                  <div className="text-gray-300 text-xs font-medium mb-1">2. Weather database initialization</div>
                  <p>
                    The weather pipeline fetches global forecast data from NOAA GFS (wind)
                    and Copernicus CMEMS (waves, currents, ice, SST, visibility). A full
                    refresh downloads and processes ~2&ndash;5 GB of GRIB2/NetCDF data into
                    the local database. The weather panel shows a
                    &ldquo;stale&rdquo; indicator until the first successful refresh completes.
                  </p>
                </div>

                <div className="bg-white/10 rounded-lg p-4">
                  <div className="text-gray-300 text-xs font-medium mb-1">3. Map tile rendering</div>
                  <p>
                    The frontend loads vector tiles from OpenStreetMap on first paint. Tile
                    loading depends on your internet connection and the initial viewport size.
                    Route drawing is available as soon as the map canvas is interactive.
                  </p>
                </div>
              </div>

              <p>
                <strong className="text-gray-300">Typical first-boot time:</strong> 30&ndash;90
                seconds before the API is fully responsive, depending on network speed. After
                the initial boot, restarts take only a few seconds because all downloaded data
                is persisted in Docker volumes.
              </p>

              <p>
                <strong className="text-gray-300">If the app feels unresponsive:</strong> Check
                the API container logs (<code className="text-xs text-ocean-400 bg-white/5 px-1 py-0.5 rounded">docker
                logs windmar-api</code>) for download progress. The weather panel&apos;s
                &ldquo;stale&rdquo; badge clears once the startup prefetch completes. There
                is no periodic refresh &mdash; after boot, weather data updates on demand when
                you activate a layer or trigger a manual resync.
              </p>
            </div>
          </section>

        </div>
      </div>
    </div>
  );
}

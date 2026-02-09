"""
Monte Carlo voyage simulation with temporal weather slices.

Runs N voyage calculations with perturbed weather to produce
P10/P50/P90 confidence intervals for ETA, fuel, and voyage time.

Two modes:
1. **Temporal** (preferred): Divides the voyage into time slices, fetches
   time-varying weather from pre-ingested DB grids (wave forecast hours 0-120h),
   and applies temporally correlated perturbations via Cholesky decomposition
   of an exponential correlation matrix.

2. **Legacy fallback**: Single weather sample per leg midpoint with uniform
   perturbation (same random factor for all legs). Used when DB weather
   is unavailable.

Perturbation model:
- Exponential temporal correlation: cov(i,j) = exp(-|ti-tj| / tau)
  where tau = 0.3 × voyage duration (~1.5 days for a 5-day voyage)
- Wind speed: log-normal, sigma=0.35, mean-corrected to E[factor]=1
- Wave height: 70% correlated with wind + 30% independent, sigma=0.20
- Current: independent correlated process, sigma=0.15
- Directions: Gaussian angular offsets, sigma=15 deg, temporally correlated

References:
- Dickson et al. (2019), "Uncertainty in marine weather routing", arXiv:1901.03840
- Aijjou et al. (2021), "A Comprehensive Approach to Account for Weather
  Uncertainties in Ship Route Optimization", JMSE 9(12):1434
"""

import logging
import time as time_mod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.optimization.voyage import VoyageCalculator, LegWeather, VoyageResult
from src.routes.rtz_parser import Route

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation result with percentiles."""
    n_simulations: int

    # ETA percentiles (ISO strings)
    eta_p10: str
    eta_p50: str
    eta_p90: str

    # Fuel percentiles (MT)
    fuel_p10: float
    fuel_p50: float
    fuel_p90: float

    # Total time percentiles (hours)
    time_p10: float
    time_p50: float
    time_p90: float

    # Performance
    computation_time_ms: float


class MonteCarloSimulator:
    """
    Run Monte Carlo voyage simulations with temporally correlated
    perturbed weather.

    When db_weather is provided (DbWeatherProvider with multi-timestep
    wave forecast), uses time-varying base weather along the route.
    Otherwise falls back to the legacy single-snapshot approach.
    """

    # Perturbation parameters
    SIGMA_WIND = 0.35       # log-normal sigma for wind speed
    SIGMA_WAVE = 0.20       # log-normal sigma for wave height
    SIGMA_CURRENT = 0.15    # log-normal sigma for current speed
    SIGMA_DIR = 15.0        # degrees, Gaussian offset for directions
    CORR_TAU = 0.3          # correlation length as fraction of voyage duration

    def __init__(self, voyage_calculator: VoyageCalculator):
        self.voyage_calculator = voyage_calculator

    def run(
        self,
        route: Route,
        calm_speed_kts: float,
        is_laden: bool,
        departure_time: datetime,
        weather_provider: Optional[Callable] = None,
        n_simulations: int = 100,
        db_weather=None,
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation.

        Args:
            route: Voyage route with waypoints.
            calm_speed_kts: Calm-water speed assumption (knots).
            is_laden: Laden or ballast condition.
            departure_time: Voyage departure time.
            weather_provider: Callable (lat, lon, time) -> LegWeather.
            n_simulations: Number of simulation runs.
            db_weather: DbWeatherProvider for multi-timestep wave grids.

        Returns:
            MonteCarloResult with P10/P50/P90 percentiles.
        """
        t0 = time_mod.time()

        # Compute route timeline (positions along route at even time intervals)
        total_dist = sum(leg.distance_nm for leg in route.legs)
        total_time_h = total_dist / max(calm_speed_kts, 0.1)
        n_slices = min(100, max(20, int(total_time_h / 1.2)))  # ~1 slice per 1.2h
        slices = self._compute_route_timeline(
            route, calm_speed_kts, departure_time, n_slices
        )

        # Pre-fetch base weather for all time slices
        base_weather = self._prefetch_base_weather(
            slices, route, weather_provider, db_weather
        )

        prefetch_ms = (time_mod.time() - t0) * 1000
        logger.info(
            f"MC: Pre-fetched {len(base_weather)} time slices in {prefetch_ms:.0f}ms "
            f"(voyage: {total_time_h:.0f}h, {total_dist:.0f}nm)"
        )

        # Build correlation matrix (Cholesky factor)
        L = self._build_correlation_cholesky(n_slices, self.CORR_TAU)

        # Run N simulations
        rng = np.random.default_rng()
        total_times: List[float] = []
        total_fuels: List[float] = []
        arrival_times: List[datetime] = []

        for sim_idx in range(n_simulations):
            # Generate temporally correlated perturbation factors
            wind_factors, wave_factors, current_factors, dir_offsets = (
                self._generate_correlated_perturbations(rng, L)
            )

            # Create perturbed weather provider
            perturbed = self._make_temporal_perturbed_provider(
                base_weather, slices, route,
                wind_factors, wave_factors, current_factors, dir_offsets,
            )

            try:
                result = self.voyage_calculator.calculate_voyage(
                    route=route,
                    calm_speed_kts=calm_speed_kts,
                    is_laden=is_laden,
                    departure_time=departure_time,
                    weather_provider=perturbed,
                )
                if 0 < result.total_time_hours < 1e6:
                    total_times.append(result.total_time_hours)
                    total_fuels.append(result.total_fuel_mt)
                    arrival_times.append(result.arrival_time)
            except Exception as e:
                logger.warning(f"MC simulation {sim_idx} failed: {e}")
                continue

        elapsed_ms = (time_mod.time() - t0) * 1000

        if len(total_times) < 3:
            raise ValueError(
                f"Only {len(total_times)} simulations succeeded out of "
                f"{n_simulations}. Cannot compute meaningful percentiles."
            )

        # Compute percentiles
        time_arr = np.array(total_times)
        fuel_arr = np.array(total_fuels)

        time_p10, time_p50, time_p90 = np.percentile(time_arr, [10, 50, 90])
        fuel_p10, fuel_p50, fuel_p90 = np.percentile(fuel_arr, [10, 50, 90])

        arrival_ts = np.array([dt.timestamp() for dt in arrival_times])
        eta_p10_ts, eta_p50_ts, eta_p90_ts = np.percentile(arrival_ts, [10, 50, 90])

        logger.info(
            f"MC: {len(total_times)}/{n_simulations} succeeded in {elapsed_ms:.0f}ms. "
            f"Fuel P10/P50/P90 = {fuel_p10:.1f}/{fuel_p50:.1f}/{fuel_p90:.1f} MT"
        )

        return MonteCarloResult(
            n_simulations=len(total_times),
            eta_p10=datetime.utcfromtimestamp(eta_p10_ts).isoformat() + "Z",
            eta_p50=datetime.utcfromtimestamp(eta_p50_ts).isoformat() + "Z",
            eta_p90=datetime.utcfromtimestamp(eta_p90_ts).isoformat() + "Z",
            fuel_p10=round(float(fuel_p10), 2),
            fuel_p50=round(float(fuel_p50), 2),
            fuel_p90=round(float(fuel_p90), 2),
            time_p10=round(float(time_p10), 2),
            time_p50=round(float(time_p50), 2),
            time_p90=round(float(time_p90), 2),
            computation_time_ms=round(elapsed_ms, 1),
        )

    # ------------------------------------------------------------------
    # Route timeline
    # ------------------------------------------------------------------

    def _compute_route_timeline(
        self,
        route: Route,
        speed_kts: float,
        departure: datetime,
        n_slices: int,
    ) -> List[Tuple[datetime, float, float]]:
        """Compute (time, lat, lon) for n_slices evenly spaced along the route.

        Uses linear interpolation along rhumb-line legs.
        """
        # Build cumulative distance table
        waypoints = []  # (cum_dist, lat, lon)
        cum = 0.0
        for leg in route.legs:
            waypoints.append((cum, leg.from_wp.lat, leg.from_wp.lon))
            cum += leg.distance_nm
        waypoints.append((cum, route.legs[-1].to_wp.lat, route.legs[-1].to_wp.lon))

        total_dist = cum
        total_time_h = total_dist / max(speed_kts, 0.1)

        slices = []
        for i in range(n_slices):
            frac = i / (n_slices - 1) if n_slices > 1 else 0.0
            dist = frac * total_dist
            t = departure + timedelta(hours=frac * total_time_h)
            lat, lon = self._interpolate_position(waypoints, dist)
            slices.append((t, lat, lon))

        return slices

    @staticmethod
    def _interpolate_position(
        waypoints: List[Tuple[float, float, float]],
        target_dist: float,
    ) -> Tuple[float, float]:
        """Linear interpolation of lat/lon at a given cumulative distance."""
        for i in range(len(waypoints) - 1):
            d0, lat0, lon0 = waypoints[i]
            d1, lat1, lon1 = waypoints[i + 1]
            if target_dist <= d1 or i == len(waypoints) - 2:
                seg_len = d1 - d0
                if seg_len <= 0:
                    return lat0, lon0
                frac = max(0.0, min(1.0, (target_dist - d0) / seg_len))
                return lat0 + frac * (lat1 - lat0), lon0 + frac * (lon1 - lon0)
        return waypoints[-1][1], waypoints[-1][2]

    # ------------------------------------------------------------------
    # Weather pre-fetch
    # ------------------------------------------------------------------

    def _prefetch_base_weather(
        self,
        slices: List[Tuple[datetime, float, float]],
        route: Route,
        weather_provider: Optional[Callable],
        db_weather,
    ) -> List[LegWeather]:
        """Pre-fetch base weather for all time slices.

        Uses DB wave grids (multi-timestep) + weather_provider for wind.
        Falls back to weather_provider-only if DB is unavailable.
        """
        if db_weather is not None:
            db_result = self._prefetch_from_db(slices, route, db_weather, weather_provider)
            if db_result is not None:
                return db_result

        # Fallback: use weather_provider at each slice
        if weather_provider:
            return self._prefetch_from_provider(slices, weather_provider)

        return [LegWeather() for _ in slices]

    def _prefetch_from_db(
        self,
        slices: List[Tuple[datetime, float, float]],
        route: Route,
        db_weather,
        wind_provider: Optional[Callable],
    ) -> Optional[List[LegWeather]]:
        """Pre-fetch from DB using shared TemporalGridWeatherProvider.

        Builds a TemporalGridWeatherProvider via RouteWeatherAssessment,
        then queries it at each slice position. Falls back to the legacy
        manual interpolation if the temporal provider is unavailable.

        Returns list of LegWeather per slice, or None if DB has no data.
        """
        try:
            from src.optimization.weather_assessment import RouteWeatherAssessment

            # Compute bounding box and time range from slices
            all_lats = [s[1] for s in slices]
            all_lons = [s[2] for s in slices]
            departure = slices[0][0]
            total_time_h = max(
                1.0,
                (slices[-1][0] - slices[0][0]).total_seconds() / 3600.0,
            )
            est_speed = route.total_distance_nm / total_time_h

            # Use RouteWeatherAssessment for provisioning
            origin_lat = all_lats[0]
            origin_lon = all_lons[0]
            dest_lat = all_lats[-1]
            dest_lon = all_lons[-1]

            assessor = RouteWeatherAssessment(db_weather=db_weather)
            assessment = assessor.assess(
                origin=(origin_lat, origin_lon),
                destination=(dest_lat, dest_lon),
                departure_time=departure,
                calm_speed_kts=est_speed,
            )
            temporal_wx = assessor.provision(assessment)

            if temporal_wx is None:
                logger.info("MC: Temporal provider unavailable, falling back to weather_provider")
                return None

            # Query temporal provider at each slice
            result = []
            for t, lat, lon in slices:
                result.append(temporal_wx.get_weather(lat, lon, t))

            logger.info(
                f"MC: Temporal pre-fetch complete — {len(result)} slices "
                f"via TemporalGridWeatherProvider"
            )
            return result

        except Exception as e:
            logger.warning(f"MC: Temporal pre-fetch failed ({e}), falling back")
            return None

    def _prefetch_from_provider(
        self,
        slices: List[Tuple[datetime, float, float]],
        weather_provider: Callable,
    ) -> List[LegWeather]:
        """Fallback: fetch weather at each slice position via the weather_provider."""
        result = []
        for t, lat, lon in slices:
            try:
                result.append(weather_provider(lat, lon, t))
            except Exception as e:
                logger.warning(f"MC weather fetch failed at ({lat:.1f}, {lon:.1f}): {e}")
                result.append(LegWeather())
        return result

    # ------------------------------------------------------------------
    # Correlation model
    # ------------------------------------------------------------------

    @staticmethod
    def _build_correlation_cholesky(n: int, tau: float) -> np.ndarray:
        """Build Cholesky factor of an exponential correlation matrix.

        cov(i, j) = exp(-|t_i - t_j| / tau)
        where t is normalized to [0, 1] over the voyage duration.

        Args:
            n: Number of time slices.
            tau: Correlation length as fraction of voyage duration.

        Returns:
            Lower-triangular Cholesky factor L such that L @ L^T = cov.
        """
        t = np.linspace(0.0, 1.0, n)
        cov = np.exp(-np.abs(t[:, None] - t[None, :]) / max(tau, 0.01))
        # Small diagonal regularization for numerical stability
        cov += 1e-8 * np.eye(n)
        return np.linalg.cholesky(cov)

    def _generate_correlated_perturbations(
        self,
        rng: np.random.Generator,
        L: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate temporally correlated perturbation factors.

        Returns:
            (wind_factors, wave_factors, current_factors, dir_offsets)
            Each is an array of length n_slices.
        """
        n = L.shape[0]

        # Correlated standard normals
        z_wind = L @ rng.standard_normal(n)
        z_wave_indep = L @ rng.standard_normal(n)
        z_current = L @ rng.standard_normal(n)
        z_dir = L @ rng.standard_normal(n)

        # Wind speed: log-normal, E[factor] = 1.0
        wind_factors = np.exp(
            self.SIGMA_WIND * z_wind - 0.5 * self.SIGMA_WIND ** 2
        )

        # Wave height: 70% correlated with wind, 30% independent
        z_wave = 0.7 * z_wind + np.sqrt(1 - 0.7 ** 2) * z_wave_indep
        wave_factors = np.exp(
            self.SIGMA_WAVE * z_wave - 0.5 * self.SIGMA_WAVE ** 2
        )

        # Current: independent correlated process
        current_factors = np.exp(
            self.SIGMA_CURRENT * z_current - 0.5 * self.SIGMA_CURRENT ** 2
        )

        # Direction offsets: Gaussian, temporally correlated
        dir_offsets = self.SIGMA_DIR * z_dir

        return wind_factors, wave_factors, current_factors, dir_offsets

    # ------------------------------------------------------------------
    # Perturbed weather provider
    # ------------------------------------------------------------------

    def _make_temporal_perturbed_provider(
        self,
        base_weather: List[LegWeather],
        slices: List[Tuple[datetime, float, float]],
        route: Route,
        wind_factors: np.ndarray,
        wave_factors: np.ndarray,
        current_factors: np.ndarray,
        dir_offsets: np.ndarray,
    ) -> Callable:
        """Create a weather provider with temporally varying perturbations.

        The VoyageCalculator queries weather at leg midpoints. This provider
        matches each query to the nearest time slice and applies that slice's
        perturbation factors to the pre-fetched base weather.
        """
        # Pre-compute slice timestamps for fast lookup
        slice_times = np.array([s[0].timestamp() for s in slices])
        slice_lats = np.array([s[1] for s in slices])
        slice_lons = np.array([s[2] for s in slices])

        def perturbed(lat: float, lon: float, t: datetime) -> LegWeather:
            # Find nearest slice by time (primary) and distance (secondary)
            t_ts = t.timestamp()
            time_dist = np.abs(slice_times - t_ts)
            # Among the 5 closest in time, pick the spatially nearest
            k = min(5, len(slices))
            candidates = np.argpartition(time_dist, k)[:k]
            spatial_dist = (slice_lats[candidates] - lat) ** 2 + (slice_lons[candidates] - lon) ** 2
            best = candidates[np.argmin(spatial_dist)]

            base = base_weather[best]
            wf = float(wind_factors[best])
            vf = float(wave_factors[best])
            cf = float(current_factors[best])
            do = float(dir_offsets[best])

            return LegWeather(
                wind_speed_ms=base.wind_speed_ms * wf,
                wind_dir_deg=(base.wind_dir_deg + do) % 360,
                sig_wave_height_m=base.sig_wave_height_m * vf,
                wave_period_s=base.wave_period_s,
                wave_dir_deg=(base.wave_dir_deg + do) % 360,
                current_speed_ms=base.current_speed_ms * cf,
                current_dir_deg=base.current_dir_deg,
                windwave_height_m=base.windwave_height_m * vf,
                windwave_period_s=base.windwave_period_s,
                windwave_dir_deg=(base.windwave_dir_deg + do) % 360,
                swell_height_m=base.swell_height_m * vf,
                swell_period_s=base.swell_period_s,
                swell_dir_deg=base.swell_dir_deg,
                has_decomposition=base.has_decomposition,
            )

        return perturbed

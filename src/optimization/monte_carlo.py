"""
Monte Carlo voyage simulation.

Runs N voyage calculations with perturbed weather to produce
P10/P50/P90 confidence intervals for ETA, fuel, and voyage time.

Perturbation strategy:
- Wind speed: log-normal multiplicative factor, sigma ~0.35
- Wave height: correlated with wind perturbation (Hs ~ wind^1.5)
- Current: small independent perturbation (sigma ~0.15)
- Wind/wave directions: small angular perturbation (sigma ~15 deg)

Performance: Base weather is fetched once for each leg midpoint,
then all N simulations use the cached values with perturbation.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
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
    Run Monte Carlo voyage simulations with perturbed weather.

    For each simulation run:
    1. Generate perturbation factors from climatological variability
    2. Create a perturbed weather provider that scales cached base weather
    3. Run VoyageCalculator.calculate_voyage()
    4. Collect ETA, fuel, and time

    Finally compute P10/P50/P90 percentiles.
    """

    def __init__(
        self,
        voyage_calculator: VoyageCalculator,
    ):
        self.voyage_calculator = voyage_calculator

    def run(
        self,
        route: Route,
        calm_speed_kts: float,
        is_laden: bool,
        departure_time: datetime,
        weather_provider: Optional[Callable] = None,
        n_simulations: int = 100,
    ) -> MonteCarloResult:
        start_time = time.time()

        # Phase 1: Pre-fetch base weather for all leg midpoints (single pass)
        base_weather: Dict[int, LegWeather] = {}
        if weather_provider:
            current_time = departure_time
            for i, leg in enumerate(route.legs):
                mid_lat = (leg.from_wp.lat + leg.to_wp.lat) / 2
                mid_lon = (leg.from_wp.lon + leg.to_wp.lon) / 2
                leg_mid_time = current_time + timedelta(
                    hours=leg.distance_nm / calm_speed_kts / 2
                )
                try:
                    base_weather[i] = weather_provider(mid_lat, mid_lon, leg_mid_time)
                except Exception as e:
                    logger.warning(f"MC base weather fetch failed for leg {i}: {e}")
                    base_weather[i] = LegWeather()
                # Advance time estimate
                current_time += timedelta(hours=leg.distance_nm / calm_speed_kts)

        logger.info(
            f"MC: Pre-fetched base weather for {len(base_weather)} legs "
            f"in {(time.time() - start_time) * 1000:.0f}ms"
        )

        # Phase 2: Run N simulations with perturbations on cached weather
        rng = np.random.default_rng()

        total_times: List[float] = []
        total_fuels: List[float] = []
        arrival_times: List[datetime] = []

        for i in range(n_simulations):
            wind_factor = rng.lognormal(mean=0.0, sigma=0.35)
            wave_corr = wind_factor ** 1.5
            wave_factor = wave_corr * rng.lognormal(mean=0.0, sigma=0.15)
            current_factor = rng.lognormal(mean=0.0, sigma=0.15)
            dir_offset = rng.normal(0.0, 15.0)

            if base_weather:
                perturbed = self._make_cached_perturbed_provider(
                    base_weather,
                    route,
                    wind_factor,
                    wave_factor,
                    current_factor,
                    dir_offset,
                )
            else:
                perturbed = None

            try:
                result = self.voyage_calculator.calculate_voyage(
                    route=route,
                    calm_speed_kts=calm_speed_kts,
                    is_laden=is_laden,
                    departure_time=departure_time,
                    weather_provider=perturbed,
                )

                if result.total_time_hours > 0 and result.total_time_hours < 1e6:
                    total_times.append(result.total_time_hours)
                    total_fuels.append(result.total_fuel_mt)
                    arrival_times.append(result.arrival_time)
            except Exception as e:
                logger.warning(f"MC simulation {i} failed: {e}")
                continue

        elapsed_ms = (time.time() - start_time) * 1000

        if len(total_times) < 3:
            raise ValueError(
                f"Only {len(total_times)} simulations succeeded out of {n_simulations}. "
                "Cannot compute meaningful percentiles."
            )

        # Compute percentiles
        time_arr = np.array(total_times)
        fuel_arr = np.array(total_fuels)

        time_p10, time_p50, time_p90 = np.percentile(time_arr, [10, 50, 90])
        fuel_p10, fuel_p50, fuel_p90 = np.percentile(fuel_arr, [10, 50, 90])

        arrival_timestamps = np.array([dt.timestamp() for dt in arrival_times])
        eta_p10_ts, eta_p50_ts, eta_p90_ts = np.percentile(arrival_timestamps, [10, 50, 90])

        logger.info(
            f"MC: {len(total_times)}/{n_simulations} succeeded in {elapsed_ms:.0f}ms. "
            f"Fuel P10/P50/P90 = {fuel_p10:.1f}/{fuel_p50:.1f}/{fuel_p90:.1f} MT"
        )

        return MonteCarloResult(
            n_simulations=len(total_times),
            eta_p10=datetime.utcfromtimestamp(eta_p10_ts).isoformat() + 'Z',
            eta_p50=datetime.utcfromtimestamp(eta_p50_ts).isoformat() + 'Z',
            eta_p90=datetime.utcfromtimestamp(eta_p90_ts).isoformat() + 'Z',
            fuel_p10=round(float(fuel_p10), 2),
            fuel_p50=round(float(fuel_p50), 2),
            fuel_p90=round(float(fuel_p90), 2),
            time_p10=round(float(time_p10), 2),
            time_p50=round(float(time_p50), 2),
            time_p90=round(float(time_p90), 2),
            computation_time_ms=round(elapsed_ms, 1),
        )

    def _make_cached_perturbed_provider(
        self,
        base_weather: Dict[int, LegWeather],
        route: Route,
        wind_factor: float,
        wave_factor: float,
        current_factor: float,
        dir_offset: float,
    ) -> Callable:
        """Create a weather provider that perturbs cached base weather.

        The VoyageCalculator queries weather at leg midpoints. We match
        the query point to the nearest leg index and apply perturbation
        to the cached base weather for that leg.
        """
        # Build lookup: list of (mid_lat, mid_lon) for each leg
        leg_midpoints = []
        for leg in route.legs:
            mid_lat = (leg.from_wp.lat + leg.to_wp.lat) / 2
            mid_lon = (leg.from_wp.lon + leg.to_wp.lon) / 2
            leg_midpoints.append((mid_lat, mid_lon))

        def perturbed(lat: float, lon: float, t: datetime) -> LegWeather:
            # Find nearest leg by midpoint distance
            best_idx = 0
            best_dist = float('inf')
            for idx, (mlat, mlon) in enumerate(leg_midpoints):
                d = (lat - mlat) ** 2 + (lon - mlon) ** 2
                if d < best_dist:
                    best_dist = d
                    best_idx = idx

            base = base_weather.get(best_idx, LegWeather())

            return LegWeather(
                wind_speed_ms=base.wind_speed_ms * wind_factor,
                wind_dir_deg=(base.wind_dir_deg + dir_offset) % 360,
                sig_wave_height_m=base.sig_wave_height_m * wave_factor,
                wave_period_s=base.wave_period_s,
                wave_dir_deg=(base.wave_dir_deg + dir_offset) % 360,
                current_speed_ms=base.current_speed_ms * current_factor,
                current_dir_deg=base.current_dir_deg,
                windwave_height_m=base.windwave_height_m * wave_factor,
                windwave_period_s=base.windwave_period_s,
                windwave_dir_deg=(base.windwave_dir_deg + dir_offset) % 360,
                swell_height_m=base.swell_height_m * wave_factor,
                swell_period_s=base.swell_period_s,
                swell_dir_deg=base.swell_dir_deg,
                has_decomposition=base.has_decomposition,
            )

        return perturbed

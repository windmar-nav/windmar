"""
Database-backed weather provider — reads pre-ingested compressed grids
from PostgreSQL and returns GridWeatherProvider instances for fast
bilinear interpolation.

Drop-in replacement for the live download → GridWeatherProvider flow.
"""

import logging
import zlib
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import psycopg2
import psycopg2.extras

from src.data.copernicus import WeatherData
from src.optimization.grid_weather_provider import GridWeatherProvider

logger = logging.getLogger(__name__)


class DbWeatherProvider:
    """Weather provider that reads pre-ingested grids from PostgreSQL."""

    def __init__(self, db_url: str):
        self.db_url = db_url

    def _get_conn(self):
        return psycopg2.connect(self.db_url)

    def get_wind_from_db(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        time: Optional[datetime] = None,
    ) -> tuple:
        """Load wind data from DB, cropped to bbox. Returns (data, ingested_at) or (None, None)."""
        return self._load_vector_data("gfs", "wind_u", "wind_v", lat_min, lat_max, lon_min, lon_max, time)

    def get_wave_from_db(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        time: Optional[datetime] = None,
    ) -> tuple:
        """Load wave data from DB, cropped to bbox. Returns (data, ingested_at) or (None, None).

        If time is given and multi-timestep wave forecast data exists,
        selects the closest available forecast hour.
        """
        run_id, ingested_at = self._find_latest_run("cmems_wave")
        if run_id is None:
            return None, None

        forecast_hour = 0
        if time is not None:
            forecast_hour = self._best_forecast_hour(run_id, time)

        conn = self._get_conn()
        try:
            hs = self._load_grid(conn, run_id, forecast_hour, "wave_hs")
            tp = self._load_grid(conn, run_id, forecast_hour, "wave_tp")
            wd = self._load_grid(conn, run_id, forecast_hour, "wave_dir")

            if hs is None:
                return None, None

            lats, lons, hs_data = hs
            lats_c, lons_c, hs_crop = self._crop_grid(lats, lons, hs_data, lat_min, lat_max, lon_min, lon_max)

            tp_crop = None
            if tp is not None:
                _, _, tp_crop = self._crop_grid(tp[0], tp[1], tp[2], lat_min, lat_max, lon_min, lon_max)

            wd_crop = None
            if wd is not None:
                _, _, wd_crop = self._crop_grid(wd[0], wd[1], wd[2], lat_min, lat_max, lon_min, lon_max)

            return WeatherData(
                parameter="wave_height",
                time=datetime.now(timezone.utc),
                lats=lats_c,
                lons=lons_c,
                values=hs_crop,
                unit="m",
                wave_period=tp_crop,
                wave_direction=wd_crop,
            ), ingested_at
        except Exception as e:
            logger.error(f"Failed to load wave data from DB: {e}")
            return None, None
        finally:
            conn.close()

    def get_current_from_db(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> tuple:
        """Load current data from DB, cropped to bbox. Returns (data, ingested_at) or (None, None)."""
        return self._load_vector_data("cmems_current", "current_u", "current_v", lat_min, lat_max, lon_min, lon_max)

    def get_ice_from_db(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        time: Optional[datetime] = None,
    ) -> tuple:
        """Load ice concentration from DB, cropped to bbox. Returns (data, ingested_at) or (None, None).

        If time is given and multi-timestep ice forecast data exists,
        selects the closest available forecast hour.
        """
        run_id, ingested_at = self._find_latest_run("cmems_ice")
        if run_id is None:
            return None, None

        forecast_hour = 0
        if time is not None:
            forecast_hour = self._best_forecast_hour(run_id, time)

        conn = self._get_conn()
        try:
            grid = self._load_grid(conn, run_id, forecast_hour, "ice_siconc")
            if grid is None:
                return None, None

            lats, lons, siconc = grid
            lats_c, lons_c, siconc_crop = self._crop_grid(
                lats, lons, siconc, lat_min, lat_max, lon_min, lon_max
            )

            return WeatherData(
                parameter="ice_concentration",
                time=datetime.now(timezone.utc),
                lats=lats_c,
                lons=lons_c,
                values=siconc_crop,
                unit="fraction",
                ice_concentration=siconc_crop,
            ), ingested_at
        except Exception as e:
            logger.error(f"Failed to load ice data from DB: {e}")
            return None, None
        finally:
            conn.close()

    def _load_vector_data(
        self,
        source: str,
        u_param: str,
        v_param: str,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        time: Optional[datetime] = None,
    ) -> tuple:
        """Load U/V component data from DB and return as (WeatherData, ingested_at).

        Returns (None, None) if data is unavailable.
        """
        run_id, ingested_at = self._find_latest_run(source)
        if run_id is None:
            return None, None

        # Pick best forecast hour for wind
        forecast_hour = 0
        if time is not None and source == "gfs":
            forecast_hour = self._best_forecast_hour(run_id, time)

        conn = self._get_conn()
        try:
            u_grid = self._load_grid(conn, run_id, forecast_hour, u_param)
            v_grid = self._load_grid(conn, run_id, forecast_hour, v_param)

            if u_grid is None or v_grid is None:
                return None, None

            lats, lons, u_data = u_grid
            _, _, v_data = v_grid

            lats_c, lons_c, u_crop = self._crop_grid(lats, lons, u_data, lat_min, lat_max, lon_min, lon_max)
            _, _, v_crop = self._crop_grid(lats, lons, v_data, lat_min, lat_max, lon_min, lon_max)

            # Compute wind speed as values for WeatherData compatibility
            speed = np.sqrt(u_crop ** 2 + v_crop ** 2)

            param_name = "wind" if "wind" in u_param else "current"
            return WeatherData(
                parameter=param_name,
                time=datetime.now(timezone.utc),
                lats=lats_c,
                lons=lons_c,
                values=speed,
                unit="m/s",
                u_component=u_crop,
                v_component=v_crop,
            ), ingested_at
        except Exception as e:
            logger.error(f"Failed to load {source} data from DB: {e}")
            return None, None
        finally:
            conn.close()

    def _find_latest_run(self, source: str) -> tuple:
        """Find the best complete forecast run ID for a source.

        Prefers runs with the most forecast hours (multi-timestep over snapshots).
        Among runs with equal hour count, picks the most recent.

        Returns (run_id, ingested_at) or (None, None).
        """
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """SELECT id, ingested_at FROM weather_forecast_runs
                   WHERE source = %s AND status = 'complete'
                   ORDER BY array_length(forecast_hours, 1) DESC NULLS LAST,
                            ingested_at DESC
                   LIMIT 1""",
                (source,),
            )
            row = cur.fetchone()
            if row:
                ingested_at = row[1]
                if ingested_at is not None and ingested_at.tzinfo is None:
                    ingested_at = ingested_at.replace(tzinfo=timezone.utc)
                return row[0], ingested_at
            return None, None
        except Exception as e:
            logger.error(f"Failed to find latest run for {source}: {e}")
            return None, None
        finally:
            conn.close()

    def _best_forecast_hour(self, run_id: int, target_time: datetime) -> int:
        """Find the closest available forecast hour to target_time."""
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """SELECT r.run_time, array_agg(DISTINCT g.forecast_hour ORDER BY g.forecast_hour)
                   FROM weather_forecast_runs r
                   JOIN weather_grid_data g ON g.run_id = r.id
                   WHERE r.id = %s
                   GROUP BY r.run_time""",
                (run_id,),
            )
            row = cur.fetchone()
            if row is None:
                return 0

            run_time, hours = row
            if not hours:
                return 0

            # Compute hours offset from run_time
            if target_time.tzinfo is None:
                target_time = target_time.replace(tzinfo=timezone.utc)
            if run_time.tzinfo is None:
                run_time = run_time.replace(tzinfo=timezone.utc)

            delta_hours = (target_time - run_time).total_seconds() / 3600.0
            # Find closest available hour
            best = min(hours, key=lambda h: abs(h - delta_hours))
            return best
        except Exception as e:
            logger.error(f"Failed to find best forecast hour: {e}")
            return 0
        finally:
            conn.close()

    def _load_grid(self, conn, run_id: int, forecast_hour: int, parameter: str):
        """Load and decompress a single grid from DB. Returns (lats, lons, data) or None."""
        cur = conn.cursor()
        cur.execute(
            """SELECT lats, lons, data, shape_rows, shape_cols
               FROM weather_grid_data
               WHERE run_id = %s AND forecast_hour = %s AND parameter = %s""",
            (run_id, forecast_hour, parameter),
        )
        row = cur.fetchone()
        if row is None:
            return None

        lats_blob, lons_blob, data_blob, rows, cols = row
        lats = self._decompress_1d(lats_blob)
        lons = self._decompress_1d(lons_blob)
        data = self._decompress_2d(data_blob, rows, cols)
        return lats, lons, data

    @staticmethod
    def _decompress_1d(blob: bytes) -> np.ndarray:
        """Decompress zlib blob to 1D float32 numpy array."""
        return np.frombuffer(zlib.decompress(blob), dtype=np.float32)

    @staticmethod
    def _decompress_2d(blob: bytes, rows: int, cols: int) -> np.ndarray:
        """Decompress zlib blob to 2D float32 numpy array."""
        return np.frombuffer(zlib.decompress(blob), dtype=np.float32).reshape(rows, cols)

    @staticmethod
    def _crop_grid(lats, lons, data, lat_min, lat_max, lon_min, lon_max):
        """Crop a global grid to a requested bounding box."""
        lats = np.asarray(lats, dtype=np.float64)
        lons = np.asarray(lons, dtype=np.float64)
        data = np.asarray(data, dtype=np.float64)

        # Handle ascending/descending latitude
        if len(lats) > 1 and lats[0] > lats[-1]:
            # Descending — flip to ascending for consistent indexing
            lats = lats[::-1]
            data = data[::-1]

        lat_mask = (lats >= lat_min) & (lats <= lat_max)
        lon_mask = (lons >= lon_min) & (lons <= lon_max)

        if not lat_mask.any() or not lon_mask.any():
            # Bbox outside grid — return full grid as fallback
            return lats, lons, data

        lats_c = lats[lat_mask]
        lons_c = lons[lon_mask]
        data_c = data[np.ix_(lat_mask, lon_mask)]
        return lats_c, lons_c, data_c

    def get_wave_grids_for_timeline(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        times: list,
    ) -> dict:
        """Load wave grids for all unique forecast hours needed by the given times.

        Efficiently loads each grid once from DB, crops to bbox, and returns
        a dict mapping forecast_hour → (lats, lons, hs_2d, tp_2d, dir_2d).

        Used by Monte Carlo to pre-fetch multi-timestep wave data in one pass.
        """
        run_id, _ = self._find_latest_run("cmems_wave")
        if run_id is None:
            return {}

        conn = self._get_conn()
        try:
            # Get run_time and available forecast hours
            cur = conn.cursor()
            cur.execute(
                """SELECT r.run_time, array_agg(DISTINCT g.forecast_hour ORDER BY g.forecast_hour)
                   FROM weather_forecast_runs r
                   JOIN weather_grid_data g ON g.run_id = r.id
                   WHERE r.id = %s AND g.parameter = 'wave_hs'
                   GROUP BY r.run_time""",
                (run_id,),
            )
            row = cur.fetchone()
            if row is None:
                return {}

            run_time, available_hours = row
            if not available_hours:
                return {}

            if run_time is not None and run_time.tzinfo is None:
                run_time = run_time.replace(tzinfo=timezone.utc)

            # Determine which forecast hours are needed
            needed_hours = set()
            for t in times:
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                delta_h = (t - run_time).total_seconds() / 3600.0
                best = min(available_hours, key=lambda h: abs(h - delta_h))
                needed_hours.add(best)

            # Load each needed forecast hour's grids
            grids = {}
            for fh in sorted(needed_hours):
                hs = self._load_grid(conn, run_id, fh, "wave_hs")
                if hs is None:
                    continue
                tp = self._load_grid(conn, run_id, fh, "wave_tp")
                wd = self._load_grid(conn, run_id, fh, "wave_dir")

                lats, lons, hs_data = hs
                lats_c, lons_c, hs_crop = self._crop_grid(
                    lats, lons, hs_data, lat_min, lat_max, lon_min, lon_max
                )

                tp_crop = None
                if tp is not None:
                    _, _, tp_crop = self._crop_grid(
                        tp[0], tp[1], tp[2], lat_min, lat_max, lon_min, lon_max
                    )

                wd_crop = None
                if wd is not None:
                    _, _, wd_crop = self._crop_grid(
                        wd[0], wd[1], wd[2], lat_min, lat_max, lon_min, lon_max
                    )

                grids[fh] = (lats_c, lons_c, hs_crop, tp_crop, wd_crop)

            logger.info(
                f"Loaded wave grids for {len(grids)} forecast hours "
                f"(available: {len(available_hours)}, needed: {len(needed_hours)})"
            )
            return grids

        except Exception as e:
            logger.error(f"Failed to load wave timeline grids: {e}")
            return {}
        finally:
            conn.close()

    def get_sst_from_db(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        time: Optional[datetime] = None,
    ) -> tuple:
        """Load SST from DB, cropped to bbox. Returns (data, ingested_at) or (None, None)."""
        run_id, ingested_at = self._find_latest_run("cmems_sst")
        if run_id is None:
            return None, None

        forecast_hour = 0
        if time is not None:
            forecast_hour = self._best_forecast_hour(run_id, time)

        conn = self._get_conn()
        try:
            grid = self._load_grid(conn, run_id, forecast_hour, "sst")
            if grid is None:
                return None, None

            lats, lons, sst_vals = grid
            lats_c, lons_c, sst_crop = self._crop_grid(
                lats, lons, sst_vals, lat_min, lat_max, lon_min, lon_max
            )

            return WeatherData(
                parameter="sst",
                time=datetime.now(timezone.utc),
                lats=lats_c,
                lons=lons_c,
                values=sst_crop,
                unit="°C",
            ), ingested_at
        except Exception as e:
            logger.error(f"Failed to load SST data from DB: {e}")
            return None, None
        finally:
            conn.close()

    def get_visibility_from_db(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        time: Optional[datetime] = None,
    ) -> tuple:
        """Load visibility from DB, cropped to bbox. Returns (data, ingested_at) or (None, None)."""
        run_id, ingested_at = self._find_latest_run("gfs_visibility")
        if run_id is None:
            return None, None

        forecast_hour = 0
        if time is not None:
            forecast_hour = self._best_forecast_hour(run_id, time)

        conn = self._get_conn()
        try:
            grid = self._load_grid(conn, run_id, forecast_hour, "visibility")
            if grid is None:
                return None, None

            lats, lons, vis_vals = grid
            lats_c, lons_c, vis_crop = self._crop_grid(
                lats, lons, vis_vals, lat_min, lat_max, lon_min, lon_max
            )

            return WeatherData(
                parameter="visibility",
                time=datetime.now(timezone.utc),
                lats=lats_c,
                lons=lons_c,
                values=vis_crop,
                unit="km",
            ), ingested_at
        except Exception as e:
            logger.error(f"Failed to load visibility data from DB: {e}")
            return None, None
        finally:
            conn.close()

    def get_available_wave_hours(self) -> tuple:
        """Return (run_time, [forecast_hours]) for the latest complete wave run."""
        run_id, _ = self._find_latest_run("cmems_wave")
        if run_id is None:
            return None, []

        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """SELECT r.run_time, array_agg(DISTINCT g.forecast_hour ORDER BY g.forecast_hour)
                   FROM weather_forecast_runs r
                   JOIN weather_grid_data g ON g.run_id = r.id
                   WHERE r.id = %s AND g.parameter = 'wave_hs'
                   GROUP BY r.run_time""",
                (run_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None, []
            return row[0], row[1] or []
        except Exception:
            return None, []
        finally:
            conn.close()

    def get_grids_for_timeline(
        self,
        source: str,
        parameters: list,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        forecast_hours: list,
    ) -> dict:
        """Load grids for arbitrary source/parameters/forecast hours.

        Generalised version of get_wave_grids_for_timeline() that works for
        any source (gfs, cmems_wave, cmems_current) and parameter set.

        Uses a single bulk SQL query and parallel decompression for fast
        loading (328 grids in ~2s vs ~16s sequential).

        Returns:
            Dict mapping parameter -> {forecast_hour -> (lats, lons, data_2d)}.
            Missing parameter/hour combos are silently skipped.
        """
        import time as _time
        from concurrent.futures import ThreadPoolExecutor

        run_id, _ = self._find_latest_run(source)
        if run_id is None:
            return {}

        conn = self._get_conn()
        try:
            # Determine which forecast hours actually exist for this run
            cur = conn.cursor()
            cur.execute(
                """SELECT array_agg(DISTINCT forecast_hour ORDER BY forecast_hour)
                   FROM weather_grid_data WHERE run_id = %s""",
                (run_id,),
            )
            row = cur.fetchone()
            available_hours = set(row[0]) if row and row[0] else set()

            # Map each requested hour to the closest available one
            needed_hours = set()
            for fh in forecast_hours:
                if available_hours:
                    best = min(available_hours, key=lambda h: abs(h - fh))
                    needed_hours.add(best)

            if not needed_hours:
                return {p: {} for p in parameters}

            # Bulk-fetch all grids in a single query
            t0 = _time.monotonic()
            cur.execute(
                """SELECT parameter, forecast_hour, lats, lons, data,
                          shape_rows, shape_cols
                   FROM weather_grid_data
                   WHERE run_id = %s
                     AND forecast_hour = ANY(%s)
                     AND parameter = ANY(%s)""",
                (run_id, sorted(needed_hours), list(parameters)),
            )
            rows = cur.fetchall()
            t_fetch = _time.monotonic() - t0

            # Parallel decompress + crop (zlib releases the GIL)
            t1 = _time.monotonic()

            def _process_row(row_tuple):
                param, fh, lats_blob, lons_blob, data_blob, nrows, ncols = row_tuple
                lats = self._decompress_1d(lats_blob)
                lons = self._decompress_1d(lons_blob)
                data = self._decompress_2d(data_blob, nrows, ncols)
                lats_c, lons_c, data_c = self._crop_grid(
                    lats, lons, data, lat_min, lat_max, lon_min, lon_max
                )
                return param, fh, lats_c, lons_c, data_c

            result: dict = {p: {} for p in parameters}
            workers = min(4, len(rows)) if len(rows) > 1 else 1
            if workers > 1:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    for param, fh, lats_c, lons_c, data_c in pool.map(_process_row, rows):
                        if param in result:
                            result[param][fh] = (lats_c, lons_c, data_c)
            else:
                for r in rows:
                    param, fh, lats_c, lons_c, data_c = _process_row(r)
                    if param in result:
                        result[param][fh] = (lats_c, lons_c, data_c)

            t_decompress = _time.monotonic() - t1
            total_grids = sum(len(v) for v in result.values())
            logger.info(
                f"get_grids_for_timeline({source}): {total_grids} grids, "
                f"{len(parameters)} params × {len(needed_hours)} hours "
                f"(fetch={t_fetch:.1f}s, decompress={t_decompress:.1f}s)"
            )
            return result

        except Exception as e:
            logger.error(f"Failed to load grids for timeline ({source}): {e}")
            return {}
        finally:
            conn.close()

    def get_available_hours_by_source(self, source: str) -> tuple:
        """Return (run_time, [forecast_hours]) for the latest complete run of a source.

        Works for any source: 'gfs', 'cmems_wave', 'cmems_current'.
        """
        run_id, _ = self._find_latest_run(source)
        if run_id is None:
            return None, []

        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """SELECT r.run_time, array_agg(DISTINCT g.forecast_hour ORDER BY g.forecast_hour)
                   FROM weather_forecast_runs r
                   JOIN weather_grid_data g ON g.run_id = r.id
                   WHERE r.id = %s
                   GROUP BY r.run_time""",
                (run_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None, []
            return row[0], row[1] or []
        except Exception:
            return None, []
        finally:
            conn.close()

    # Source definitions: key, label, expected frame count
    SOURCE_DEFS = {
        "gfs": {"label": "Wind", "expected_frames": 41},
        "cmems_wave": {"label": "Waves", "expected_frames": 41},
        "cmems_current": {"label": "Currents", "expected_frames": 41},
        "cmems_ice": {"label": "Ice", "expected_frames": 10},
        # "cmems_sst" disabled — global 0.083 deg SST download too large for current pipeline
        "gfs_visibility": {"label": "Visibility", "expected_frames": 41},
    }

    def get_health(self, max_age_hours: float = 12.0) -> dict:
        """Check health of all 6 weather sources in the database.

        Returns a structured dict with per-source status and overall health.
        "Complete" = frame count >= 75% of expected.
        "Fresh" = age < max_age_hours.
        "Healthy" = present AND complete AND fresh.
        """
        conn = self._get_conn()
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            sources_status = {}

            for source_key, sdef in self.SOURCE_DEFS.items():
                cur.execute(
                    """SELECT id, ingested_at, array_length(forecast_hours, 1) AS frame_count,
                              lat_min, lat_max, lon_min, lon_max
                       FROM weather_forecast_runs
                       WHERE source = %s AND status = 'complete'
                       ORDER BY array_length(forecast_hours, 1) DESC NULLS LAST,
                                ingested_at DESC
                       LIMIT 1""",
                    (source_key,),
                )
                row = cur.fetchone()

                if row is None:
                    sources_status[source_key] = {
                        "label": sdef["label"],
                        "present": False,
                        "complete": False,
                        "fresh": False,
                        "healthy": False,
                        "frame_count": 0,
                        "expected_frames": sdef["expected_frames"],
                        "age_hours": None,
                        "bounds": None,
                    }
                    continue

                ingested_at = row["ingested_at"]
                if ingested_at.tzinfo is None:
                    ingested_at = ingested_at.replace(tzinfo=timezone.utc)
                age_hours = (datetime.now(timezone.utc) - ingested_at).total_seconds() / 3600.0

                frame_count = row["frame_count"] or 0
                threshold = sdef["expected_frames"] * 0.75
                is_complete = frame_count >= threshold
                is_fresh = age_hours < max_age_hours

                sources_status[source_key] = {
                    "label": sdef["label"],
                    "present": True,
                    "complete": is_complete,
                    "fresh": is_fresh,
                    "healthy": is_complete and is_fresh,
                    "frame_count": frame_count,
                    "expected_frames": sdef["expected_frames"],
                    "age_hours": round(age_hours, 1),
                    "bounds": {
                        "lat_min": float(row["lat_min"]),
                        "lat_max": float(row["lat_max"]),
                        "lon_min": float(row["lon_min"]),
                        "lon_max": float(row["lon_max"]),
                    },
                }

            all_healthy = all(s["healthy"] for s in sources_status.values())
            db_bounds = self.get_db_bounds()

            return {
                "healthy": all_healthy,
                "db_bounds": db_bounds,
                "sources": sources_status,
            }
        except Exception as e:
            logger.error(f"Failed to get health status: {e}")
            return {"healthy": False, "db_bounds": None, "sources": {}, "error": str(e)}
        finally:
            conn.close()

    def get_db_bounds(self) -> dict | None:
        """Get the union bounding box across all complete runs.

        Returns {"lat_min", "lat_max", "lon_min", "lon_max"} or None if no data.
        """
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """SELECT MIN(lat_min), MAX(lat_max), MIN(lon_min), MAX(lon_max)
                   FROM weather_forecast_runs
                   WHERE status = 'complete'"""
            )
            row = cur.fetchone()
            if row is None or row[0] is None:
                return None
            return {
                "lat_min": float(row[0]),
                "lat_max": float(row[1]),
                "lon_min": float(row[2]),
                "lon_max": float(row[3]),
            }
        except Exception:
            return None
        finally:
            conn.close()

    def has_data_for_source(self, source: str, max_age_hours: int = 24) -> bool:
        """Check if a given source has a recent completed run with enough frames.

        Returns True only if a complete run exists that is recent AND has at
        least 75% of the expected frame count. This prevents skipping sources
        that were partially ingested before a container restart.
        """
        source_def = self.SOURCE_DEFS.get(source)
        if not source_def:
            return False
        expected = source_def["expected_frames"]
        min_frames = max(2, int(expected * 0.75))

        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """SELECT r.ingested_at,
                          (SELECT COUNT(*) FROM weather_grid_data g WHERE g.run_id = r.id)
                   FROM weather_forecast_runs r
                   WHERE r.source = %s AND r.status = 'complete'
                   ORDER BY r.ingested_at DESC LIMIT 1""",
                (source,),
            )
            row = cur.fetchone()
            if not row:
                return False
            ingested_at, frame_count = row
            if ingested_at.tzinfo is None:
                ingested_at = ingested_at.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - ingested_at).total_seconds() / 3600
            if age >= max_age_hours:
                return False
            if frame_count < min_frames:
                return False
            return True
        except Exception:
            return False
        finally:
            conn.close()

    def has_data(self) -> bool:
        """Check if any complete weather data exists in the database."""
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM weather_forecast_runs WHERE status = 'complete'"
            )
            count = cur.fetchone()[0]
            return count > 0
        except Exception:
            return False
        finally:
            conn.close()

    def get_freshness(self) -> Optional[dict]:
        """Get age of the latest weather data for frontend indicator."""
        conn = self._get_conn()
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                """SELECT source, ingested_at, status
                   FROM weather_forecast_runs
                   WHERE status = 'complete'
                   ORDER BY ingested_at DESC
                   LIMIT 3"""
            )
            rows = cur.fetchall()
            if not rows:
                return None

            latest = rows[0]["ingested_at"]
            now = datetime.now(timezone.utc)
            if latest.tzinfo is None:
                latest = latest.replace(tzinfo=timezone.utc)
            age_hours = (now - latest).total_seconds() / 3600.0

            return {
                "latest_ingestion": latest.isoformat(),
                "age_hours": round(age_hours, 1),
                "sources": {r["source"]: r["ingested_at"].isoformat() for r in rows},
            }
        except Exception as e:
            logger.error(f"Failed to get freshness: {e}")
            return None
        finally:
            conn.close()

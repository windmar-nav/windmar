"""
Weather ingestion service — downloads weather grids and stores compressed
blobs in PostgreSQL for fast route optimization.

Replaces live downloads with pre-fetched grids served from the database.
"""

import logging
import zlib
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)


class WeatherIngestionService:
    """Downloads weather grids and stores compressed blobs in PostgreSQL."""

    # Global bounding box (full ocean coverage)
    # Use -179.75/179.75 for longitude to avoid GFS 0/360 wrap-around issues
    LAT_MIN = -85.0
    LAT_MAX = 85.0
    LON_MIN = -179.75
    LON_MAX = 179.75
    GRID_RESOLUTION = 0.5  # degrees

    def __init__(self, db_url: str, copernicus_provider, gfs_provider):
        self.db_url = db_url
        self.copernicus_provider = copernicus_provider
        self.gfs_provider = gfs_provider

    def _get_conn(self):
        return psycopg2.connect(self.db_url)

    def ingest_all(self):
        """Run full ingestion cycle: waves + currents (+ wind if bbox provided).

        Wind is skipped from global pre-ingestion because GFS's GRIB caching
        is already fast (~2s per fetch). CMEMS wave/current are the real
        bottleneck (30-60s each) and benefit most from pre-ingestion.
        """
        logger.info("Starting weather ingestion cycle")
        self.ingest_waves()
        self.ingest_currents()
        self._supersede_old_runs()
        logger.info("Weather ingestion cycle complete")

    def ingest_wind(self):
        """Fetch GFS wind grids for forecast hours 0-120 (3-hourly)."""
        source = "gfs"
        run_time = datetime.now(timezone.utc)
        conn = self._get_conn()
        try:
            cur = conn.cursor()

            # Create forecast run record
            cur.execute(
                """INSERT INTO weather_forecast_runs
                   (source, run_time, status, grid_resolution,
                    lat_min, lat_max, lon_min, lon_max, forecast_hours)
                   VALUES (%s, %s, 'ingesting', %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (source, run_time) DO UPDATE
                   SET status = 'ingesting', ingested_at = NOW()
                   RETURNING id""",
                (source, run_time, self.GRID_RESOLUTION,
                 self.LAT_MIN, self.LAT_MAX, self.LON_MIN, self.LON_MAX,
                 self.gfs_provider.FORECAST_HOURS),
            )
            run_id = cur.fetchone()[0]
            conn.commit()

            ingested_hours = []
            for fh in self.gfs_provider.FORECAST_HOURS:
                try:
                    wind_data = self.gfs_provider.fetch_wind_data(
                        self.LAT_MIN, self.LAT_MAX,
                        self.LON_MIN, self.LON_MAX,
                        forecast_hour=fh,
                    )
                    if wind_data is None:
                        logger.warning(f"GFS wind f{fh:03d} returned None, skipping")
                        continue

                    lats_blob = self._compress(np.asarray(wind_data.lats))
                    lons_blob = self._compress(np.asarray(wind_data.lons))
                    rows = len(wind_data.lats)
                    cols = len(wind_data.lons)

                    # Store wind_u and wind_v
                    for param, arr in [
                        ("wind_u", wind_data.u_component),
                        ("wind_v", wind_data.v_component),
                    ]:
                        if arr is None:
                            continue
                        cur.execute(
                            """INSERT INTO weather_grid_data
                               (run_id, forecast_hour, parameter, lats, lons, data, shape_rows, shape_cols)
                               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                               ON CONFLICT (run_id, forecast_hour, parameter)
                               DO UPDATE SET data = EXCLUDED.data,
                                            lats = EXCLUDED.lats,
                                            lons = EXCLUDED.lons,
                                            shape_rows = EXCLUDED.shape_rows,
                                            shape_cols = EXCLUDED.shape_cols""",
                            (run_id, fh, param, lats_blob, lons_blob,
                             self._compress(np.asarray(arr)), rows, cols),
                        )

                    ingested_hours.append(fh)
                    conn.commit()
                    logger.debug(f"Ingested GFS wind f{fh:03d}")

                except Exception as e:
                    logger.error(f"Failed to ingest GFS wind f{fh:03d}: {e}")
                    conn.rollback()

            # Mark run complete
            status = "complete" if ingested_hours else "failed"
            cur.execute(
                "UPDATE weather_forecast_runs SET status = %s, forecast_hours = %s WHERE id = %s",
                (status, ingested_hours, run_id),
            )
            conn.commit()
            logger.info(f"GFS wind ingestion {status}: {len(ingested_hours)} hours")

        except Exception as e:
            logger.error(f"Wind ingestion failed: {e}")
            conn.rollback()
        finally:
            conn.close()

    def ingest_waves(self):
        """Fetch CMEMS wave forecast (Hs, Tp, Dir) for all available forecast hours.

        Tries fetch_wave_forecast() first (0-120h, 3-hourly = up to 41 timesteps).
        Falls back to fetch_wave_data() (single snapshot at h=0) if forecast
        fetch is unavailable.
        """
        source = "cmems_wave"
        run_time = datetime.now(timezone.utc)

        # Try multi-timestep forecast first
        forecast_frames = None
        if hasattr(self.copernicus_provider, "fetch_wave_forecast"):
            try:
                logger.info("Attempting CMEMS wave forecast download (0-120h)...")
                forecast_frames = self.copernicus_provider.fetch_wave_forecast(
                    self.LAT_MIN, self.LAT_MAX, self.LON_MIN, self.LON_MAX,
                )
            except Exception as e:
                logger.warning(f"Wave forecast fetch failed, falling back to snapshot: {e}")

        conn = self._get_conn()
        try:
            cur = conn.cursor()

            if forecast_frames:
                # Multi-timestep path
                available_hours = sorted(forecast_frames.keys())
                cur.execute(
                    """INSERT INTO weather_forecast_runs
                       (source, run_time, status, grid_resolution,
                        lat_min, lat_max, lon_min, lon_max, forecast_hours)
                       VALUES (%s, %s, 'ingesting', %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (source, run_time) DO UPDATE
                       SET status = 'ingesting', ingested_at = NOW()
                       RETURNING id""",
                    (source, run_time, self.GRID_RESOLUTION,
                     self.LAT_MIN, self.LAT_MAX, self.LON_MIN, self.LON_MAX,
                     available_hours),
                )
                run_id = cur.fetchone()[0]
                conn.commit()

                ingested_hours = []
                for fh in available_hours:
                    try:
                        wd = forecast_frames[fh]
                        lats_blob = self._compress(np.asarray(wd.lats))
                        lons_blob = self._compress(np.asarray(wd.lons))
                        rows = len(wd.lats)
                        cols = len(wd.lons)

                        for param, arr in [
                            ("wave_hs", wd.values),
                            ("wave_tp", wd.wave_period),
                            ("wave_dir", wd.wave_direction),
                            ("swell_hs", wd.swell_height),
                            ("swell_tp", wd.swell_period),
                            ("swell_dir", wd.swell_direction),
                            ("windwave_hs", wd.windwave_height),
                            ("windwave_tp", wd.windwave_period),
                            ("windwave_dir", wd.windwave_direction),
                        ]:
                            if arr is None:
                                continue
                            cur.execute(
                                """INSERT INTO weather_grid_data
                                   (run_id, forecast_hour, parameter, lats, lons, data, shape_rows, shape_cols)
                                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                                   ON CONFLICT (run_id, forecast_hour, parameter)
                                   DO UPDATE SET data = EXCLUDED.data,
                                                lats = EXCLUDED.lats,
                                                lons = EXCLUDED.lons,
                                                shape_rows = EXCLUDED.shape_rows,
                                                shape_cols = EXCLUDED.shape_cols""",
                                (run_id, fh, param, lats_blob, lons_blob,
                                 self._compress(np.asarray(arr)), rows, cols),
                            )
                        ingested_hours.append(fh)
                        conn.commit()
                        logger.debug(f"Ingested CMEMS wave f{fh:03d}")
                    except Exception as e:
                        logger.error(f"Failed to ingest wave f{fh:03d}: {e}")
                        conn.rollback()

                status = "complete" if ingested_hours else "failed"
                cur.execute(
                    "UPDATE weather_forecast_runs SET status = %s, forecast_hours = %s WHERE id = %s",
                    (status, ingested_hours, run_id),
                )
                conn.commit()
                logger.info(
                    f"CMEMS wave forecast ingestion {status}: "
                    f"{len(ingested_hours)}/{len(available_hours)} hours"
                )
            else:
                # Single-snapshot fallback (hour 0 only)
                cur.execute(
                    """INSERT INTO weather_forecast_runs
                       (source, run_time, status, grid_resolution,
                        lat_min, lat_max, lon_min, lon_max, forecast_hours)
                       VALUES (%s, %s, 'ingesting', %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (source, run_time) DO UPDATE
                       SET status = 'ingesting', ingested_at = NOW()
                       RETURNING id""",
                    (source, run_time, self.GRID_RESOLUTION,
                     self.LAT_MIN, self.LAT_MAX, self.LON_MIN, self.LON_MAX,
                     [0]),
                )
                run_id = cur.fetchone()[0]
                conn.commit()

                wave_data = self.copernicus_provider.fetch_wave_data(
                    self.LAT_MIN, self.LAT_MAX, self.LON_MIN, self.LON_MAX,
                )
                if wave_data is None:
                    cur.execute(
                        "UPDATE weather_forecast_runs SET status = 'failed' WHERE id = %s",
                        (run_id,),
                    )
                    conn.commit()
                    logger.warning("CMEMS wave fetch returned None")
                    return

                lats_blob = self._compress(np.asarray(wave_data.lats))
                lons_blob = self._compress(np.asarray(wave_data.lons))
                rows = len(wave_data.lats)
                cols = len(wave_data.lons)

                for param, arr in [
                    ("wave_hs", wave_data.values),
                    ("wave_tp", wave_data.wave_period),
                    ("wave_dir", wave_data.wave_direction),
                    ("swell_hs", wave_data.swell_height),
                    ("swell_tp", wave_data.swell_period),
                    ("swell_dir", wave_data.swell_direction),
                    ("windwave_hs", wave_data.windwave_height),
                    ("windwave_tp", wave_data.windwave_period),
                    ("windwave_dir", wave_data.windwave_direction),
                ]:
                    if arr is None:
                        continue
                    cur.execute(
                        """INSERT INTO weather_grid_data
                           (run_id, forecast_hour, parameter, lats, lons, data, shape_rows, shape_cols)
                           VALUES (%s, 0, %s, %s, %s, %s, %s, %s)
                           ON CONFLICT (run_id, forecast_hour, parameter)
                           DO UPDATE SET data = EXCLUDED.data,
                                        lats = EXCLUDED.lats,
                                        lons = EXCLUDED.lons,
                                        shape_rows = EXCLUDED.shape_rows,
                                        shape_cols = EXCLUDED.shape_cols""",
                        (run_id, param, lats_blob, lons_blob,
                         self._compress(np.asarray(arr)), rows, cols),
                    )

                cur.execute(
                    "UPDATE weather_forecast_runs SET status = 'complete' WHERE id = %s",
                    (run_id,),
                )
                conn.commit()
                logger.info("CMEMS wave snapshot ingestion complete (forecast unavailable)")

        except Exception as e:
            logger.error(f"Wave ingestion failed: {e}")
            conn.rollback()
        finally:
            conn.close()

    def ingest_currents(self):
        """Fetch CMEMS current data.

        Tries multi-timestep forecast first (0-120h, 3-hourly).
        Falls back to single snapshot if forecast fetch is unavailable.
        """
        source = "cmems_current"
        run_time = datetime.now(timezone.utc)

        # Try multi-timestep forecast first
        forecast_frames = None
        if hasattr(self.copernicus_provider, "fetch_current_forecast"):
            try:
                logger.info("Attempting CMEMS current forecast download (0-120h)...")
                forecast_frames = self.copernicus_provider.fetch_current_forecast(
                    self.LAT_MIN, self.LAT_MAX, self.LON_MIN, self.LON_MAX,
                )
            except Exception as e:
                logger.warning(f"Current forecast fetch failed, falling back to snapshot: {e}")

        conn = self._get_conn()
        try:
            cur = conn.cursor()

            if forecast_frames:
                # Multi-timestep path
                available_hours = sorted(forecast_frames.keys())
                cur.execute(
                    """INSERT INTO weather_forecast_runs
                       (source, run_time, status, grid_resolution,
                        lat_min, lat_max, lon_min, lon_max, forecast_hours)
                       VALUES (%s, %s, 'ingesting', %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (source, run_time) DO UPDATE
                       SET status = 'ingesting', ingested_at = NOW()
                       RETURNING id""",
                    (source, run_time, self.GRID_RESOLUTION,
                     self.LAT_MIN, self.LAT_MAX, self.LON_MIN, self.LON_MAX,
                     available_hours),
                )
                run_id = cur.fetchone()[0]
                conn.commit()

                ingested_hours = []
                for fh in available_hours:
                    try:
                        cd = forecast_frames[fh]
                        lats_blob = self._compress(np.asarray(cd.lats))
                        lons_blob = self._compress(np.asarray(cd.lons))
                        rows = len(cd.lats)
                        cols = len(cd.lons)

                        for param, arr in [
                            ("current_u", cd.u_component),
                            ("current_v", cd.v_component),
                        ]:
                            if arr is None:
                                continue
                            cur.execute(
                                """INSERT INTO weather_grid_data
                                   (run_id, forecast_hour, parameter, lats, lons, data, shape_rows, shape_cols)
                                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                                   ON CONFLICT (run_id, forecast_hour, parameter)
                                   DO UPDATE SET data = EXCLUDED.data,
                                                lats = EXCLUDED.lats,
                                                lons = EXCLUDED.lons,
                                                shape_rows = EXCLUDED.shape_rows,
                                                shape_cols = EXCLUDED.shape_cols""",
                                (run_id, fh, param, lats_blob, lons_blob,
                                 self._compress(np.asarray(arr)), rows, cols),
                            )
                        ingested_hours.append(fh)
                        conn.commit()
                        logger.debug(f"Ingested CMEMS current f{fh:03d}")
                    except Exception as e:
                        logger.error(f"Failed to ingest current f{fh:03d}: {e}")
                        conn.rollback()

                status = "complete" if ingested_hours else "failed"
                cur.execute(
                    "UPDATE weather_forecast_runs SET status = %s, forecast_hours = %s WHERE id = %s",
                    (status, ingested_hours, run_id),
                )
                conn.commit()
                logger.info(
                    f"CMEMS current forecast ingestion {status}: "
                    f"{len(ingested_hours)}/{len(available_hours)} hours"
                )
            else:
                # Single-snapshot fallback
                cur.execute(
                    """INSERT INTO weather_forecast_runs
                       (source, run_time, status, grid_resolution,
                        lat_min, lat_max, lon_min, lon_max, forecast_hours)
                       VALUES (%s, %s, 'ingesting', %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (source, run_time) DO UPDATE
                       SET status = 'ingesting', ingested_at = NOW()
                       RETURNING id""",
                    (source, run_time, self.GRID_RESOLUTION,
                     self.LAT_MIN, self.LAT_MAX, self.LON_MIN, self.LON_MAX,
                     [0]),
                )
                run_id = cur.fetchone()[0]
                conn.commit()

                current_data = self.copernicus_provider.fetch_current_data(
                    self.LAT_MIN, self.LAT_MAX, self.LON_MIN, self.LON_MAX,
                )
                if current_data is None:
                    cur.execute(
                        "UPDATE weather_forecast_runs SET status = 'failed' WHERE id = %s",
                        (run_id,),
                    )
                    conn.commit()
                    logger.warning("CMEMS current fetch returned None")
                    return

                lats_blob = self._compress(np.asarray(current_data.lats))
                lons_blob = self._compress(np.asarray(current_data.lons))
                rows = len(current_data.lats)
                cols = len(current_data.lons)

                for param, arr in [
                    ("current_u", current_data.u_component),
                    ("current_v", current_data.v_component),
                ]:
                    if arr is None:
                        continue
                    cur.execute(
                        """INSERT INTO weather_grid_data
                           (run_id, forecast_hour, parameter, lats, lons, data, shape_rows, shape_cols)
                           VALUES (%s, 0, %s, %s, %s, %s, %s, %s)
                           ON CONFLICT (run_id, forecast_hour, parameter)
                           DO UPDATE SET data = EXCLUDED.data,
                                        lats = EXCLUDED.lats,
                                        lons = EXCLUDED.lons,
                                        shape_rows = EXCLUDED.shape_rows,
                                        shape_cols = EXCLUDED.shape_cols""",
                        (run_id, param, lats_blob, lons_blob,
                         self._compress(np.asarray(arr)), rows, cols),
                    )

                cur.execute(
                    "UPDATE weather_forecast_runs SET status = 'complete' WHERE id = %s",
                    (run_id,),
                )
                conn.commit()
                logger.info("CMEMS current snapshot ingestion complete (forecast unavailable)")

        except Exception as e:
            logger.error(f"Current ingestion failed: {e}")
            conn.rollback()
        finally:
            conn.close()

    def _compress(self, arr: np.ndarray) -> bytes:
        """zlib-compress a numpy array (stored as float32 to halve size)."""
        return zlib.compress(arr.astype(np.float32).tobytes())

    def _supersede_old_runs(self):
        """Mark runs older than 24h as 'superseded'."""
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
            cur.execute(
                """UPDATE weather_forecast_runs
                   SET status = 'superseded'
                   WHERE status = 'complete'
                     AND ingested_at < %s""",
                (cutoff,),
            )
            superseded = cur.rowcount
            conn.commit()
            if superseded > 0:
                logger.info(f"Superseded {superseded} old weather runs")
        except Exception as e:
            logger.error(f"Failed to supersede old runs: {e}")
            conn.rollback()
        finally:
            conn.close()

    def get_latest_status(self) -> dict:
        """Get status of the latest ingestion runs."""
        conn = self._get_conn()
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                """SELECT source, run_time, ingested_at, status,
                          forecast_hours, grid_resolution
                   FROM weather_forecast_runs
                   WHERE status IN ('complete', 'ingesting')
                   ORDER BY ingested_at DESC
                   LIMIT 10""",
            )
            rows = cur.fetchall()

            # Count total grid rows
            cur.execute("SELECT COUNT(*) FROM weather_grid_data")
            grid_count = cur.fetchone()["count"]

            return {
                "runs": [
                    {
                        "source": r["source"],
                        "run_time": r["run_time"].isoformat() if r["run_time"] else None,
                        "ingested_at": r["ingested_at"].isoformat() if r["ingested_at"] else None,
                        "status": r["status"],
                        "forecast_hours": r["forecast_hours"],
                        "grid_resolution": r["grid_resolution"],
                    }
                    for r in rows
                ],
                "total_grid_rows": grid_count,
            }
        except Exception as e:
            logger.error(f"Failed to get ingestion status: {e}")
            return {"runs": [], "total_grid_rows": 0, "error": str(e)}
        finally:
            conn.close()

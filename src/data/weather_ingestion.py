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

    def ingest_all(self, force: bool = False):
        """Run full ingestion cycle: wind + waves + currents + ice + visibility.

        Downloads GFS wind (41 forecast hours, ~2-3 min with rate limiting),
        CMEMS waves, CMEMS currents, CMEMS ice, and GFS visibility
        into PostgreSQL for sub-second route optimization queries.
        SST disabled — global 0.083 deg download too large for current pipeline.

        Args:
            force: If True, bypass freshness checks and re-ingest all sources.
        """
        logger.info(f"Starting weather ingestion cycle (force={force})")
        self.ingest_wind(force=force)
        self.ingest_waves(force=force)
        self.ingest_currents(force=force)
        self.ingest_ice(force=force)
        # SST disabled — copernicusmarine.subset() downloads 4+ GB for global grid
        self.ingest_visibility(force=force)
        self._supersede_old_runs()
        self.cleanup_orphaned_grid_data()
        logger.info("Weather ingestion cycle complete")

    def ingest_wind(self, force: bool = False):
        """Fetch GFS wind grids for forecast hours 0-120 (3-hourly).

        Downloads 41 GRIB files from NOAA NOMADS with 2s rate limiting
        between requests. Cached GRIBs are reused (no download needed).
        Skips if a recent multi-timestep run already exists in the DB.
        """
        import time as _time

        source = "gfs"

        if not force and self._has_multistep_run(source):
            logger.debug("Skipping wind ingestion — multi-timestep GFS run exists in DB")
            return

        run_time = datetime.now(timezone.utc)
        conn = self._get_conn()
        try:
            cur = conn.cursor()

            # Supersede any existing complete GFS runs
            cur.execute(
                """UPDATE weather_forecast_runs SET status = 'superseded'
                   WHERE source = %s AND status = 'complete'""",
                (source,),
            )

            # Create forecast run record
            cur.execute(
                """INSERT INTO weather_forecast_runs
                   (source, run_time, status, grid_resolution,
                    lat_min, lat_max, lon_min, lon_max, forecast_hours)
                   VALUES (%s, %s, 'ingesting', %s, %s, %s, %s, %s, %s)
                   RETURNING id""",
                (source, run_time, self.GRID_RESOLUTION,
                 self.LAT_MIN, self.LAT_MAX, self.LON_MIN, self.LON_MAX,
                 self.gfs_provider.FORECAST_HOURS),
            )
            run_id = cur.fetchone()[0]
            conn.commit()

            ingested_hours = []
            for i, fh in enumerate(self.gfs_provider.FORECAST_HOURS):
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
                    logger.debug(f"Ingested GFS wind f{fh:03d} ({i+1}/{len(self.gfs_provider.FORECAST_HOURS)})")

                    # Rate-limit NOMADS requests (2s between downloads)
                    if i < len(self.gfs_provider.FORECAST_HOURS) - 1:
                        _time.sleep(2)

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
            logger.info(f"GFS wind ingestion {status}: {len(ingested_hours)}/{len(self.gfs_provider.FORECAST_HOURS)} hours")

        except Exception as e:
            logger.error(f"Wind ingestion failed: {e}")
            conn.rollback()
        finally:
            conn.close()

    def _has_multistep_run(self, source: str, max_age_hours: float = 12.0) -> bool:
        """Check if a recent multi-timestep (>1 hour) complete run exists for source."""
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """SELECT 1 FROM weather_forecast_runs
                   WHERE source = %s AND status = 'complete'
                     AND array_length(forecast_hours, 1) > 1
                     AND ingested_at > NOW() - INTERVAL '%s hours'
                   LIMIT 1""",
                (source, max_age_hours),
            )
            return cur.fetchone() is not None
        except Exception:
            return False
        finally:
            conn.close()

    def ingest_waves(self, force: bool = False):
        """Fetch CMEMS wave snapshot (Hs, Tp, Dir) including swell decomposition.

        Skips if a multi-timestep forecast run already exists in the DB
        (to avoid replacing it with a single snapshot).
        """
        source = "cmems_wave"
        if not force and self._has_multistep_run(source):
            logger.debug("Skipping wave snapshot — multi-timestep run exists in DB")
            return

        run_time = datetime.now(timezone.utc)

        conn = self._get_conn()
        try:
            cur = conn.cursor()

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
            logger.info("CMEMS wave snapshot ingestion complete")

        except Exception as e:
            logger.error(f"Wave ingestion failed: {e}")
            conn.rollback()
        finally:
            conn.close()

    def ingest_currents(self, force: bool = False):
        """Fetch CMEMS current data (single snapshot).

        Skips if a multi-timestep forecast run already exists in the DB.
        """
        source = "cmems_current"
        if not force and self._has_multistep_run(source):
            logger.debug("Skipping current snapshot — multi-timestep run exists in DB")
            return

        run_time = datetime.now(timezone.utc)

        conn = self._get_conn()
        try:
            cur = conn.cursor()

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
            logger.info("CMEMS current snapshot ingestion complete")

        except Exception as e:
            logger.error(f"Current ingestion failed: {e}")
            conn.rollback()
        finally:
            conn.close()

    def ingest_ice(self, force: bool = False):
        """Fetch CMEMS ice concentration snapshot.

        Skips if a complete ice run already exists in the DB (any timestep count).
        """
        source = "cmems_ice"
        if not force and self._has_multistep_run(source):
            logger.debug("Skipping ice snapshot — run exists in DB")
            return

        run_time = datetime.now(timezone.utc)

        conn = self._get_conn()
        try:
            cur = conn.cursor()

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

            ice_data = self.copernicus_provider.fetch_ice_data(
                self.LAT_MIN, self.LAT_MAX, self.LON_MIN, self.LON_MAX,
            )
            if ice_data is None:
                cur.execute(
                    "UPDATE weather_forecast_runs SET status = 'failed' WHERE id = %s",
                    (run_id,),
                )
                conn.commit()
                logger.warning("CMEMS ice fetch returned None")
                return

            lats_blob = self._compress(np.asarray(ice_data.lats))
            lons_blob = self._compress(np.asarray(ice_data.lons))
            rows = len(ice_data.lats)
            cols = len(ice_data.lons)

            arr = ice_data.ice_concentration if ice_data.ice_concentration is not None else ice_data.values
            if arr is not None:
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
                    (run_id, "ice_siconc", lats_blob, lons_blob,
                     self._compress(np.asarray(arr)), rows, cols),
                )

            cur.execute(
                "UPDATE weather_forecast_runs SET status = 'complete' WHERE id = %s",
                (run_id,),
            )
            conn.commit()
            logger.info("CMEMS ice snapshot ingestion complete")

        except Exception as e:
            logger.error(f"Ice ingestion failed: {e}")
            conn.rollback()
        finally:
            conn.close()

    def ingest_wave_forecast_frames(self, frames: dict):
        """Store multi-timestep wave forecast frames into PostgreSQL.

        Args:
            frames: Dict mapping forecast_hour (int) -> WeatherData.
                    Each WeatherData has values (wave_hs), wave_period, wave_direction,
                    and optionally swell/windwave decomposition.
        """
        if not frames:
            return

        source = "cmems_wave"
        run_time = datetime.now(timezone.utc)
        forecast_hours = sorted(frames.keys())
        conn = self._get_conn()
        try:
            cur = conn.cursor()

            # Supersede any existing complete wave runs before creating new one
            cur.execute(
                """UPDATE weather_forecast_runs SET status = 'superseded'
                   WHERE source = %s AND status = 'complete'""",
                (source,),
            )

            # Create forecast run record with ALL forecast hours
            cur.execute(
                """INSERT INTO weather_forecast_runs
                   (source, run_time, status, grid_resolution,
                    lat_min, lat_max, lon_min, lon_max, forecast_hours)
                   VALUES (%s, %s, 'ingesting', %s, %s, %s, %s, %s, %s)
                   RETURNING id""",
                (source, run_time, 0.083,
                 self.LAT_MIN, self.LAT_MAX, self.LON_MIN, self.LON_MAX,
                 forecast_hours),
            )
            run_id = cur.fetchone()[0]
            conn.commit()

            ingested_count = 0
            for fh in forecast_hours:
                wd = frames[fh]
                try:
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

                    ingested_count += 1
                    conn.commit()
                except Exception as e:
                    logger.error(f"Failed to ingest wave forecast f{fh:03d}: {e}")
                    conn.rollback()

            status = "complete" if ingested_count > 0 else "failed"
            cur.execute(
                "UPDATE weather_forecast_runs SET status = %s WHERE id = %s",
                (status, run_id),
            )
            conn.commit()
            logger.info(
                f"Wave forecast DB ingestion {status}: "
                f"{ingested_count}/{len(forecast_hours)} hours"
            )

        except Exception as e:
            logger.error(f"Wave forecast frame ingestion failed: {e}")
            conn.rollback()
        finally:
            conn.close()

    def ingest_current_forecast_frames(self, frames: dict):
        """Store multi-timestep current forecast frames into PostgreSQL.

        Args:
            frames: Dict mapping forecast_hour (int) -> WeatherData.
                    Each WeatherData has u_component and v_component.
        """
        if not frames:
            return

        source = "cmems_current"
        run_time = datetime.now(timezone.utc)
        forecast_hours = sorted(frames.keys())
        conn = self._get_conn()
        try:
            cur = conn.cursor()

            # Supersede any existing complete current runs
            cur.execute(
                """UPDATE weather_forecast_runs SET status = 'superseded'
                   WHERE source = %s AND status = 'complete'""",
                (source,),
            )

            cur.execute(
                """INSERT INTO weather_forecast_runs
                   (source, run_time, status, grid_resolution,
                    lat_min, lat_max, lon_min, lon_max, forecast_hours)
                   VALUES (%s, %s, 'ingesting', %s, %s, %s, %s, %s, %s)
                   RETURNING id""",
                (source, run_time, 0.083,
                 self.LAT_MIN, self.LAT_MAX, self.LON_MIN, self.LON_MAX,
                 forecast_hours),
            )
            run_id = cur.fetchone()[0]
            conn.commit()

            ingested_count = 0
            for fh in forecast_hours:
                wd = frames[fh]
                try:
                    lats_blob = self._compress(np.asarray(wd.lats))
                    lons_blob = self._compress(np.asarray(wd.lons))
                    rows = len(wd.lats)
                    cols = len(wd.lons)

                    for param, arr in [
                        ("current_u", wd.u_component),
                        ("current_v", wd.v_component),
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

                    ingested_count += 1
                    conn.commit()
                except Exception as e:
                    logger.error(f"Failed to ingest current forecast f{fh:03d}: {e}")
                    conn.rollback()

            status = "complete" if ingested_count > 0 else "failed"
            cur.execute(
                "UPDATE weather_forecast_runs SET status = %s WHERE id = %s",
                (status, run_id),
            )
            conn.commit()
            logger.info(
                f"Current forecast DB ingestion {status}: "
                f"{ingested_count}/{len(forecast_hours)} hours"
            )

        except Exception as e:
            logger.error(f"Current forecast frame ingestion failed: {e}")
            conn.rollback()
        finally:
            conn.close()

    def ingest_ice_forecast_frames(self, frames: dict):
        """Store multi-timestep ice forecast frames into PostgreSQL.

        Args:
            frames: Dict mapping forecast_hour (int) -> WeatherData.
                    Each WeatherData has ice_concentration (siconc).
                    Expected hours: 0, 24, 48, ..., 216 (10 daily steps).
        """
        if not frames:
            return

        source = "cmems_ice"
        run_time = datetime.now(timezone.utc)
        forecast_hours = sorted(frames.keys())
        conn = self._get_conn()
        try:
            cur = conn.cursor()

            # Supersede any existing complete ice runs
            cur.execute(
                """UPDATE weather_forecast_runs SET status = 'superseded'
                   WHERE source = %s AND status = 'complete'""",
                (source,),
            )

            cur.execute(
                """INSERT INTO weather_forecast_runs
                   (source, run_time, status, grid_resolution,
                    lat_min, lat_max, lon_min, lon_max, forecast_hours)
                   VALUES (%s, %s, 'ingesting', %s, %s, %s, %s, %s, %s)
                   RETURNING id""",
                (source, run_time, 0.083,
                 self.LAT_MIN, self.LAT_MAX, self.LON_MIN, self.LON_MAX,
                 forecast_hours),
            )
            run_id = cur.fetchone()[0]
            conn.commit()

            ingested_count = 0
            for fh in forecast_hours:
                wd = frames[fh]
                try:
                    lats_blob = self._compress(np.asarray(wd.lats))
                    lons_blob = self._compress(np.asarray(wd.lons))
                    rows = len(wd.lats)
                    cols = len(wd.lons)

                    arr = wd.ice_concentration if wd.ice_concentration is not None else wd.values
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
                        (run_id, fh, "ice_siconc", lats_blob, lons_blob,
                         self._compress(np.asarray(arr)), rows, cols),
                    )

                    ingested_count += 1
                    conn.commit()
                except Exception as e:
                    logger.error(f"Failed to ingest ice forecast f{fh:03d}: {e}")
                    conn.rollback()

            status = "complete" if ingested_count > 0 else "failed"
            cur.execute(
                "UPDATE weather_forecast_runs SET status = %s WHERE id = %s",
                (status, run_id),
            )
            conn.commit()
            logger.info(
                f"Ice forecast DB ingestion {status}: "
                f"{ingested_count}/{len(forecast_hours)} hours"
            )

        except Exception as e:
            logger.error(f"Ice forecast frame ingestion failed: {e}")
            conn.rollback()
        finally:
            conn.close()

    def ingest_sst(self, force: bool = False):
        """Fetch CMEMS SST forecast (0-120h, 3-hourly) and store in PostgreSQL.

        Skips if a multi-timestep SST run already exists in the DB.
        """
        source = "cmems_sst"
        if not force and self._has_multistep_run(source):
            logger.debug("Skipping SST ingestion — multi-timestep run exists in DB")
            return

        logger.info("CMEMS SST forecast ingestion starting")
        try:
            result = self.copernicus_provider.fetch_sst_forecast(
                self.LAT_MIN, self.LAT_MAX, self.LON_MIN, self.LON_MAX,
            )
            if not result:
                logger.warning("CMEMS SST forecast fetch returned empty")
                return
            self.ingest_sst_forecast_frames(result)
        except Exception as e:
            logger.error(f"SST ingestion failed: {e}")

    def ingest_visibility(self, force: bool = False):
        """Fetch GFS visibility forecast (0-120h, 3-hourly) and store in PostgreSQL.

        Skips if a multi-timestep visibility run already exists in the DB.
        """
        source = "gfs_visibility"
        if not force and self._has_multistep_run(source):
            logger.debug("Skipping visibility ingestion — multi-timestep run exists in DB")
            return

        logger.info("GFS visibility forecast ingestion starting")
        try:
            result = self.gfs_provider.fetch_visibility_forecast(
                self.LAT_MIN, self.LAT_MAX, self.LON_MIN, self.LON_MAX,
            )
            if not result:
                logger.warning("GFS visibility forecast fetch returned empty")
                return
            self.ingest_visibility_forecast_frames(result)
        except Exception as e:
            logger.error(f"Visibility ingestion failed: {e}")

    def ingest_sst_forecast_frames(self, frames: dict):
        """Store multi-timestep SST forecast frames into PostgreSQL.

        Args:
            frames: Dict mapping forecast_hour (int) -> WeatherData.
                    Each WeatherData has sst or values field (°C).
        """
        if not frames:
            return

        source = "cmems_sst"
        run_time = datetime.now(timezone.utc)
        forecast_hours = sorted(frames.keys())
        conn = self._get_conn()
        try:
            cur = conn.cursor()

            # Supersede any existing complete SST runs
            cur.execute(
                """UPDATE weather_forecast_runs SET status = 'superseded'
                   WHERE source = %s AND status = 'complete'""",
                (source,),
            )

            cur.execute(
                """INSERT INTO weather_forecast_runs
                   (source, run_time, status, grid_resolution,
                    lat_min, lat_max, lon_min, lon_max, forecast_hours)
                   VALUES (%s, %s, 'ingesting', %s, %s, %s, %s, %s, %s)
                   RETURNING id""",
                (source, run_time, 0.083,
                 self.LAT_MIN, self.LAT_MAX, self.LON_MIN, self.LON_MAX,
                 forecast_hours),
            )
            run_id = cur.fetchone()[0]
            conn.commit()

            ingested_count = 0
            for fh in forecast_hours:
                wd = frames[fh]
                try:
                    lats_blob = self._compress(np.asarray(wd.lats))
                    lons_blob = self._compress(np.asarray(wd.lons))
                    rows = len(wd.lats)
                    cols = len(wd.lons)

                    arr = wd.sst if wd.sst is not None else wd.values
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
                        (run_id, fh, "sst", lats_blob, lons_blob,
                         self._compress(np.asarray(arr)), rows, cols),
                    )

                    ingested_count += 1
                    conn.commit()
                except Exception as e:
                    logger.error(f"Failed to ingest SST forecast f{fh:03d}: {e}")
                    conn.rollback()

            status = "complete" if ingested_count > 0 else "failed"
            cur.execute(
                "UPDATE weather_forecast_runs SET status = %s WHERE id = %s",
                (status, run_id),
            )
            conn.commit()
            logger.info(
                f"SST forecast DB ingestion {status}: "
                f"{ingested_count}/{len(forecast_hours)} hours"
            )

        except Exception as e:
            logger.error(f"SST forecast frame ingestion failed: {e}")
            conn.rollback()
        finally:
            conn.close()

    def ingest_visibility_forecast_frames(self, frames: dict):
        """Store multi-timestep visibility forecast frames into PostgreSQL.

        Args:
            frames: Dict mapping forecast_hour (int) -> WeatherData.
                    Each WeatherData has visibility or values field (km).
        """
        if not frames:
            return

        source = "gfs_visibility"
        run_time = datetime.now(timezone.utc)
        forecast_hours = sorted(frames.keys())
        conn = self._get_conn()
        try:
            cur = conn.cursor()

            # Supersede any existing complete visibility runs
            cur.execute(
                """UPDATE weather_forecast_runs SET status = 'superseded'
                   WHERE source = %s AND status = 'complete'""",
                (source,),
            )

            cur.execute(
                """INSERT INTO weather_forecast_runs
                   (source, run_time, status, grid_resolution,
                    lat_min, lat_max, lon_min, lon_max, forecast_hours)
                   VALUES (%s, %s, 'ingesting', %s, %s, %s, %s, %s, %s)
                   RETURNING id""",
                (source, run_time, self.GRID_RESOLUTION,
                 self.LAT_MIN, self.LAT_MAX, self.LON_MIN, self.LON_MAX,
                 forecast_hours),
            )
            run_id = cur.fetchone()[0]
            conn.commit()

            ingested_count = 0
            for fh in forecast_hours:
                wd = frames[fh]
                try:
                    lats_blob = self._compress(np.asarray(wd.lats))
                    lons_blob = self._compress(np.asarray(wd.lons))
                    rows = len(wd.lats)
                    cols = len(wd.lons)

                    arr = wd.visibility if wd.visibility is not None else wd.values
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
                        (run_id, fh, "visibility", lats_blob, lons_blob,
                         self._compress(np.asarray(arr)), rows, cols),
                    )

                    ingested_count += 1
                    conn.commit()
                except Exception as e:
                    logger.error(f"Failed to ingest visibility forecast f{fh:03d}: {e}")
                    conn.rollback()

            status = "complete" if ingested_count > 0 else "failed"
            cur.execute(
                "UPDATE weather_forecast_runs SET status = %s WHERE id = %s",
                (status, run_id),
            )
            conn.commit()
            logger.info(
                f"Visibility forecast DB ingestion {status}: "
                f"{ingested_count}/{len(forecast_hours)} hours"
            )

        except Exception as e:
            logger.error(f"Visibility forecast frame ingestion failed: {e}")
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

    def cleanup_orphaned_grid_data(self):
        """Delete grid data rows belonging to superseded, failed, or stale ingesting runs.

        This reclaims TOAST storage from dead runs. Should be called after
        _supersede_old_runs() in each ingestion cycle.

        Note: PostgreSQL does not release disk space from TOAST deletes until
        autovacuum runs (or manual VACUUM). Large deletes may cause temporary
        I/O spikes.
        """
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            # Delete grid data for superseded and failed runs
            cur.execute(
                """DELETE FROM weather_grid_data
                   WHERE run_id IN (
                       SELECT id FROM weather_forecast_runs
                       WHERE status IN ('superseded', 'failed')
                   )"""
            )
            deleted_dead = cur.rowcount

            # Delete grid data for ingesting runs older than 6h (stale/abandoned)
            cutoff = datetime.now(timezone.utc) - timedelta(hours=6)
            cur.execute(
                """DELETE FROM weather_grid_data
                   WHERE run_id IN (
                       SELECT id FROM weather_forecast_runs
                       WHERE status = 'ingesting'
                         AND ingested_at < %s
                   )""",
                (cutoff,),
            )
            deleted_stale = cur.rowcount

            # Now delete the orphaned run metadata too
            cur.execute(
                """DELETE FROM weather_forecast_runs
                   WHERE status IN ('superseded', 'failed')"""
            )
            deleted_runs_dead = cur.rowcount

            cur.execute(
                """DELETE FROM weather_forecast_runs
                   WHERE status = 'ingesting'
                     AND ingested_at < %s""",
                (cutoff,),
            )
            deleted_runs_stale = cur.rowcount

            conn.commit()
            total_grids = deleted_dead + deleted_stale
            total_runs = deleted_runs_dead + deleted_runs_stale
            if total_grids > 0 or total_runs > 0:
                logger.info(
                    f"Orphan cleanup: deleted {total_grids} grid rows "
                    f"and {total_runs} run records "
                    f"(superseded/failed={deleted_dead}, stale_ingesting={deleted_stale})"
                )
        except Exception as e:
            logger.error(f"Failed to clean up orphaned grid data: {e}")
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

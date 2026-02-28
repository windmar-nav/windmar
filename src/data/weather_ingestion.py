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

    # Default CMEMS viewport bounds.
    # Waves use subset() (server-side download) — can handle wide bbox.
    # Currents/SST use open_dataset() — limited to moderate bbox to avoid
    # slow S3 chunk downloads.  GFS wind/visibility are global at 0.5°.
    CMEMS_DEFAULT_LAT_MIN = -60.0
    CMEMS_DEFAULT_LAT_MAX = 60.0
    CMEMS_DEFAULT_LON_MIN = -80.0
    CMEMS_DEFAULT_LON_MAX = 140.0
    # SST — moderate bbox (open_dataset), covers Atl + Med + Indian.
    SST_DEFAULT_LAT_MIN = -40.0
    SST_DEFAULT_LAT_MAX = 65.0
    SST_DEFAULT_LON_MIN = -80.0
    SST_DEFAULT_LON_MAX = 80.0
    # Ice — Arctic (50-85°N), moderate lon range.
    ICE_DEFAULT_LAT_MIN = 50.0
    ICE_DEFAULT_LAT_MAX = 85.0
    ICE_DEFAULT_LON_MIN = -80.0
    ICE_DEFAULT_LON_MAX = 80.0

    def ingest_all(self, force: bool = False,
                   lat_min: Optional[float] = None, lat_max: Optional[float] = None,
                   lon_min: Optional[float] = None, lon_max: Optional[float] = None):
        """Run full ingestion cycle: wind + waves + currents + ice + sst + visibility.

        Downloads GFS wind and visibility globally (0.5° resolution).
        Downloads CMEMS waves, currents, ice, and SST for the given viewport bounds
        (0.083° resolution — viewport-bounded to keep data volume manageable).

        Args:
            force: If True, bypass freshness checks and re-ingest all sources.
            lat_min/lat_max/lon_min/lon_max: Viewport bounds for CMEMS sources.
                Defaults to CMEMS_DEFAULT_* class attributes.
        """
        _lat_min = lat_min if lat_min is not None else self.CMEMS_DEFAULT_LAT_MIN
        _lat_max = lat_max if lat_max is not None else self.CMEMS_DEFAULT_LAT_MAX
        _lon_min = lon_min if lon_min is not None else self.CMEMS_DEFAULT_LON_MIN
        _lon_max = lon_max if lon_max is not None else self.CMEMS_DEFAULT_LON_MAX
        _ice_lat_min = lat_min if lat_min is not None else self.ICE_DEFAULT_LAT_MIN
        _ice_lat_max = lat_max if lat_max is not None else self.ICE_DEFAULT_LAT_MAX
        _ice_lon_min = lon_min if lon_min is not None else self.ICE_DEFAULT_LON_MIN
        _ice_lon_max = lon_max if lon_max is not None else self.ICE_DEFAULT_LON_MAX
        _sst_lat_min = lat_min if lat_min is not None else self.SST_DEFAULT_LAT_MIN
        _sst_lat_max = lat_max if lat_max is not None else self.SST_DEFAULT_LAT_MAX
        _sst_lon_min = lon_min if lon_min is not None else self.SST_DEFAULT_LON_MIN
        _sst_lon_max = lon_max if lon_max is not None else self.SST_DEFAULT_LON_MAX

        logger.info(f"Starting weather ingestion cycle (force={force})")
        self.ingest_wind(force=force)
        self.ingest_waves(force=force, lat_min=_lat_min, lat_max=_lat_max,
                          lon_min=_lon_min, lon_max=_lon_max)
        self.ingest_currents(force=force, lat_min=_lat_min, lat_max=_lat_max,
                             lon_min=_lon_min, lon_max=_lon_max)
        self.ingest_ice(force=force, lat_min=_ice_lat_min, lat_max=_ice_lat_max,
                        lon_min=_ice_lon_min, lon_max=_ice_lon_max)
        self.ingest_sst(force=force, lat_min=_sst_lat_min, lat_max=_sst_lat_max,
                        lon_min=_sst_lon_min, lon_max=_sst_lon_max)
        self.ingest_visibility(force=force)
        self._supersede_old_runs()
        self.cleanup_orphaned_grid_data()
        logger.info("Weather ingestion cycle complete")

    def ingest_wind(self, force: bool = False):
        """Fetch GFS wind grids for forecast hours 0-120 (3-hourly).

        Downloads 41 GRIB files from NOAA NOMADS with 2s rate limiting
        between requests. Cached GRIBs are reused (no download needed).
        Skips if a recent multi-timestep run already exists in the DB.

        Old runs are only superseded AFTER the new run proves it has at
        least as many hours — prevents data loss when NOMADS is still
        publishing a new GFS cycle.
        """
        import time as _time

        source = "gfs"

        if not force and self._has_multistep_run(source):
            logger.debug("Skipping wind ingestion — multi-timestep GFS run exists in DB")
            return

        # Count hours in existing best run (for deferred supersede)
        old_hour_count = self._count_best_run_hours(source)

        run_time = datetime.now(timezone.utc)
        conn = self._get_conn()
        try:
            cur = conn.cursor()

            # Create new forecast run (do NOT supersede old runs yet)
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

            # Only supersede old runs if new run has at least as many hours
            if len(ingested_hours) >= old_hour_count:
                cur.execute(
                    """UPDATE weather_forecast_runs SET status = 'superseded'
                       WHERE source = %s AND status = 'complete' AND id != %s""",
                    (source, run_id),
                )
                status = "complete" if ingested_hours else "failed"
                logger.info(f"GFS wind: new run ({len(ingested_hours)}h) >= old ({old_hour_count}h) — superseded old runs")
            else:
                # New run has fewer hours — mark it failed, keep old run
                status = "failed"
                logger.warning(
                    f"GFS wind: new run ({len(ingested_hours)}h) < old ({old_hour_count}h) "
                    f"— keeping old run, marking new as failed"
                )

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

    def _count_best_run_hours(self, source: str) -> int:
        """Return the hour count of the best existing complete run for a source.

        Used by deferred-supersede logic: new runs only replace old ones
        if they have at least as many forecast hours.
        """
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """SELECT COALESCE(array_length(forecast_hours, 1), 0)
                   FROM weather_forecast_runs
                   WHERE source = %s AND status = 'complete'
                   ORDER BY array_length(forecast_hours, 1) DESC NULLS LAST,
                            run_time DESC
                   LIMIT 1""",
                (source,),
            )
            row = cur.fetchone()
            return row[0] if row else 0
        except Exception:
            return 0
        finally:
            conn.close()

    def ingest_waves(self, force: bool = False,
                     lat_min: Optional[float] = None, lat_max: Optional[float] = None,
                     lon_min: Optional[float] = None, lon_max: Optional[float] = None):
        """Fetch CMEMS wave forecast (0-120h, 3-hourly) and store all frames.

        Downloads the full multi-timestep wave forecast including swell
        decomposition.  Skips if a recent multi-timestep run already exists.

        Args:
            force: Bypass freshness guard.
            lat_min/lat_max/lon_min/lon_max: Viewport bounds (defaults to class globals).
        """
        source = "cmems_wave"
        if not force and self._has_multistep_run(source):
            logger.debug("Skipping wave ingestion — multi-timestep run exists in DB")
            return

        _lat_min = lat_min if lat_min is not None else self.CMEMS_DEFAULT_LAT_MIN
        _lat_max = lat_max if lat_max is not None else self.CMEMS_DEFAULT_LAT_MAX
        _lon_min = lon_min if lon_min is not None else self.CMEMS_DEFAULT_LON_MIN
        _lon_max = lon_max if lon_max is not None else self.CMEMS_DEFAULT_LON_MAX

        logger.info("CMEMS wave forecast ingestion starting")
        try:
            result = self.copernicus_provider.fetch_wave_forecast(
                _lat_min, _lat_max, _lon_min, _lon_max,
            )
            if not result:
                logger.warning("CMEMS wave forecast fetch returned empty")
                return
            self.ingest_wave_forecast_frames(result)
        except Exception as e:
            logger.error(f"Wave forecast ingestion failed: {e}")

    def ingest_currents(self, force: bool = False,
                        lat_min: Optional[float] = None, lat_max: Optional[float] = None,
                        lon_min: Optional[float] = None, lon_max: Optional[float] = None):
        """Fetch CMEMS current forecast (0-120h, 3-hourly) and store all frames.

        Downloads the full multi-timestep current forecast (u/v components).
        Skips if a recent multi-timestep run already exists.

        Args:
            force: Bypass freshness guard.
            lat_min/lat_max/lon_min/lon_max: Viewport bounds (defaults to class globals).
        """
        source = "cmems_current"
        if not force and self._has_multistep_run(source):
            logger.debug("Skipping current ingestion — multi-timestep run exists in DB")
            return

        _lat_min = lat_min if lat_min is not None else self.CMEMS_DEFAULT_LAT_MIN
        _lat_max = lat_max if lat_max is not None else self.CMEMS_DEFAULT_LAT_MAX
        _lon_min = lon_min if lon_min is not None else self.CMEMS_DEFAULT_LON_MIN
        _lon_max = lon_max if lon_max is not None else self.CMEMS_DEFAULT_LON_MAX

        logger.info("CMEMS current forecast ingestion starting")
        try:
            result = self.copernicus_provider.fetch_current_forecast(
                _lat_min, _lat_max, _lon_min, _lon_max,
            )
            if not result:
                logger.warning("CMEMS current forecast fetch returned empty")
                return
            self.ingest_current_forecast_frames(result)
        except Exception as e:
            logger.error(f"Current forecast ingestion failed: {e}")

    def ingest_ice(self, force: bool = False,
                   lat_min: Optional[float] = None, lat_max: Optional[float] = None,
                   lon_min: Optional[float] = None, lon_max: Optional[float] = None):
        """Fetch CMEMS ice forecast (10-day daily) and store all frames.

        Downloads multi-timestep ice concentration forecast (0, 24, 48, ..., 216h).
        Skips if a recent multi-timestep run already exists.

        Args:
            force: Bypass freshness guard.
            lat_min/lat_max/lon_min/lon_max: Viewport bounds (defaults to class globals).
        """
        source = "cmems_ice"
        if not force and self._has_multistep_run(source):
            logger.debug("Skipping ice ingestion — multi-timestep run exists in DB")
            return

        _lat_min = lat_min if lat_min is not None else self.ICE_DEFAULT_LAT_MIN
        _lat_max = lat_max if lat_max is not None else self.ICE_DEFAULT_LAT_MAX
        _lon_min = lon_min if lon_min is not None else self.ICE_DEFAULT_LON_MIN
        _lon_max = lon_max if lon_max is not None else self.ICE_DEFAULT_LON_MAX

        logger.info("CMEMS ice forecast ingestion starting")
        try:
            result = self.copernicus_provider.fetch_ice_forecast(
                _lat_min, _lat_max, _lon_min, _lon_max,
            )
            if not result:
                logger.warning("CMEMS ice forecast fetch returned empty")
                return
            self.ingest_ice_forecast_frames(result)
        except Exception as e:
            logger.error(f"Ice forecast ingestion failed: {e}")

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
        old_hour_count = self._count_best_run_hours(source)
        conn = self._get_conn()
        try:
            cur = conn.cursor()

            # Create forecast run record (defer supersede until we know new run is better)
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

            # Only supersede old runs if new run has at least 70% of old frame count
            min_required = max(1, int(old_hour_count * 0.7))
            if ingested_count >= min_required:
                cur.execute(
                    """UPDATE weather_forecast_runs SET status = 'superseded'
                       WHERE source = %s AND status = 'complete' AND id != %s""",
                    (source, run_id),
                )
                status = "complete" if ingested_count > 0 else "failed"
            else:
                status = "failed"
                logger.warning(
                    f"Wave: new run ({ingested_count}h) < min required ({min_required}h from old {old_hour_count}h) — keeping old"
                )

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
        old_hour_count = self._count_best_run_hours(source)
        conn = self._get_conn()
        try:
            cur = conn.cursor()

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

            min_required = max(1, int(old_hour_count * 0.7))
            if ingested_count >= min_required:
                cur.execute(
                    """UPDATE weather_forecast_runs SET status = 'superseded'
                       WHERE source = %s AND status = 'complete' AND id != %s""",
                    (source, run_id),
                )
                status = "complete" if ingested_count > 0 else "failed"
            else:
                status = "failed"
                logger.warning(
                    f"Current: new run ({ingested_count}h) < min required ({min_required}h from old {old_hour_count}h) — keeping old"
                )

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
        old_hour_count = self._count_best_run_hours(source)
        conn = self._get_conn()
        try:
            cur = conn.cursor()

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

            min_required = max(1, int(old_hour_count * 0.7))
            if ingested_count >= min_required:
                cur.execute(
                    """UPDATE weather_forecast_runs SET status = 'superseded'
                       WHERE source = %s AND status = 'complete' AND id != %s""",
                    (source, run_id),
                )
                status = "complete" if ingested_count > 0 else "failed"
            else:
                status = "failed"
                logger.warning(
                    f"Ice: new run ({ingested_count}h) < min required ({min_required}h from old {old_hour_count}h) — keeping old"
                )

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

    def ingest_sst(self, force: bool = False,
                   lat_min: Optional[float] = None, lat_max: Optional[float] = None,
                   lon_min: Optional[float] = None, lon_max: Optional[float] = None):
        """Fetch CMEMS SST forecast (0-120h, 3-hourly) and store in PostgreSQL.

        Downloads viewport-bounded data via open_dataset() at native 0.083° —
        same pattern as waves/currents/ice.

        Args:
            force: Bypass freshness guard.
            lat_min/lat_max/lon_min/lon_max: Viewport bounds (defaults to class globals).
        """
        source = "cmems_sst"
        if not force and self._has_multistep_run(source):
            logger.debug("Skipping SST ingestion — multi-timestep run exists in DB")
            return

        _lat_min = lat_min if lat_min is not None else self.SST_DEFAULT_LAT_MIN
        _lat_max = lat_max if lat_max is not None else self.SST_DEFAULT_LAT_MAX
        _lon_min = lon_min if lon_min is not None else self.SST_DEFAULT_LON_MIN
        _lon_max = lon_max if lon_max is not None else self.SST_DEFAULT_LON_MAX

        logger.info("CMEMS SST forecast ingestion starting")
        try:
            result = self.copernicus_provider.fetch_sst_forecast(
                _lat_min, _lat_max, _lon_min, _lon_max,
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
        old_hour_count = self._count_best_run_hours(source)
        conn = self._get_conn()
        try:
            cur = conn.cursor()

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

            min_required = max(1, int(old_hour_count * 0.7))
            if ingested_count >= min_required:
                cur.execute(
                    """UPDATE weather_forecast_runs SET status = 'superseded'
                       WHERE source = %s AND status = 'complete' AND id != %s""",
                    (source, run_id),
                )
                status = "complete" if ingested_count > 0 else "failed"
            else:
                status = "failed"
                logger.warning(
                    f"SST: new run ({ingested_count}h) < min required ({min_required}h from old {old_hour_count}h) — keeping old"
                )

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
        old_hour_count = self._count_best_run_hours(source)
        conn = self._get_conn()
        try:
            cur = conn.cursor()

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

            min_required = max(1, int(old_hour_count * 0.7))
            if ingested_count >= min_required:
                cur.execute(
                    """UPDATE weather_forecast_runs SET status = 'superseded'
                       WHERE source = %s AND status = 'complete' AND id != %s""",
                    (source, run_id),
                )
                status = "complete" if ingested_count > 0 else "failed"
            else:
                status = "failed"
                logger.warning(
                    f"Visibility: new run ({ingested_count}h) < min required ({min_required}h from old {old_hour_count}h) — keeping old"
                )

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

    def _supersede_old_runs(self, source: str | None = None):
        """Mark runs older than 24h as 'superseded'.

        Args:
            source: If provided, only supersede runs for this source.
                    If None, supersede across all sources (legacy behavior).
        """
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
            if source is not None:
                cur.execute(
                    """UPDATE weather_forecast_runs
                       SET status = 'superseded'
                       WHERE status = 'complete'
                         AND source = %s
                         AND ingested_at < %s""",
                    (source, cutoff),
                )
            else:
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
                scope = source or "all"
                logger.info(f"Superseded {superseded} old weather runs (scope={scope})")
        except Exception as e:
            logger.error(f"Failed to supersede old runs: {e}")
            conn.rollback()
        finally:
            conn.close()

    def cleanup_orphaned_grid_data(self, source: str | None = None):
        """Delete grid data rows belonging to superseded, failed, or stale ingesting runs.

        This reclaims TOAST storage from dead runs. Should be called after
        _supersede_old_runs() in each ingestion cycle.

        Args:
            source: If provided, only clean up runs for this source.
                    If None, clean up across all sources (legacy behavior).

        Note: PostgreSQL does not release disk space from TOAST deletes until
        autovacuum runs (or manual VACUUM). Large deletes may cause temporary
        I/O spikes.
        """
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            source_clause = "AND source = %s" if source else ""
            source_params: tuple = (source,) if source else ()

            # Delete grid data for superseded and failed runs
            cur.execute(
                f"""DELETE FROM weather_grid_data
                   WHERE run_id IN (
                       SELECT id FROM weather_forecast_runs
                       WHERE status IN ('superseded', 'failed')
                       {source_clause}
                   )""",
                source_params,
            )
            deleted_dead = cur.rowcount

            # Delete grid data for ingesting runs older than 6h (stale/abandoned)
            cutoff = datetime.now(timezone.utc) - timedelta(hours=6)
            cur.execute(
                f"""DELETE FROM weather_grid_data
                   WHERE run_id IN (
                       SELECT id FROM weather_forecast_runs
                       WHERE status = 'ingesting'
                         AND ingested_at < %s
                         {source_clause}
                   )""",
                (cutoff, *source_params),
            )
            deleted_stale = cur.rowcount

            # Now delete the orphaned run metadata too
            cur.execute(
                f"""DELETE FROM weather_forecast_runs
                   WHERE status IN ('superseded', 'failed')
                   {source_clause}""",
                source_params,
            )
            deleted_runs_dead = cur.rowcount

            cur.execute(
                f"""DELETE FROM weather_forecast_runs
                   WHERE status = 'ingesting'
                     AND ingested_at < %s
                     {source_clause}""",
                (cutoff, *source_params),
            )
            deleted_runs_stale = cur.rowcount

            conn.commit()
            total_grids = deleted_dead + deleted_stale
            total_runs = deleted_runs_dead + deleted_runs_stale
            if total_grids > 0 or total_runs > 0:
                scope = source or "all"
                logger.info(
                    f"Orphan cleanup (scope={scope}): deleted {total_grids} grid rows "
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

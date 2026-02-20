"""
Generic forecast layer manager for WINDMAR weather prefetch pipelines.

Eliminates ~2,000 LOC of near-identical boilerplate across 6 weather layers
(wind, wave, current, ice, sst, visibility) by providing:
  - Per-layer state management (running flag, threading lock, Redis sync)
  - File-based cache I/O (atomic writes, JSON)
  - Shared utility functions (cache completeness, bounds coverage, covering cache)
  - Prefetch wrapper with lock + Redis acquire/release
  - Trigger/frames endpoint helpers

Each layer supplies its own fetch + frame-building logic as a callback.
"""
import json
import logging
import re
import threading
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_CACHE_ROOT = Path("/tmp/windmar_cache")


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def is_cache_complete(cache_file: Path) -> bool:
    """Check if a forecast cache file has all expected frames.

    Reads only the first 512 bytes to extract cached_hours / total_hours
    without parsing the entire multi-MB JSON payload.  Returns True if
    the file looks complete (or if the check is inconclusive).
    """
    try:
        with open(cache_file, "rb") as fh:
            header = fh.read(512)
        m_cached = re.search(rb'"cached_hours"\s*:\s*(\d+)', header)
        m_total = re.search(rb'"total_hours"\s*:\s*(\d+)', header)
        if m_cached and m_total:
            return int(m_cached.group(1)) >= int(m_total.group(1))
    except Exception:
        pass
    return True  # inconclusive → serve it


def cache_covers_bounds(
    cached_data: dict,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    min_coverage: float = 0.8,
) -> bool:
    """Check whether cached data covers the requested bounding box.

    Returns True if the cached lat/lon range covers at least *min_coverage*
    fraction of the requested span on both axes.
    """
    lats = cached_data.get("lats", [])
    lons = cached_data.get("lons", [])
    if not lats or not lons:
        return False
    data_lat_min, data_lat_max = min(lats), max(lats)
    data_lon_min, data_lon_max = min(lons), max(lons)

    req_lat_span = max(lat_max - lat_min, 0.01)
    req_lon_span = max(lon_max - lon_min, 0.01)

    lat_overlap = max(0, min(data_lat_max, lat_max) - max(data_lat_min, lat_min))
    lon_overlap = max(0, min(data_lon_max, lon_max) - max(data_lon_min, lon_min))

    lat_cov = lat_overlap / req_lat_span
    lon_cov = lon_overlap / req_lon_span

    covers = lat_cov >= min_coverage and lon_cov >= min_coverage
    if not covers:
        logger.info(
            f"Cache coverage insufficient: lat {lat_cov:.0%} lon {lon_cov:.0%} "
            f"(need {min_coverage:.0%}). cached=[{data_lat_min:.1f}-{data_lat_max:.1f},"
            f"{data_lon_min:.1f}-{data_lon_max:.1f}] requested=[{lat_min:.1f}-{lat_max:.1f},"
            f"{lon_min:.1f}-{lon_max:.1f}]"
        )
    return covers


def find_covering_cache(
    cache_dir: Path,
    prefix: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> Optional[Path]:
    """Find a cached JSON file whose bounds fully cover the requested viewport.

    Cache filenames follow the pattern: {prefix}_{lat_min}_{lat_max}_{lon_min}_{lon_max}.json
    Returns the first file whose bounds enclose the requested area, or None.
    """
    pattern = re.compile(rf"^{re.escape(prefix)}_(-?\d+)_(-?\d+)_(-?\d+)_(-?\d+)\.json$")
    for f in cache_dir.glob(f"{prefix}_*.json"):
        m = pattern.match(f.name)
        if not m:
            continue
        c_lat_min, c_lat_max, c_lon_min, c_lon_max = (float(m.group(i)) for i in range(1, 5))
        if c_lat_min <= lat_min and c_lat_max >= lat_max and c_lon_min <= lon_min and c_lon_max >= lon_max:
            return f
    return None


# ---------------------------------------------------------------------------
# ForecastLayerManager
# ---------------------------------------------------------------------------

class ForecastLayerManager:
    """Per-layer state + cache management for forecast prefetch pipelines.

    Each weather layer (wind, wave, current, ice, sst, visibility) gets one
    instance.  The manager handles thread locking, Redis distributed sync,
    file-based JSON cache I/O, and prefetch lifecycle — leaving only the
    per-layer fetch + frame-building logic to the caller.
    """

    def __init__(self, name: str, *, cache_subdir: str = None, use_redis: bool = True):
        self.name = name
        self._cache_dir = _CACHE_ROOT / (cache_subdir or name)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._running = False
        self._lock: Optional[threading.Lock] = None
        self._use_redis = use_redis
        self._redis_lock_key = f"windmar:{name}_prefetch_lock"
        self._redis_status_key = f"windmar:{name}_prefetch_running"
        # Wind-specific: last successful (run_date, run_hour)
        self.last_run = None

    # -- state management --------------------------------------------------

    def get_lock(self) -> threading.Lock:
        if self._lock is None:
            self._lock = threading.Lock()
        return self._lock

    @property
    def is_running(self) -> bool:
        """Check if prefetch is running (Redis first, local fallback)."""
        if self._use_redis:
            from api.weather_service import _get_redis
            r = _get_redis()
            if r is not None:
                try:
                    return r.exists(self._redis_status_key) > 0
                except Exception:
                    pass
        return self._running

    @is_running.setter
    def is_running(self, value: bool):
        self._running = value

    # -- cache I/O ---------------------------------------------------------

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    def make_cache_key(self, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> str:
        return f"{self.name}_{lat_min:.0f}_{lat_max:.0f}_{lon_min:.0f}_{lon_max:.0f}"

    def cache_path(self, cache_key: str) -> Path:
        return self._cache_dir / f"{cache_key}.json"

    def cache_get(self, cache_key: str) -> Optional[dict]:
        p = self.cache_path(cache_key)
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                return None
        return None

    def cache_put(self, cache_key: str, data: dict) -> None:
        p = self.cache_path(cache_key)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.rename(p)  # atomic on same filesystem

    # -- prefetch lifecycle ------------------------------------------------

    def run_prefetch(
        self,
        do_fn: Callable[["ForecastLayerManager", float, float, float, float], None],
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> None:
        """Execute a prefetch function with full lock + Redis sync lifecycle.

        ``do_fn(manager, lat_min, lat_max, lon_min, lon_max)`` contains the
        per-layer logic (fetch, build frames, cache, ingest).  This wrapper
        handles: thread lock, Redis distributed lock, running-flag management,
        and cleanup on completion/error.
        """
        pflock = self.get_lock()
        if not pflock.acquire(blocking=False):
            return

        r = None
        if self._use_redis:
            from api.weather_service import _get_redis
            r = _get_redis()

        try:
            if r is not None:
                acquired = r.set(self._redis_lock_key, "1", nx=True, ex=1200)
                if not acquired:
                    return
                r.setex(self._redis_status_key, 1200, "1")

            self._running = True
            do_fn(self, lat_min, lat_max, lon_min, lon_max)

        except Exception as e:
            logger.error(f"{self.name.capitalize()} forecast prefetch failed: {e}")
        finally:
            self._running = False
            if r is not None:
                try:
                    r.delete(self._redis_lock_key, self._redis_status_key)
                except Exception:
                    pass
            pflock.release()

    # -- endpoint helpers --------------------------------------------------

    def trigger_response(
        self,
        background_tasks,
        do_fn: Callable,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> dict:
        """Standard trigger-prefetch response: check running → lock → start."""
        if self.is_running:
            return {"status": "already_running", "message": f"{self.name.capitalize()} prefetch is already in progress"}

        lock = self.get_lock()
        if not lock.acquire(blocking=False):
            return {"status": "already_running", "message": f"{self.name.capitalize()} prefetch is already in progress"}
        lock.release()

        background_tasks.add_task(self.run_prefetch, do_fn, lat_min, lat_max, lon_min, lon_max)
        return {"status": "started", "message": f"{self.name.capitalize()} forecast prefetch triggered in background"}

    def serve_frames_file(
        self,
        cache_key: str,
        lat_min: float = 0,
        lat_max: float = 0,
        lon_min: float = 0,
        lon_max: float = 0,
        use_covering: bool = False,
    ):
        """Try to serve cached frames as a raw Response.  Returns None if no file found.

        Caller is responsible for the DB-rebuild fallback.
        """
        from starlette.responses import Response as RawResponse

        cache_file = self.cache_path(cache_key)
        if not cache_file.exists() and use_covering:
            cache_file = find_covering_cache(self._cache_dir, self.name, lat_min, lat_max, lon_min, lon_max)

        if cache_file and cache_file.exists():
            if is_cache_complete(cache_file):
                return RawResponse(content=cache_file.read_bytes(), media_type="application/json")
            logger.warning("%s cache %s is partial — removing", self.name.capitalize(), cache_file.name)
            cache_file.unlink(missing_ok=True)

        return None


# ---------------------------------------------------------------------------
# Layer instances (module-level singletons)
# ---------------------------------------------------------------------------

wind_layer = ForecastLayerManager("wind", use_redis=False)
wave_layer = ForecastLayerManager("wave")
current_layer = ForecastLayerManager("current")
ice_layer = ForecastLayerManager("ice")
sst_layer = ForecastLayerManager("sst")
vis_layer = ForecastLayerManager("vis")

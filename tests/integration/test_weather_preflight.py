"""
Pre-production weather preflight test suite.

Runs sequentially against the LIVE API (default http://localhost:8003)
to verify that every weather endpoint returns valid data before deploying
to production.

Usage:
    # Against local containers (default)
    pytest tests/integration/test_weather_preflight.py -v

    # Against a custom host
    WINDMAR_API_URL=https://demo-windmar.slmar.co \
        pytest tests/integration/test_weather_preflight.py -v

    # Standalone (no pytest)
    python tests/integration/test_weather_preflight.py

Requires a running Windmar API with ingested weather data.
Skipped in CI (no live API available).
"""

import gzip
import json
import os
import sys

import httpx
import pytest

API_URL = os.environ.get("WINDMAR_API_URL", "http://localhost:8003")
TIMEOUT = 30.0

# Viewport inside the union bbox of ADRS 1+2 + ADRS 4 (28-72N, -30-42E)
_VIEWPORT = {"lat_min": 40, "lat_max": 60, "lon_min": -10, "lon_max": 20}

# All 7 registered weather fields
_ALL_FIELDS = ("wind", "waves", "swell", "currents", "sst", "visibility", "ice")

# Fields expected to have data when ADRS 1+2 + ADRS 4 are selected
# (ice has no bbox for these areas → not_applicable)
_DATA_FIELDS = ("wind", "waves", "swell", "currents", "sst", "visibility")

# Minimum frames expected per field
_MIN_FRAMES = {"ice": 9}
_DEFAULT_MIN_FRAMES = 41


def _api_available() -> bool:
    """Check if the live API is reachable."""
    try:
        r = httpx.get(f"{API_URL}/api/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


skip_no_api = pytest.mark.skipif(
    not _api_available(),
    reason=f"Live API not reachable at {API_URL}",
)


def _get(path: str, params: dict = None) -> httpx.Response:
    """GET helper with timeout and Accept header to avoid gzip responses."""
    return httpx.get(
        f"{API_URL}{path}",
        params=params,
        headers={"Accept-Encoding": "identity"},
        timeout=TIMEOUT,
    )


def _json(resp: httpx.Response) -> dict:
    """Parse response as JSON.

    httpx auto-decompresses gzip, so resp.text is always plain JSON.
    The server may send pre-compressed .gz files with Content-Encoding: gzip
    but httpx handles this transparently.
    """
    return json.loads(resp.text)


# ============================================================================
# 1. Health & readiness
# ============================================================================


@skip_no_api
class TestHealthAndReadiness:
    """Verify the API is healthy and weather data is ready."""

    def test_api_health(self):
        resp = _get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("healthy", "degraded")
        assert data["components"]["database"]["status"] == "healthy"
        assert data["components"]["redis"]["status"] == "healthy"

    def test_weather_health(self):
        resp = _get("/api/weather/health")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    def test_readiness_structure(self):
        resp = _get("/api/weather/readiness")
        assert resp.status_code == 200
        data = resp.json()

        # Top-level keys
        assert "global_fields" in data
        assert "areas" in data
        assert "all_ready" in data
        assert "selected_areas" in data
        assert "available_areas" in data

        # Global fields present
        for gf in ("wind", "visibility"):
            assert gf in data["global_fields"]
            f = data["global_fields"][gf]
            assert "status" in f
            assert "frames" in f
            assert "expected" in f

        # At least one area selected
        assert len(data["selected_areas"]) >= 1

    def test_readiness_areas_have_fields(self):
        resp = _get("/api/weather/readiness")
        data = resp.json()

        for area_id, area in data["areas"].items():
            assert "label" in area
            assert "fields" in area
            assert "all_ready" in area
            for fname in ("waves", "currents", "sst", "ice"):
                assert fname in area["fields"], (
                    f"Missing {fname} in area {area_id}"
                )
                f = area["fields"][fname]
                assert f["status"] in ("ready", "missing", "not_applicable")


# ============================================================================
# 2. ADRS area configuration
# ============================================================================


@skip_no_api
class TestADRSAreas:
    """Verify ADRS area endpoints return correct structure."""

    def test_ocean_areas_list(self):
        resp = _get("/api/weather/ocean-areas")
        assert resp.status_code == 200
        data = resp.json()
        assert "areas" in data
        assert "selected" in data
        assert len(data["areas"]) >= 2

        for area in data["areas"]:
            assert "id" in area
            assert "label" in area
            assert "bbox" in area
            assert len(area["bbox"]) == 4
            assert "disabled" in area

    def test_selected_areas(self):
        resp = _get("/api/weather/selected-areas")
        assert resp.status_code == 200
        data = resp.json()
        assert "selected" in data
        assert isinstance(data["selected"], list)
        assert len(data["selected"]) >= 1

    def test_available_areas_have_adrs_ids(self):
        resp = _get("/api/weather/ocean-areas")
        data = resp.json()
        ids = {a["id"] for a in data["areas"]}
        assert "adrs_1_2" in ids
        assert "adrs_4" in ids


# ============================================================================
# 3. Per-field status
# ============================================================================


@skip_no_api
class TestFieldStatus:
    """Verify /api/weather/{field}/status for all fields."""

    @pytest.mark.parametrize("field", _ALL_FIELDS)
    def test_field_status_structure(self, field: str):
        resp = _get(f"/api/weather/{field}/status", params=_VIEWPORT)
        assert resp.status_code == 200
        data = resp.json()
        assert "total_hours" in data
        assert "cached_hours" in data
        assert "complete" in data
        assert "prefetch_running" in data

    @pytest.mark.parametrize("field", _DATA_FIELDS)
    def test_data_field_has_cached_hours(self, field: str):
        """Fields with ingested data should report cached_hours > 0."""
        resp = _get(f"/api/weather/{field}/status", params=_VIEWPORT)
        data = resp.json()
        assert data["cached_hours"] > 0, (
            f"{field} has 0 cached hours — data not ingested?"
        )


# ============================================================================
# 4. Per-field frames (the actual data payloads)
# ============================================================================


@skip_no_api
class TestFieldFrames:
    """Verify /api/weather/{field}/frames returns valid data."""

    @pytest.mark.parametrize("field", _DATA_FIELDS)
    def test_frames_response_ok(self, field: str):
        resp = _get(f"/api/weather/{field}/frames", params=_VIEWPORT)
        assert resp.status_code == 200
        data = _json(resp)
        assert "frames" in data
        assert "source" in data

    @pytest.mark.parametrize("field", _DATA_FIELDS)
    def test_frames_have_expected_count(self, field: str):
        resp = _get(f"/api/weather/{field}/frames", params=_VIEWPORT)
        data = _json(resp)
        frames = data.get("frames", {})
        expected = _MIN_FRAMES.get(field, _DEFAULT_MIN_FRAMES)
        assert len(frames) >= expected, (
            f"{field}: got {len(frames)} frames, expected >= {expected}"
        )

    @pytest.mark.parametrize("field", ("waves", "currents", "sst", "visibility"))
    def test_frames_have_lat_lon_grid(self, field: str):
        """CMEMS/GFS scalar/vector fields must include lats, lons, ny, nx."""
        resp = _get(f"/api/weather/{field}/frames", params=_VIEWPORT)
        data = _json(resp)
        assert "lats" in data, f"{field}: missing lats"
        assert "lons" in data, f"{field}: missing lons"
        assert "ny" in data, f"{field}: missing ny"
        assert "nx" in data, f"{field}: missing nx"
        assert len(data["lats"]) == data["ny"]
        assert len(data["lons"]) == data["nx"]

    @pytest.mark.parametrize("field", ("waves", "currents", "sst", "visibility"))
    def test_frames_bbox_covers_viewport(self, field: str):
        """Cached data must cover the requested viewport."""
        resp = _get(f"/api/weather/{field}/frames", params=_VIEWPORT)
        data = _json(resp)
        lats = data.get("lats", [])
        lons = data.get("lons", [])
        if not lats or not lons:
            pytest.skip(f"{field}: no lat/lon grid returned")

        assert min(lats) <= _VIEWPORT["lat_min"] + 1, (
            f"{field}: lat_min {min(lats)} doesn't cover viewport {_VIEWPORT['lat_min']}"
        )
        assert max(lats) >= _VIEWPORT["lat_max"] - 1, (
            f"{field}: lat_max {max(lats)} doesn't cover viewport {_VIEWPORT['lat_max']}"
        )
        assert min(lons) <= _VIEWPORT["lon_min"] + 1, (
            f"{field}: lon_min {min(lons)} doesn't cover viewport {_VIEWPORT['lon_min']}"
        )
        assert max(lons) >= _VIEWPORT["lon_max"] - 1, (
            f"{field}: lon_max {max(lons)} doesn't cover viewport {_VIEWPORT['lon_max']}"
        )

    def test_wind_frames_leaflet_velocity_format(self):
        """Wind uses leaflet-velocity format with header/data arrays."""
        resp = _get("/api/weather/wind/frames", params=_VIEWPORT)
        data = _json(resp)
        frames = data.get("frames", {})
        if not frames:
            pytest.fail("wind: no frames returned")

        first_key = next(iter(frames))
        frame = frames[first_key]
        assert isinstance(frame, list), "wind frame should be a list of components"
        assert len(frame) >= 2, "wind frame should have >= 2 components (u, v)"
        header = frame[0].get("header", {})
        assert "nx" in header
        assert "ny" in header
        assert "la1" in header
        assert "lo1" in header

    @pytest.mark.parametrize("field", ("waves",))
    def test_wave_frame_has_decomposition(self, field: str):
        """Wave frames should contain wave_hs, swell_hs, etc."""
        resp = _get(f"/api/weather/{field}/frames", params=_VIEWPORT)
        data = _json(resp)
        frames = data.get("frames", {})
        if not frames:
            pytest.fail(f"{field}: no frames returned")

        first_key = next(iter(frames))
        frame = frames[first_key]
        assert isinstance(frame, dict), f"{field}: frame should be a dict"
        # At minimum wave_hs must be present
        assert "data" in frame or "wave_hs" in frame, (
            f"{field}: frame missing data or wave_hs key"
        )


# ============================================================================
# 5. Per-field debug diagnostics
# ============================================================================


@skip_no_api
class TestFieldDebug:
    """Verify /api/weather/{field}/debug returns cache diagnostics."""

    @pytest.mark.parametrize("field", _DATA_FIELDS)
    def test_debug_response_structure(self, field: str):
        resp = _get(f"/api/weather/{field}/debug", params=_VIEWPORT)
        assert resp.status_code == 200
        data = resp.json()
        assert "field" in data
        assert data["field"] == field

        # If cache exists, validate diagnostic fields
        if data.get("status") != "no_cache":
            assert "frame_count" in data
            assert "checks" in data
            assert "all_checks_pass" in data

    @pytest.mark.parametrize("field", _DATA_FIELDS)
    def test_debug_checks_pass(self, field: str):
        """All cache consistency checks should pass."""
        resp = _get(f"/api/weather/{field}/debug", params=_VIEWPORT)
        data = resp.json()
        if data.get("status") == "no_cache":
            pytest.skip(f"{field}: no cache at this viewport")

        failed = [c for c in data.get("checks", []) if not c["pass"]]
        assert not failed, (
            f"{field}: cache checks failed: "
            + ", ".join(f"{c['check']}: {c['detail']}" for c in failed)
        )


# ============================================================================
# 6. Cross-field alignment
# ============================================================================


@skip_no_api
class TestCrossFieldAlignment:
    """Verify that different fields are spatially and temporally aligned."""

    def test_cmems_fields_share_bbox(self):
        """Waves, currents, and SST should cover the same geographic extent."""
        extents = {}
        for field in ("waves", "currents", "sst"):
            resp = _get(f"/api/weather/{field}/frames", params=_VIEWPORT)
            data = _json(resp)
            lats = data.get("lats", [])
            lons = data.get("lons", [])
            if not lats or not lons:
                pytest.skip(f"{field}: no grid data")
            extents[field] = {
                "lat_min": min(lats),
                "lat_max": max(lats),
                "lon_min": min(lons),
                "lon_max": max(lons),
            }

        fields = list(extents.keys())
        for i in range(len(fields) - 1):
            a, b = fields[i], fields[i + 1]
            ea, eb = extents[a], extents[b]
            # Allow 2° tolerance for different resolutions/subsampling
            assert abs(ea["lat_min"] - eb["lat_min"]) < 2, (
                f"{a} vs {b}: lat_min mismatch {ea['lat_min']} vs {eb['lat_min']}"
            )
            assert abs(ea["lat_max"] - eb["lat_max"]) < 2, (
                f"{a} vs {b}: lat_max mismatch {ea['lat_max']} vs {eb['lat_max']}"
            )
            assert abs(ea["lon_min"] - eb["lon_min"]) < 2, (
                f"{a} vs {b}: lon_min mismatch {ea['lon_min']} vs {eb['lon_min']}"
            )
            assert abs(ea["lon_max"] - eb["lon_max"]) < 2, (
                f"{a} vs {b}: lon_max mismatch {ea['lon_max']} vs {eb['lon_max']}"
            )

    def test_all_fields_have_same_frame_count(self):
        """All GFS-cadence fields should have 41 frames."""
        for field in ("wind", "waves", "currents", "visibility"):
            resp = _get(f"/api/weather/{field}/frames", params=_VIEWPORT)
            data = _json(resp)
            frames = data.get("frames", {})
            assert len(frames) == 41, (
                f"{field}: expected 41 frames, got {len(frames)}"
            )

    def test_ice_not_applicable_for_non_arctic_areas(self):
        """Ice should be not_applicable when no Arctic area is selected."""
        resp = _get("/api/weather/readiness")
        data = resp.json()
        for area_id, area in data["areas"].items():
            if area_id in ("adrs_1_2", "adrs_4"):
                ice = area["fields"].get("ice", {})
                assert ice.get("status") == "not_applicable", (
                    f"Ice should be not_applicable for {area_id}, got {ice.get('status')}"
                )


# ============================================================================
# 7. Freshness
# ============================================================================


@skip_no_api
class TestFreshness:
    """Verify weather freshness reporting."""

    def test_freshness_endpoint(self):
        resp = _get("/api/weather/freshness")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        # After ingestion, status should be "ok"
        if data["status"] == "ok":
            assert "age_hours" in data
            assert "color" in data
            assert data["color"] in ("green", "yellow", "red")


# ============================================================================
# 8. Resync status (non-destructive check)
# ============================================================================


@skip_no_api
class TestResyncStatus:
    """Verify resync status endpoint (read-only, no trigger)."""

    def test_resync_status(self):
        resp = _get("/api/weather/resync-status")
        assert resp.status_code == 200
        data = resp.json()
        assert "active" in data


# ============================================================================
# Standalone runner
# ============================================================================


def _run_standalone():
    """Run all checks as a standalone script with colored output."""
    passed = 0
    failed = 0
    skipped = 0

    if not _api_available():
        print(f"\n  SKIP  API not reachable at {API_URL}")
        print("  Set WINDMAR_API_URL or start containers first.\n")
        sys.exit(1)

    checks = [
        ("Health: API", lambda: _get("/api/health").status_code == 200),
        ("Health: Weather", lambda: _get("/api/weather/health").status_code == 200),
        ("Readiness: structure", lambda: "global_fields" in _get("/api/weather/readiness").json()),
        ("ADRS: ocean areas", lambda: len(_get("/api/weather/ocean-areas").json()["areas"]) >= 2),
        ("ADRS: selected areas", lambda: len(_get("/api/weather/selected-areas").json()["selected"]) >= 1),
    ]

    for field in _DATA_FIELDS:
        checks.append((
            f"Status: {field}",
            lambda f=field: _get(f"/api/weather/{f}/status", params=_VIEWPORT).json().get("cached_hours", 0) > 0,
        ))

    for field in _DATA_FIELDS:
        min_frames = _MIN_FRAMES.get(field, _DEFAULT_MIN_FRAMES)
        checks.append((
            f"Frames: {field} (>= {min_frames})",
            lambda f=field, mf=min_frames: len(
                _json(_get(f"/api/weather/{f}/frames", params=_VIEWPORT)).get("frames", {})
            ) >= mf,
        ))

    for field in _DATA_FIELDS:
        checks.append((
            f"Debug: {field} checks pass",
            lambda f=field: _get(f"/api/weather/{f}/debug", params=_VIEWPORT).json().get("all_checks_pass", False)
            or _get(f"/api/weather/{f}/debug", params=_VIEWPORT).json().get("status") == "no_cache",
        ))

    print(f"\n  Windmar Weather Preflight — {API_URL}\n")
    print(f"  {'Check':<40} Result")
    print(f"  {'─' * 40} ──────")

    for name, check_fn in checks:
        try:
            ok = check_fn()
            if ok:
                print(f"  {name:<40} PASS")
                passed += 1
            else:
                print(f"  {name:<40} FAIL")
                failed += 1
        except Exception as e:
            print(f"  {name:<40} ERROR: {e}")
            failed += 1

    print(f"\n  ─────────────────────────────────────────────────")
    print(f"  Total: {passed + failed + skipped}  Passed: {passed}  Failed: {failed}  Skipped: {skipped}")

    if failed:
        print(f"\n  PREFLIGHT FAILED — do NOT deploy.\n")
        sys.exit(1)
    else:
        print(f"\n  PREFLIGHT PASSED — safe to deploy.\n")
        sys.exit(0)


if __name__ == "__main__":
    _run_standalone()

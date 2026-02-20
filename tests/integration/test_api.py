"""
Integration tests for WINDMAR API.

Tests major API endpoints after the router-split refactoring.
Fixtures (db, client) provided by tests/conftest.py.
"""
import os

import pytest


# ============================================================================
# Public Endpoint Tests
# ============================================================================

def test_root_endpoint(client):
    """Test API root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "WINDMAR API"
    assert data["version"] == "2.1.0"
    assert data["status"] == "operational"


def test_health_check(client):
    """Test health check endpoint returns valid structure.

    In test env Redis is unavailable so status may be 'unhealthy' or 'degraded'.
    """
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded", "unhealthy")
    assert "timestamp" in data


# ============================================================================
# Vessel Endpoint Tests
# ============================================================================

def test_get_default_vessel_specs(client):
    """Test getting default vessel specifications."""
    response = client.get("/api/vessel/specs")
    assert response.status_code == 200
    data = response.json()
    assert data["dwt"] > 0
    assert data["loa"] > 0
    assert data["beam"] > 0


def test_update_vessel_specs(client):
    """Test updating vessel specifications."""
    vessel_data = {
        "dwt": 50000.0,
        "loa": 185.0,
        "beam": 32.5,
        "draft_laden": 12.0,
        "draft_ballast": 6.8,
        "mcr_kw": 9000.0,
        "sfoc_at_mcr": 170.0,
        "service_speed_laden": 14.5,
        "service_speed_ballast": 15.0,
    }

    response = client.post("/api/vessel/specs", json=vessel_data)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"

    # Verify persistence
    response = client.get("/api/vessel/specs")
    data = response.json()
    assert data["dwt"] == 50000.0
    assert data["loa"] == 185.0


# ============================================================================
# Vessel Predict Tests (replaces /api/fuel/calculate)
# ============================================================================

def test_predict_calm_weather(client):
    """Test performance prediction in calm weather."""
    request_data = {
        "calm_speed_kts": 14.5,
        "is_laden": True,
    }

    response = client.post("/api/vessel/predict", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert data["stw_kts"] > 0
    assert data["fuel_per_day_mt"] > 0
    assert data["power_kw"] > 0
    assert data["mode"] == "calm_speed"


def test_predict_with_wind(client):
    """Test performance prediction with head wind."""
    # Head wind
    wind_response = client.post("/api/vessel/predict", json={
        "calm_speed_kts": 14.5,
        "is_laden": True,
        "wind_speed_kts": 20.0,
        "wind_relative_deg": 0.0,  # Head wind
    })
    assert wind_response.status_code == 200
    wind_data = wind_response.json()
    assert wind_data["fuel_per_day_mt"] > 0

    # Calm reference
    calm_response = client.post("/api/vessel/predict", json={
        "calm_speed_kts": 14.5,
        "is_laden": True,
    })
    calm_data = calm_response.json()

    # Head wind should increase fuel consumption
    assert wind_data["fuel_per_day_mt"] > calm_data["fuel_per_day_mt"], \
        "Head wind should increase fuel consumption"


def test_predict_with_waves(client):
    """Test performance prediction with waves."""
    request_data = {
        "calm_speed_kts": 14.5,
        "is_laden": True,
        "wind_speed_kts": 25.0,
        "wind_relative_deg": 0.0,
        "wave_height_m": 3.0,
        "wave_relative_deg": 0.0,
    }

    response = client.post("/api/vessel/predict", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert data["fuel_per_day_mt"] > 0
    assert data["resistance_breakdown_kn"]["waves"] > 0


def test_predict_engine_load_mode(client):
    """Test performance prediction in engine load mode."""
    request_data = {
        "engine_load_pct": 85.0,
        "is_laden": True,
    }

    response = client.post("/api/vessel/predict", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert data["stw_kts"] > 0
    assert data["mode"] == "engine_load"


# ============================================================================
# Model Status & Curves
# ============================================================================

def test_model_status(client):
    """Test vessel model status endpoint."""
    response = client.get("/api/vessel/model-status")
    assert response.status_code == 200
    data = response.json()
    assert "specifications" in data
    assert "calibration" in data
    assert "computed" in data


def test_model_curves(client):
    """Test vessel model curves endpoint."""
    response = client.get("/api/vessel/model-curves")
    assert response.status_code == 200
    data = response.json()
    assert "speed_range_kts" in data
    assert len(data["speed_range_kts"]) > 0


def test_fuel_scenarios(client):
    """Test vessel fuel scenarios endpoint."""
    response = client.get("/api/vessel/fuel-scenarios")
    assert response.status_code == 200
    data = response.json()
    assert "scenarios" in data
    assert len(data["scenarios"]) > 0


# ============================================================================
# Error Handling Tests
# ============================================================================

def test_404_endpoint(client):
    """Test 404 for non-existent endpoint."""
    response = client.get("/api/nonexistent")
    assert response.status_code == 404


# ============================================================================
# Authentication Tests (when enabled)
# ============================================================================

@pytest.mark.skipif(
    os.environ.get("AUTH_ENABLED") == "false",
    reason="Authentication is disabled in test environment",
)
def test_authentication_required(client):
    """Test that endpoints require authentication when enabled."""
    response = client.post("/api/vessel/specs", json={"dwt": 50000.0})
    assert response.status_code == 401


@pytest.mark.skipif(
    os.environ.get("AUTH_ENABLED") == "false",
    reason="Authentication is disabled in test environment",
)
def test_valid_api_key(client, api_key):
    """Test access with valid API key."""
    headers = {"X-API-Key": api_key}
    response = client.get("/api/vessel/specs", headers=headers)
    assert response.status_code == 200


@pytest.mark.skipif(
    os.environ.get("AUTH_ENABLED") == "false",
    reason="Authentication is disabled in test environment",
)
def test_invalid_api_key(client):
    """Test access with invalid API key."""
    headers = {"X-API-Key": "invalid_key"}
    response = client.post("/api/vessel/specs", json={"dwt": 50000.0}, headers=headers)
    assert response.status_code == 401

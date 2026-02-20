"""
Integration tests for WINDMAR API.

Tests all major API endpoints with authentication and database.
Fixtures (db, client, api_key, test_vessel) provided by tests/conftest.py.
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
    """Test health check endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
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


def test_create_vessel(client):
    """Test creating a new vessel."""
    vessel_data = {
        "name": "Test Tanker",
        "dwt": 50000.0,
        "loa": 185.0,
        "beam": 32.5,
        "draft_laden": 12.0,
        "draft_ballast": 6.8,
        "mcr_kw": 9000.0,
        "sfoc_at_mcr": 170.0,
        "service_speed_laden": 14.5,
        "service_speed_ballast": 15.0
    }

    response = client.post("/api/vessel/specs", json=vessel_data)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "vessel_id" in data


def test_list_vessels(client, test_vessel):
    """Test listing all vessels."""
    response = client.get("/api/vessels")
    assert response.status_code == 200
    data = response.json()
    assert "vessels" in data
    assert len(data["vessels"]) >= 1
    assert any(v["name"] == "Test Vessel" for v in data["vessels"])


def test_get_specific_vessel(client, test_vessel):
    """Test getting specific vessel specifications."""
    response = client.get(f"/api/vessel/specs?vessel_id={test_vessel.id}")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Test Vessel"
    assert data["dwt"] == 49000.0


# ============================================================================
# Fuel Calculation Tests
# ============================================================================

def test_fuel_calculation_calm_weather(client):
    """Test fuel calculation in calm weather."""
    request_data = {
        "speed_kts": 14.5,
        "is_laden": True,
        "distance_nm": 348.0,
        "wind_speed_ms": None
    }

    response = client.post("/api/fuel/calculate", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert data["fuel_mt"] > 0
    assert data["power_kw"] > 0
    assert data["time_hours"] > 0
    assert "fuel_breakdown" in data
    assert "resistance_breakdown_kn" in data


def test_fuel_calculation_with_wind(client):
    """Test fuel calculation with wind."""
    request_data = {
        "speed_kts": 14.5,
        "is_laden": True,
        "distance_nm": 348.0,
        "wind_speed_ms": 10.0,
        "wind_dir_deg": 0.0,
        "heading_deg": 0.0
    }

    response = client.post("/api/fuel/calculate", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert data["fuel_mt"] > 0

    # Fuel with head wind should be higher than calm
    calm_response = client.post("/api/fuel/calculate", json={
        "speed_kts": 14.5,
        "is_laden": True,
        "distance_nm": 348.0
    })
    calm_fuel = calm_response.json()["fuel_mt"]
    assert data["fuel_mt"] > calm_fuel, "Head wind should increase fuel consumption"


def test_fuel_calculation_with_waves(client):
    """Test fuel calculation with waves."""
    request_data = {
        "speed_kts": 14.5,
        "is_laden": True,
        "distance_nm": 348.0,
        "wind_speed_ms": 12.5,
        "wind_dir_deg": 0.0,
        "heading_deg": 0.0,
        "sig_wave_height_m": 3.0,
        "wave_dir_deg": 0.0
    }

    response = client.post("/api/fuel/calculate", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert data["fuel_mt"] > 0
    assert data["fuel_breakdown"]["waves"] > 0


def test_fuel_calculation_invalid_speed(client):
    """Test fuel calculation with invalid speed."""
    request_data = {
        "speed_kts": 30.0,  # Too high
        "is_laden": True,
        "distance_nm": 348.0
    }

    response = client.post("/api/fuel/calculate", json=request_data)
    assert response.status_code == 422  # Validation error


# ============================================================================
# Route Optimization Tests
# ============================================================================

def test_route_optimization_great_circle(client):
    """Test route optimization using great circle (no weather)."""
    request_data = {
        "start": {
            "latitude": 51.5,
            "longitude": -0.1
        },
        "end": {
            "latitude": 40.7,
            "longitude": -74.0
        },
        "use_weather": False,
        "is_laden": True
    }

    response = client.post("/api/routes/optimize", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "route_id" in data
    assert len(data["waypoints"]) >= 2
    assert data["total_distance_nm"] > 0
    assert data["total_time_hours"] > 0
    assert data["total_fuel_mt"] > 0
    assert data["optimization_method"] == "Great circle"


def test_route_optimization_with_vessel(client, test_vessel):
    """Test route optimization with specific vessel."""
    request_data = {
        "start": {
            "latitude": 51.5,
            "longitude": -0.1
        },
        "end": {
            "latitude": 48.8,
            "longitude": 2.3
        },
        "vessel_id": str(test_vessel.id),
        "use_weather": False,
        "is_laden": True
    }

    response = client.post("/api/routes/optimize", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert data["total_distance_nm"] > 0


def test_route_optimization_invalid_coordinates(client):
    """Test route optimization with invalid coordinates."""
    request_data = {
        "start": {
            "latitude": 91.0,  # Invalid
            "longitude": -0.1
        },
        "end": {
            "latitude": 40.7,
            "longitude": -74.0
        },
        "use_weather": False
    }

    response = client.post("/api/routes/optimize", json=request_data)
    assert response.status_code == 422  # Validation error


def test_list_routes(client, test_vessel):
    """Test listing routes."""
    # First create a route
    request_data = {
        "start": {"latitude": 51.5, "longitude": -0.1},
        "end": {"latitude": 48.8, "longitude": 2.3},
        "vessel_id": str(test_vessel.id),
        "use_weather": False
    }
    client.post("/api/routes/optimize", json=request_data)

    # Then list routes
    response = client.get("/api/routes")
    assert response.status_code == 200
    data = response.json()
    assert "routes" in data
    assert len(data["routes"]) >= 1


def test_list_routes_by_vessel(client, test_vessel):
    """Test listing routes filtered by vessel."""
    # Create a route for the test vessel
    request_data = {
        "start": {"latitude": 51.5, "longitude": -0.1},
        "end": {"latitude": 48.8, "longitude": 2.3},
        "vessel_id": str(test_vessel.id),
        "use_weather": False
    }
    client.post("/api/routes/optimize", json=request_data)

    # List routes for this vessel
    response = client.get(f"/api/routes?vessel_id={test_vessel.id}")
    assert response.status_code == 200
    data = response.json()
    assert "routes" in data
    assert all(r["vessel_id"] == str(test_vessel.id) for r in data["routes"] if r["vessel_id"])


# ============================================================================
# Error Handling Tests
# ============================================================================

def test_404_endpoint(client):
    """Test 404 for non-existent endpoint."""
    response = client.get("/api/nonexistent")
    assert response.status_code == 404


def test_nonexistent_vessel(client):
    """Test getting non-existent vessel."""
    fake_id = "00000000-0000-0000-0000-000000000000"
    response = client.get(f"/api/vessel/specs?vessel_id={fake_id}")
    assert response.status_code == 500  # Will be caught by error handler


# ============================================================================
# Authentication Tests (when enabled)
# ============================================================================

@pytest.mark.skipif(
    os.environ.get("AUTH_ENABLED") == "false",
    reason="Authentication is disabled in test environment"
)
def test_authentication_required(client):
    """Test that endpoints require authentication when enabled."""
    response = client.get("/api/vessels")
    assert response.status_code == 401


@pytest.mark.skipif(
    os.environ.get("AUTH_ENABLED") == "false",
    reason="Authentication is disabled in test environment"
)
def test_valid_api_key(client, api_key):
    """Test access with valid API key."""
    headers = {"X-API-Key": api_key}
    response = client.get("/api/vessels", headers=headers)
    assert response.status_code == 200


@pytest.mark.skipif(
    os.environ.get("AUTH_ENABLED") == "false",
    reason="Authentication is disabled in test environment"
)
def test_invalid_api_key(client):
    """Test access with invalid API key."""
    headers = {"X-API-Key": "invalid_key"}
    response = client.get("/api/vessels", headers=headers)
    assert response.status_code == 401

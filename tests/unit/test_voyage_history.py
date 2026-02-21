"""
Unit tests for Voyage History module.

Tests CRUD operations, noon report generation, departure/arrival reports,
and PDF generation.
"""

import uuid
from datetime import datetime, timedelta

import pytest

from api.models import Voyage, VoyageLeg
from api.reports.noon_reports import generate_noon_reports
from api.reports.templates import build_departure_report, build_arrival_report


# ============================================================================
# Fixtures
# ============================================================================


def _make_voyage(db, n_legs=5, voyage_hours=120, name="Test Voyage"):
    """Create a test voyage with legs spanning multiple days."""
    departure = datetime(2026, 3, 1, 8, 0)
    hours_per_leg = voyage_hours / n_legs
    distance_per_leg = 100.0
    fuel_per_leg = 5.0

    voyage = Voyage(
        name=name,
        departure_port="Rotterdam",
        arrival_port="Singapore",
        departure_time=departure,
        arrival_time=departure + timedelta(hours=voyage_hours),
        total_distance_nm=distance_per_leg * n_legs,
        total_time_hours=voyage_hours,
        total_fuel_mt=fuel_per_leg * n_legs,
        avg_sog_kts=round(distance_per_leg * n_legs / voyage_hours, 1),
        avg_stw_kts=round(distance_per_leg * n_legs / voyage_hours + 0.3, 1),
        calm_speed_kts=14.0,
        is_laden=True,
        vessel_specs_snapshot={"name": "Test Vessel", "deadweight": 49000},
        cii_estimate={"rating": "C", "attained_cii": 5.12, "required_cii": 5.50},
        notes="Test voyage notes",
    )

    cum_hours = 0.0
    for i in range(n_legs):
        leg = VoyageLeg(
            leg_index=i,
            from_name=f"WP{i}",
            from_lat=51.9 - i * 2,
            from_lon=4.5 + i * 10,
            to_name=f"WP{i + 1}",
            to_lat=51.9 - (i + 1) * 2,
            to_lon=4.5 + (i + 1) * 10,
            distance_nm=distance_per_leg,
            bearing_deg=120.0 + i * 5,
            wind_speed_kts=15.0 + i,
            wind_dir_deg=225.0,
            wave_height_m=1.5 + i * 0.2,
            wave_dir_deg=220.0,
            current_speed_ms=0.3,
            current_dir_deg=180.0,
            calm_speed_kts=14.0,
            stw_kts=13.5 - i * 0.1,
            sog_kts=12.8 - i * 0.15,
            speed_loss_pct=5.0 + i,
            time_hours=hours_per_leg,
            departure_time=departure + timedelta(hours=cum_hours),
            arrival_time=departure + timedelta(hours=cum_hours + hours_per_leg),
            fuel_mt=fuel_per_leg,
            power_kw=5000.0,
            data_source="forecast",
        )
        voyage.legs.append(leg)
        cum_hours += hours_per_leg

    db.add(voyage)
    db.commit()
    db.refresh(voyage)
    return voyage


# ============================================================================
# CRUD Tests
# ============================================================================


class TestVoyageCRUD:
    """Test voyage save, list, get, delete via API."""

    def test_save_voyage(self, client, api_key):
        """POST /api/voyages saves a voyage and returns summary."""
        payload = {
            "name": "Rotterdam-Singapore",
            "departure_port": "Rotterdam",
            "arrival_port": "Singapore",
            "departure_time": "2026-03-01T08:00:00",
            "arrival_time": "2026-03-06T08:00:00",
            "total_distance_nm": 500.0,
            "total_time_hours": 120.0,
            "total_fuel_mt": 25.0,
            "avg_sog_kts": 4.2,
            "avg_stw_kts": 4.5,
            "calm_speed_kts": 14.0,
            "is_laden": True,
            "vessel_specs_snapshot": {"name": "Test Vessel", "deadweight": 49000},
            "cii_estimate": {"rating": "C", "attained_cii": 5.12},
            "notes": "Test voyage",
            "legs": [
                {
                    "leg_index": 0,
                    "from_name": "WP0",
                    "from_lat": 51.9,
                    "from_lon": 4.5,
                    "to_name": "WP1",
                    "to_lat": 49.9,
                    "to_lon": 14.5,
                    "distance_nm": 250.0,
                    "time_hours": 60.0,
                    "fuel_mt": 12.5,
                    "sog_kts": 4.2,
                    "wind_speed_kts": 15.0,
                    "wave_height_m": 1.5,
                },
                {
                    "leg_index": 1,
                    "from_name": "WP1",
                    "from_lat": 49.9,
                    "from_lon": 14.5,
                    "to_name": "WP2",
                    "to_lat": 47.9,
                    "to_lon": 24.5,
                    "distance_nm": 250.0,
                    "time_hours": 60.0,
                    "fuel_mt": 12.5,
                    "sog_kts": 4.2,
                    "wind_speed_kts": 16.0,
                    "wave_height_m": 1.7,
                },
            ],
        }

        resp = client.post("/api/voyages", json=payload, headers={"X-API-Key": api_key})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["name"] == "Rotterdam-Singapore"
        assert data["total_distance_nm"] == 500.0
        assert "id" in data

    def test_list_voyages(self, client, db):
        """GET /api/voyages returns paginated list."""
        _make_voyage(db, name="Voyage A")
        _make_voyage(db, name="Voyage B")

        resp = client.get("/api/voyages")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["voyages"]) == 2
        # Ordered by departure_time desc (both same, order stable)
        assert all("id" in v for v in data["voyages"])

    def test_list_voyages_search(self, client, db):
        """GET /api/voyages?name=... filters by name."""
        _make_voyage(db, name="Atlantic Crossing")
        _make_voyage(db, name="Pacific Voyage")

        resp = client.get("/api/voyages", params={"name": "atlantic"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["voyages"][0]["name"] == "Atlantic Crossing"

    def test_get_voyage_detail(self, client, db):
        """GET /api/voyages/{id} returns full voyage with legs."""
        voyage = _make_voyage(db)
        resp = client.get(f"/api/voyages/{voyage.id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Test Voyage"
        assert len(data["legs"]) == 5
        assert data["legs"][0]["leg_index"] == 0
        assert data["legs"][4]["leg_index"] == 4

    def test_get_voyage_not_found(self, client):
        """GET /api/voyages/{id} returns 404 for nonexistent voyage."""
        fake_id = str(uuid.uuid4())
        resp = client.get(f"/api/voyages/{fake_id}")
        assert resp.status_code == 404

    def test_delete_voyage(self, client, db, api_key):
        """DELETE /api/voyages/{id} removes voyage and legs."""
        voyage = _make_voyage(db)
        vid = str(voyage.id)

        resp = client.delete(f"/api/voyages/{vid}", headers={"X-API-Key": api_key})
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        # Verify deleted
        resp2 = client.get(f"/api/voyages/{vid}")
        assert resp2.status_code == 404

    def test_delete_voyage_cascade_legs(self, client, db, api_key):
        """Deleting a voyage also deletes its legs."""
        voyage = _make_voyage(db, n_legs=3)
        vid = str(voyage.id)

        # Verify legs exist
        legs_before = db.query(VoyageLeg).filter(VoyageLeg.voyage_id == voyage.id).count()
        assert legs_before == 3

        client.delete(f"/api/voyages/{vid}", headers={"X-API-Key": api_key})

        legs_after = db.query(VoyageLeg).filter(VoyageLeg.voyage_id == voyage.id).count()
        assert legs_after == 0


# ============================================================================
# Noon Report Tests
# ============================================================================


class TestNoonReports:
    """Test noon report generation from voyage legs."""

    def test_noon_reports_5_day_voyage(self, db):
        """5-day voyage produces 4 noon reports (at 24h, 48h, 72h, 96h)."""
        voyage = _make_voyage(db, n_legs=5, voyage_hours=120)
        reports = generate_noon_reports(voyage)
        assert len(reports) == 4  # 24h, 48h, 72h, 96h (120h is arrival)

    def test_noon_reports_short_voyage(self, db):
        """Voyage under 24 hours produces zero noon reports."""
        voyage = _make_voyage(db, n_legs=2, voyage_hours=20)
        reports = generate_noon_reports(voyage)
        assert len(reports) == 0

    def test_noon_reports_interpolation(self, db):
        """Noon reports have interpolated positions between legs."""
        voyage = _make_voyage(db, n_legs=5, voyage_hours=120)
        reports = generate_noon_reports(voyage)

        for r in reports:
            assert "lat" in r
            assert "lon" in r
            assert "timestamp" in r
            assert r["cumulative_distance_nm"] > 0
            assert r["cumulative_fuel_mt"] > 0

    def test_noon_reports_cumulative_increasing(self, db):
        """Cumulative distance and fuel increase monotonically."""
        voyage = _make_voyage(db, n_legs=10, voyage_hours=240)
        reports = generate_noon_reports(voyage)
        assert len(reports) > 1

        for i in range(1, len(reports)):
            assert reports[i]["cumulative_distance_nm"] > reports[i - 1]["cumulative_distance_nm"]
            assert reports[i]["cumulative_fuel_mt"] > reports[i - 1]["cumulative_fuel_mt"]

    def test_noon_reports_api_endpoint(self, client, db):
        """GET /api/voyages/{id}/noon-reports returns JSON noon reports."""
        voyage = _make_voyage(db, n_legs=5, voyage_hours=120)
        resp = client.get(f"/api/voyages/{voyage.id}/noon-reports")
        assert resp.status_code == 200
        data = resp.json()
        assert data["voyage_id"] == str(voyage.id)
        assert len(data["reports"]) == 4

    def test_noon_reports_no_legs(self, db):
        """Voyage with no legs produces empty reports."""
        voyage = Voyage(
            name="Empty",
            departure_time=datetime(2026, 3, 1),
            arrival_time=datetime(2026, 3, 5),
            total_distance_nm=0,
            total_time_hours=96,
            total_fuel_mt=0,
            calm_speed_kts=14.0,
            is_laden=True,
        )
        db.add(voyage)
        db.commit()
        db.refresh(voyage)
        reports = generate_noon_reports(voyage)
        assert reports == []


# ============================================================================
# Template Tests
# ============================================================================


class TestReportTemplates:
    """Test departure and arrival report builders."""

    def test_departure_report(self, db):
        """Departure report has all required fields."""
        voyage = _make_voyage(db)
        dep = build_departure_report(voyage)
        assert dep["departure_port"] == "Rotterdam"
        assert dep["loading_condition"] == "Laden"
        assert dep["planned_distance_nm"] == voyage.total_distance_nm
        assert dep["vessel_name"] == "Test Vessel"
        assert dep["weather_at_departure"] is not None
        assert "wind_speed_kts" in dep["weather_at_departure"]

    def test_departure_report_ballast(self, db):
        """Ballast condition is correctly reflected."""
        voyage = _make_voyage(db)
        voyage.is_laden = False
        db.commit()
        dep = build_departure_report(voyage)
        assert dep["loading_condition"] == "Ballast"

    def test_arrival_report(self, db):
        """Arrival report has all required fields."""
        voyage = _make_voyage(db)
        arr = build_arrival_report(voyage)
        assert arr["arrival_port"] == "Singapore"
        assert arr["total_fuel_consumed_mt"] == voyage.total_fuel_mt
        assert arr["actual_voyage_time_hours"] == voyage.total_time_hours
        assert arr["cii_estimate"] is not None
        assert arr["weather_summary"] is not None
        assert "wind_speed_kts" in arr["weather_summary"]

    def test_reports_api_endpoint(self, client, db):
        """GET /api/voyages/{id}/reports returns all three report types."""
        voyage = _make_voyage(db)
        resp = client.get(f"/api/voyages/{voyage.id}/reports")
        assert resp.status_code == 200
        data = resp.json()
        assert "departure_report" in data
        assert "arrival_report" in data
        assert "noon_reports" in data
        assert data["departure_report"]["departure_port"] == "Rotterdam"
        assert data["arrival_report"]["arrival_port"] == "Singapore"


# ============================================================================
# PDF Generation Tests
# ============================================================================


class TestPDFGeneration:
    """Test PDF report generation."""

    def test_pdf_generation(self, db):
        """PDF is generated as bytes with non-zero length."""
        from api.reports.pdf_generator import generate_voyage_pdf

        voyage = _make_voyage(db, n_legs=5, voyage_hours=120)
        noon = generate_noon_reports(voyage)
        dep = build_departure_report(voyage)
        arr = build_arrival_report(voyage)

        pdf_bytes = generate_voyage_pdf(voyage, noon, dep, arr)
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 1000  # PDF should have meaningful content
        assert pdf_bytes[:5] == b"%PDF-"  # Valid PDF header

    def test_pdf_no_noon_reports(self, db):
        """PDF generation works even with no noon reports."""
        from api.reports.pdf_generator import generate_voyage_pdf

        voyage = _make_voyage(db, n_legs=2, voyage_hours=20)
        dep = build_departure_report(voyage)
        arr = build_arrival_report(voyage)

        pdf_bytes = generate_voyage_pdf(voyage, [], dep, arr)
        assert pdf_bytes[:5] == b"%PDF-"

    def test_pdf_api_endpoint(self, client, db):
        """GET /api/voyages/{id}/pdf returns valid PDF response."""
        voyage = _make_voyage(db, n_legs=3, voyage_hours=72)
        resp = client.get(f"/api/voyages/{voyage.id}/pdf")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/pdf"
        assert resp.content[:5] == b"%PDF-"

    def test_pdf_many_legs(self, db):
        """PDF handles many legs (page break logic)."""
        from api.reports.pdf_generator import generate_voyage_pdf

        voyage = _make_voyage(db, n_legs=50, voyage_hours=600)
        noon = generate_noon_reports(voyage)
        dep = build_departure_report(voyage)
        arr = build_arrival_report(voyage)

        pdf_bytes = generate_voyage_pdf(voyage, noon, dep, arr)
        assert pdf_bytes[:5] == b"%PDF-"
        assert len(pdf_bytes) > 5000  # Should have multiple pages

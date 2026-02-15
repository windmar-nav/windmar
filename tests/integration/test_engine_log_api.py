"""Integration tests for engine log API endpoints."""

import io
import os
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

os.environ["ENVIRONMENT"] = "development"
os.environ["AUTH_ENABLED"] = "false"
os.environ["RATE_LIMIT_ENABLED"] = "false"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["DB_ECHO"] = "false"

from sqlalchemy import create_engine as _real_create_engine
from sqlalchemy.orm import sessionmaker


def _patched_create_engine(url, **kwargs):
    """Create engine, stripping pool params invalid for SQLite."""
    if url.startswith("sqlite"):
        kwargs.pop("pool_size", None)
        kwargs.pop("max_overflow", None)
        kwargs.pop("pool_pre_ping", None)
        kwargs.setdefault("connect_args", {"check_same_thread": False})
    return _real_create_engine(url, **kwargs)


# Patch create_engine before api.database imports it
with patch("sqlalchemy.create_engine", _patched_create_engine):
    # Clear any cached imports so patch applies
    for mod in list(sys.modules.keys()):
        if mod.startswith("api.database"):
            del sys.modules[mod]
    from api.database import Base, get_db, engine as _app_engine

from fastapi.testclient import TestClient
from api.main import app
from api.models import EngineLogEntry
from src.database.engine_log_parser import COLUMN_MAP, DATA_START_ROW

# Use the (now SQLite-backed) app engine for tests
test_engine = _app_engine
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
Base.metadata.create_all(bind=test_engine)


def _build_elog_bytes(rows, sheet_name="E log"):
    ncols = 138
    total_rows = DATA_START_ROW + len(rows)
    data = [[None] * ncols for _ in range(total_rows)]
    data[2][8] = "MAIN ENGINE PARAMETERS"
    data[3][10] = "RPM"
    data[2][83] = "HFO CONSUMPTION (MT)"
    data[2][90] = "MGO CONSUMPTION (MT)"
    for i, row_data in enumerate(rows):
        for j, val in enumerate(row_data):
            if j < ncols:
                data[DATA_START_ROW + i][j] = val
    df = pd.DataFrame(data)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
    buf.seek(0)
    return buf.read()


def _make_row(**kwargs):
    row = [None] * 138
    for field, value in kwargs.items():
        if field in COLUMN_MAP:
            row[COLUMN_MAP[field]] = value
    return row


@pytest.fixture
def db():
    connection = test_engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def client(db):
    def override_get_db():
        try:
            yield db
        finally:
            pass
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as tc:
        yield tc
    app.dependency_overrides.clear()


@pytest.fixture
def sample_elog_bytes():
    rows = [
        _make_row(date=datetime(2025, 8, 1), datetime_combined=datetime(2025, 8, 1, 12, 0),
                   event="Noon", rpm=55, lapse_hours=24, place="At Sea", speed_stw=12.5,
                   me_power_kw=3500, hfo_total_mt=6.0, mgo_total_mt=0.5,
                   rob_vlsfo_mt=500, rob_mgo_mt=80, rh_me=23.5, rh_ae_total=24),
        _make_row(date=datetime(2025, 8, 2), datetime_combined=datetime(2025, 8, 2, 12, 0),
                   event="Noon", rpm=60, lapse_hours=24, place="At Sea", speed_stw=13.0,
                   me_power_kw=4000, hfo_total_mt=7.0, mgo_total_mt=0.6,
                   rob_vlsfo_mt=493, rob_mgo_mt=79.4, rh_me=24, rh_ae_total=24),
        _make_row(date=datetime(2025, 8, 3), datetime_combined=datetime(2025, 8, 3, 8, 0),
                   event="SOSP", rpm=0, lapse_hours=20, place="Singapore",
                   hfo_total_mt=0.5, mgo_total_mt=0.3,
                   rob_vlsfo_mt=492.5, rob_mgo_mt=79.1, rh_me=0, rh_ae_total=20),
    ]
    return _build_elog_bytes(rows)


class TestUpload:
    def test_upload_success(self, client, sample_elog_bytes):
        resp = client.post("/api/engine-log/upload",
                           files={"file": ("test_elog.xlsx", sample_elog_bytes)})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["imported"] == 3
        assert data["batch_id"] is not None
        assert data["events_summary"]["NOON"] == 2

    def test_upload_empty_file(self, client):
        resp = client.post("/api/engine-log/upload", files={"file": ("empty.xlsx", b"")})
        assert resp.status_code == 400

    def test_upload_invalid_format(self, client):
        resp = client.post("/api/engine-log/upload",
                           files={"file": ("test.xlsx", b"not an excel file")})
        assert resp.status_code == 400

    def test_upload_wrong_sheet(self, client, sample_elog_bytes):
        resp = client.post("/api/engine-log/upload?sheet_name=Nonexistent",
                           files={"file": ("test_elog.xlsx", sample_elog_bytes)})
        assert resp.status_code == 400


class TestQuery:
    @pytest.fixture(autouse=True)
    def _upload(self, client, sample_elog_bytes):
        resp = client.post("/api/engine-log/upload",
                           files={"file": ("test_elog.xlsx", sample_elog_bytes)})
        self.batch_id = resp.json()["batch_id"]

    def test_query_all(self, client):
        assert len(client.get("/api/engine-log/entries").json()) == 3

    def test_query_by_event(self, client):
        entries = client.get("/api/engine-log/entries?event=NOON").json()
        assert len(entries) == 2
        assert all(e["event"] == "NOON" for e in entries)

    def test_query_by_date_range(self, client):
        entries = client.get(
            "/api/engine-log/entries?date_from=2025-08-02T00:00:00&date_to=2025-08-02T23:59:59"
        ).json()
        assert len(entries) == 1

    def test_query_by_min_rpm(self, client):
        assert len(client.get("/api/engine-log/entries?min_rpm=55").json()) == 2

    def test_query_by_batch_id(self, client):
        assert len(client.get(f"/api/engine-log/entries?batch_id={self.batch_id}").json()) == 3

    def test_query_pagination(self, client):
        assert len(client.get("/api/engine-log/entries?limit=2&offset=0").json()) == 2
        assert len(client.get("/api/engine-log/entries?limit=2&offset=2").json()) == 1

    def test_entries_ordered_by_timestamp(self, client):
        entries = client.get("/api/engine-log/entries").json()
        timestamps = [e["timestamp"] for e in entries]
        assert timestamps == sorted(timestamps)


class TestSummary:
    @pytest.fixture(autouse=True)
    def _upload(self, client, sample_elog_bytes):
        resp = client.post("/api/engine-log/upload",
                           files={"file": ("test_elog.xlsx", sample_elog_bytes)})
        self.batch_id = resp.json()["batch_id"]

    def test_summary(self, client):
        data = client.get("/api/engine-log/summary").json()
        assert data["total_entries"] == 3
        assert data["events_breakdown"]["NOON"] == 2
        assert data["fuel_summary"]["hfo_mt"] > 0

    def test_summary_by_batch(self, client):
        data = client.get(f"/api/engine-log/summary?batch_id={self.batch_id}").json()
        assert data["total_entries"] == 3
        assert len(data["batches"]) == 1

    def test_summary_empty(self, client, db):
        db.query(EngineLogEntry).delete()
        db.commit()
        assert client.get("/api/engine-log/summary").json()["total_entries"] == 0


class TestDelete:
    @pytest.fixture(autouse=True)
    def _upload(self, client, sample_elog_bytes):
        resp = client.post("/api/engine-log/upload",
                           files={"file": ("test_elog.xlsx", sample_elog_bytes)})
        self.batch_id = resp.json()["batch_id"]

    def test_delete_batch(self, client):
        resp = client.delete(f"/api/engine-log/batch/{self.batch_id}")
        assert resp.status_code == 200
        assert resp.json()["deleted_count"] == 3
        assert len(client.get("/api/engine-log/entries").json()) == 0

    def test_delete_nonexistent_batch(self, client):
        assert client.delete(
            "/api/engine-log/batch/00000000-0000-0000-0000-000000000000"
        ).status_code == 404

    def test_delete_invalid_uuid(self, client):
        assert client.delete("/api/engine-log/batch/not-a-uuid").status_code == 400


class TestFullLifecycle:
    def test_full_lifecycle(self, client, sample_elog_bytes):
        # Upload
        upload = client.post("/api/engine-log/upload",
                             files={"file": ("lifecycle.xlsx", sample_elog_bytes)})
        assert upload.status_code == 200
        batch_id = upload.json()["batch_id"]

        # Query
        assert len(client.get("/api/engine-log/entries").json()) == 3
        assert len(client.get("/api/engine-log/entries?event=NOON").json()) == 2

        # Summary
        summary = client.get("/api/engine-log/summary").json()
        assert summary["total_entries"] == 3
        assert summary["avg_rpm_at_sea"] is not None

        # Delete
        assert client.delete(f"/api/engine-log/batch/{batch_id}").json()["deleted_count"] == 3
        assert client.get("/api/engine-log/entries").json() == []
        assert client.get("/api/engine-log/summary").json()["total_entries"] == 0

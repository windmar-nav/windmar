"""Integration tests for engine log API endpoints.

Fixtures (db, client, sample_elog_bytes) provided by tests/conftest.py.
"""

from datetime import datetime

import pytest

from api.models import EngineLogEntry


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


class TestDeduplication:
    def test_duplicate_upload_skips_existing(self, client, sample_elog_bytes):
        """Uploading the same file twice should deduplicate entries."""
        resp1 = client.post("/api/engine-log/upload",
                            files={"file": ("test_elog.xlsx", sample_elog_bytes)})
        assert resp1.status_code == 200
        assert resp1.json()["imported"] == 3
        assert resp1.json()["skipped"] == 0

        resp2 = client.post("/api/engine-log/upload",
                            files={"file": ("test_elog.xlsx", sample_elog_bytes)})
        assert resp2.status_code == 200
        assert resp2.json()["imported"] == 0
        assert resp2.json()["skipped"] == 3

        # Total entries should be 3, not 6
        entries = client.get("/api/engine-log/entries").json()
        assert len(entries) == 3


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

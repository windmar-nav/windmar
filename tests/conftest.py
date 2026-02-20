"""
Shared pytest fixtures for WINDMAR tests.

CRITICAL: Database patching must occur at module-import time so SQLite
engine creation (without pool_size/max_overflow params) happens before
api.database is imported anywhere. The _patched_create_engine wrapper
strips pool params that are invalid for SQLite.
"""

import io
import os
import sys
from datetime import datetime
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Section 1: Environment setup (before ANY api.* imports)
# ---------------------------------------------------------------------------
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("DB_ECHO", "false")

# ---------------------------------------------------------------------------
# Section 2: Patch SQLAlchemy engine creation for SQLite compatibility
# ---------------------------------------------------------------------------
from sqlalchemy import create_engine as _real_create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


def _patched_create_engine(url, **kwargs):
    """Create engine, stripping pool params invalid for SQLite.

    Uses StaticPool so all connections share the same in-memory database.
    """
    if str(url).startswith("sqlite"):
        kwargs.pop("pool_size", None)
        kwargs.pop("max_overflow", None)
        kwargs.pop("pool_pre_ping", None)
        kwargs.setdefault("connect_args", {"check_same_thread": False})
        kwargs["poolclass"] = StaticPool
    return _real_create_engine(url, **kwargs)


# Apply patch before api.database is imported
_patcher = patch("sqlalchemy.create_engine", _patched_create_engine)
_patcher.start()

# Clear any cached api.database imports so patch takes effect
for _mod in list(sys.modules.keys()):
    if _mod.startswith("api.database"):
        del sys.modules[_mod]

from api.database import Base, get_db, engine as test_engine  # noqa: E402
import api.models  # noqa: E402,F401 — ensure all ORM models are registered

TestingSessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=test_engine
)
Base.metadata.create_all(bind=test_engine)

# ---------------------------------------------------------------------------
# Section 3: Core database + client fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db():
    """Create a test database session with transaction isolation."""
    connection = test_engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def client(db):
    """Create a FastAPI TestClient with database dependency override."""
    from api.main import app

    def override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Section 4: API model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def api_key(db):
    """Create a test API key for authenticated endpoints."""
    from api.auth import generate_api_key, hash_api_key
    from api.models import APIKey

    plain_key = generate_api_key()
    key_hash = hash_api_key(plain_key)

    api_key_obj = APIKey(
        key_hash=key_hash,
        name="Test Key",
        is_active=True,
        rate_limit=1000,
    )
    db.add(api_key_obj)
    db.commit()
    db.refresh(api_key_obj)

    return plain_key


@pytest.fixture
def test_vessel(db):
    """Create a test vessel record in the database."""
    from api.models import VesselSpec

    vessel = VesselSpec(
        name="Test Vessel",
        length=183.0,
        beam=32.0,
        draft=11.8,
        displacement=51450.0,
        deadweight=49000.0,
        engine_power=8840.0,
        service_speed=14.5,
        fuel_type="HFO",
    )
    db.add(vessel)
    db.commit()
    db.refresh(vessel)

    return vessel


# ---------------------------------------------------------------------------
# Section 5: Engine log test helpers + fixtures
# ---------------------------------------------------------------------------


def _build_elog_bytes(rows, sheet_name="E log"):
    """Build Excel file bytes from row data for engine log upload testing."""
    import pandas as pd

    from src.database.engine_log_parser import DATA_START_ROW

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
    """Create a single engine log row with named fields mapped to column indices."""
    from src.database.engine_log_parser import COLUMN_MAP

    row = [None] * 138
    for field, value in kwargs.items():
        if field in COLUMN_MAP:
            row[COLUMN_MAP[field]] = value
    return row


@pytest.fixture
def sample_elog_bytes():
    """Create sample engine log Excel file with 3 entries (2 noon + 1 SOSP)."""
    rows = [
        _make_row(
            date=datetime(2025, 8, 1),
            datetime_combined=datetime(2025, 8, 1, 12, 0),
            event="Noon", rpm=55, lapse_hours=24, place="At Sea",
            speed_stw=12.5, me_power_kw=3500, hfo_total_mt=6.0,
            mgo_total_mt=0.5, rob_vlsfo_mt=500, rob_mgo_mt=80,
            rh_me=23.5, rh_ae_total=24,
        ),
        _make_row(
            date=datetime(2025, 8, 2),
            datetime_combined=datetime(2025, 8, 2, 12, 0),
            event="Noon", rpm=60, lapse_hours=24, place="At Sea",
            speed_stw=13.0, me_power_kw=4000, hfo_total_mt=7.0,
            mgo_total_mt=0.6, rob_vlsfo_mt=493, rob_mgo_mt=79.4,
            rh_me=24, rh_ae_total=24,
        ),
        _make_row(
            date=datetime(2025, 8, 3),
            datetime_combined=datetime(2025, 8, 3, 8, 0),
            event="SOSP", rpm=0, lapse_hours=20, place="Singapore",
            hfo_total_mt=0.5, mgo_total_mt=0.3,
            rob_vlsfo_mt=492.5, rob_mgo_mt=79.1, rh_me=0, rh_ae_total=20,
        ),
    ]
    return _build_elog_bytes(rows)

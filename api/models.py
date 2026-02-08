"""
SQLAlchemy models for WINDMAR database.
"""
from sqlalchemy import (
    Column, String, Float, Integer, Boolean, DateTime, ForeignKey, Text, JSON
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from api.database import Base


class APIKey(Base):
    """API key for authentication."""

    __tablename__ = "api_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key_hash = Column(String(255), nullable=False, unique=True, index=True)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_used_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    rate_limit = Column(Integer, default=1000)
    extra_metadata = Column("metadata", JSON, nullable=True)

    def __repr__(self):
        return f"<APIKey(name='{self.name}', active={self.is_active})>"


class VesselSpec(Base):
    """Vessel specifications."""

    __tablename__ = "vessel_specs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    length = Column(Float, nullable=False)
    beam = Column(Float, nullable=False)
    draft = Column(Float, nullable=False)
    displacement = Column(Float, nullable=False)
    deadweight = Column(Float, nullable=False)
    block_coefficient = Column(Float, nullable=True)
    midship_coefficient = Column(Float, nullable=True)
    waterplane_coefficient = Column(Float, nullable=True)
    lcb_fraction = Column(Float, nullable=True)
    propeller_diameter = Column(Float, nullable=True)
    max_speed = Column(Float, nullable=True)
    service_speed = Column(Float, nullable=True)
    engine_power = Column(Float, nullable=True)
    fuel_type = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    created_by = Column(UUID(as_uuid=True), nullable=True)
    extra_metadata = Column("metadata", JSON, nullable=True)

    # Relationships
    routes = relationship("Route", back_populates="vessel")
    calibration_data = relationship("CalibrationData", back_populates="vessel")
    noon_reports = relationship("NoonReport", back_populates="vessel")

    def __repr__(self):
        return f"<VesselSpec(name='{self.name}', length={self.length})>"


class Route(Base):
    """Optimized route calculation."""

    __tablename__ = "routes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    vessel_id = Column(UUID(as_uuid=True), ForeignKey("vessel_specs.id"), nullable=True, index=True)
    origin_lat = Column(Float, nullable=False)
    origin_lon = Column(Float, nullable=False)
    destination_lat = Column(Float, nullable=False)
    destination_lon = Column(Float, nullable=False)
    departure_time = Column(DateTime, nullable=False)
    route_data = Column(JSON, nullable=False)
    total_distance = Column(Float, nullable=True)
    total_time = Column(Float, nullable=True)
    fuel_consumption = Column(Float, nullable=True)
    calculation_time = Column(Float, nullable=True)
    weather_data_source = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    created_by = Column(UUID(as_uuid=True), nullable=True)
    extra_metadata = Column("metadata", JSON, nullable=True)

    # Relationships
    vessel = relationship("VesselSpec", back_populates="routes")
    noon_reports = relationship("NoonReport", back_populates="route")

    def __repr__(self):
        return f"<Route(id={self.id}, vessel_id={self.vessel_id})>"


class CalibrationData(Base):
    """Vessel performance calibration data from actual operations."""

    __tablename__ = "calibration_data"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    vessel_id = Column(UUID(as_uuid=True), ForeignKey("vessel_specs.id"), nullable=True, index=True)
    speed = Column(Float, nullable=False)
    fuel_consumption = Column(Float, nullable=False)
    wind_speed = Column(Float, nullable=True)
    wind_direction = Column(Float, nullable=True)
    wave_height = Column(Float, nullable=True)
    current_speed = Column(Float, nullable=True)
    current_direction = Column(Float, nullable=True)
    recorded_at = Column(DateTime, nullable=False, index=True)
    data_source = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    extra_metadata = Column("metadata", JSON, nullable=True)

    # Relationships
    vessel = relationship("VesselSpec", back_populates="calibration_data")

    def __repr__(self):
        return f"<CalibrationData(vessel_id={self.vessel_id}, speed={self.speed})>"


class NoonReport(Base):
    """Noon report from vessel operations."""

    __tablename__ = "noon_reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    vessel_id = Column(UUID(as_uuid=True), ForeignKey("vessel_specs.id"), nullable=True, index=True)
    route_id = Column(UUID(as_uuid=True), ForeignKey("routes.id"), nullable=True, index=True)
    position_lat = Column(Float, nullable=False)
    position_lon = Column(Float, nullable=False)
    speed_over_ground = Column(Float, nullable=True)
    speed_through_water = Column(Float, nullable=True)
    course = Column(Float, nullable=True)
    fuel_consumed = Column(Float, nullable=True)
    distance_made_good = Column(Float, nullable=True)
    weather_conditions = Column(JSON, nullable=True)
    report_time = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    extra_metadata = Column("metadata", JSON, nullable=True)

    # Relationships
    vessel = relationship("VesselSpec", back_populates="noon_reports")
    route = relationship("Route", back_populates="noon_reports")

    def __repr__(self):
        return f"<NoonReport(vessel_id={self.vessel_id}, time={self.report_time})>"

"""
Real-time sensor data API for WINDMAR.

Provides:
- REST endpoints for sensor data
- WebSocket streaming for live updates
- Data recording and playback

Based on MIROS-style monitoring dashboard requirements.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Query
from pydantic import BaseModel, Field

# Import sensor modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sensors.sbg_ellipse import SBGEllipseN, SBGSimulator, SBGData, ConnectionType
from src.sensors.timeseries import SensorDataStore


logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/live", tags=["live"])


# ============================================================================
# Request/Response Models
# ============================================================================

class SensorConfig(BaseModel):
    """Sensor connection configuration."""
    connection_type: str = "serial"  # serial, tcp, udp, simulator
    port: str = "/dev/ttyUSB0"
    baudrate: int = 115200
    host: str = "192.168.1.1"
    tcp_port: int = 1234


class SensorStatus(BaseModel):
    """Sensor connection status."""
    connected: bool
    streaming: bool
    connection_type: str
    message_count: int
    parse_errors: int
    last_message_time: Optional[str]


class LiveDataResponse(BaseModel):
    """Current sensor data."""
    timestamp: str
    position: Dict
    velocity: Dict
    attitude: Dict
    motion: Dict
    status: Dict


class TimeSeriesRequest(BaseModel):
    """Time series data request."""
    channel: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    window_seconds: Optional[float] = 300.0


class TimeSeriesResponse(BaseModel):
    """Time series data response."""
    channel: str
    timestamps: List[str]
    values: List[float]
    statistics: Dict


class MotionStatistics(BaseModel):
    """Motion statistics response."""
    heave: Dict
    roll: Dict
    pitch: Dict
    sog: Dict
    motion_severity_index: float
    sample_count: int
    window_seconds: float


# ============================================================================
# Global State
# ============================================================================

# Sensor instance (None until configured)
_sensor: Optional[SBGEllipseN] = None
_simulator: Optional[SBGSimulator] = None
_data_store: Optional[SensorDataStore] = None
_use_simulator: bool = False

# WebSocket connections
_ws_clients: List[WebSocket] = []
_broadcast_task: Optional[asyncio.Task] = None


def _get_sensor():
    """Get current sensor instance."""
    if _use_simulator and _simulator:
        return _simulator
    return _sensor


def _get_data_store() -> SensorDataStore:
    """Get or create data store."""
    global _data_store
    if _data_store is None:
        _data_store = SensorDataStore(
            db_path="data/sensor_data.db",
            buffer_size=36000,  # 1 hour at 10Hz
            buffer_age_seconds=3600.0,
        )
    return _data_store


def _on_sensor_data(data: SBGData) -> None:
    """Callback for new sensor data."""
    store = _get_data_store()
    store.store(data)


# ============================================================================
# REST Endpoints
# ============================================================================

@router.get("/status", response_model=SensorStatus)
async def get_sensor_status():
    """Get current sensor connection status."""
    sensor = _get_sensor()

    if sensor is None:
        return SensorStatus(
            connected=False,
            streaming=False,
            connection_type="none",
            message_count=0,
            parse_errors=0,
            last_message_time=None,
        )

    if isinstance(sensor, SBGSimulator):
        return SensorStatus(
            connected=True,
            streaming=True,
            connection_type="simulator",
            message_count=0,
            parse_errors=0,
            last_message_time=datetime.now(timezone.utc).isoformat(),
        )

    stats = sensor.get_statistics()
    return SensorStatus(
        connected=stats["connected"],
        streaming=stats["streaming"],
        connection_type="serial",  # TODO: track actual type
        message_count=stats["message_count"],
        parse_errors=stats["parse_errors"],
        last_message_time=stats["last_message_time"],
    )


@router.post("/connect")
async def connect_sensor(config: SensorConfig):
    """Connect to SBG sensor or start simulator."""
    global _sensor, _simulator, _use_simulator

    # Disconnect existing
    if _sensor and _sensor.is_connected:
        _sensor.disconnect()
    if _simulator:
        _simulator.stop()

    if config.connection_type == "simulator":
        # Use simulator for testing
        _simulator = SBGSimulator(
            base_latitude=51.9225,
            base_longitude=4.4792,
            heading=180.0,
            speed_kts=12.0,
            sea_state=4,
        )
        _simulator.add_callback(_on_sensor_data)
        _simulator.start(rate_hz=10.0)
        _use_simulator = True

        # Start data store persistence
        _get_data_store().start_persistence()

        logger.info("Started SBG simulator")
        return {"status": "connected", "type": "simulator"}

    else:
        # Real sensor connection
        conn_type = {
            "serial": ConnectionType.SERIAL,
            "tcp": ConnectionType.TCP,
            "udp": ConnectionType.UDP,
        }.get(config.connection_type, ConnectionType.SERIAL)

        _sensor = SBGEllipseN(
            connection_type=conn_type,
            port=config.port,
            baudrate=config.baudrate,
            host=config.host,
            tcp_port=config.tcp_port,
        )

        if not _sensor.connect():
            raise HTTPException(status_code=500, detail="Failed to connect to sensor")

        _sensor.add_callback(_on_sensor_data)
        _sensor.start_streaming()
        _use_simulator = False

        # Start data store persistence
        _get_data_store().start_persistence()

        logger.info(f"Connected to SBG sensor: {config.connection_type}")
        return {"status": "connected", "type": config.connection_type}


@router.post("/disconnect")
async def disconnect_sensor():
    """Disconnect from sensor."""
    global _sensor, _simulator, _use_simulator

    if _sensor:
        _sensor.disconnect()
        _sensor = None

    if _simulator:
        _simulator.stop()
        _simulator = None

    _use_simulator = False

    # Stop persistence
    if _data_store:
        _data_store.stop_persistence()

    logger.info("Disconnected from sensor")
    return {"status": "disconnected"}


@router.get("/data", response_model=LiveDataResponse)
async def get_live_data():
    """Get current sensor data."""
    sensor = _get_sensor()

    if sensor is None:
        raise HTTPException(status_code=400, detail="Sensor not connected")

    data = sensor.get_latest_data()
    return LiveDataResponse(
        timestamp=data.timestamp.isoformat(),
        position={
            "latitude": data.latitude,
            "longitude": data.longitude,
            "altitude": data.altitude,
        },
        velocity={
            "sog_kts": data.sog,
            "cog_deg": data.cog,
        },
        attitude={
            "roll_deg": data.roll,
            "pitch_deg": data.pitch,
            "heading_deg": data.heading,
        },
        motion={
            "heave_m": data.heave,
            "surge_m": data.surge,
            "sway_m": data.sway,
        },
        status={
            "gnss_fix": data.gnss_fix,
            "satellites": data.num_satellites,
            "hdop": data.hdop,
        },
    )


@router.get("/timeseries/{channel}", response_model=TimeSeriesResponse)
async def get_timeseries(
    channel: str,
    window_seconds: float = Query(default=300.0, ge=10, le=86400),
):
    """Get time series data for a channel."""
    store = _get_data_store()

    try:
        start = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)
        timestamps, values = store.get_channel(channel, start=start)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    stats = store.get_statistics(channel, window_seconds=window_seconds)

    return TimeSeriesResponse(
        channel=channel,
        timestamps=[t.isoformat() for t in timestamps],
        values=values,
        statistics=stats,
    )


@router.get("/timeseries")
async def get_all_timeseries(
    window_seconds: float = Query(default=300.0, ge=10, le=86400),
    channels: str = Query(default="sog,heading,roll,pitch,heave"),
):
    """Get time series data for multiple channels."""
    store = _get_data_store()
    channel_list = [c.strip() for c in channels.split(",")]

    start = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)
    result = {}

    for channel in channel_list:
        try:
            timestamps, values = store.get_channel(channel, start=start)
            stats = store.get_statistics(channel, window_seconds=window_seconds)
            result[channel] = {
                "timestamps": [t.isoformat() for t in timestamps],
                "values": values,
                "statistics": stats,
            }
        except ValueError:
            continue

    return result


@router.get("/motion/statistics", response_model=MotionStatistics)
async def get_motion_statistics(
    window_seconds: float = Query(default=60.0, ge=10, le=3600),
):
    """Get computed motion statistics."""
    sensor = _get_sensor()

    if sensor is None:
        raise HTTPException(status_code=400, detail="Sensor not connected")

    if isinstance(sensor, SBGSimulator):
        # For simulator, compute from data store
        store = _get_data_store()
        stats = store.get_all_statistics(window_seconds=window_seconds)

        heave_stats = stats.get("heave", {})
        roll_stats = stats.get("roll", {})
        pitch_stats = stats.get("pitch", {})
        sog_stats = stats.get("sog", {})

        # Compute motion severity
        motion_severity = min(10, (
            heave_stats.get("std", 0) * 2 +
            roll_stats.get("std", 0) * 0.5 +
            pitch_stats.get("std", 0) * 0.5 +
            sog_stats.get("std", 0) * 0.2
        ))

        return MotionStatistics(
            heave={
                "mean": heave_stats.get("mean", 0),
                "std": heave_stats.get("std", 0),
                "max": heave_stats.get("max", 0),
                "significant": heave_stats.get("std", 0) * 4,
            },
            roll=roll_stats,
            pitch=pitch_stats,
            sog=sog_stats,
            motion_severity_index=motion_severity,
            sample_count=heave_stats.get("count", 0),
            window_seconds=window_seconds,
        )

    # Real sensor has compute_motion_statistics method
    stats = sensor.compute_motion_statistics(window_seconds=window_seconds)

    if "error" in stats:
        raise HTTPException(status_code=400, detail=stats["error"])

    return MotionStatistics(**stats)


@router.get("/channels")
async def get_available_channels():
    """Get list of available data channels."""
    return {
        "channels": SensorDataStore.CHANNELS,
        "descriptions": {
            "latitude": "Latitude (degrees)",
            "longitude": "Longitude (degrees)",
            "altitude": "Altitude above ellipsoid (meters)",
            "sog": "Speed Over Ground (knots)",
            "cog": "Course Over Ground (degrees)",
            "heading": "True heading (degrees)",
            "roll": "Roll angle (degrees, + = starboard down)",
            "pitch": "Pitch angle (degrees, + = bow up)",
            "heave": "Heave displacement (meters, + = up)",
            "surge": "Surge displacement (meters, + = forward)",
            "sway": "Sway displacement (meters, + = starboard)",
            "roll_rate": "Roll rate (deg/s)",
            "pitch_rate": "Pitch rate (deg/s)",
            "yaw_rate": "Yaw rate (deg/s)",
            "accel_x": "Forward acceleration (m/s²)",
            "accel_y": "Starboard acceleration (m/s²)",
            "accel_z": "Downward acceleration (m/s²)",
        }
    }


@router.post("/export")
async def export_data(
    format: str = Query(default="csv"),
    window_hours: float = Query(default=1.0, ge=0.1, le=24),
):
    """Export recorded data."""
    store = _get_data_store()

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=window_hours)

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"data/exports/sensor_data_{timestamp_str}.csv"

    Path("data/exports").mkdir(parents=True, exist_ok=True)

    rows = store.export_csv(filename, start=start, end=end)

    return {
        "status": "exported",
        "filename": filename,
        "rows": rows,
        "start": start.isoformat(),
        "end": end.isoformat(),
    }


# ============================================================================
# WebSocket Streaming
# ============================================================================

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time data streaming.

    Sends JSON data packets at 10Hz with current sensor state.
    """
    await websocket.accept()
    _ws_clients.append(websocket)
    logger.info(f"WebSocket client connected. Total: {len(_ws_clients)}")

    try:
        while True:
            # Send data at ~10Hz
            sensor = _get_sensor()
            if sensor:
                data = sensor.get_latest_data()
                await websocket.send_json(data.to_dict())

            # Also handle incoming messages (for configuration)
            try:
                msg = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=0.1
                )
                # Handle commands if needed
                logger.debug(f"Received WS message: {msg}")
            except asyncio.TimeoutError:
                pass

            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.warning(f"WebSocket error: {e}")
    finally:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)


# ============================================================================
# Integration with Main API
# ============================================================================

def include_in_app(app):
    """Include this router in the main FastAPI app."""
    app.include_router(router)
    logger.info("Included live sensor API routes")

"""
SBG Ellipse N INS/GNSS sensor integration.

Provides real-time vessel motion, position, and heading data via:
- Serial (RS232/USB) connection
- Ethernet (TCP/UDP) connection
- NMEA 0183 protocol parsing
- sbgECom binary protocol (optional)

Reference: https://www.sbg-systems.com/ins/ellipse-n/
"""

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple
from collections import deque
from enum import Enum

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

try:
    import socket
    SOCKET_AVAILABLE = True
except ImportError:
    SOCKET_AVAILABLE = False

import numpy as np


logger = logging.getLogger(__name__)


class ConnectionType(Enum):
    """SBG connection types."""
    SERIAL = "serial"
    TCP = "tcp"
    UDP = "udp"


@dataclass
class SBGData:
    """
    Data packet from SBG Ellipse N sensor.

    Contains all relevant navigation and motion data.
    """
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Position (GNSS)
    latitude: float = 0.0          # degrees
    longitude: float = 0.0         # degrees
    altitude: float = 0.0          # meters (above ellipsoid)

    # Velocity
    sog: float = 0.0               # Speed Over Ground (knots)
    cog: float = 0.0               # Course Over Ground (degrees)
    velocity_north: float = 0.0    # m/s
    velocity_east: float = 0.0     # m/s
    velocity_down: float = 0.0     # m/s

    # Attitude (roll, pitch, yaw)
    roll: float = 0.0              # degrees (positive = starboard down)
    pitch: float = 0.0             # degrees (positive = bow up)
    heading: float = 0.0           # degrees (true north)

    # Heave and motion
    heave: float = 0.0             # meters (positive = up)
    surge: float = 0.0             # meters (positive = forward)
    sway: float = 0.0              # meters (positive = starboard)

    # Angular rates
    roll_rate: float = 0.0         # deg/s
    pitch_rate: float = 0.0        # deg/s
    yaw_rate: float = 0.0          # deg/s

    # Accelerations
    accel_x: float = 0.0           # m/s² (forward)
    accel_y: float = 0.0           # m/s² (starboard)
    accel_z: float = 0.0           # m/s² (down)

    # Status
    gnss_fix: int = 0              # 0=no fix, 1=2D, 2=3D, 4=RTK float, 5=RTK fixed
    ins_status: int = 0            # Solution status
    num_satellites: int = 0
    hdop: float = 99.9

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "position": {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "altitude": self.altitude,
            },
            "velocity": {
                "sog_kts": self.sog,
                "cog_deg": self.cog,
                "north_ms": self.velocity_north,
                "east_ms": self.velocity_east,
                "down_ms": self.velocity_down,
            },
            "attitude": {
                "roll_deg": self.roll,
                "pitch_deg": self.pitch,
                "heading_deg": self.heading,
            },
            "motion": {
                "heave_m": self.heave,
                "surge_m": self.surge,
                "sway_m": self.sway,
            },
            "rates": {
                "roll_rate_dps": self.roll_rate,
                "pitch_rate_dps": self.pitch_rate,
                "yaw_rate_dps": self.yaw_rate,
            },
            "acceleration": {
                "x_ms2": self.accel_x,
                "y_ms2": self.accel_y,
                "z_ms2": self.accel_z,
            },
            "status": {
                "gnss_fix": self.gnss_fix,
                "ins_status": self.ins_status,
                "satellites": self.num_satellites,
                "hdop": self.hdop,
            },
        }


class NMEAParser:
    """
    Parser for NMEA 0183 messages from SBG Ellipse N.

    Supported sentences:
    - GGA: Position fix
    - RMC: Recommended minimum data
    - HDT: True heading
    - VTG: Course and speed over ground
    - PASHR: Attitude (roll, pitch, heading)
    - PSBGI: SBG proprietary INS data
    """

    @staticmethod
    def parse_sentence(sentence: str) -> Optional[Dict]:
        """Parse a single NMEA sentence."""
        sentence = sentence.strip()

        if not sentence.startswith('$'):
            return None

        # Validate checksum
        if '*' in sentence:
            data_part, checksum = sentence[1:].split('*')
            calculated = NMEAParser._calculate_checksum(data_part)
            if calculated.upper() != checksum.upper():
                logger.warning(f"NMEA checksum mismatch: {sentence}")
                return None
        else:
            data_part = sentence[1:]

        fields = data_part.split(',')
        sentence_type = fields[0]

        try:
            if sentence_type.endswith('GGA'):
                return NMEAParser._parse_gga(fields)
            elif sentence_type.endswith('RMC'):
                return NMEAParser._parse_rmc(fields)
            elif sentence_type.endswith('HDT'):
                return NMEAParser._parse_hdt(fields)
            elif sentence_type.endswith('VTG'):
                return NMEAParser._parse_vtg(fields)
            elif sentence_type == 'PASHR':
                return NMEAParser._parse_pashr(fields)
            elif sentence_type == 'PSBGI':
                return NMEAParser._parse_psbgi(fields)
        except (ValueError, IndexError) as e:
            logger.debug(f"Failed to parse {sentence_type}: {e}")

        return None

    @staticmethod
    def _calculate_checksum(data: str) -> str:
        """Calculate NMEA checksum."""
        checksum = 0
        for char in data:
            checksum ^= ord(char)
        return f"{checksum:02X}"

    @staticmethod
    def _parse_coordinate(value: str, direction: str) -> float:
        """Parse NMEA coordinate (DDMM.MMMM or DDDMM.MMMM)."""
        if not value:
            return 0.0

        if len(value.split('.')[0]) <= 4:
            # Latitude: DDMM.MMMM
            degrees = int(value[:2])
            minutes = float(value[2:])
        else:
            # Longitude: DDDMM.MMMM
            degrees = int(value[:3])
            minutes = float(value[3:])

        result = degrees + minutes / 60.0

        if direction in ('S', 'W'):
            result = -result

        return result

    @staticmethod
    def _parse_gga(fields: List[str]) -> Dict:
        """Parse GGA sentence (Position fix)."""
        return {
            "type": "position",
            "latitude": NMEAParser._parse_coordinate(fields[2], fields[3]) if fields[2] else 0,
            "longitude": NMEAParser._parse_coordinate(fields[4], fields[5]) if fields[4] else 0,
            "gnss_fix": int(fields[6]) if fields[6] else 0,
            "num_satellites": int(fields[7]) if fields[7] else 0,
            "hdop": float(fields[8]) if fields[8] else 99.9,
            "altitude": float(fields[9]) if fields[9] else 0,
        }

    @staticmethod
    def _parse_rmc(fields: List[str]) -> Dict:
        """Parse RMC sentence (Recommended minimum)."""
        sog_knots = float(fields[7]) if fields[7] else 0
        cog = float(fields[8]) if fields[8] else 0

        return {
            "type": "velocity",
            "latitude": NMEAParser._parse_coordinate(fields[3], fields[4]) if fields[3] else 0,
            "longitude": NMEAParser._parse_coordinate(fields[5], fields[6]) if fields[5] else 0,
            "sog": sog_knots,
            "cog": cog,
            "valid": fields[2] == 'A',
        }

    @staticmethod
    def _parse_hdt(fields: List[str]) -> Dict:
        """Parse HDT sentence (True heading)."""
        return {
            "type": "heading",
            "heading": float(fields[1]) if fields[1] else 0,
        }

    @staticmethod
    def _parse_vtg(fields: List[str]) -> Dict:
        """Parse VTG sentence (Course and speed)."""
        return {
            "type": "velocity",
            "cog": float(fields[1]) if fields[1] else 0,
            "sog": float(fields[5]) if fields[5] else 0,  # Speed in knots
        }

    @staticmethod
    def _parse_pashr(fields: List[str]) -> Dict:
        """Parse PASHR sentence (Attitude)."""
        return {
            "type": "attitude",
            "heading": float(fields[2]) if fields[2] else 0,
            "roll": float(fields[4]) if fields[4] else 0,
            "pitch": float(fields[5]) if fields[5] else 0,
            "heave": float(fields[6]) if fields[6] else 0,
        }

    @staticmethod
    def _parse_psbgi(fields: List[str]) -> Dict:
        """Parse PSBGI sentence (SBG proprietary INS data)."""
        # PSBGI format varies - this is a common variant
        return {
            "type": "ins",
            "roll": float(fields[1]) if len(fields) > 1 and fields[1] else 0,
            "pitch": float(fields[2]) if len(fields) > 2 and fields[2] else 0,
            "heading": float(fields[3]) if len(fields) > 3 and fields[3] else 0,
            "heave": float(fields[4]) if len(fields) > 4 and fields[4] else 0,
            "surge": float(fields[5]) if len(fields) > 5 and fields[5] else 0,
            "sway": float(fields[6]) if len(fields) > 6 and fields[6] else 0,
        }


class SBGEllipseN:
    """
    SBG Ellipse N sensor interface.

    Connects to the sensor via serial or Ethernet and provides
    real-time navigation and motion data.

    Example usage:
        sensor = SBGEllipseN(port="/dev/ttyUSB0", baudrate=115200)
        sensor.connect()
        sensor.start_streaming()

        while True:
            data = sensor.get_latest_data()
            print(f"Position: {data.latitude}, {data.longitude}")
            print(f"Heading: {data.heading}°, SOG: {data.sog} kts")
            print(f"Roll: {data.roll}°, Pitch: {data.pitch}°, Heave: {data.heave}m")
    """

    def __init__(
        self,
        connection_type: ConnectionType = ConnectionType.SERIAL,
        # Serial settings
        port: str = "/dev/ttyUSB0",
        baudrate: int = 115200,
        # Ethernet settings
        host: str = "192.168.1.1",
        tcp_port: int = 1234,
        # Data settings
        buffer_size: int = 1000,
        output_rate_hz: float = 10.0,
    ):
        """
        Initialize SBG Ellipse N interface.

        Args:
            connection_type: Serial, TCP, or UDP
            port: Serial port (e.g., /dev/ttyUSB0, COM3)
            baudrate: Serial baudrate (typically 115200 or 921600)
            host: IP address for Ethernet connection
            tcp_port: TCP/UDP port number
            buffer_size: Size of data buffer for time series
            output_rate_hz: Expected sensor output rate
        """
        self.connection_type = connection_type
        self.port = port
        self.baudrate = baudrate
        self.host = host
        self.tcp_port = tcp_port
        self.buffer_size = buffer_size
        self.output_rate_hz = output_rate_hz

        # Connection state
        self._connection = None
        self._connected = False
        self._streaming = False
        self._read_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Data storage
        self._current_data = SBGData()
        self._data_buffer: deque = deque(maxlen=buffer_size)
        self._data_lock = threading.Lock()

        # Callbacks
        self._callbacks: List[Callable[[SBGData], None]] = []

        # Statistics
        self._message_count = 0
        self._last_message_time: Optional[datetime] = None
        self._parse_errors = 0

        # NMEA parser
        self._nmea_parser = NMEAParser()
        self._nmea_buffer = ""

    def connect(self) -> bool:
        """
        Establish connection to sensor.

        Returns:
            True if connection successful
        """
        if self._connected:
            logger.warning("Already connected")
            return True

        try:
            if self.connection_type == ConnectionType.SERIAL:
                if not SERIAL_AVAILABLE:
                    raise ImportError("pyserial not installed. Run: pip install pyserial")

                self._connection = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    timeout=1.0,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                )
                logger.info(f"Connected to SBG via serial: {self.port} @ {self.baudrate}")

            elif self.connection_type == ConnectionType.TCP:
                self._connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._connection.connect((self.host, self.tcp_port))
                self._connection.settimeout(1.0)
                logger.info(f"Connected to SBG via TCP: {self.host}:{self.tcp_port}")

            elif self.connection_type == ConnectionType.UDP:
                self._connection = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self._connection.bind(('', self.tcp_port))
                self._connection.settimeout(1.0)
                logger.info(f"Listening for SBG UDP on port {self.tcp_port}")

            self._connected = True
            return True

        except Exception as e:
            logger.error(f"Failed to connect to SBG: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from sensor."""
        self.stop_streaming()

        if self._connection:
            try:
                self._connection.close()
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")

        self._connection = None
        self._connected = False
        logger.info("Disconnected from SBG")

    def start_streaming(self) -> bool:
        """
        Start reading data from sensor.

        Returns:
            True if streaming started
        """
        if not self._connected:
            logger.error("Not connected")
            return False

        if self._streaming:
            logger.warning("Already streaming")
            return True

        self._stop_event.clear()
        self._streaming = True

        self._read_thread = threading.Thread(
            target=self._read_loop,
            daemon=True,
            name="SBG-Reader"
        )
        self._read_thread.start()

        logger.info("Started SBG data streaming")
        return True

    def stop_streaming(self) -> None:
        """Stop reading data from sensor."""
        if not self._streaming:
            return

        self._stop_event.set()
        self._streaming = False

        if self._read_thread:
            self._read_thread.join(timeout=2.0)
            self._read_thread = None

        logger.info("Stopped SBG data streaming")

    def _read_loop(self) -> None:
        """Main reading loop (runs in separate thread)."""
        while not self._stop_event.is_set():
            try:
                data = self._read_data()
                if data:
                    self._process_data(data)
            except Exception as e:
                logger.debug(f"Read error: {e}")
                time.sleep(0.01)

    def _read_data(self) -> Optional[bytes]:
        """Read raw data from connection."""
        if self.connection_type == ConnectionType.SERIAL:
            if self._connection.in_waiting > 0:
                return self._connection.read(self._connection.in_waiting)
        elif self.connection_type in (ConnectionType.TCP, ConnectionType.UDP):
            try:
                data, _ = self._connection.recvfrom(4096)
                return data
            except socket.timeout:
                return None
        return None

    def _process_data(self, raw_data: bytes) -> None:
        """Process raw data and extract NMEA sentences."""
        try:
            text = raw_data.decode('ascii', errors='ignore')
        except (UnicodeDecodeError, AttributeError):
            return

        self._nmea_buffer += text

        # Process complete sentences
        while '\n' in self._nmea_buffer:
            line, self._nmea_buffer = self._nmea_buffer.split('\n', 1)
            line = line.strip()

            if line.startswith('$'):
                parsed = self._nmea_parser.parse_sentence(line)
                if parsed:
                    self._update_data(parsed)
                    self._message_count += 1
                    self._last_message_time = datetime.now(timezone.utc)
                else:
                    self._parse_errors += 1

    def _update_data(self, parsed: Dict) -> None:
        """Update current data with parsed values."""
        with self._data_lock:
            data_type = parsed.get("type", "")

            if data_type == "position":
                self._current_data.latitude = parsed.get("latitude", self._current_data.latitude)
                self._current_data.longitude = parsed.get("longitude", self._current_data.longitude)
                self._current_data.altitude = parsed.get("altitude", self._current_data.altitude)
                self._current_data.gnss_fix = parsed.get("gnss_fix", self._current_data.gnss_fix)
                self._current_data.num_satellites = parsed.get("num_satellites", self._current_data.num_satellites)
                self._current_data.hdop = parsed.get("hdop", self._current_data.hdop)

            elif data_type == "velocity":
                self._current_data.sog = parsed.get("sog", self._current_data.sog)
                self._current_data.cog = parsed.get("cog", self._current_data.cog)
                if "latitude" in parsed and parsed["latitude"]:
                    self._current_data.latitude = parsed["latitude"]
                    self._current_data.longitude = parsed["longitude"]

            elif data_type == "heading":
                self._current_data.heading = parsed.get("heading", self._current_data.heading)

            elif data_type == "attitude":
                self._current_data.roll = parsed.get("roll", self._current_data.roll)
                self._current_data.pitch = parsed.get("pitch", self._current_data.pitch)
                self._current_data.heading = parsed.get("heading", self._current_data.heading)
                self._current_data.heave = parsed.get("heave", self._current_data.heave)

            elif data_type == "ins":
                self._current_data.roll = parsed.get("roll", self._current_data.roll)
                self._current_data.pitch = parsed.get("pitch", self._current_data.pitch)
                self._current_data.heading = parsed.get("heading", self._current_data.heading)
                self._current_data.heave = parsed.get("heave", self._current_data.heave)
                self._current_data.surge = parsed.get("surge", self._current_data.surge)
                self._current_data.sway = parsed.get("sway", self._current_data.sway)

            # Update timestamp
            self._current_data.timestamp = datetime.now(timezone.utc)

            # Add to buffer
            self._data_buffer.append(SBGData(**vars(self._current_data)))

            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(self._current_data)
                except Exception as e:
                    logger.warning(f"Callback error: {e}")

    def get_latest_data(self) -> SBGData:
        """Get the most recent data from the sensor."""
        with self._data_lock:
            return SBGData(**vars(self._current_data))

    def get_buffer(self, seconds: Optional[float] = None) -> List[SBGData]:
        """
        Get buffered data.

        Args:
            seconds: If provided, return only data from last N seconds

        Returns:
            List of SBGData objects
        """
        with self._data_lock:
            data = list(self._data_buffer)

        if seconds is not None:
            cutoff = datetime.now(timezone.utc).timestamp() - seconds
            data = [d for d in data if d.timestamp.timestamp() > cutoff]

        return data

    def add_callback(self, callback: Callable[[SBGData], None]) -> None:
        """Add a callback function to be called on new data."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[SBGData], None]) -> None:
        """Remove a callback function."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def get_statistics(self) -> Dict:
        """Get connection and data statistics."""
        return {
            "connected": self._connected,
            "streaming": self._streaming,
            "message_count": self._message_count,
            "parse_errors": self._parse_errors,
            "buffer_size": len(self._data_buffer),
            "last_message_time": self._last_message_time.isoformat() if self._last_message_time else None,
        }

    def compute_motion_statistics(self, window_seconds: float = 60.0) -> Dict:
        """
        Compute motion statistics from buffered data.

        Args:
            window_seconds: Time window for statistics

        Returns:
            Dictionary with motion statistics
        """
        data = self.get_buffer(seconds=window_seconds)

        if len(data) < 10:
            return {"error": "Insufficient data"}

        # Extract arrays
        heave = np.array([d.heave for d in data])
        roll = np.array([d.roll for d in data])
        pitch = np.array([d.pitch for d in data])
        sog = np.array([d.sog for d in data])

        # Compute statistics
        stats = {
            "heave": {
                "mean": float(np.mean(heave)),
                "std": float(np.std(heave)),
                "max": float(np.max(np.abs(heave))),
                "significant": float(4 * np.std(heave)),  # Significant heave height
            },
            "roll": {
                "mean": float(np.mean(roll)),
                "std": float(np.std(roll)),
                "max": float(np.max(np.abs(roll))),
            },
            "pitch": {
                "mean": float(np.mean(pitch)),
                "std": float(np.std(pitch)),
                "max": float(np.max(np.abs(pitch))),
            },
            "sog": {
                "mean": float(np.mean(sog)),
                "std": float(np.std(sog)),
                "min": float(np.min(sog)),
                "max": float(np.max(sog)),
            },
            "sample_count": len(data),
            "window_seconds": window_seconds,
        }

        # Estimate motion severity index (0-10 scale)
        # Based on combined motion parameters
        motion_severity = min(10, (
            np.std(heave) * 2 +       # Heave contribution
            np.std(roll) * 0.5 +      # Roll contribution
            np.std(pitch) * 0.5 +     # Pitch contribution
            np.std(sog) * 0.2         # SOG variation
        ))
        stats["motion_severity_index"] = float(motion_severity)

        return stats

    @property
    def is_connected(self) -> bool:
        """Check if sensor is connected."""
        return self._connected

    @property
    def is_streaming(self) -> bool:
        """Check if actively streaming data."""
        return self._streaming


class SBGSimulator:
    """
    Simulator for SBG Ellipse N sensor.

    Generates realistic motion data for testing without hardware.
    """

    def __init__(
        self,
        base_latitude: float = 51.9225,
        base_longitude: float = 4.4792,
        heading: float = 180.0,
        speed_kts: float = 12.0,
        sea_state: int = 4,  # Douglas scale 0-9
    ):
        """
        Initialize simulator.

        Args:
            base_latitude: Starting latitude
            base_longitude: Starting longitude
            heading: Vessel heading (degrees)
            speed_kts: Speed over ground (knots)
            sea_state: Sea state (Douglas scale)
        """
        self.latitude = base_latitude
        self.longitude = base_longitude
        self.heading = heading
        self.speed_kts = speed_kts
        self.sea_state = sea_state

        self._start_time = time.time()
        self._callbacks: List[Callable[[SBGData], None]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self, rate_hz: float = 10.0) -> None:
        """Start generating simulated data."""
        self._running = True
        self._thread = threading.Thread(
            target=self._generate_loop,
            args=(rate_hz,),
            daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop generating data."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _generate_loop(self, rate_hz: float) -> None:
        """Generate data at specified rate."""
        interval = 1.0 / rate_hz

        while self._running:
            data = self._generate_data()
            for callback in self._callbacks:
                try:
                    callback(data)
                except Exception as e:
                    logger.warning(f"Callback error: {e}")
            time.sleep(interval)

    def _generate_data(self) -> SBGData:
        """Generate realistic simulated data."""
        t = time.time() - self._start_time

        # Sea state parameters (simplified model)
        # Higher sea state = more motion
        wave_height = self.sea_state * 0.5  # meters
        wave_period = 6 + self.sea_state * 0.5  # seconds

        # Generate motion
        heave = wave_height * 0.5 * math.sin(2 * math.pi * t / wave_period)
        roll = self.sea_state * 2 * math.sin(2 * math.pi * t / (wave_period * 1.2) + 0.5)
        pitch = self.sea_state * 1.5 * math.sin(2 * math.pi * t / (wave_period * 0.8) + 1.0)

        # Add some noise
        heave += np.random.normal(0, wave_height * 0.1)
        roll += np.random.normal(0, self.sea_state * 0.2)
        pitch += np.random.normal(0, self.sea_state * 0.15)

        # Update position based on speed and heading
        dt = 0.1  # seconds
        speed_ms = self.speed_kts * 0.51444

        # Simple position update (flat earth approximation for small distances)
        dlat = speed_ms * math.cos(math.radians(self.heading)) * dt / 111320
        dlon = speed_ms * math.sin(math.radians(self.heading)) * dt / (111320 * math.cos(math.radians(self.latitude)))

        self.latitude += dlat
        self.longitude += dlon

        # SOG varies slightly with waves
        sog = self.speed_kts + np.random.normal(0, 0.2)

        return SBGData(
            timestamp=datetime.now(timezone.utc),
            latitude=self.latitude,
            longitude=self.longitude,
            altitude=0.0,
            sog=sog,
            cog=self.heading,
            heading=self.heading + np.random.normal(0, 0.5),
            roll=roll,
            pitch=pitch,
            heave=heave,
            gnss_fix=5,  # RTK fixed
            num_satellites=12,
            hdop=0.8,
        )

    def add_callback(self, callback: Callable[[SBGData], None]) -> None:
        """Add callback for new data."""
        self._callbacks.append(callback)

    def get_latest_data(self) -> SBGData:
        """Get current simulated data."""
        return self._generate_data()

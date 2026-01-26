"""
SBG Ellipse N NMEA Parser.

Reads NMEA sentences from SBG Ellipse N INS via USB serial port.
Supports both standard NMEA and SBG proprietary sentences.

Usage:
    from src.sensors.sbg_nmea import SBGNmeaParser

    # Find your port: ls /dev/tty* (Linux) or ls /dev/cu.* (Mac)
    parser = SBGNmeaParser(port="/dev/ttyUSB0", baudrate=115200)
    parser.start()

    while True:
        data = parser.get_latest()
        print(f"Heave: {data.heave_m}, Roll: {data.roll_deg}")
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional
from queue import Queue, Empty

logger = logging.getLogger(__name__)


@dataclass
class AttitudeData:
    """Roll, pitch, yaw from EKF."""
    timestamp: datetime
    roll_deg: float       # Roll angle (positive = starboard down)
    pitch_deg: float      # Pitch angle (positive = bow up)
    heading_deg: float    # True heading
    roll_std: float = 0.0
    pitch_std: float = 0.0
    heading_std: float = 0.0
    valid: bool = True


@dataclass
class IMUData:
    """Raw IMU accelerations and angular rates."""
    timestamp: datetime
    accel_x: float  # m/s² (forward)
    accel_y: float  # m/s² (starboard)
    accel_z: float  # m/s² (down)
    gyro_x: float   # rad/s (roll rate)
    gyro_y: float   # rad/s (pitch rate)
    gyro_z: float   # rad/s (yaw rate)
    valid: bool = True


@dataclass
class PositionData:
    """GPS position and velocity."""
    timestamp: datetime
    latitude: float   # degrees
    longitude: float  # degrees
    altitude_m: float = 0.0
    speed_kts: float = 0.0
    course_deg: float = 0.0
    valid: bool = True


@dataclass
class ShipMotionData:
    """Aggregated ship motion state."""
    timestamp: datetime
    # Attitude
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    heading_deg: float = 0.0
    # Heave (vertical motion)
    heave_m: float = 0.0
    heave_velocity_ms: float = 0.0
    heave_period_s: float = 0.0
    # Position/velocity
    latitude: float = 0.0
    longitude: float = 0.0
    speed_kts: float = 0.0
    course_deg: float = 0.0
    # Raw accelerations (for wave resistance estimation)
    accel_z: float = 0.0  # Vertical acceleration
    # Status
    valid: bool = True


class SBGNmeaParser:
    """
    Parser for SBG Ellipse N NMEA output.

    Supported sentences:
    - $PSBGA: Attitude (roll, pitch, heading)
    - $PSBGI: IMU data (accelerations, gyros)
    - $GPRMC: Position, speed, course
    - $GPGGA: Position, altitude
    - $PHTRO: Pitch and roll (standard)
    - $HEAVE: Heave measurement (if available)
    """

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 115200,
        timeout: float = 1.0,
    ):
        """
        Initialize parser.

        Args:
            port: Serial port (e.g., /dev/ttyUSB0, /dev/cu.usbserial-*, COM3)
            baudrate: Baud rate (SBG default: 115200)
            timeout: Read timeout in seconds
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout

        self._serial = None
        self._running = False
        self._thread = None

        # Latest data
        self._attitude = AttitudeData(datetime.now(), 0, 0, 0)
        self._imu = IMUData(datetime.now(), 0, 0, 0, 0, 0, 0)
        self._position = PositionData(datetime.now(), 0, 0)
        self._heave = 0.0
        self._heave_period = 0.0

        # Data lock
        self._lock = threading.Lock()

        # Callbacks
        self._callbacks: List[Callable[[ShipMotionData], None]] = []

        # Data queue for async processing
        self._queue: Queue = Queue(maxsize=1000)

        # Statistics
        self._stats = {
            "sentences_parsed": 0,
            "parse_errors": 0,
            "checksum_errors": 0,
        }

    def start(self) -> bool:
        """Start reading from serial port."""
        try:
            import serial
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
            )
            logger.info(f"Connected to {self.port} @ {self.baudrate} baud")
        except ImportError:
            logger.error("pyserial not installed. Run: pip install pyserial")
            return False
        except Exception as e:
            logger.error(f"Failed to open {self.port}: {e}")
            return False

        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        """Stop reading and close port."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._serial:
            self._serial.close()
            self._serial = None
        logger.info("SBG parser stopped")

    def get_latest(self) -> ShipMotionData:
        """Get latest aggregated ship motion data."""
        with self._lock:
            return ShipMotionData(
                timestamp=datetime.now(),
                roll_deg=self._attitude.roll_deg,
                pitch_deg=self._attitude.pitch_deg,
                heading_deg=self._attitude.heading_deg,
                heave_m=self._heave,
                heave_period_s=self._heave_period,
                latitude=self._position.latitude,
                longitude=self._position.longitude,
                speed_kts=self._position.speed_kts,
                course_deg=self._position.course_deg,
                accel_z=self._imu.accel_z,
                valid=self._attitude.valid and self._position.valid,
            )

    def add_callback(self, callback: Callable[[ShipMotionData], None]):
        """Register callback for new data."""
        self._callbacks.append(callback)

    def get_stats(self) -> Dict:
        """Get parsing statistics."""
        return self._stats.copy()

    def _read_loop(self):
        """Main read loop (runs in thread)."""
        buffer = ""

        while self._running:
            try:
                # Read available data
                if self._serial.in_waiting > 0:
                    data = self._serial.read(self._serial.in_waiting)
                    buffer += data.decode('ascii', errors='ignore')

                    # Process complete sentences
                    while '\r\n' in buffer:
                        line, buffer = buffer.split('\r\n', 1)
                        if line.startswith('$'):
                            self._parse_sentence(line)
                else:
                    time.sleep(0.01)  # Small sleep to prevent busy loop

            except Exception as e:
                logger.error(f"Read error: {e}")
                time.sleep(0.1)

    def _parse_sentence(self, sentence: str):
        """Parse a single NMEA sentence."""
        try:
            # Verify checksum
            if '*' in sentence:
                data, checksum = sentence.rsplit('*', 1)
                if not self._verify_checksum(data, checksum):
                    self._stats["checksum_errors"] += 1
                    return
                sentence = data

            # Remove $ prefix
            if sentence.startswith('$'):
                sentence = sentence[1:]

            # Split into fields
            fields = sentence.split(',')
            msg_type = fields[0]

            # Parse by message type
            if msg_type == 'PSBGA':
                self._parse_psbga(fields)
            elif msg_type == 'PSBGI':
                self._parse_psbgi(fields)
            elif msg_type in ('GPRMC', 'GNRMC'):
                self._parse_rmc(fields)
            elif msg_type in ('GPGGA', 'GNGGA'):
                self._parse_gga(fields)
            elif msg_type == 'PHTRO':
                self._parse_phtro(fields)
            elif msg_type == 'HEAVE':
                self._parse_heave(fields)

            self._stats["sentences_parsed"] += 1

            # Notify callbacks
            if self._callbacks:
                data = self.get_latest()
                for cb in self._callbacks:
                    try:
                        cb(data)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

        except Exception as e:
            self._stats["parse_errors"] += 1
            logger.debug(f"Parse error for '{sentence}': {e}")

    def _parse_psbga(self, fields: List[str]):
        """
        Parse SBG proprietary attitude message.

        Format: $PSBGA,time,heading,roll,pitch,status*CS
        """
        if len(fields) < 5:
            return

        with self._lock:
            self._attitude = AttitudeData(
                timestamp=datetime.now(),
                heading_deg=self._safe_float(fields[2]),
                roll_deg=self._safe_float(fields[3]),
                pitch_deg=self._safe_float(fields[4]),
                valid=True,
            )

    def _parse_psbgi(self, fields: List[str]):
        """
        Parse SBG proprietary IMU message.

        Format: $PSBGI,time,ax,ay,az,gx,gy,gz*CS
        Accelerations in m/s², angular rates in rad/s
        """
        if len(fields) < 8:
            return

        with self._lock:
            self._imu = IMUData(
                timestamp=datetime.now(),
                accel_x=self._safe_float(fields[2]),
                accel_y=self._safe_float(fields[3]),
                accel_z=self._safe_float(fields[4]),
                gyro_x=self._safe_float(fields[5]),
                gyro_y=self._safe_float(fields[6]),
                gyro_z=self._safe_float(fields[7]),
                valid=True,
            )

    def _parse_rmc(self, fields: List[str]):
        """
        Parse standard RMC (Recommended Minimum) message.

        Format: $GPRMC,time,status,lat,N/S,lon,E/W,speed,course,date,...*CS
        """
        if len(fields) < 8:
            return

        status = fields[2]
        if status != 'A':  # A = valid, V = warning
            return

        lat = self._parse_lat_lon(fields[3], fields[4])
        lon = self._parse_lat_lon(fields[5], fields[6])
        speed_kts = self._safe_float(fields[7])
        course = self._safe_float(fields[8]) if len(fields) > 8 else 0.0

        with self._lock:
            self._position.latitude = lat
            self._position.longitude = lon
            self._position.speed_kts = speed_kts
            self._position.course_deg = course
            self._position.timestamp = datetime.now()
            self._position.valid = True

    def _parse_gga(self, fields: List[str]):
        """
        Parse standard GGA (Fix) message.

        Format: $GPGGA,time,lat,N/S,lon,E/W,quality,sats,hdop,alt,M,...*CS
        """
        if len(fields) < 10:
            return

        quality = self._safe_int(fields[6])
        if quality == 0:  # No fix
            return

        lat = self._parse_lat_lon(fields[2], fields[3])
        lon = self._parse_lat_lon(fields[4], fields[5])
        alt = self._safe_float(fields[9])

        with self._lock:
            self._position.latitude = lat
            self._position.longitude = lon
            self._position.altitude_m = alt
            self._position.timestamp = datetime.now()
            self._position.valid = True

    def _parse_phtro(self, fields: List[str]):
        """
        Parse standard PHTRO (Pitch/Roll) message.

        Format: $PHTRO,pitch,P,roll,R*CS
        """
        if len(fields) < 5:
            return

        pitch = self._safe_float(fields[1])
        if fields[2] == 'B':  # Bow down
            pitch = -pitch

        roll = self._safe_float(fields[3])
        if fields[4].startswith('P'):  # Port down
            roll = -roll

        with self._lock:
            self._attitude.pitch_deg = pitch
            self._attitude.roll_deg = roll
            self._attitude.timestamp = datetime.now()

    def _parse_heave(self, fields: List[str]):
        """Parse heave message (format may vary)."""
        if len(fields) < 2:
            return

        with self._lock:
            self._heave = self._safe_float(fields[1])

    def _parse_lat_lon(self, value: str, direction: str) -> float:
        """Convert NMEA lat/lon to decimal degrees."""
        if not value:
            return 0.0

        try:
            # Format: DDDMM.MMMM or DDMM.MMMM
            if len(value) > 5:
                if '.' in value:
                    dot_pos = value.index('.')
                    degrees = float(value[:dot_pos-2])
                    minutes = float(value[dot_pos-2:])
                else:
                    degrees = float(value[:-7])
                    minutes = float(value[-7:])

                result = degrees + minutes / 60.0

                if direction in ('S', 'W'):
                    result = -result

                return result
        except (ValueError, IndexError) as e:
            logger.debug(f"Failed to parse lat/lon '{value}': {e}")
        return 0.0

    def _verify_checksum(self, data: str, checksum: str) -> bool:
        """Verify NMEA checksum."""
        try:
            # Remove $ if present
            if data.startswith('$'):
                data = data[1:]

            # Calculate XOR of all characters
            calc = 0
            for char in data:
                calc ^= ord(char)

            # Compare with provided checksum
            return calc == int(checksum, 16)
        except (ValueError, TypeError) as e:
            logger.debug(f"Checksum verification failed: {e}")
            return False

    def _safe_float(self, value: str) -> float:
        """Safely convert to float."""
        try:
            return float(value) if value else 0.0
        except (ValueError, TypeError):
            return 0.0

    def _safe_int(self, value: str) -> int:
        """Safely convert to int."""
        try:
            return int(value) if value else 0
        except (ValueError, TypeError):
            return 0


class SBGSimulator:
    """
    Simulates SBG NMEA output for testing without hardware.

    Usage:
        sim = SBGSimulator()
        sim.start()  # Creates virtual port

        # Connect parser to virtual port
        parser = SBGNmeaParser(port=sim.port)
    """

    def __init__(self, wave_height_m: float = 2.0, wave_period_s: float = 8.0):
        self.wave_height = wave_height_m
        self.wave_period = wave_period_s
        self._running = False
        self._thread = None
        self._callbacks: List[Callable[[str], None]] = []

    def add_output_callback(self, callback: Callable[[str], None]):
        """Add callback for simulated sentences."""
        self._callbacks.append(callback)

    def start(self):
        """Start simulation."""
        self._running = True
        self._thread = threading.Thread(target=self._sim_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop simulation."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _sim_loop(self):
        """Generate simulated NMEA sentences."""
        import math

        t = 0
        while self._running:
            # Simulate wave motion
            omega = 2 * math.pi / self.wave_period
            heave = (self.wave_height / 2) * math.sin(omega * t)
            roll = 5.0 * math.sin(omega * t * 0.8)  # Roll slightly out of phase
            pitch = 2.0 * math.sin(omega * t * 1.2)

            # Generate sentences
            sentences = [
                self._make_psbga(45.0, roll, pitch),
                self._make_psbgi(0.1, 0.05, 9.81 + heave * omega**2, 0.01, 0.02, 0.005),
                self._make_gprmc(48.8566, 2.3522, 12.5, 270.0),
            ]

            for sentence in sentences:
                for cb in self._callbacks:
                    cb(sentence)

            t += 0.1
            time.sleep(0.1)

    def _make_psbga(self, heading: float, roll: float, pitch: float) -> str:
        """Create PSBGA sentence."""
        data = f"PSBGA,120000.00,{heading:.2f},{roll:.2f},{pitch:.2f},0"
        checksum = self._calc_checksum(data)
        return f"${data}*{checksum:02X}"

    def _make_psbgi(self, ax: float, ay: float, az: float,
                    gx: float, gy: float, gz: float) -> str:
        """Create PSBGI sentence."""
        data = f"PSBGI,120000.00,{ax:.4f},{ay:.4f},{az:.4f},{gx:.4f},{gy:.4f},{gz:.4f}"
        checksum = self._calc_checksum(data)
        return f"${data}*{checksum:02X}"

    def _make_gprmc(self, lat: float, lon: float, speed: float, course: float) -> str:
        """Create GPRMC sentence."""
        lat_deg = int(abs(lat))
        lat_min = (abs(lat) - lat_deg) * 60
        lat_dir = 'N' if lat >= 0 else 'S'

        lon_deg = int(abs(lon))
        lon_min = (abs(lon) - lon_deg) * 60
        lon_dir = 'E' if lon >= 0 else 'W'

        data = (f"GPRMC,120000.00,A,"
                f"{lat_deg:02d}{lat_min:07.4f},{lat_dir},"
                f"{lon_deg:03d}{lon_min:07.4f},{lon_dir},"
                f"{speed:.1f},{course:.1f},140125,,,A")
        checksum = self._calc_checksum(data)
        return f"${data}*{checksum:02X}"

    def _calc_checksum(self, data: str) -> int:
        """Calculate NMEA checksum."""
        result = 0
        for char in data:
            result ^= ord(char)
        return result


def list_serial_ports() -> List[str]:
    """List available serial ports."""
    import glob
    import sys

    if sys.platform.startswith('linux'):
        ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/cu.*') + glob.glob('/dev/tty.usb*')
    elif sys.platform.startswith('win'):
        ports = [f'COM{i}' for i in range(1, 20)]
    else:
        ports = []

    return sorted(ports)


# Quick test function
def test_parser():
    """Test with simulator."""
    print("Starting SBG NMEA test with simulator...")

    # Create simulator
    sim = SBGSimulator(wave_height_m=3.0, wave_period_s=10.0)

    # Collect sentences
    sentences = []
    def collect(s):
        sentences.append(s)
    sim.add_output_callback(collect)

    # Start simulator
    sim.start()
    time.sleep(1.0)
    sim.stop()

    # Create parser properly (without starting serial connection)
    parser = SBGNmeaParser(port="/dev/null")

    # Parse collected sentences directly
    for sentence in sentences:
        parser._parse_sentence(sentence)

    data = parser.get_latest()
    print(f"\nParsed {parser._stats['sentences_parsed']} sentences")
    print(f"Latest data:")
    print(f"  Roll: {data.roll_deg:.2f}°")
    print(f"  Pitch: {data.pitch_deg:.2f}°")
    print(f"  Heading: {data.heading_deg:.2f}°")
    print(f"  Position: {data.latitude:.4f}, {data.longitude:.4f}")
    print(f"  Speed: {data.speed_kts:.1f} kts")
    print(f"  Accel Z: {data.accel_z:.2f} m/s²")


if __name__ == "__main__":
    test_parser()

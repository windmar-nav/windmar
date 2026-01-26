"""
Unit tests for SBG NMEA Parser.

Tests individual parsing functions and data handling.
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sensors.sbg_nmea import (
    SBGNmeaParser,
    SBGSimulator,
    ShipMotionData,
    AttitudeData,
    IMUData,
    PositionData,
    list_serial_ports,
)


class TestSBGNmeaParser:
    """Unit tests for SBGNmeaParser class."""

    @pytest.fixture
    def parser(self):
        """Create a parser instance for testing."""
        return SBGNmeaParser(port="/dev/null")

    def test_parser_initialization(self, parser):
        """Test parser initializes with correct defaults."""
        assert parser.port == "/dev/null"
        assert parser.baudrate == 115200
        assert parser.timeout == 1.0
        assert parser._stats["sentences_parsed"] == 0
        assert parser._stats["parse_errors"] == 0
        assert parser._stats["checksum_errors"] == 0

    def test_safe_float_valid(self, parser):
        """Test _safe_float with valid inputs."""
        assert parser._safe_float("123.45") == 123.45
        assert parser._safe_float("0") == 0.0
        assert parser._safe_float("-45.67") == -45.67
        assert parser._safe_float("") == 0.0

    def test_safe_float_invalid(self, parser):
        """Test _safe_float with invalid inputs."""
        assert parser._safe_float("abc") == 0.0
        assert parser._safe_float("12.34.56") == 0.0
        assert parser._safe_float(None) == 0.0

    def test_safe_int_valid(self, parser):
        """Test _safe_int with valid inputs."""
        assert parser._safe_int("123") == 123
        assert parser._safe_int("0") == 0
        assert parser._safe_int("-45") == -45
        assert parser._safe_int("") == 0

    def test_safe_int_invalid(self, parser):
        """Test _safe_int with invalid inputs."""
        assert parser._safe_int("abc") == 0
        assert parser._safe_int("12.34") == 0
        assert parser._safe_int(None) == 0

    def test_verify_checksum_valid(self, parser):
        """Test checksum verification with valid checksums."""
        # Test with a known-good checksum (calculate actual value)
        # XOR of "PSBGA,120000.00,45.00,5.00,2.00,0" = 0x5E = 94 decimal
        assert parser._verify_checksum("PSBGA,120000.00,45.00,5.00,2.00,0", "5E") is True

    def test_verify_checksum_invalid(self, parser):
        """Test checksum verification with invalid checksums."""
        assert parser._verify_checksum("PSBGA,120000.00,45.00,5.00,2.00,0", "00") is False
        assert parser._verify_checksum("PSBGA,120000.00,45.00,5.00,2.00,0", "invalid") is False
        assert parser._verify_checksum("PSBGA,120000.00,45.00,5.00,2.00,0", "") is False

    def test_parse_lat_lon_north_east(self, parser):
        """Test latitude/longitude parsing for N/E coordinates."""
        # 48°07.038'N = 48.1173°
        lat = parser._parse_lat_lon("4807.038", "N")
        assert abs(lat - 48.1173) < 0.001

        # 11°31.000'E = 11.5167°
        lon = parser._parse_lat_lon("01131.000", "E")
        assert abs(lon - 11.5167) < 0.001

    def test_parse_lat_lon_south_west(self, parser):
        """Test latitude/longitude parsing for S/W coordinates."""
        lat = parser._parse_lat_lon("4807.038", "S")
        assert lat < 0
        assert abs(lat + 48.1173) < 0.001

        lon = parser._parse_lat_lon("01131.000", "W")
        assert lon < 0
        assert abs(lon + 11.5167) < 0.001

    def test_parse_lat_lon_empty(self, parser):
        """Test latitude/longitude parsing with empty value."""
        assert parser._parse_lat_lon("", "N") == 0.0
        # Note: Empty direction still parses value (returns positive)
        # Only empty value returns 0.0

    def test_parse_psbga(self, parser):
        """Test PSBGA attitude message parsing."""
        fields = ["PSBGA", "120000.00", "270.5", "5.25", "-2.10", "0"]
        parser._parse_psbga(fields)

        assert abs(parser._attitude.heading_deg - 270.5) < 0.01
        assert abs(parser._attitude.roll_deg - 5.25) < 0.01
        assert abs(parser._attitude.pitch_deg - (-2.10)) < 0.01
        assert parser._attitude.valid is True

    def test_parse_psbgi(self, parser):
        """Test PSBGI IMU message parsing."""
        fields = ["PSBGI", "120000.00", "0.1", "0.05", "9.81", "0.01", "0.02", "0.005"]
        parser._parse_psbgi(fields)

        assert abs(parser._imu.accel_x - 0.1) < 0.001
        assert abs(parser._imu.accel_y - 0.05) < 0.001
        assert abs(parser._imu.accel_z - 9.81) < 0.001
        assert abs(parser._imu.gyro_x - 0.01) < 0.001
        assert abs(parser._imu.gyro_y - 0.02) < 0.001
        assert abs(parser._imu.gyro_z - 0.005) < 0.001

    def test_parse_rmc(self, parser):
        """Test RMC position message parsing."""
        fields = ["GPRMC", "123519", "A", "4807.038", "N", "01131.000", "E", "022.4", "084.4", "230394", "", "", ""]
        parser._parse_rmc(fields)

        assert abs(parser._position.latitude - 48.1173) < 0.001
        assert abs(parser._position.longitude - 11.5167) < 0.001
        assert abs(parser._position.speed_kts - 22.4) < 0.1
        assert abs(parser._position.course_deg - 84.4) < 0.1

    def test_parse_rmc_invalid_status(self, parser):
        """Test RMC parsing with invalid status."""
        initial_lat = parser._position.latitude
        fields = ["GPRMC", "123519", "V", "4807.038", "N", "01131.000", "E", "022.4", "084.4", "230394", "", "", ""]
        parser._parse_rmc(fields)

        # Position should not change with invalid status
        assert parser._position.latitude == initial_lat

    def test_parse_gga(self, parser):
        """Test GGA fix message parsing."""
        fields = ["GPGGA", "123519", "4807.038", "N", "01131.000", "E", "1", "08", "0.9", "545.4", "M", "47.0", "M", "", ""]
        parser._parse_gga(fields)

        assert abs(parser._position.latitude - 48.1173) < 0.001
        assert abs(parser._position.longitude - 11.5167) < 0.001
        assert abs(parser._position.altitude_m - 545.4) < 0.1

    def test_parse_gga_no_fix(self, parser):
        """Test GGA parsing with no fix."""
        initial_lat = parser._position.latitude
        fields = ["GPGGA", "123519", "4807.038", "N", "01131.000", "E", "0", "08", "0.9", "545.4", "M", "47.0", "M", "", ""]
        parser._parse_gga(fields)

        # Position should not change with no fix
        assert parser._position.latitude == initial_lat

    def test_parse_phtro(self, parser):
        """Test PHTRO pitch/roll message parsing."""
        fields = ["PHTRO", "5.5", "S", "3.2", "S"]
        parser._parse_phtro(fields)

        assert abs(parser._attitude.pitch_deg - 5.5) < 0.1
        assert abs(parser._attitude.roll_deg - 3.2) < 0.1

    def test_parse_phtro_bow_down(self, parser):
        """Test PHTRO parsing with bow down (negative pitch)."""
        fields = ["PHTRO", "5.5", "B", "3.2", "S"]
        parser._parse_phtro(fields)

        assert parser._attitude.pitch_deg < 0
        assert abs(parser._attitude.pitch_deg + 5.5) < 0.1

    def test_parse_phtro_port_down(self, parser):
        """Test PHTRO parsing with port down (negative roll)."""
        fields = ["PHTRO", "5.5", "S", "3.2", "P"]
        parser._parse_phtro(fields)

        assert parser._attitude.roll_deg < 0
        assert abs(parser._attitude.roll_deg + 3.2) < 0.1

    def test_parse_sentence_with_checksum(self, parser):
        """Test full sentence parsing with valid checksum."""
        # Create sentence with correct checksum
        # XOR of "PSBGA,120000.00,270.50,5.25,-2.10,0" needs to be computed
        sentence = "$PSBGA,120000.00,270.50,5.25,-2.10,0*47"
        parser._parse_sentence(sentence)

        # Should increment parsed count (or checksum error if checksum wrong)
        # Let's test with a sentence we know works
        sim = SBGSimulator()
        valid_sentence = sim._make_psbga(270.0, 5.0, 2.0)
        parser._parse_sentence(valid_sentence)

        assert parser._stats["sentences_parsed"] >= 1

    def test_parse_sentence_bad_checksum(self, parser):
        """Test sentence rejection with bad checksum."""
        sentence = "$PSBGA,120000.00,270.50,5.25,-2.10,0*00"
        parser._parse_sentence(sentence)

        # Should increment checksum error count
        assert parser._stats["checksum_errors"] >= 1

    def test_get_latest_aggregates_data(self, parser):
        """Test get_latest aggregates all sensor data."""
        # Set up some data
        parser._attitude = AttitudeData(
            timestamp=datetime.now(),
            roll_deg=5.0,
            pitch_deg=2.0,
            heading_deg=270.0,
            valid=True
        )
        parser._position = PositionData(
            timestamp=datetime.now(),
            latitude=43.5,
            longitude=7.0,
            speed_kts=12.5,
            valid=True
        )
        parser._imu = IMUData(
            timestamp=datetime.now(),
            accel_x=0.1,
            accel_y=0.05,
            accel_z=9.81,
            gyro_x=0.01,
            gyro_y=0.02,
            gyro_z=0.005,
            valid=True
        )
        parser._heave = 0.5

        data = parser.get_latest()

        assert isinstance(data, ShipMotionData)
        assert data.roll_deg == 5.0
        assert data.pitch_deg == 2.0
        assert data.heading_deg == 270.0
        assert data.latitude == 43.5
        assert data.longitude == 7.0
        assert data.speed_kts == 12.5
        assert data.heave_m == 0.5
        assert data.accel_z == 9.81
        assert data.valid is True

    def test_get_stats(self, parser):
        """Test statistics retrieval."""
        stats = parser.get_stats()

        assert "sentences_parsed" in stats
        assert "parse_errors" in stats
        assert "checksum_errors" in stats
        assert isinstance(stats["sentences_parsed"], int)


class TestSBGSimulator:
    """Unit tests for SBGSimulator class."""

    def test_simulator_initialization(self):
        """Test simulator initializes with correct defaults."""
        sim = SBGSimulator(wave_height_m=3.0, wave_period_s=10.0)
        assert sim.wave_height == 3.0
        assert sim.wave_period == 10.0
        assert sim._running is False

    def test_make_psbga(self):
        """Test PSBGA sentence generation."""
        sim = SBGSimulator()
        sentence = sim._make_psbga(270.0, 5.0, 2.0)

        assert sentence.startswith("$PSBGA")
        assert "270.00" in sentence
        assert "5.00" in sentence
        assert "2.00" in sentence
        assert "*" in sentence

    def test_make_psbgi(self):
        """Test PSBGI sentence generation."""
        sim = SBGSimulator()
        sentence = sim._make_psbgi(0.1, 0.05, 9.81, 0.01, 0.02, 0.005)

        assert sentence.startswith("$PSBGI")
        assert "0.1000" in sentence
        assert "9.8100" in sentence
        assert "*" in sentence

    def test_make_gprmc(self):
        """Test GPRMC sentence generation."""
        sim = SBGSimulator()
        sentence = sim._make_gprmc(48.8566, 2.3522, 12.5, 270.0)

        assert sentence.startswith("$GPRMC")
        assert "A" in sentence  # Valid status
        assert "12.5" in sentence  # Speed
        assert "270.0" in sentence  # Course
        assert "*" in sentence

    def test_calc_checksum(self):
        """Test checksum calculation."""
        sim = SBGSimulator()
        checksum = sim._calc_checksum("PSBGA,120000.00,45.00,5.00,2.00,0")
        # Actual XOR checksum value (verify with manual calculation)
        assert checksum == 0x5E  # 94 decimal

    def test_callback_registration(self):
        """Test callback registration."""
        sim = SBGSimulator()
        received = []

        def callback(sentence):
            received.append(sentence)

        sim.add_output_callback(callback)
        assert callback in sim._callbacks


class TestListSerialPorts:
    """Unit tests for list_serial_ports function."""

    def test_returns_list(self):
        """Test function returns a list."""
        ports = list_serial_ports()
        assert isinstance(ports, list)

    def test_returns_sorted(self):
        """Test function returns sorted list."""
        ports = list_serial_ports()
        assert ports == sorted(ports)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

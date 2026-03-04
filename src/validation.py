"""
Input validation utilities for WINDMAR.

Provides validation functions with clear error messages for all user inputs.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union


class ValidationError(Exception):
    """Raised when input validation fails."""

    def __init__(self, field: str, message: str, value=None):
        self.field = field
        self.message = message
        self.value = value
        super().__init__(f"{field}: {message}")


@dataclass
class ValidationResult:
    """Result of a validation check."""

    is_valid: bool
    errors: List[ValidationError]

    @classmethod
    def success(cls) -> "ValidationResult":
        return cls(is_valid=True, errors=[])

    @classmethod
    def failure(cls, errors: List[ValidationError]) -> "ValidationResult":
        return cls(is_valid=False, errors=errors)


def validate_speed(speed_kts: float, field_name: str = "speed_kts") -> None:
    """
    Validate vessel speed in knots.

    Args:
        speed_kts: Speed value in knots
        field_name: Field name for error messages

    Raises:
        ValidationError: If speed is invalid
    """
    if speed_kts is None:
        raise ValidationError(field_name, "Speed is required", speed_kts)

    if not isinstance(speed_kts, (int, float)):
        raise ValidationError(field_name, "Speed must be a number", speed_kts)

    if speed_kts <= 0:
        raise ValidationError(
            field_name,
            "Speed must be positive (got {:.2f} kts)".format(speed_kts),
            speed_kts,
        )

    if speed_kts > 25:
        raise ValidationError(
            field_name,
            "Speed exceeds maximum safe limit of 25 knots (got {:.2f} kts)".format(
                speed_kts
            ),
            speed_kts,
        )


def validate_distance(distance_nm: float, field_name: str = "distance_nm") -> None:
    """
    Validate distance in nautical miles.

    Args:
        distance_nm: Distance value in nautical miles
        field_name: Field name for error messages

    Raises:
        ValidationError: If distance is invalid
    """
    if distance_nm is None:
        raise ValidationError(field_name, "Distance is required", distance_nm)

    if not isinstance(distance_nm, (int, float)):
        raise ValidationError(field_name, "Distance must be a number", distance_nm)

    if distance_nm < 0:
        raise ValidationError(
            field_name,
            "Distance cannot be negative (got {:.2f} nm)".format(distance_nm),
            distance_nm,
        )

    if distance_nm > 20000:
        raise ValidationError(
            field_name,
            "Distance exceeds maximum reasonable value of 20000 nm (got {:.2f} nm)".format(
                distance_nm
            ),
            distance_nm,
        )


def validate_coordinates(
    lat: float, lon: float, lat_field: str = "latitude", lon_field: str = "longitude"
) -> None:
    """
    Validate geographic coordinates.

    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        lat_field: Field name for latitude errors
        lon_field: Field name for longitude errors

    Raises:
        ValidationError: If coordinates are invalid
    """
    if lat is None:
        raise ValidationError(lat_field, "Latitude is required", lat)

    if lon is None:
        raise ValidationError(lon_field, "Longitude is required", lon)

    if not isinstance(lat, (int, float)):
        raise ValidationError(lat_field, "Latitude must be a number", lat)

    if not isinstance(lon, (int, float)):
        raise ValidationError(lon_field, "Longitude must be a number", lon)

    if lat < -90 or lat > 90:
        raise ValidationError(
            lat_field,
            "Latitude must be between -90 and 90 degrees (got {:.4f})".format(lat),
            lat,
        )

    if lon < -180 or lon > 180:
        raise ValidationError(
            lon_field,
            "Longitude must be between -180 and 180 degrees (got {:.4f})".format(lon),
            lon,
        )


def validate_position(
    position: Tuple[float, float], field_name: str = "position"
) -> None:
    """
    Validate a position tuple (lat, lon).

    Args:
        position: Tuple of (latitude, longitude)
        field_name: Field name for error messages

    Raises:
        ValidationError: If position is invalid
    """
    if position is None:
        raise ValidationError(field_name, "Position is required", position)

    if not isinstance(position, (tuple, list)) or len(position) != 2:
        raise ValidationError(
            field_name, "Position must be a tuple of (latitude, longitude)", position
        )

    validate_coordinates(
        position[0], position[1], f"{field_name}.latitude", f"{field_name}.longitude"
    )


def validate_weather(weather: Optional[Dict[str, float]]) -> None:
    """
    Validate weather conditions dictionary.

    Args:
        weather: Weather conditions dict or None

    Raises:
        ValidationError: If weather values are invalid
    """
    if weather is None:
        return  # Weather is optional

    if not isinstance(weather, dict):
        raise ValidationError("weather", "Weather must be a dictionary", weather)

    # Validate wind speed
    if "wind_speed_ms" in weather:
        wind_speed = weather["wind_speed_ms"]
        if not isinstance(wind_speed, (int, float)):
            raise ValidationError(
                "wind_speed_ms", "Wind speed must be a number", wind_speed
            )
        if wind_speed < 0:
            raise ValidationError(
                "wind_speed_ms",
                "Wind speed cannot be negative (got {:.2f} m/s)".format(wind_speed),
                wind_speed,
            )
        if wind_speed > 50:
            raise ValidationError(
                "wind_speed_ms",
                "Wind speed exceeds hurricane force (got {:.2f} m/s, max 50 m/s)".format(
                    wind_speed
                ),
                wind_speed,
            )

    # Validate wind direction
    if "wind_dir_deg" in weather:
        wind_dir = weather["wind_dir_deg"]
        if not isinstance(wind_dir, (int, float)):
            raise ValidationError(
                "wind_dir_deg", "Wind direction must be a number", wind_dir
            )

    # Validate wave height
    if "sig_wave_height_m" in weather:
        wave_height = weather["sig_wave_height_m"]
        if not isinstance(wave_height, (int, float)):
            raise ValidationError(
                "sig_wave_height_m", "Wave height must be a number", wave_height
            )
        if wave_height < 0:
            raise ValidationError(
                "sig_wave_height_m",
                "Wave height cannot be negative (got {:.2f} m)".format(wave_height),
                wave_height,
            )
        if wave_height > 20:
            raise ValidationError(
                "sig_wave_height_m",
                "Wave height exceeds extreme conditions (got {:.2f} m, max 20 m)".format(
                    wave_height
                ),
                wave_height,
            )

    # Validate wave direction
    if "wave_dir_deg" in weather:
        wave_dir = weather["wave_dir_deg"]
        if not isinstance(wave_dir, (int, float)):
            raise ValidationError(
                "wave_dir_deg", "Wave direction must be a number", wave_dir
            )

    # Validate heading
    if "heading_deg" in weather:
        heading = weather["heading_deg"]
        if not isinstance(heading, (int, float)):
            raise ValidationError("heading_deg", "Heading must be a number", heading)


def validate_vessel_specs(specs: Dict) -> None:
    """
    Validate vessel specifications dictionary.

    Args:
        specs: Vessel specifications dict

    Raises:
        ValidationError: If any specification is invalid
    """
    required_fields = [
        ("dwt", 1000, 500000, "tonnes"),
        ("loa", 50, 400, "meters"),
        ("beam", 10, 80, "meters"),
        ("draft_laden", 3, 25, "meters"),
        ("draft_ballast", 2, 20, "meters"),
        ("mcr_kw", 1000, 100000, "kW"),
        ("sfoc_at_mcr", 100, 250, "g/kWh"),
        ("service_speed_laden", 8, 25, "knots"),
        ("service_speed_ballast", 8, 25, "knots"),
    ]

    for field, min_val, max_val, unit in required_fields:
        if field not in specs:
            continue  # Optional field

        value = specs[field]
        if not isinstance(value, (int, float)):
            raise ValidationError(field, f"{field} must be a number", value)

        if value < min_val or value > max_val:
            raise ValidationError(
                field,
                f"{field} must be between {min_val} and {max_val} {unit} (got {value})",
                value,
            )

    # Cross-field validations
    if "draft_laden" in specs and "draft_ballast" in specs:
        if specs["draft_laden"] < specs["draft_ballast"]:
            raise ValidationError(
                "draft_laden",
                "Laden draft ({:.1f}m) must be greater than ballast draft ({:.1f}m)".format(
                    specs["draft_laden"], specs["draft_ballast"]
                ),
                specs["draft_laden"],
            )

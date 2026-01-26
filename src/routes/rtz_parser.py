"""
RTZ Route File Parser.

Parses RTZ (Route Plan Exchange Format) files used by ECDIS systems.
RTZ is an XML-based format defined by IEC 61174.

Security Note:
    Uses defusedxml to prevent XXE (XML External Entity) attacks.
    Never use standard xml.etree.ElementTree for untrusted input.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime

# Use defusedxml to prevent XXE attacks
# See: https://owasp.org/www-community/vulnerabilities/XML_External_Entity_(XXE)_Processing
try:
    import defusedxml.ElementTree as ET
except ImportError:
    # Fallback with security warning - should never happen in production
    import xml.etree.ElementTree as ET
    import warnings
    warnings.warn(
        "defusedxml not installed! XML parsing is vulnerable to XXE attacks. "
        "Install with: pip install defusedxml",
        SecurityWarning
    )

logger = logging.getLogger(__name__)


@dataclass
class Waypoint:
    """A waypoint in a route."""
    id: int
    name: str
    lat: float
    lon: float
    radius: Optional[float] = None  # Turn radius in nm


@dataclass
class RouteLeg:
    """A leg between two waypoints."""
    from_wp: Waypoint
    to_wp: Waypoint
    distance_nm: float
    bearing_deg: float


@dataclass
class Route:
    """A complete route with waypoints."""
    name: str
    waypoints: List[Waypoint]

    @property
    def legs(self) -> List[RouteLeg]:
        """Calculate legs between waypoints."""
        legs = []
        for i in range(len(self.waypoints) - 1):
            wp1 = self.waypoints[i]
            wp2 = self.waypoints[i + 1]
            dist = haversine_distance(wp1.lat, wp1.lon, wp2.lat, wp2.lon)
            bearing = calculate_bearing(wp1.lat, wp1.lon, wp2.lat, wp2.lon)
            legs.append(RouteLeg(wp1, wp2, dist, bearing))
        return legs

    @property
    def total_distance_nm(self) -> float:
        """Total route distance in nautical miles."""
        return sum(leg.distance_nm for leg in self.legs)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great circle distance between two points.

    Args:
        lat1, lon1: First point in degrees
        lat2, lon2: Second point in degrees

    Returns:
        Distance in nautical miles
    """
    import math

    R = 3440.065  # Earth radius in nm

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate initial bearing from point 1 to point 2.

    Args:
        lat1, lon1: First point in degrees
        lat2, lon2: Second point in degrees

    Returns:
        Bearing in degrees (0-360)
    """
    import math

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)

    x = math.sin(dlon) * math.cos(lat2_rad)
    y = (math.cos(lat1_rad) * math.sin(lat2_rad) -
         math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon))

    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


def parse_rtz_file(file_path: Path) -> Route:
    """
    Parse an RTZ route file.

    Args:
        file_path: Path to RTZ file

    Returns:
        Route object with waypoints

    Raises:
        ValueError: If file format is invalid
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Handle namespace
    ns = {'rtz': 'http://www.cirm.org/RTZ/1/1'}

    # Try with namespace first, then without
    route_info = root.find('.//rtz:routeInfo', ns)
    if route_info is None:
        route_info = root.find('.//routeInfo')

    route_name = "Unnamed Route"
    if route_info is not None:
        route_name = route_info.get('routeName', 'Unnamed Route')

    # Find waypoints
    waypoints_elem = root.find('.//rtz:waypoints', ns)
    if waypoints_elem is None:
        waypoints_elem = root.find('.//waypoints')

    if waypoints_elem is None:
        raise ValueError("No waypoints found in RTZ file")

    waypoints = []
    wp_list = waypoints_elem.findall('rtz:waypoint', ns)
    if not wp_list:
        wp_list = waypoints_elem.findall('waypoint')

    for i, wp_elem in enumerate(wp_list):
        # Get position element
        pos = wp_elem.find('rtz:position', ns)
        if pos is None:
            pos = wp_elem.find('position')

        if pos is None:
            continue

        lat = float(pos.get('lat', 0))
        lon = float(pos.get('lon', 0))
        name = wp_elem.get('name', f'WP{i+1}')
        radius = wp_elem.get('radius')

        waypoints.append(Waypoint(
            id=i,
            name=name,
            lat=lat,
            lon=lon,
            radius=float(radius) if radius else None
        ))

    if not waypoints:
        raise ValueError("No valid waypoints found in RTZ file")

    logger.info(f"Parsed RTZ route '{route_name}' with {len(waypoints)} waypoints")
    return Route(name=route_name, waypoints=waypoints)


def parse_rtz_string(rtz_content: str) -> Route:
    """
    Parse RTZ content from string.

    Args:
        rtz_content: RTZ XML content as string

    Returns:
        Route object with waypoints
    """
    root = ET.fromstring(rtz_content)

    # Handle namespace
    ns = {'rtz': 'http://www.cirm.org/RTZ/1/1'}

    route_info = root.find('.//rtz:routeInfo', ns)
    if route_info is None:
        route_info = root.find('.//routeInfo')

    route_name = "Unnamed Route"
    if route_info is not None:
        route_name = route_info.get('routeName', 'Unnamed Route')

    waypoints_elem = root.find('.//rtz:waypoints', ns)
    if waypoints_elem is None:
        waypoints_elem = root.find('.//waypoints')

    if waypoints_elem is None:
        raise ValueError("No waypoints found in RTZ content")

    waypoints = []
    wp_list = waypoints_elem.findall('rtz:waypoint', ns)
    if not wp_list:
        wp_list = waypoints_elem.findall('waypoint')

    for i, wp_elem in enumerate(wp_list):
        pos = wp_elem.find('rtz:position', ns)
        if pos is None:
            pos = wp_elem.find('position')

        if pos is None:
            continue

        lat = float(pos.get('lat', 0))
        lon = float(pos.get('lon', 0))
        name = wp_elem.get('name', f'WP{i+1}')

        waypoints.append(Waypoint(
            id=i,
            name=name,
            lat=lat,
            lon=lon
        ))

    return Route(name=route_name, waypoints=waypoints)


def create_route_from_waypoints(
    waypoints: List[Tuple[float, float]],
    name: str = "Custom Route"
) -> Route:
    """
    Create a Route from a list of (lat, lon) tuples.

    Args:
        waypoints: List of (lat, lon) tuples
        name: Route name

    Returns:
        Route object
    """
    wps = [
        Waypoint(id=i, name=f'WP{i+1}', lat=lat, lon=lon)
        for i, (lat, lon) in enumerate(waypoints)
    ]
    return Route(name=name, waypoints=wps)

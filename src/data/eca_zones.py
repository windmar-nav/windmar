"""
Emission Control Area (ECA) zone definitions.

Contains polygon boundaries for IMO-designated ECAs where stricter
emissions standards apply (MARPOL Annex VI).

ECA Zones require:
- Fuel sulfur content <= 0.1% (since 2015)
- Typically requires use of VLSFO, MGO, or scrubbers

Current designated ECAs:
1. Baltic Sea ECA (SOx)
2. North Sea ECA (SOx)
3. North American ECA (SOx and NOx)
4. US Caribbean ECA (SOx and NOx)
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import math


@dataclass
class ECAZone:
    """Represents an Emission Control Area."""

    name: str
    code: str
    polygon: List[Tuple[float, float]]  # List of (lat, lon) coordinates
    sox_limit: float = 0.1  # Sulfur limit in %
    nox_tier: Optional[int] = None  # NOx tier requirement (III for NAm/Caribbean)
    effective_date: str = "2015-01-01"
    color: str = "#ff6b6b"  # Display color for map
    zone_type: str = "eca"  # "eca" (full) or "seca" (sulfur-only)

    def contains_point(self, lat: float, lon: float) -> bool:
        """
        Check if a point is inside this ECA zone using ray casting algorithm.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            True if point is inside the zone
        """
        n = len(self.polygon)
        inside = False

        p1_lat, p1_lon = self.polygon[0]
        for i in range(1, n + 1):
            p2_lat, p2_lon = self.polygon[i % n]
            if lat > min(p1_lat, p2_lat):
                if lat <= max(p1_lat, p2_lat):
                    if lon <= max(p1_lon, p2_lon):
                        if p1_lat != p2_lat:
                            lon_inters = (lat - p1_lat) * (p2_lon - p1_lon) / (p2_lat - p1_lat) + p1_lon
                        if p1_lon == p2_lon or lon <= lon_inters:
                            inside = not inside
            p1_lat, p1_lon = p2_lat, p2_lon

        return inside

    def to_geojson(self) -> dict:
        """Convert to GeoJSON feature format."""
        coordinates = [[lon, lat] for lat, lon in self.polygon]
        # Close the polygon
        if coordinates[0] != coordinates[-1]:
            coordinates.append(coordinates[0])

        return {
            "type": "Feature",
            "properties": {
                "name": self.name,
                "code": self.code,
                "sox_limit": self.sox_limit,
                "nox_tier": self.nox_tier,
                "effective_date": self.effective_date,
                "color": self.color,
                "zone_type": self.zone_type,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [coordinates],
            },
        }


# Baltic Sea ECA - SOx control
# Covers the Baltic Sea including the Gulf of Bothnia, Gulf of Finland,
# and entrance to the Baltic up to Skagen
# Simplified polygon covering the entire sea area
BALTIC_SEA_ECA = ECAZone(
    name="Baltic Sea SECA",
    code="BALTIC",
    color="#eab308",
    zone_type="seca",
    polygon=[
        # Simplified boundary covering Baltic Sea
        (53.5, 9.5),    # SW - near Kiel
        (54.0, 10.0),   # Denmark straits
        (54.5, 12.0),   # South Denmark
        (54.0, 14.5),   # Poland
        (54.5, 19.5),   # Kaliningrad
        (55.5, 21.0),   # Lithuania
        (56.5, 21.0),   # Latvia
        (58.0, 24.5),   # Estonia
        (60.5, 28.5),   # Gulf of Finland
        (60.5, 30.5),   # East end
        (66.0, 26.0),   # Gulf of Bothnia north
        (66.0, 22.0),   # Sweden north
        (63.0, 18.0),   # Sweden mid
        (59.5, 17.5),   # Stockholm area
        (57.5, 12.0),   # South Sweden
        (57.75, 10.5),  # Skagen
        (56.0, 8.0),    # Jutland west
        (53.5, 9.5),    # Back to start
    ],
)


# North Sea ECA - SOx control
# Covers the North Sea, English Channel, and approaches
# Simplified polygon covering the water area
NORTH_SEA_ECA = ECAZone(
    name="North Sea SECA",
    code="NORTHSEA",
    color="#eab308",
    zone_type="seca",
    polygon=[
        # Simplified boundary covering North Sea and English Channel
        (48.0, -6.0),    # SW - Atlantic approach
        (49.0, -5.0),    # English Channel west
        (50.0, -2.0),    # Channel mid
        (50.5, 1.0),     # Dover Strait
        (51.0, 3.0),     # Belgium coast
        (52.0, 5.0),     # Netherlands (Rotterdam area)
        (53.5, 7.0),     # Germany
        (55.0, 9.0),     # Denmark
        (57.5, 10.0),    # Denmark/Skagen
        (58.5, 10.5),    # Sweden
        (62.0, 3.0),     # Norway north
        (62.0, -2.0),    # Norwegian Sea
        (60.0, -4.0),    # Scotland north
        (58.5, -5.0),    # Scotland west
        (55.0, -6.0),    # Ireland
        (52.0, -6.0),    # Celtic Sea
        (50.0, -6.0),    # SW approach
        (48.0, -6.0),    # Back to start
    ],
)


# North American ECA - SOx and NOx control
# Covers waters within 200 nautical miles of US and Canadian coasts (Atlantic/Gulf)
# Simplified polygon approximating the 200nm zone
NORTH_AMERICAN_ECA = ECAZone(
    name="North American ECA",
    code="NAMERICA",
    color="#ff6b6b",
    nox_tier=3,
    polygon=[
        # Atlantic coast simplified
        (50.0, -67.0),   # Canada/Maine
        (45.0, -64.0),   # Nova Scotia
        (42.0, -66.0),   # Massachusetts
        (40.0, -70.0),   # New York
        (37.0, -73.0),   # Virginia
        (33.0, -76.0),   # Carolinas
        (30.0, -78.0),   # Georgia
        (26.0, -78.0),   # Florida
        (24.5, -80.0),   # Florida Keys
        (24.5, -84.0),   # Gulf of Mexico
        (26.0, -86.0),   # Gulf
        (29.0, -89.0),   # Louisiana
        (29.0, -94.0),   # Texas
        (26.0, -97.0),   # South Texas
        # Offshore boundary (200nm approximation)
        (24.0, -99.0),
        (27.0, -98.0),
        (30.0, -96.0),
        (33.0, -93.0),
        (36.0, -88.0),
        (38.0, -82.0),
        (40.0, -78.0),
        (43.0, -75.0),
        (46.0, -71.0),
        (50.0, -67.0),
    ],
)


# North American Pacific ECA
# Covers US and Canadian Pacific coast
# Simplified polygon approximating the 200nm zone
NORTH_AMERICAN_PACIFIC_ECA = ECAZone(
    name="North American Pacific ECA",
    code="NAMERICA_PAC",
    color="#ff6b6b",
    nox_tier=3,
    polygon=[
        # Pacific coast simplified (vertices must include Seattle/Puget Sound)
        (55.0, -130.0),  # British Columbia
        (50.0, -126.0),  # Vancouver Island
        (48.5, -122.0),  # Puget Sound north
        (47.0, -122.0),  # Puget Sound south / Seattle
        (46.0, -124.0),  # Washington coast
        (45.0, -124.0),  # Oregon
        (42.0, -124.0),  # California north
        (38.0, -123.0),  # San Francisco
        (34.0, -118.0),  # Los Angeles
        (32.5, -117.0),  # San Diego
        # 200nm offshore boundary
        (32.0, -121.0),
        (34.0, -124.0),
        (38.0, -127.0),
        (42.0, -130.0),
        (46.0, -132.0),
        (50.0, -134.0),
        (55.0, -136.0),
        (56.0, -134.0),
        (55.0, -130.0),
    ],
)


# US Caribbean ECA - SOx and NOx control
# Covers waters around Puerto Rico and US Virgin Islands
US_CARIBBEAN_ECA = ECAZone(
    name="US Caribbean ECA",
    code="USCARIB",
    color="#feca57",
    nox_tier=3,
    polygon=[
        # Puerto Rico and USVI
        (20.50, -68.50),
        (20.50, -67.50),
        (19.50, -65.00),
        (18.00, -64.00),
        (17.00, -64.50),
        (16.50, -65.00),
        (16.50, -67.00),
        (17.00, -68.00),
        (18.00, -68.50),
        (19.00, -68.50),
        (20.50, -68.50),
    ],
)


# Mediterranean Sea ECA - SOx control (MEPC 80, entered force 2025-05-01)
# Covers entire Mediterranean Sea including Black Sea approaches
MEDITERRANEAN_ECA = ECAZone(
    name="Mediterranean Sea ECA",
    code="MEDSEA",
    color="#22c55e",
    zone_type="seca",
    effective_date="2025-05-01",
    polygon=[
        # Western entrance (Strait of Gibraltar)
        (35.8, -5.8),
        (36.2, -5.3),
        # Spanish coast
        (37.0, -2.0),
        (38.0, 0.0),
        (39.5, 0.5),
        (41.0, 2.0),
        # Gulf of Lion
        (43.5, 4.0),
        (43.5, 6.0),
        # Ligurian Sea / Italian Riviera
        (44.0, 9.5),
        # Tyrrhenian Sea
        (42.0, 12.0),
        # South Italy
        (38.0, 16.0),
        # Adriatic entrance
        (40.0, 19.0),
        # Adriatic north
        (45.5, 14.0),
        (45.0, 13.0),
        # Back down Adriatic
        (41.0, 17.0),
        # Greece west
        (38.0, 20.0),
        # Crete
        (35.0, 24.0),
        # Eastern Med
        (35.0, 28.0),
        (36.5, 30.0),
        # Turkish coast
        (36.5, 32.0),
        (36.0, 36.0),
        # Syria/Lebanon
        (35.0, 36.0),
        (33.0, 35.5),
        # Israel
        (32.0, 34.5),
        # Egypt
        (31.0, 32.5),
        # Libya coast
        (32.0, 25.0),
        (33.0, 20.0),
        (34.0, 12.0),
        # Tunisia
        (37.0, 10.0),
        # Algeria
        (37.0, 6.0),
        (36.5, 2.0),
        (36.0, 0.0),
        # Morocco
        (35.5, -2.0),
        (35.8, -5.8),  # Back to Gibraltar
    ],
)


# All defined ECA zones
ECA_ZONES = [
    BALTIC_SEA_ECA,
    NORTH_SEA_ECA,
    NORTH_AMERICAN_ECA,
    NORTH_AMERICAN_PACIFIC_ECA,
    US_CARIBBEAN_ECA,
    MEDITERRANEAN_ECA,
]


class ECAManager:
    """Manager for checking ECA zone constraints."""

    def __init__(self, zones: List[ECAZone] = None):
        """
        Initialize ECA manager.

        Args:
            zones: List of ECA zones to manage. Defaults to all defined zones.
        """
        self.zones = zones or ECA_ZONES

    def get_zone_at_point(self, lat: float, lon: float) -> Optional[ECAZone]:
        """
        Get the ECA zone at a specific point.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            ECAZone if point is in an ECA, None otherwise
        """
        for zone in self.zones:
            if zone.contains_point(lat, lon):
                return zone
        return None

    def is_in_eca(self, lat: float, lon: float) -> bool:
        """
        Check if a point is in any ECA zone.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            True if point is in any ECA zone
        """
        return self.get_zone_at_point(lat, lon) is not None

    def get_zones_for_route(
        self, waypoints: List[Tuple[float, float]]
    ) -> List[ECAZone]:
        """
        Get all ECA zones that a route passes through.

        Args:
            waypoints: List of (lat, lon) coordinates

        Returns:
            List of unique ECA zones crossed by the route
        """
        zones_crossed = set()
        for lat, lon in waypoints:
            zone = self.get_zone_at_point(lat, lon)
            if zone:
                zones_crossed.add(zone.code)

        return [z for z in self.zones if z.code in zones_crossed]

    def get_eca_distance(
        self, waypoints: List[Tuple[float, float]]
    ) -> Tuple[float, float]:
        """
        Calculate distance inside and outside ECA zones.

        Args:
            waypoints: List of (lat, lon) coordinates

        Returns:
            Tuple of (distance_in_eca_nm, distance_outside_eca_nm)
        """
        from src.optimization.router import MaritimeRouter

        eca_distance = 0.0
        non_eca_distance = 0.0

        for i in range(len(waypoints) - 1):
            lat1, lon1 = waypoints[i]
            lat2, lon2 = waypoints[i + 1]

            # Calculate segment distance
            segment_dist = MaritimeRouter._distance_nm(lat1, lon1, lat2, lon2)

            # Check midpoint of segment
            mid_lat = (lat1 + lat2) / 2
            mid_lon = (lon1 + lon2) / 2

            if self.is_in_eca(mid_lat, mid_lon):
                eca_distance += segment_dist
            else:
                non_eca_distance += segment_dist

        return eca_distance, non_eca_distance

    def to_geojson_collection(self) -> dict:
        """
        Convert all zones to a GeoJSON FeatureCollection.

        Returns:
            GeoJSON FeatureCollection
        """
        return {
            "type": "FeatureCollection",
            "features": [zone.to_geojson() for zone in self.zones],
        }


# Singleton instance
eca_manager = ECAManager()

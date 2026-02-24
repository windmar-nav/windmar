"""
Regulatory Zone Management for WINDMAR.

Handles maritime regulatory zones including:
- ECA (Emission Control Areas) - affects fuel type/cost
- HRA (High Risk Areas) - piracy, security concerns
- TSS (Traffic Separation Schemes) - mandatory routing
- Exclusion zones - military, environmental, restricted
- Custom user-defined zones

Zone interaction levels:
- MANDATORY: Route must pass through (e.g., TSS, canal)
- EXCLUSION: Route must avoid (e.g., military, environmental)
- PENALTY: Route can pass but with cost penalty (e.g., ECA fuel cost)
- ADVISORY: Information only, no routing impact
"""

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import uuid

from .tss_zones import TSS_METADATA, TSS_ZONES

logger = logging.getLogger(__name__)


class ZoneType(Enum):
    """Type of regulatory zone."""
    ECA = "eca"  # Emission Control Area
    SECA = "seca"  # Sulfur Emission Control Area
    HRA = "hra"  # High Risk Area (piracy)
    TSS = "tss"  # Traffic Separation Scheme
    VTS = "vts"  # Vessel Traffic Service
    EXCLUSION = "exclusion"  # General exclusion (military, etc.)
    ENVIRONMENTAL = "environmental"  # Marine protected area
    ICE = "ice"  # Ice zone
    CANAL = "canal"  # Canal/strait requiring transit
    CUSTOM = "custom"  # User-defined


class ZoneInteraction(Enum):
    """How the optimizer should interact with the zone."""
    MANDATORY = "mandatory"  # Must pass through
    EXCLUSION = "exclusion"  # Must avoid
    PENALTY = "penalty"  # Can pass with cost penalty
    ADVISORY = "advisory"  # Information only


@dataclass
class ZoneProperties:
    """Properties of a regulatory zone."""
    name: str
    zone_type: ZoneType
    interaction: ZoneInteraction

    # Penalty factor for PENALTY zones (1.0 = no penalty, 2.0 = double cost)
    penalty_factor: float = 1.0

    # Additional metadata
    authority: Optional[str] = None  # Governing authority (IMO, national, etc.)
    effective_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    notes: Optional[str] = None

    # For ECA zones
    fuel_requirement: Optional[str] = None  # e.g., "0.1% sulfur"

    # For HRA zones
    security_level: Optional[int] = None  # 1-3


@dataclass
class Zone:
    """A regulatory zone defined by a polygon."""
    id: str
    properties: ZoneProperties
    # Polygon as list of (lat, lon) coordinates (closed ring)
    coordinates: List[Tuple[float, float]]
    # Optional: interior rings (holes in polygon)
    holes: List[List[Tuple[float, float]]] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_builtin: bool = False  # True for hardcoded zones

    def to_geojson(self) -> Dict:
        """Convert to GeoJSON Feature."""
        # GeoJSON uses [lon, lat] order
        coords = [[lon, lat] for lat, lon in self.coordinates]

        geometry = {
            "type": "Polygon",
            "coordinates": [coords]
        }

        # Add holes if any
        for hole in self.holes:
            hole_coords = [[lon, lat] for lat, lon in hole]
            geometry["coordinates"].append(hole_coords)

        return {
            "type": "Feature",
            "id": self.id,
            "properties": {
                "name": self.properties.name,
                "zone_type": self.properties.zone_type.value,
                "interaction": self.properties.interaction.value,
                "penalty_factor": self.properties.penalty_factor,
                "authority": self.properties.authority,
                "notes": self.properties.notes,
                "is_builtin": self.is_builtin,
            },
            "geometry": geometry
        }

    @classmethod
    def from_geojson(cls, feature: Dict) -> 'Zone':
        """Create Zone from GeoJSON Feature."""
        props = feature.get("properties", {})
        geom = feature.get("geometry", {})

        # Extract coordinates (convert from [lon, lat] to (lat, lon))
        coords_raw = geom.get("coordinates", [[]])
        coordinates = [(lat, lon) for lon, lat in coords_raw[0]]

        # Extract holes if any
        holes = []
        for ring in coords_raw[1:]:
            hole = [(lat, lon) for lon, lat in ring]
            holes.append(hole)

        zone_props = ZoneProperties(
            name=props.get("name", "Unknown"),
            zone_type=ZoneType(props.get("zone_type", "custom")),
            interaction=ZoneInteraction(props.get("interaction", "advisory")),
            penalty_factor=props.get("penalty_factor", 1.0),
            authority=props.get("authority"),
            notes=props.get("notes"),
        )

        return cls(
            id=feature.get("id", str(uuid.uuid4())),
            properties=zone_props,
            coordinates=coordinates,
            holes=holes,
            is_builtin=props.get("is_builtin", False),
        )


class ZoneChecker:
    """
    Checks points and paths against regulatory zones.

    Uses ray casting algorithm for point-in-polygon tests.
    """

    def __init__(self):
        self.zones: Dict[str, Zone] = {}
        self._load_builtin_zones()

    def _load_builtin_zones(self):
        """Load hardcoded regulatory zones."""
        # Baltic Sea SECA (SOx-only)
        self.add_zone(Zone(
            id="seca_baltic",
            properties=ZoneProperties(
                name="Baltic Sea SECA",
                zone_type=ZoneType.SECA,
                interaction=ZoneInteraction.PENALTY,
                penalty_factor=1.15,  # ~15% fuel cost increase for low-sulfur
                authority="IMO MEPC",
                fuel_requirement="0.1% sulfur",
            ),
            coordinates=[
                (66.0, 4.0), (66.0, 30.5), (59.0, 30.5), (59.0, 23.0),
                (56.0, 23.0), (54.0, 14.0), (54.0, 4.0), (57.5, 4.0),
                (57.5, 10.5), (58.0, 10.5), (66.0, 4.0)
            ],
            is_builtin=True,
        ))

        # North Sea SECA (SOx-only)
        self.add_zone(Zone(
            id="seca_north_sea",
            properties=ZoneProperties(
                name="North Sea SECA",
                zone_type=ZoneType.SECA,
                interaction=ZoneInteraction.PENALTY,
                penalty_factor=1.15,
                authority="IMO MEPC",
                fuel_requirement="0.1% sulfur",
            ),
            coordinates=[
                (62.0, -4.0), (62.0, 4.0), (57.5, 4.0), (57.5, 10.5),
                (51.0, 4.0), (51.0, 2.0), (49.0, -2.0), (49.0, -5.0),
                (62.0, -4.0)
            ],
            is_builtin=True,
        ))

        # English Channel SECA (part of North Sea SECA)
        self.add_zone(Zone(
            id="seca_english_channel",
            properties=ZoneProperties(
                name="English Channel SECA",
                zone_type=ZoneType.SECA,
                interaction=ZoneInteraction.PENALTY,
                penalty_factor=1.15,
                authority="IMO MEPC",
                fuel_requirement="0.1% sulfur",
            ),
            coordinates=[
                (51.0, 2.0), (51.0, -5.0), (48.5, -5.0), (48.5, -2.0),
                (49.0, -2.0), (51.0, 2.0)
            ],
            is_builtin=True,
        ))

        # North American ECA (US/Canada coasts)
        self.add_zone(Zone(
            id="eca_north_america_east",
            properties=ZoneProperties(
                name="North American ECA (East Coast)",
                zone_type=ZoneType.ECA,
                interaction=ZoneInteraction.PENALTY,
                penalty_factor=1.15,
                authority="IMO MEPC / EPA",
                fuel_requirement="0.1% sulfur",
            ),
            coordinates=[
                (50.0, -67.0), (50.0, -60.0), (40.0, -60.0), (25.0, -77.0),
                (25.0, -82.0), (30.0, -82.0), (30.0, -85.0), (35.0, -85.0),
                (40.0, -74.0), (45.0, -67.0), (50.0, -67.0)
            ],
            is_builtin=True,
        ))

        # Caribbean ECA (US)
        self.add_zone(Zone(
            id="eca_caribbean",
            properties=ZoneProperties(
                name="US Caribbean ECA",
                zone_type=ZoneType.ECA,
                interaction=ZoneInteraction.PENALTY,
                penalty_factor=1.15,
                authority="IMO MEPC / EPA",
                fuel_requirement="0.1% sulfur",
            ),
            coordinates=[
                (20.0, -68.0), (20.0, -64.0), (17.0, -64.0), (17.0, -68.0),
                (20.0, -68.0)
            ],
            is_builtin=True,
        ))

        # Indian Ocean High Risk Area (BMP5 — current ITF/IMO boundaries)
        # Updated polygon per BMP5 2024 guidance
        self.add_zone(Zone(
            id="hra_indian_ocean",
            properties=ZoneProperties(
                name="Indian Ocean HRA (BMP5)",
                zone_type=ZoneType.HRA,
                interaction=ZoneInteraction.PENALTY,
                penalty_factor=1.5,
                authority="IMO/UKMTO/MSCHOA",
                security_level=3,
                notes="Armed guards and BMP5 recommended. Covers Gulf of Aden, Arabian Sea, Somali Basin.",
            ),
            coordinates=[
                # BMP5 HRA boundary (approximate)
                (26.0, 57.0),   # Oman/Strait of Hormuz approach
                (26.0, 78.0),   # India west coast offshore
                (10.0, 78.0),   # Sri Lanka approach
                (5.0, 78.0),    # South of India
                (-5.0, 55.0),   # Seychelles area
                (-5.0, 40.0),   # East Africa offshore
                (2.0, 40.0),    # Kenya/Somalia coast
                (12.0, 44.0),   # Gulf of Aden west
                (12.0, 49.0),   # Gulf of Aden east / Socotra
                (15.5, 52.0),   # Oman south coast
                (22.0, 60.0),   # Arabian Sea
                (26.0, 57.0),   # Back to start
            ],
            is_builtin=True,
        ))

        # Gulf of Guinea HRA (BMP West Africa / IMO)
        self.add_zone(Zone(
            id="hra_gulf_of_guinea",
            properties=ZoneProperties(
                name="Gulf of Guinea HRA",
                zone_type=ZoneType.HRA,
                interaction=ZoneInteraction.PENALTY,
                penalty_factor=1.3,
                authority="IMO/MDAT-GoG",
                security_level=2,
                notes="Piracy and armed robbery risk. MDAT-GoG reporting zone.",
            ),
            coordinates=[
                # Gulf of Guinea voluntary reporting area
                (6.0, -5.0),     # Ivory Coast offshore
                (6.0, 9.0),      # Nigeria/Cameroon offshore
                (2.0, 9.0),      # Equatorial Guinea
                (-2.0, 9.0),     # Gabon
                (-2.0, -5.0),    # Open Atlantic
                (6.0, -5.0),     # Back to start
            ],
            is_builtin=True,
        ))

        # Load all TSS zones from tss_zones.py (20 zones worldwide)
        for key, coords in TSS_ZONES.items():
            meta = TSS_METADATA.get(key, {})
            self.add_zone(Zone(
                id=f"tss_{key}",
                properties=ZoneProperties(
                    name=meta.get("name", key.replace("_", " ").title()),
                    zone_type=ZoneType.TSS,
                    interaction=ZoneInteraction.MANDATORY,
                    authority=meta.get("authority", "IMO"),
                    notes=meta.get("notes"),
                ),
                coordinates=coords,
                is_builtin=True,
            ))

        # Suez Canal
        self.add_zone(Zone(
            id="canal_suez",
            properties=ZoneProperties(
                name="Suez Canal",
                zone_type=ZoneType.CANAL,
                interaction=ZoneInteraction.MANDATORY,
                authority="Suez Canal Authority",
                notes="Transit fees apply",
            ),
            coordinates=[
                (31.3, 32.3), (31.3, 32.4), (29.9, 32.6), (29.9, 32.5),
                (31.3, 32.3)
            ],
            is_builtin=True,
        ))

        # Mediterranean Sea SECA (MEPC 80 — entered force 2025-05-01)
        self.add_zone(Zone(
            id="seca_mediterranean",
            properties=ZoneProperties(
                name="Mediterranean Sea SECA",
                zone_type=ZoneType.SECA,
                interaction=ZoneInteraction.PENALTY,
                penalty_factor=1.15,
                authority="IMO MEPC 80",
                fuel_requirement="0.1% sulfur",
                notes="MARPOL Annex VI designation, effective May 2025",
            ),
            coordinates=[
                (46.0, -6.0), (46.0, 36.0), (30.0, 36.0), (30.0, -6.0),
                (46.0, -6.0)
            ],
            is_builtin=True,
        ))

        logger.info(f"Loaded {len(self.zones)} built-in regulatory zones")

    def add_zone(self, zone: Zone):
        """Add a zone to the checker."""
        self.zones[zone.id] = zone

    def remove_zone(self, zone_id: str) -> bool:
        """Remove a zone. Returns True if removed."""
        if zone_id in self.zones:
            zone = self.zones[zone_id]
            if zone.is_builtin:
                logger.warning(f"Cannot remove built-in zone: {zone_id}")
                return False
            del self.zones[zone_id]
            return True
        return False

    def get_zone(self, zone_id: str) -> Optional[Zone]:
        """Get a zone by ID."""
        return self.zones.get(zone_id)

    def get_all_zones(self) -> List[Zone]:
        """Get all zones."""
        return list(self.zones.values())

    def get_zones_by_type(self, zone_type: ZoneType) -> List[Zone]:
        """Get zones of a specific type."""
        return [z for z in self.zones.values() if z.properties.zone_type == zone_type]

    def point_in_polygon(
        self,
        lat: float,
        lon: float,
        polygon: List[Tuple[float, float]],
    ) -> bool:
        """
        Check if point is inside polygon using ray casting.

        Args:
            lat, lon: Point to test
            polygon: List of (lat, lon) vertices

        Returns:
            True if point is inside polygon
        """
        n = len(polygon)
        inside = False

        j = n - 1
        for i in range(n):
            yi, xi = polygon[i]
            yj, xj = polygon[j]

            if ((yi > lat) != (yj > lat)) and \
               (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
                inside = not inside

            j = i

        return inside

    def point_in_zone(self, lat: float, lon: float, zone: Zone) -> bool:
        """Check if point is inside a zone (considering holes)."""
        # Check main polygon
        if not self.point_in_polygon(lat, lon, zone.coordinates):
            return False

        # Check if in any hole (should be outside holes)
        for hole in zone.holes:
            if self.point_in_polygon(lat, lon, hole):
                return False

        return True

    def get_zones_at_point(self, lat: float, lon: float) -> List[Zone]:
        """Get all zones containing a point."""
        result = []
        for zone in self.zones.values():
            if self.point_in_zone(lat, lon, zone):
                result.append(zone)
        return result

    def check_path_zones(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
        num_checks: int = 10,
    ) -> Dict[str, List[Zone]]:
        """
        Check which zones a path segment passes through.

        Returns dict with:
        - 'mandatory': Zones that must be transited
        - 'exclusion': Zones that must be avoided
        - 'penalty': Zones with cost penalties
        - 'advisory': Information-only zones
        """
        result = {
            'mandatory': [],
            'exclusion': [],
            'penalty': [],
            'advisory': [],
        }

        seen_zones = set()

        for i in range(num_checks + 1):
            t = i / num_checks
            lat = lat1 + t * (lat2 - lat1)
            lon = lon1 + t * (lon2 - lon1)

            for zone in self.get_zones_at_point(lat, lon):
                if zone.id not in seen_zones:
                    seen_zones.add(zone.id)
                    interaction = zone.properties.interaction.value
                    if interaction in result:
                        result[interaction].append(zone)

        return result

    @staticmethod
    def _compute_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Compute initial bearing from (lat1,lon1) to (lat2,lon2) in degrees [0,360)."""
        lat1_r = math.radians(lat1)
        lat2_r = math.radians(lat2)
        dlon = math.radians(lon2 - lon1)
        x = math.sin(dlon) * math.cos(lat2_r)
        y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
        brg = math.degrees(math.atan2(x, y))
        return brg % 360.0

    @staticmethod
    def _angular_deviation(bearing: float, target: float) -> float:
        """Minimum angular difference between two bearings in degrees [0, 180]."""
        diff = abs(bearing - target) % 360
        return min(diff, 360 - diff)

    def _check_tss_heading(
        self,
        zone: Zone,
        bearing_deg: float,
    ) -> Tuple[float, Optional[str]]:
        """
        Check if a segment bearing aligns with a TSS zone's required direction.

        For bidirectional zones, either the stated direction or its reciprocal
        (±180°) is acceptable.

        Returns:
            (penalty_multiplier, warning_message_or_None)
        """
        zone_key = zone.id.removeprefix("tss_")
        meta = TSS_METADATA.get(zone_key, {})
        direction = meta.get("direction_deg")
        if direction is None:
            return 0.4, f"Mandatory TSS zone: {zone.properties.name}"

        tolerance = meta.get("tolerance_deg", 20)
        bidirectional = meta.get("bidirectional", True)

        # Check alignment with primary direction
        dev = self._angular_deviation(bearing_deg, direction)
        if dev <= tolerance:
            # Aligned with primary direction — incentivize
            return 0.4, None

        # Check reciprocal for bidirectional zones
        if bidirectional:
            dev_recip = self._angular_deviation(bearing_deg, (direction + 180) % 360)
            if dev_recip <= tolerance:
                return 0.4, None

        # Misaligned — heavy penalty (potential wrong-way transit or crossing)
        return 10.0, f"TSS heading violation: {zone.properties.name} (bearing {bearing_deg:.0f}° vs required {direction:.0f}°±{tolerance}°)"

    def get_path_penalty(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> Tuple[float, List[str]]:
        """
        Calculate penalty factor for a path segment.

        Returns:
            Tuple of (penalty_factor, list of warning messages)
        """
        zones = self.check_path_zones(lat1, lon1, lat2, lon2)

        # Check for exclusion zones (forbidden)
        if zones['exclusion']:
            names = [z.properties.name for z in zones['exclusion']]
            return float('inf'), [f"Path crosses exclusion zone: {', '.join(names)}"]

        # Calculate penalty from penalty zones
        penalty = 1.0
        warnings = []

        for zone in zones['penalty']:
            penalty *= zone.properties.penalty_factor
            warnings.append(f"Transiting {zone.properties.name} (+{(zone.properties.penalty_factor-1)*100:.0f}% cost)")

        # TSS heading alignment check for mandatory zones
        if zones['mandatory']:
            bearing = self._compute_bearing(lat1, lon1, lat2, lon2)
            for zone in zones['mandatory']:
                if zone.properties.zone_type == ZoneType.TSS:
                    tss_mult, tss_warn = self._check_tss_heading(zone, bearing)
                    penalty *= tss_mult
                    if tss_warn:
                        warnings.append(tss_warn)

        # Add advisory info
        for zone in zones['advisory']:
            warnings.append(f"Advisory: {zone.properties.name}")

        return penalty, warnings

    def export_geojson(self) -> Dict:
        """Export all zones as GeoJSON FeatureCollection."""
        return {
            "type": "FeatureCollection",
            "features": [zone.to_geojson() for zone in self.zones.values()]
        }

    def import_geojson(self, geojson: Dict):
        """Import zones from GeoJSON FeatureCollection."""
        if geojson.get("type") != "FeatureCollection":
            raise ValueError("Expected GeoJSON FeatureCollection")

        for feature in geojson.get("features", []):
            zone = Zone.from_geojson(feature)
            # Don't overwrite built-in zones
            if zone.id not in self.zones or not self.zones[zone.id].is_builtin:
                self.zones[zone.id] = zone

    def save_custom_zones(self, filepath: Path):
        """Save custom (non-builtin) zones to file."""
        custom_zones = [z for z in self.zones.values() if not z.is_builtin]
        geojson = {
            "type": "FeatureCollection",
            "features": [z.to_geojson() for z in custom_zones]
        }

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(geojson, f, indent=2, default=str)

    def load_custom_zones(self, filepath: Path):
        """Load custom zones from file."""
        if not filepath.exists():
            return

        with open(filepath, 'r') as f:
            geojson = json.load(f)

        self.import_geojson(geojson)


# Global zone checker instance
_zone_checker: Optional[ZoneChecker] = None


def get_zone_checker() -> ZoneChecker:
    """Get the global zone checker instance."""
    global _zone_checker
    if _zone_checker is None:
        _zone_checker = ZoneChecker()
    return _zone_checker

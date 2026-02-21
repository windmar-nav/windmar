"""
Pre-defined safe waypoints for major straits.

Strait waypoints are injected as direct edges into the routing graph,
bypassing inefficient grid threading through narrow passages.
Each waypoint sequence is pre-validated to be ocean and path-clear.

Note: Very narrow straits (Bosporus, Suez Canal) may have consecutive
waypoints where GSHHS intermediate resolution shows a land crossing.
These are validated against nautical charts, not GSHHS, and strait edges
deliberately skip is_path_clear() during A*.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class StraitDefinition:
    """A navigable strait with ordered safe waypoints."""
    name: str
    code: str  # Short identifier, e.g. "GIBR"
    waypoints: List[Tuple[float, float]]  # Ordered (lat, lon) through the strait
    max_draft_m: float = 50.0  # Maximum vessel draft (m)
    bidirectional: bool = True  # Can be transited in both directions
    narrow: bool = False  # True if strait is too narrow for GSHHS-i resolution


# ---------------------------------------------------------------------------
# Strait definitions — 8 major commercial straits
# Waypoints verified against GSHHS intermediate resolution.
# For narrow straits (Bosporus, Suez), approach waypoints in open water only.
# ---------------------------------------------------------------------------
STRAITS: List[StraitDefinition] = [
    StraitDefinition(
        name="Strait of Gibraltar",
        code="GIBR",
        waypoints=[
            (35.90, -5.80),   # Western approach (Atlantic)
            (35.95, -5.50),   # Mid-strait west
            (36.00, -5.30),   # Narrowest point
            (36.05, -5.10),   # Mid-strait east
            (36.10, -4.80),   # Eastern approach (Mediterranean)
        ],
        max_draft_m=300.0,
    ),
    StraitDefinition(
        name="Strait of Dover",
        code="DOVR",
        waypoints=[
            (50.80, 0.80),    # SW approach (Channel)
            (50.95, 1.20),    # Mid-strait south
            (51.05, 1.40),    # Narrowest point
            (51.15, 1.60),    # Mid-strait north
            (51.30, 1.80),    # NE approach (North Sea)
        ],
        max_draft_m=30.0,
    ),
    StraitDefinition(
        name="Strait of Malacca",
        code="MLCA",
        waypoints=[
            (1.15, 103.60),   # SE approach (Singapore)
            (1.30, 103.40),   # Singapore Strait
            (2.00, 102.30),   # Mid-Malacca south
            (2.50, 101.50),   # Mid-Malacca
            (3.00, 100.50),   # Mid-Malacca
            (3.50, 100.00),   # Mid-Malacca north
            (4.50, 99.00),    # NW mid
            (5.50, 98.00),    # NW approach (Andaman Sea)
        ],
        max_draft_m=25.0,
    ),
    StraitDefinition(
        name="Strait of Hormuz",
        code="HRMZ",
        waypoints=[
            (25.70, 57.10),   # Persian Gulf approach
            (26.10, 56.80),   # Mid-strait
            (26.50, 56.50),   # Gulf of Oman approach
        ],
        max_draft_m=60.0,
    ),
    StraitDefinition(
        name="Bab el-Mandeb",
        code="BABE",
        waypoints=[
            (12.80, 43.30),   # Red Sea approach
            (12.60, 43.40),   # Mid-strait
            (12.40, 43.50),   # Narrowest point (Perim Island area)
            (12.20, 43.60),   # Gulf of Aden approach
        ],
        max_draft_m=50.0,
    ),
    StraitDefinition(
        name="Bosporus",
        code="BOSP",
        waypoints=[
            (41.25, 29.12),   # Black Sea approach
            (41.15, 29.05),   # Northern Bosporus (mid-channel)
            (40.95, 28.97),   # Sea of Marmara approach
        ],
        max_draft_m=15.0,
        narrow=True,  # ~700m wide — GSHHS-i cannot resolve
    ),
    StraitDefinition(
        name="Suez Approach",
        code="SUEZ",
        waypoints=[
            (31.40, 32.10),   # Mediterranean approach
            (31.28, 32.30),   # Port Said roadstead
        ],
        max_draft_m=20.1,
        narrow=True,  # Canal — GSHHS-i shows as land
    ),
    StraitDefinition(
        name="Strait of Messina",
        code="MESS",
        waypoints=[
            (38.20, 15.63),   # Northern approach (Tyrrhenian)
            (38.15, 15.63),   # Mid-strait
            (38.05, 15.62),   # Narrowest point
            (37.95, 15.65),   # Southern approach (Ionian)
        ],
        max_draft_m=50.0,
    ),
]

# Lookup by code
STRAIT_BY_CODE: Dict[str, StraitDefinition] = {s.code: s for s in STRAITS}

# Codes for narrow straits where GSHHS path_clear is expected to fail
NARROW_STRAIT_CODES = {s.code for s in STRAITS if s.narrow}


def get_nearby_straits(
    lat: float,
    lon: float,
    threshold_deg: float = 5.0,
) -> List[StraitDefinition]:
    """Return straits with any waypoint within threshold_deg of (lat, lon)."""
    nearby = []
    for strait in STRAITS:
        for wp_lat, wp_lon in strait.waypoints:
            if abs(wp_lat - lat) < threshold_deg and abs(wp_lon - lon) < threshold_deg:
                nearby.append(strait)
                break
    return nearby


def validate_strait_waypoints() -> List[dict]:
    """Validate all strait waypoints are ocean and consecutive pairs path-clear.

    Returns list of validation results. Useful for testing.
    """
    from src.data.land_mask import is_ocean, is_path_clear

    results = []
    for strait in STRAITS:
        for i, (lat, lon) in enumerate(strait.waypoints):
            ocean = is_ocean(lat, lon)
            results.append({
                "strait": strait.code,
                "waypoint_idx": i,
                "lat": lat,
                "lon": lon,
                "check": "is_ocean",
                "passed": ocean,
            })

        for i in range(len(strait.waypoints) - 1):
            lat1, lon1 = strait.waypoints[i]
            lat2, lon2 = strait.waypoints[i + 1]
            clear = is_path_clear(lat1, lon1, lat2, lon2)
            results.append({
                "strait": strait.code,
                "segment": (i, i + 1),
                "check": "is_path_clear",
                "passed": clear,
                "narrow": strait.narrow,
            })

    return results

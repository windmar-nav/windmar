"""
Tests for TSS mandatory zone enforcement in the optimizer.

Verifies that:
1. Paths through MANDATORY zones receive a cost incentive (factor < 1.0)
2. Paths through EXCLUSION zones are blocked (factor = inf)
3. Paths through PENALTY zones receive a cost increase (factor > 1.0)
4. Paths outside any zone have neutral cost (factor = 1.0)
"""

import pytest
from src.data.regulatory_zones import (
    Zone,
    ZoneChecker,
    ZoneInteraction,
    ZoneProperties,
    ZoneType,
)


@pytest.fixture
def zone_checker():
    """Create a clean ZoneChecker with only test zones (no builtins)."""
    checker = ZoneChecker.__new__(ZoneChecker)
    checker.zones = {}
    checker._enforced_types = None  # enforce all types

    # Test mandatory zone (isolated area in South Pacific to avoid builtin overlap)
    checker.add_zone(Zone(
        id="tss_test_isolated",
        properties=ZoneProperties(
            name="Test Mandatory TSS",
            zone_type=ZoneType.TSS,
            interaction=ZoneInteraction.MANDATORY,
        ),
        coordinates=[
            (-20.0, 170.0), (-20.0, 171.0),
            (-21.0, 171.0), (-21.0, 170.0),
            (-20.0, 170.0),
        ],
        is_builtin=False,
    ))

    # Test penalty zone
    checker.add_zone(Zone(
        id="eca_test",
        properties=ZoneProperties(
            name="Test ECA",
            zone_type=ZoneType.ECA,
            interaction=ZoneInteraction.PENALTY,
            penalty_factor=1.3,
        ),
        coordinates=[
            (-22.0, 170.0), (-22.0, 171.0),
            (-23.0, 171.0), (-23.0, 170.0),
            (-22.0, 170.0),
        ],
        is_builtin=False,
    ))

    # Test exclusion zone
    checker.add_zone(Zone(
        id="hra_test",
        properties=ZoneProperties(
            name="Test HRA",
            zone_type=ZoneType.HRA,
            interaction=ZoneInteraction.EXCLUSION,
        ),
        coordinates=[
            (-25.0, 170.0), (-25.0, 171.0),
            (-25.5, 171.0), (-25.5, 170.0),
            (-25.0, 170.0),
        ],
        is_builtin=False,
    ))

    return checker


class TestMandatoryZoneIncentive:
    """Verify paths through mandatory (TSS) zones get a cost reduction."""

    def test_path_through_mandatory_zone_has_incentive(self, zone_checker):
        """A path through a mandatory zone should have penalty < 1.0."""
        penalty, warnings = zone_checker.get_path_penalty(
            -20.5, 170.2, -20.5, 170.8  # crosses the test TSS
        )
        assert penalty < 1.0, f"Expected incentive (< 1.0), got {penalty}"
        assert any("mandatory" in w.lower() for w in warnings)

    def test_path_outside_all_zones_is_neutral(self, zone_checker):
        """A path outside any zone should have penalty = 1.0."""
        penalty, warnings = zone_checker.get_path_penalty(
            -30.0, 160.0, -30.0, 161.0  # open ocean, no zones
        )
        assert penalty == 1.0
        assert len(warnings) == 0

    def test_path_through_penalty_zone_has_surcharge(self, zone_checker):
        """A path through a penalty zone should have penalty > 1.0."""
        penalty, warnings = zone_checker.get_path_penalty(
            -22.5, 170.2, -22.5, 170.8  # crosses the ECA zone
        )
        assert penalty > 1.0, f"Expected surcharge (> 1.0), got {penalty}"

    def test_path_through_exclusion_zone_is_blocked(self, zone_checker):
        """A path through an exclusion zone should have penalty = inf."""
        penalty, warnings = zone_checker.get_path_penalty(
            -25.2, 170.2, -25.2, 170.8  # crosses the HRA zone
        )
        assert penalty == float('inf')
        assert any("exclusion" in w.lower() for w in warnings)

    def test_mandatory_incentive_value(self, zone_checker):
        """Mandatory zone cost reduction should be 0.4 (60% incentive)."""
        penalty, _ = zone_checker.get_path_penalty(
            -20.5, 170.2, -20.5, 170.8
        )
        assert abs(penalty - 0.4) < 0.01, f"Expected ~0.4, got {penalty}"

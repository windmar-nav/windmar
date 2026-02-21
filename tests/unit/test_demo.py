"""Tests for demo mode helpers and tiered access."""
import pytest
from unittest.mock import patch, MagicMock

import bcrypt
from fastapi import HTTPException

from api.demo import (
    limit_demo_frames,
    get_user_tier,
    is_demo_user,
    require_not_demo,
    _parse_hashes,
    _check_key_against_hashes,
)


# ============================================================================
# Helpers
# ============================================================================

def _hash(plain: str) -> str:
    """Create a bcrypt hash for testing."""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt(rounds=4)).decode("utf-8")


def _make_request(api_key: str = "") -> MagicMock:
    """Create a mock Request with an X-API-Key header."""
    request = MagicMock()
    request.headers = {"X-API-Key": api_key} if api_key else {}
    return request


# ============================================================================
# limit_demo_frames tests (unchanged from before)
# ============================================================================

class TestLimitDemoFrames:
    """Tests for limit_demo_frames()."""

    def test_filters_frames_to_step_12(self):
        result = {
            "cached_hours": 41,
            "frames": {str(h): {"data": h} for h in range(0, 121, 3)},
        }
        filtered = limit_demo_frames(result)
        assert set(filtered["frames"].keys()) == {
            "0", "12", "24", "36", "48", "60", "72", "84", "96", "108", "120"
        }
        assert filtered["cached_hours"] == 11

    def test_custom_step(self):
        result = {
            "frames": {str(h): {} for h in range(0, 121, 3)},
        }
        with patch("api.demo.settings") as mock_settings:
            mock_settings.demo_forecast_step = 24
            filtered = limit_demo_frames(result)
        assert set(filtered["frames"].keys()) == {
            "0", "24", "48", "72", "96", "120"
        }

    def test_passthrough_non_dict(self):
        from starlette.responses import Response
        resp = Response(content=b"test")
        assert limit_demo_frames(resp) is resp

    def test_passthrough_no_frames_key(self):
        result = {"data": "something"}
        assert limit_demo_frames(result) is result

    def test_empty_frames(self):
        result = {"frames": {}, "cached_hours": 0}
        filtered = limit_demo_frames(result)
        assert filtered["frames"] == {}
        assert filtered["cached_hours"] == 0


# ============================================================================
# _parse_hashes tests
# ============================================================================

class TestParseHashes:
    def test_empty_string(self):
        assert _parse_hashes("") == []

    def test_single_hash(self):
        assert _parse_hashes("$2b$04$abc") == ["$2b$04$abc"]

    def test_multiple_hashes(self):
        result = _parse_hashes("$2b$04$abc, $2b$04$def , $2b$04$ghi")
        assert result == ["$2b$04$abc", "$2b$04$def", "$2b$04$ghi"]

    def test_strips_whitespace(self):
        result = _parse_hashes("  $2b$04$abc  ,  $2b$04$def  ")
        assert result == ["$2b$04$abc", "$2b$04$def"]

    def test_skips_empty_entries(self):
        result = _parse_hashes("$2b$04$abc,,,$2b$04$def")
        assert result == ["$2b$04$abc", "$2b$04$def"]


# ============================================================================
# _check_key_against_hashes tests
# ============================================================================

class TestCheckKeyAgainstHashes:
    def test_match(self):
        h = _hash("mykey")
        assert _check_key_against_hashes("mykey", [h]) is True

    def test_no_match(self):
        h = _hash("mykey")
        assert _check_key_against_hashes("wrongkey", [h]) is False

    def test_match_among_multiple(self):
        h1 = _hash("key1")
        h2 = _hash("key2")
        h3 = _hash("key3")
        assert _check_key_against_hashes("key2", [h1, h2, h3]) is True

    def test_empty_hashes(self):
        assert _check_key_against_hashes("anykey", []) is False

    def test_invalid_hash_skipped(self):
        h_valid = _hash("goodkey")
        assert _check_key_against_hashes("goodkey", ["not_a_hash", h_valid]) is True


# ============================================================================
# get_user_tier tests
# ============================================================================

class TestGetUserTier:
    def test_no_key_returns_anonymous(self):
        req = _make_request("")
        with patch("api.demo.settings") as s:
            s.full_api_key_hashes = ""
            s.demo_api_key_hashes = ""
            s.demo_api_key_hash = None
            assert get_user_tier(req) == "anonymous"

    def test_full_key_returns_full(self):
        h = _hash("full-key-1")
        req = _make_request("full-key-1")
        with patch("api.demo.settings") as s:
            s.full_api_key_hashes = h
            s.demo_api_key_hashes = ""
            s.demo_api_key_hash = None
            assert get_user_tier(req) == "full"

    def test_demo_key_returns_demo(self):
        h = _hash("demo-key-1")
        req = _make_request("demo-key-1")
        with patch("api.demo.settings") as s:
            s.full_api_key_hashes = ""
            s.demo_api_key_hashes = h
            s.demo_api_key_hash = None
            assert get_user_tier(req) == "demo"

    def test_legacy_key_returns_demo(self):
        h = _hash("windmar-demo-2026")
        req = _make_request("windmar-demo-2026")
        with patch("api.demo.settings") as s:
            s.full_api_key_hashes = ""
            s.demo_api_key_hashes = ""
            s.demo_api_key_hash = h
            assert get_user_tier(req) == "demo"

    def test_full_takes_priority_over_demo(self):
        """If a key matches both full and demo hashes, full wins."""
        h = _hash("shared-key")
        req = _make_request("shared-key")
        with patch("api.demo.settings") as s:
            s.full_api_key_hashes = h
            s.demo_api_key_hashes = h
            s.demo_api_key_hash = None
            assert get_user_tier(req) == "full"

    def test_wrong_key_returns_anonymous(self):
        h = _hash("real-key")
        req = _make_request("wrong-key")
        with patch("api.demo.settings") as s:
            s.full_api_key_hashes = ""
            s.demo_api_key_hashes = h
            s.demo_api_key_hash = None
            assert get_user_tier(req) == "anonymous"

    def test_multiple_demo_keys(self):
        h1 = _hash("demo-key-1")
        h2 = _hash("demo-key-2")
        req = _make_request("demo-key-2")
        with patch("api.demo.settings") as s:
            s.full_api_key_hashes = ""
            s.demo_api_key_hashes = f"{h1},{h2}"
            s.demo_api_key_hash = None
            assert get_user_tier(req) == "demo"


# ============================================================================
# is_demo_user tests
# ============================================================================

class TestIsDemoUser:
    def test_demo_key_is_demo_user(self):
        h = _hash("demo-key")
        req = _make_request("demo-key")
        with patch("api.demo.settings") as s:
            s.full_api_key_hashes = ""
            s.demo_api_key_hashes = h
            s.demo_api_key_hash = None
            assert is_demo_user(req) is True

    def test_full_key_is_not_demo_user(self):
        h = _hash("full-key")
        req = _make_request("full-key")
        with patch("api.demo.settings") as s:
            s.full_api_key_hashes = h
            s.demo_api_key_hashes = ""
            s.demo_api_key_hash = None
            assert is_demo_user(req) is False

    def test_anonymous_is_not_demo_user(self):
        req = _make_request("")
        with patch("api.demo.settings") as s:
            s.full_api_key_hashes = ""
            s.demo_api_key_hashes = ""
            s.demo_api_key_hash = None
            assert is_demo_user(req) is False


# ============================================================================
# require_not_demo guard tests
# ============================================================================

class TestRequireNotDemo:
    """Security tests for the require_not_demo guard."""

    def test_non_demo_deployment_always_passes(self):
        guard = require_not_demo("Test feature")
        req = _make_request("")
        with patch("api.demo.settings") as s:
            s.demo_mode = False
            # Should not raise for any tier
            guard(req)

    def test_full_tier_passes(self):
        h = _hash("full-key")
        guard = require_not_demo("Test feature")
        req = _make_request("full-key")
        with patch("api.demo.settings") as s:
            s.demo_mode = True
            s.full_api_key_hashes = h
            s.demo_api_key_hashes = ""
            s.demo_api_key_hash = None
            # Should not raise
            guard(req)

    def test_demo_tier_blocked(self):
        h = _hash("demo-key")
        guard = require_not_demo("Test feature")
        req = _make_request("demo-key")
        with patch("api.demo.settings") as s:
            s.demo_mode = True
            s.full_api_key_hashes = ""
            s.demo_api_key_hashes = h
            s.demo_api_key_hash = None
            with pytest.raises(HTTPException) as exc_info:
                guard(req)
            assert exc_info.value.status_code == 403

    def test_anonymous_blocked(self):
        """CRITICAL: anonymous requests must be blocked in demo mode."""
        guard = require_not_demo("Test feature")
        req = _make_request("")
        with patch("api.demo.settings") as s:
            s.demo_mode = True
            s.full_api_key_hashes = ""
            s.demo_api_key_hashes = ""
            s.demo_api_key_hash = None
            with pytest.raises(HTTPException) as exc_info:
                guard(req)
            assert exc_info.value.status_code == 403

    def test_invalid_key_blocked(self):
        """Invalid keys must be blocked (not just missing keys)."""
        h = _hash("real-key")
        guard = require_not_demo("Test feature")
        req = _make_request("wrong-key")
        with patch("api.demo.settings") as s:
            s.demo_mode = True
            s.full_api_key_hashes = ""
            s.demo_api_key_hashes = h
            s.demo_api_key_hash = None
            with pytest.raises(HTTPException) as exc_info:
                guard(req)
            assert exc_info.value.status_code == 403


# ============================================================================
# Constant-time check tests
# ============================================================================

class TestConstantTimeChecks:
    """Verify _check_key_against_hashes doesn't return early."""

    def test_checks_all_hashes_even_after_match(self):
        """The function must check ALL hashes, not return on first match."""
        h1 = _hash("key1")
        h2 = _hash("key2")
        h3 = _hash("key3")
        # key1 matches h1 (first hash), but h2 and h3 must still be checked
        with patch("bcrypt.checkpw", wraps=bcrypt.checkpw) as mock_check:
            result = _check_key_against_hashes("key1", [h1, h2, h3])
            assert result is True
            # All 3 hashes must have been checked
            assert mock_check.call_count == 3

    def test_no_match_checks_all(self):
        h1 = _hash("key1")
        h2 = _hash("key2")
        with patch("bcrypt.checkpw", wraps=bcrypt.checkpw) as mock_check:
            result = _check_key_against_hashes("wrong", [h1, h2])
            assert result is False
            assert mock_check.call_count == 2

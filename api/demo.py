"""
Demo mode guards for WINDMAR API.

Provides FastAPI dependencies and helpers that block or stub endpoints
when DEMO_MODE=true. Supports tiered access: demo keys get restricted
features (frame limiting, blocked writes), full keys unlock everything.
"""

import logging

import bcrypt
from fastapi import HTTPException, Request

from api.config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# Tier resolution
# ============================================================================

def _parse_hashes(csv: str) -> list[str]:
    """Parse comma-separated bcrypt hashes, stripping whitespace."""
    if not csv:
        return []
    return [h.strip() for h in csv.split(",") if h.strip()]


def _check_key_against_hashes(plain_key: str, hashes: list[str]) -> bool:
    """Check a plain-text key against a list of bcrypt hashes.

    Always iterates ALL hashes to prevent timing oracles.
    """
    key_bytes = plain_key.encode("utf-8")
    matched = False
    for h in hashes:
        try:
            if bcrypt.checkpw(key_bytes, h.encode("utf-8")):
                matched = True
            # Do NOT return early — constant-time iteration
        except Exception:
            continue
    return matched


def get_user_tier(request: Request) -> str:
    """Resolve user tier from X-API-Key header.

    Returns ``"full"``, ``"demo"``, or ``"anonymous"``.
    All hash lists are checked unconditionally to prevent timing oracles.
    """
    api_key = request.headers.get("X-API-Key", "")
    if not api_key:
        return "anonymous"

    # Check ALL tiers unconditionally (constant-time)
    full_hashes = _parse_hashes(settings.full_api_key_hashes)
    is_full = bool(full_hashes) and _check_key_against_hashes(api_key, full_hashes)

    demo_hashes = _parse_hashes(settings.demo_api_key_hashes)
    is_demo = bool(demo_hashes) and _check_key_against_hashes(api_key, demo_hashes)

    # Legacy single demo key (backwards compat)
    is_legacy = False
    if settings.demo_api_key_hash:
        try:
            is_legacy = bcrypt.checkpw(
                api_key.encode("utf-8"),
                settings.demo_api_key_hash.encode("utf-8"),
            )
        except Exception:
            pass

    # Priority: full > demo > legacy > anonymous
    if is_full:
        return "full"
    if is_demo or is_legacy:
        return "demo"
    return "anonymous"


def is_demo_user(request: Request) -> bool:
    """Return True if the current request comes from a demo-tier user."""
    return get_user_tier(request) == "demo"


# ============================================================================
# FastAPI Depends guards
# ============================================================================

def require_not_demo(feature_name: str = "This feature"):
    """FastAPI Depends() guard — only full-tier users pass on demo deployments.

    On non-demo deployments (DEMO_MODE=false), always passes.
    On demo deployments, only ``"full"`` tier passes; both ``"demo"``
    and ``"anonymous"`` (missing/invalid key) get 403.
    """

    def _guard(request: Request):
        if not settings.demo_mode:
            return
        tier = get_user_tier(request)
        if tier != "full":
            raise HTTPException(
                status_code=403,
                detail=f"{feature_name} is disabled in demo mode.",
            )

    return _guard


def demo_mode_response(feature_name: str = "This feature"):
    """Return a 200 JSON stub for non-critical endpoints in demo mode."""
    return {
        "status": "demo",
        "message": f"{feature_name} is disabled in demo mode. "
        "Pre-loaded weather data is served from the database snapshot.",
    }


def is_demo() -> bool:
    """Check if demo mode is active (global deployment flag)."""
    return settings.demo_mode


def limit_demo_frames(result: dict) -> dict:
    """Filter forecast frames to keep only every Nth hour in demo mode.

    Reduces 41 frames (0-120h @ 3h) to ~11 frames (0, 12, 24, ..., 120)
    based on ``settings.demo_forecast_step``.  Passes through non-dict
    results and results without a ``frames`` key unchanged.
    """
    if not isinstance(result, dict) or "frames" not in result:
        return result

    step = settings.demo_forecast_step
    frames = result["frames"]
    filtered = {
        k: v for k, v in frames.items()
        if int(k) % step == 0
    }
    result["frames"] = filtered
    if "cached_hours" in result:
        result["cached_hours"] = len(filtered)
    return result

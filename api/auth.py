"""
Authentication and authorization for WINDMAR API.
"""
from fastapi import HTTPException, Security, status, Depends
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session
from datetime import datetime, timezone
import bcrypt
import secrets
import logging

from api.database import get_db
from api.models import APIKey
from api.config import settings

logger = logging.getLogger(__name__)

# API Key security scheme
api_key_header = APIKeyHeader(
    name=settings.api_key_header,
    auto_error=False
)


def generate_api_key() -> str:
    """
    Generate a secure random API key.

    Returns:
        str: Secure random API key (32 bytes, hex encoded)
    """
    return secrets.token_urlsafe(32)


def hash_api_key(api_key: str) -> str:
    """
    Hash an API key using bcrypt.

    Args:
        api_key: Plain text API key

    Returns:
        str: Hashed API key
    """
    return bcrypt.hashpw(
        api_key.encode('utf-8'),
        bcrypt.gensalt(rounds=settings.bcrypt_rounds)
    ).decode('utf-8')


def verify_api_key(plain_key: str, hashed_key: str) -> bool:
    """
    Verify an API key against its hash.

    Args:
        plain_key: Plain text API key
        hashed_key: Hashed API key

    Returns:
        bool: True if key matches hash
    """
    try:
        return bcrypt.checkpw(
            plain_key.encode('utf-8'),
            hashed_key.encode('utf-8')
        )
    except Exception as e:
        logger.error(f"API key verification error: {e}")
        return False


async def get_api_key(
    api_key: str = Security(api_key_header),
    db: Session = Depends(get_db)
) -> APIKey:
    """
    Validate API key from request header.

    Args:
        api_key: API key from request header
        db: Database session

    Returns:
        APIKey: Valid API key model

    Raises:
        HTTPException: If API key is invalid or missing
    """
    # If authentication is disabled, skip validation
    if not settings.auth_enabled:
        logger.warning("Authentication is disabled!")
        return None

    # Check if API key is provided
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is required",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Query all active API keys
    api_keys = db.query(APIKey).filter(
        APIKey.is_active == True
    ).all()

    # Check each key
    for key_obj in api_keys:
        if verify_api_key(api_key, key_obj.key_hash):
            # Check expiration
            if key_obj.expires_at and key_obj.expires_at < datetime.now(timezone.utc):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key has expired",
                )

            # Update last used timestamp
            key_obj.last_used_at = datetime.now(timezone.utc)
            db.commit()

            logger.info(f"API key authenticated: {key_obj.name}")
            return key_obj

    # No matching key found
    logger.warning(f"Invalid API key attempted")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key",
    )


async def get_optional_api_key(
    api_key: str = Security(api_key_header),
    db: Session = Depends(get_db)
) -> APIKey | None:
    """
    Validate API key but don't require it.
    Used for endpoints that have both authenticated and public access.

    Args:
        api_key: API key from request header
        db: Database session

    Returns:
        APIKey | None: Valid API key model or None
    """
    if not api_key:
        return None

    try:
        return await get_api_key(api_key, db)
    except HTTPException:
        return None


def create_api_key_in_db(
    db: Session,
    name: str,
    rate_limit: int = 1000,
    expires_at: datetime = None,
    metadata: dict = None
) -> tuple[str, APIKey]:
    """
    Create a new API key in the database.

    Args:
        db: Database session
        name: Name for the API key
        rate_limit: Rate limit for this key
        expires_at: Expiration timestamp
        metadata: Additional metadata

    Returns:
        tuple: (plain_text_key, api_key_model)
    """
    # Generate new API key
    plain_key = generate_api_key()
    key_hash = hash_api_key(plain_key)

    # Create database record
    api_key_obj = APIKey(
        key_hash=key_hash,
        name=name,
        rate_limit=rate_limit,
        expires_at=expires_at,
        metadata=metadata or {}
    )

    db.add(api_key_obj)
    db.commit()
    db.refresh(api_key_obj)

    logger.info(f"Created API key: {name}")

    # Return plain key (only time it's visible!) and model
    return plain_key, api_key_obj


def revoke_api_key(db: Session, key_id: str) -> bool:
    """
    Revoke an API key.

    Args:
        db: Database session
        key_id: UUID of API key to revoke

    Returns:
        bool: True if key was revoked
    """
    api_key = db.query(APIKey).filter(APIKey.id == key_id).first()
    if not api_key:
        return False

    api_key.is_active = False
    db.commit()

    logger.info(f"Revoked API key: {api_key.name}")
    return True

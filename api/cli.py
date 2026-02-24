#!/usr/bin/env python3
"""
WINDMAR API CLI Tool.

Command-line interface for administrative tasks:
- API key management (create, list, revoke)
- Database operations
- Health checks

Usage:
    python -m api.cli create-api-key --name "My App"
    python -m api.cli list-api-keys
    python -m api.cli revoke-api-key --id <uuid>
    python -m api.cli check-health
"""
import argparse
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

# Ensure imports work
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_api_key(
    name: str,
    rate_limit: int = 1000,
    expires_days: Optional[int] = None
) -> None:
    """Create a new API key."""
    from api.database import get_db_context
    from api.auth import create_api_key_in_db

    expires_at = None
    if expires_days:
        expires_at = datetime.now(timezone.utc) + timedelta(days=expires_days)

    with get_db_context() as db:
        plain_key, api_key_obj = create_api_key_in_db(
            db=db,
            name=name,
            rate_limit=rate_limit,
            expires_at=expires_at,
        )

    print("\n" + "=" * 60)
    print("API KEY CREATED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nName: {name}")
    print(f"Key ID: {api_key_obj.id}")
    print(f"Rate Limit: {rate_limit} requests/hour")
    if expires_at:
        print(f"Expires: {expires_at.isoformat()}")
    else:
        print("Expires: Never")
    print(f"\n{'*' * 60}")
    print(f"API KEY: {plain_key}")
    print(f"{'*' * 60}")
    print("\nSAVE THIS KEY NOW - IT CANNOT BE RETRIEVED LATER!")
    print("=" * 60 + "\n")


def list_api_keys() -> None:
    """List all API keys."""
    from api.database import get_db_context
    from api.models import APIKey

    with get_db_context() as db:
        keys = db.query(APIKey).order_by(APIKey.created_at.desc()).all()

    if not keys:
        print("\nNo API keys found.")
        return

    print("\n" + "=" * 80)
    print("API KEYS")
    print("=" * 80)
    print(f"{'ID':<36} {'Name':<20} {'Active':<8} {'Rate Limit':<12} {'Last Used':<20}")
    print("-" * 80)

    for key in keys:
        last_used = key.last_used_at.strftime("%Y-%m-%d %H:%M") if key.last_used_at else "Never"
        print(
            f"{str(key.id):<36} "
            f"{key.name[:18]:<20} "
            f"{'Yes' if key.is_active else 'No':<8} "
            f"{key.rate_limit:<12} "
            f"{last_used:<20}"
        )

    print("=" * 80)
    print(f"Total: {len(keys)} key(s)\n")


def revoke_api_key(key_id: str) -> None:
    """Revoke an API key."""
    from api.database import get_db_context
    from api.auth import revoke_api_key as revoke_key

    with get_db_context() as db:
        success = revoke_key(db, key_id)

    if success:
        print(f"\nAPI key {key_id} has been revoked.")
    else:
        print(f"\nError: API key {key_id} not found.")
        sys.exit(1)


def check_health() -> None:
    """Check API health."""
    import requests

    url = "http://localhost:8000/api/health"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"\nAPI Status: {data.get('status', 'unknown')}")
            print(f"Version: {data.get('version', 'unknown')}")
            print(f"Timestamp: {data.get('timestamp', 'unknown')}")
        else:
            print(f"\nAPI returned status code: {response.status_code}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to API. Is the server running?")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


def init_db() -> None:
    """Initialize the database."""
    from api.database import init_db as do_init

    print("Initializing database...")
    do_init()
    print("Database initialized successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="WINDMAR API CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Create an API key:
    python -m api.cli create-api-key --name "Production App"

  Create an API key that expires in 90 days:
    python -m api.cli create-api-key --name "Trial Key" --expires-days 90

  List all API keys:
    python -m api.cli list-api-keys

  Revoke an API key:
    python -m api.cli revoke-api-key --id 12345678-1234-1234-1234-123456789abc

  Check API health:
    python -m api.cli check-health

  Initialize database:
    python -m api.cli init-db
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # create-api-key
    create_parser = subparsers.add_parser("create-api-key", help="Create a new API key")
    create_parser.add_argument("--name", required=True, help="Name for the API key")
    create_parser.add_argument(
        "--rate-limit",
        type=int,
        default=1000,
        help="Rate limit (requests per hour, default: 1000)"
    )
    create_parser.add_argument(
        "--expires-days",
        type=int,
        help="Number of days until expiration (default: never)"
    )

    # list-api-keys
    subparsers.add_parser("list-api-keys", help="List all API keys")

    # revoke-api-key
    revoke_parser = subparsers.add_parser("revoke-api-key", help="Revoke an API key")
    revoke_parser.add_argument("--id", required=True, help="UUID of the API key to revoke")

    # check-health
    subparsers.add_parser("check-health", help="Check API health")

    # init-db
    subparsers.add_parser("init-db", help="Initialize the database")

    args = parser.parse_args()

    if args.command == "create-api-key":
        create_api_key(args.name, args.rate_limit, args.expires_days)
    elif args.command == "list-api-keys":
        list_api_keys()
    elif args.command == "revoke-api-key":
        revoke_api_key(args.id)
    elif args.command == "check-health":
        check_health()
    elif args.command == "init-db":
        init_db()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

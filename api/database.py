"""
Database connection and session management.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator
import logging

from api.config import settings

logger = logging.getLogger(__name__)

# Create SQLAlchemy engine
engine = create_engine(
    settings.database_url,
    echo=settings.db_echo,
    pool_pre_ping=True,  # Verify connections before using
    pool_size=10,
    max_overflow=20
)

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Create declarative base for models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency for database sessions.

    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.

    Usage:
        with get_db_context() as db:
            db.query(Model).all()

    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        logger.error(f"Database context error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    """
    Initialize database tables.
    Creates all tables defined in models.
    """
    logger.info("Initializing database...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized successfully")


def drop_db():
    """
    Drop all database tables.
    WARNING: This will delete all data!
    """
    logger.warning("Dropping all database tables...")
    Base.metadata.drop_all(bind=engine)
    logger.info("Database tables dropped")

"""Database Session and Engine Management Module."""

import logging
from typing import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from ml.api import config as api_config

logger = logging.getLogger(__name__)


def build_engine():
    """Build SQLAlchemy engine based on DATABASE_URL configuration."""
    db_url = api_config.DATABASE_URL
    is_sqlite = db_url.startswith("sqlite")

    if is_sqlite:
        engine_kwargs = {
            "connect_args": {"check_same_thread": False},
            "pool_pre_ping": True
        }
    else:
        engine_kwargs = {
            "pool_size": api_config.DATABASE_POOL_SIZE,
            "max_overflow": api_config.DATABASE_MAX_OVERFLOW,
            "pool_timeout": api_config.DATABASE_POOL_TIMEOUT,
            "pool_recycle": api_config.DATABASE_POOL_RECYCLE,
            "pool_pre_ping": True
        }

    return create_engine(db_url, **engine_kwargs)


engine = build_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for yielding database session per request."""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def check_db_health(db: Session | None = None) -> bool:
    """Execute lightweight connectivity query (SELECT 1) to verify database status."""
    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True

    try:
        db.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.warning("Database connectivity check failed: %s", str(e))
        return False
    finally:
        if close_session and db is not None:
            db.close()

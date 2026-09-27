"""API Configuration management for FastAPI Recommendation backend."""

import os
import logging
from pathlib import Path
from urllib.parse import urlparse, urlunparse
from dotenv import load_dotenv
from ml.src import config as ml_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# Environment & Debug settings
ENV = os.getenv("ENVIRONMENT", os.getenv("ENV", "development")).lower()
DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "t") if ENV != "production" else False
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

API_TITLE = "ML Recommendation System REST API"
API_DESCRIPTION = (
    "Production-oriented REST API providing Content-Based, Item-Based Collaborative Filtering, "
    "and Hybrid movie recommendations built on MovieLens ml-latest-small."
)
API_VERSION = "1.0.0"
API_V1_STR = "/api/v1"

# Inherit data paths from ml/src/config.py
PROCESSED_DATA_DIR = ml_config.PROCESSED_DATA_DIR
MOVIES_CSV_PATH = PROCESSED_DATA_DIR / "movies_processed.csv"
RATINGS_CSV_PATH = PROCESSED_DATA_DIR / "ratings_processed.csv"

# Production-safe CORS configuration
raw_origins = os.getenv("ALLOWED_ORIGINS", os.getenv("CORS_ORIGINS", "*"))
if raw_origins.strip() == "*":
    ALLOWED_ORIGINS = ["*"]
else:
    ALLOWED_ORIGINS = [o.strip() for o in raw_origins.split(",") if o.strip()]

# Database configuration
raw_db_url = os.getenv(
    "DATABASE_URL",
    "sqlite:///./recommendations.db"
)
if raw_db_url and raw_db_url.startswith("DATABASE_URL="):
    raw_db_url = raw_db_url.split("DATABASE_URL=", 1)[1]

DATABASE_URL = raw_db_url

DATABASE_POOL_SIZE = int(os.getenv("DATABASE_POOL_SIZE", "5"))
DATABASE_MAX_OVERFLOW = int(os.getenv("DATABASE_MAX_OVERFLOW", "10"))
DATABASE_POOL_TIMEOUT = int(os.getenv("DATABASE_POOL_TIMEOUT", "30"))
DATABASE_POOL_RECYCLE = int(os.getenv("DATABASE_POOL_RECYCLE", "1800"))


def get_sanitized_db_url(url: str | None = None) -> str:
    """Mask sensitive database passwords in connection URLs for safe logging."""
    target_url = url or DATABASE_URL
    if not target_url:
        return "None"
    try:
        parsed = urlparse(target_url)
        if parsed.password:
            netloc = parsed.netloc.replace(f":{parsed.password}@", ":****@")
            return urlunparse(parsed._replace(netloc=netloc))
        return target_url
    except Exception:
        return "masked_db_url"


def validate_config() -> None:
    """Validate required environment configuration at application startup."""
    logger = logging.getLogger(__name__)
    logger.info("Application environment mode: %s (debug=%s)", ENV, DEBUG)
    logger.info("Database URL configured: %s", get_sanitized_db_url())

    if ENV == "production":
        if "sqlite" in DATABASE_URL.lower() and not os.getenv("ALLOW_SQLITE_IN_PROD"):
            raise ValueError(
                "Production mode detected (ENV=production), but SQLite is configured as DATABASE_URL. "
                "Production deployments must use PostgreSQL."
            )

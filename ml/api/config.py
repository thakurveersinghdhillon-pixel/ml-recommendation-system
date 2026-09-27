"""API Configuration management for FastAPI Recommendation backend."""

import os
from pathlib import Path
from dotenv import load_dotenv
from ml.src import config as ml_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

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

# CORS configuration
ALLOWED_ORIGINS = ["*"]

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

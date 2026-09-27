"""API Configuration management for FastAPI Recommendation backend."""

import os
from pathlib import Path
from ml.src import config as ml_config

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

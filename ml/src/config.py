"""Configuration management for the ML recommendation system pipeline."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Path configuration
# config.py is located in ml/src/ -> parent is ml/ -> parent.parent is project root
SRC_DIR = Path(__file__).resolve().parent
ML_DIR = SRC_DIR.parent
PROJECT_ROOT = ML_DIR.parent

# Load environment variables if .env file exists
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# Data directories
RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", ML_DIR / "data" / "raw"))
PROCESSED_DATA_DIR = Path(os.getenv("PROCESSED_DATA_DIR", ML_DIR / "data" / "processed"))

# Dataset config
MOVIELENS_URL = os.getenv(
    "MOVIELENS_URL",
    "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
)

# Schema validation rules
REQUIRED_MOVIE_COLUMNS = ["movieId", "title", "genres"]
REQUIRED_RATING_COLUMNS = ["userId", "movieId", "rating", "timestamp"]

MIN_RATING = 0.5
MAX_RATING = 5.0

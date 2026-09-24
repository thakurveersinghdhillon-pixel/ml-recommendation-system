"""Data preprocessing module for MovieLens dataset."""

import logging
import re
import pandas as pd
import numpy as np

from ml.src import config
from ml.src.data_validator import validate_movies, validate_ratings

logger = logging.getLogger(__name__)


def extract_year(title: str) -> float:
    """Extract four-digit release year in parentheses from title string.

    Example:
        'Toy Story (1995)' -> 1995.0
        'No Year Movie' -> np.nan

    Args:
        title: Movie title string.

    Returns:
        Extracted year as float (or NaN if missing/unparseable).
    """
    if not isinstance(title, str):
        return np.nan

    match = re.search(r"\((\d{4})\)", title)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return np.nan
    return np.nan


def split_genres(genres_str: str) -> list[str]:
    """Split pipe-separated genres string into a list of strings.

    Example:
        'Action|Adventure|Sci-Fi' -> ['Action', 'Adventure', 'Sci-Fi']
        '(no genres listed)' -> ['(no genres listed)']

    Args:
        genres_str: Pipe-separated genres string.

    Returns:
        List of genre names.
    """
    if not isinstance(genres_str, str) or not genres_str.strip():
        return ["(no genres listed)"]

    genres = [g.strip() for g in genres_str.split("|") if g.strip()]
    return genres if genres else ["(no genres listed)"]


def preprocess_movies(movies_df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess movies DataFrame.

    Validates schema, extracts release year, parses genres into list format,
    and preserves movie titles and unique IDs.

    Args:
        movies_df: Raw movies DataFrame.

    Returns:
        Preprocessed movies DataFrame copy.

    Raises:
        ValueError: If validation fails.
    """
    val_res = validate_movies(movies_df)
    if not val_res["passed"]:
        raise ValueError(f"Movies preprocessing failed validation: {val_res['errors']}")

    df = movies_df.copy()

    # Extract year from title
    df["year"] = df["title"].apply(extract_year)

    # Parse pipe-separated genres into list
    df["genres_list"] = df["genres"].apply(split_genres)

    return df


def preprocess_ratings(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess ratings DataFrame.

    Validates required columns, score bounds [0.5, 5.0], non-null keys,
    and converts Unix timestamps to pandas datetime objects.

    Args:
        ratings_df: Raw ratings DataFrame.

    Returns:
        Preprocessed ratings DataFrame copy.

    Raises:
        ValueError: If validation fails.
    """
    val_res = validate_ratings(ratings_df)
    if not val_res["passed"]:
        raise ValueError(f"Ratings preprocessing failed validation: {val_res['errors']}")

    df = ratings_df.copy()

    # Convert timestamp to datetime representation
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

    return df


def preprocess_dataset(
    movies_df: pd.DataFrame,
    ratings_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform dataset-level preprocessing, duplicate auditing, and referential integrity checks.

    Args:
        movies_df: Raw movies DataFrame.
        ratings_df: Raw ratings DataFrame.

    Returns:
        Tuple of (movies_processed, ratings_processed).

    Raises:
        ValueError: If duplicate keys or broken referential integrity are found.
    """
    movies_processed = preprocess_movies(movies_df)
    ratings_processed = preprocess_ratings(ratings_df)

    # Audit duplicate movie IDs
    if movies_processed["movieId"].duplicated().any():
        raise ValueError("Duplicate movieId values found during dataset preprocessing.")

    # Audit referential integrity
    valid_movie_ids = set(movies_processed["movieId"].unique())
    rated_movie_ids = set(ratings_processed["movieId"].unique())
    unmatched = rated_movie_ids - valid_movie_ids

    if unmatched:
        raise ValueError(
            f"Referential integrity failure: {len(unmatched)} rated movie ID(s) "
            f"do not exist in movie catalog: {unmatched}"
        )

    logger.info("Dataset preprocessing completed successfully.")
    return movies_processed, ratings_processed

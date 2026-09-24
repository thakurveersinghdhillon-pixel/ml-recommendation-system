"""Data validation module for MovieLens dataset."""

import logging
import pandas as pd

from ml.src import config

logger = logging.getLogger(__name__)


def validate_movies(movies_df: pd.DataFrame) -> dict:
    """Validate movies DataFrame schema, completeness, and key uniqueness.

    Args:
        movies_df: Pandas DataFrame containing movie records.

    Returns:
        Dictionary containing validation status ('passed': bool), errors, warnings, and metrics.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Check required columns
    missing_cols = [col for col in config.REQUIRED_MOVIE_COLUMNS if col not in movies_df.columns]
    if missing_cols:
        errors.append(f"Movies DataFrame missing required columns: {missing_cols}")
        return {"passed": False, "errors": errors, "warnings": warnings, "metrics": {}}

    # Check null values in key columns
    null_movie_ids = int(movies_df["movieId"].isnull().sum())
    if null_movie_ids > 0:
        errors.append(f"Found {null_movie_ids} null value(s) in 'movieId'.")

    null_titles = int(movies_df["title"].isnull().sum())
    if null_titles > 0:
        errors.append(f"Found {null_titles} null value(s) in 'title'.")

    null_genres = int(movies_df["genres"].isnull().sum())
    if null_genres > 0:
        errors.append(f"Found {null_genres} null value(s) in 'genres'.")

    # Check movieId uniqueness
    duplicate_ids = int(movies_df["movieId"].duplicated().sum())
    if duplicate_ids > 0:
        errors.append(f"Found {duplicate_ids} duplicate 'movieId' entry/entries.")

    # Check total row duplicates
    duplicate_rows = int(movies_df.duplicated().sum())
    if duplicate_rows > 0:
        warnings.append(f"Found {duplicate_rows} duplicate row(s) in movies data.")

    metrics = {
        "total_movies": len(movies_df),
        "unique_movie_ids": int(movies_df["movieId"].nunique()),
        "null_movie_ids": null_movie_ids,
        "null_titles": null_titles,
        "null_genres": null_genres,
        "duplicate_movie_ids": duplicate_ids,
        "duplicate_rows": duplicate_rows,
    }

    passed = len(errors) == 0
    return {"passed": passed, "errors": errors, "warnings": warnings, "metrics": metrics}


def validate_ratings(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame | None = None
) -> dict:
    """Validate ratings DataFrame schema, value ranges, duplicate user-movie pairs,
    and foreign key integrity against movies DataFrame.

    Args:
        ratings_df: Pandas DataFrame containing rating records.
        movies_df: Optional Pandas DataFrame of movies for referential integrity checks.

    Returns:
        Dictionary containing validation status ('passed': bool), errors, warnings, and metrics.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Check required columns
    missing_cols = [col for col in config.REQUIRED_RATING_COLUMNS if col not in ratings_df.columns]
    if missing_cols:
        errors.append(f"Ratings DataFrame missing required columns: {missing_cols}")
        return {"passed": False, "errors": errors, "warnings": warnings, "metrics": {}}

    # Check null values in key columns
    null_user_ids = int(ratings_df["userId"].isnull().sum())
    if null_user_ids > 0:
        errors.append(f"Found {null_user_ids} null value(s) in 'userId'.")

    null_movie_ids = int(ratings_df["movieId"].isnull().sum())
    if null_movie_ids > 0:
        errors.append(f"Found {null_movie_ids} null value(s) in 'movieId'.")

    null_ratings = int(ratings_df["rating"].isnull().sum())
    if null_ratings > 0:
        errors.append(f"Found {null_ratings} null value(s) in 'rating'.")

    # Check valid rating range
    if null_ratings < len(ratings_df):
        invalid_ratings = ratings_df[
            (ratings_df["rating"] < config.MIN_RATING) |
            (ratings_df["rating"] > config.MAX_RATING)
        ]
        invalid_count = len(invalid_ratings)
        if invalid_count > 0:
            errors.append(
                f"Found {invalid_count} rating(s) outside allowed range "
                f"[{config.MIN_RATING}, {config.MAX_RATING}]."
            )

    # Check duplicate (userId, movieId) pairs
    duplicate_user_movie = int(ratings_df.duplicated(subset=["userId", "movieId"]).sum())
    if duplicate_user_movie > 0:
        warnings.append(
            f"Found {duplicate_user_movie} duplicate (userId, movieId) rating pair(s)."
        )

    # Referential integrity check
    missing_movies_in_catalog = 0
    if movies_df is not None and "movieId" in movies_df.columns:
        valid_movie_ids = set(movies_df["movieId"].dropna().unique())
        rated_movie_ids = set(ratings_df["movieId"].dropna().unique())
        unmatched_ids = rated_movie_ids - valid_movie_ids
        missing_movies_in_catalog = len(unmatched_ids)

        if missing_movies_in_catalog > 0:
            errors.append(
                f"Found {missing_movies_in_catalog} rated movie ID(s) that do not exist "
                "in the movies catalog."
            )

    metrics = {
        "total_ratings": len(ratings_df),
        "unique_users": int(ratings_df["userId"].nunique()),
        "unique_rated_movies": int(ratings_df["movieId"].nunique()),
        "null_user_ids": null_user_ids,
        "null_movie_ids": null_movie_ids,
        "null_ratings": null_ratings,
        "invalid_ratings": len(invalid_ratings) if 'invalid_ratings' in locals() else 0,
        "duplicate_user_movie_pairs": duplicate_user_movie,
        "unmatched_movie_ids": missing_movies_in_catalog,
    }

    passed = len(errors) == 0
    return {"passed": passed, "errors": errors, "warnings": warnings, "metrics": metrics}


def validate_dataset(movies_df: pd.DataFrame, ratings_df: pd.DataFrame) -> dict:
    """Validate both movies and ratings DataFrames.

    Args:
        movies_df: Pandas DataFrame containing movies.
        ratings_df: Pandas DataFrame containing ratings.

    Returns:
        Summary validation dictionary.
    """
    movies_val = validate_movies(movies_df)
    ratings_val = validate_ratings(ratings_df, movies_df=movies_df)

    all_errors = movies_val["errors"] + ratings_val["errors"]
    all_warnings = movies_val["warnings"] + ratings_val["warnings"]
    passed = movies_val["passed"] and ratings_val["passed"]

    return {
        "passed": passed,
        "movies": movies_val,
        "ratings": ratings_val,
        "errors": all_errors,
        "warnings": all_warnings,
    }

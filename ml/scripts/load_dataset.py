"""CLI script to load, validate, and summarize the MovieLens dataset."""

import sys
from pathlib import Path

# Ensure ML package is importable when script is run directly
script_dir = Path(__file__).resolve().parent
ml_dir = script_dir.parent
project_root = ml_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.src.data_loader import load_dataset
from ml.src.data_validator import validate_dataset


def main():
    print("Loading MovieLens dataset...")
    movies_df, ratings_df = load_dataset(auto_download=True)

    # Perform validation
    val_summary = validate_dataset(movies_df, ratings_df)

    # Calculate actual dataset statistics
    num_users = int(ratings_df["userId"].nunique())
    num_movies = int(movies_df["movieId"].nunique())
    num_ratings = len(ratings_df)
    num_unique_ratings = int(ratings_df["rating"].nunique())

    mean_rating = float(ratings_df["rating"].mean())
    min_rating = float(ratings_df["rating"].min())
    max_rating = float(ratings_df["rating"].max())

    movies_cols = list(movies_df.columns)
    ratings_cols = list(ratings_df.columns)

    movies_missing = dict(movies_df.isnull().sum())
    ratings_missing = dict(ratings_df.isnull().sum())

    movies_duplicate_rows = int(movies_df.duplicated().sum())
    ratings_duplicate_rows = int(ratings_df.duplicated().sum())

    # Referentially valid movies check
    rated_movie_ids = set(ratings_df["movieId"].unique())
    catalog_movie_ids = set(movies_df["movieId"].unique())
    unmatched_movies = rated_movie_ids - catalog_movie_ids
    all_rated_movies_exist = (len(unmatched_movies) == 0)

    # Print requested overview
    print("========================================")
    print("MovieLens Dataset")
    print("========================================")
    print()
    print(f"Number of users: {num_users}")
    print(f"Number of movies: {num_movies}")
    print(f"Number of ratings: {num_ratings}")
    print(f"Number of unique rating values: {num_unique_ratings}")
    print()
    print("Rating statistics:")
    print(f"Mean rating: {mean_rating:.2f}")
    print(f"Minimum rating: {min_rating:.1f}")
    print(f"Maximum rating: {max_rating:.1f}")
    print()
    print(f"Movies columns:\n  {movies_cols}")
    print(f"Ratings columns:\n  {ratings_cols}")
    print()
    print("Data Quality & Integrity Checks:")
    print(f"  Movies missing values per column: {movies_missing}")
    print(f"  Ratings missing values per column: {ratings_missing}")
    print(f"  Movies duplicate rows: {movies_duplicate_rows}")
    print(f"  Ratings duplicate rows: {ratings_duplicate_rows}")
    print(f"  Every rated movie exists in movies catalog: {all_rated_movies_exist}")

    if val_summary["warnings"]:
        print("\nWarnings:")
        for warning in val_summary["warnings"]:
            print(f"  - {warning}")

    print()
    if val_summary["passed"]:
        print("Dataset validation: PASSED")
    else:
        print("Dataset validation: FAILED")
        print("Errors:")
        for err in val_summary["errors"]:
            print(f"  - {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()

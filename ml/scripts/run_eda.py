"""CLI script to run data preprocessing, EDA analysis, plot generation, and save processed datasets."""

import json
import sys
from pathlib import Path

# Ensure ML package is importable when script is run directly
script_dir = Path(__file__).resolve().parent
ml_dir = script_dir.parent
project_root = ml_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.src import config
from ml.src.data_loader import load_dataset
from ml.src.preprocessing import preprocess_dataset
from ml.src.eda import (
    compute_user_statistics,
    compute_movie_statistics,
    compute_genre_statistics,
    build_user_item_matrix,
    calculate_sparsity,
    generate_all_eda_plots,
)


def main():
    print("=" * 60)
    print("PHASE 2 — DATA PREPROCESSING + EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print()

    # 1. Load Raw Dataset
    print("[1/6] Loading raw dataset...")
    movies_raw, ratings_raw = load_dataset(auto_download=False)
    print(f"      Loaded {len(movies_raw):,} raw movies and {len(ratings_raw):,} raw ratings.")

    # 2. Preprocess Dataset
    print("[2/6] Running data preprocessing...")
    movies_proc, ratings_proc = preprocess_dataset(movies_raw, ratings_raw)
    print("      Preprocessing & validation passed successfully.")

    # 3. Compute Statistical Summaries
    print("[3/6] Computing user, movie, and genre statistics...")
    user_stats = compute_user_statistics(ratings_proc)
    movie_stats = compute_movie_statistics(movies_proc, ratings_proc)
    genre_stats = compute_genre_statistics(movies_proc, ratings_proc)

    # 4. User-Item Interaction Matrix & Sparsity
    print("[4/6] Constructing user-item interaction matrix & calculating sparsity...")
    user_item_matrix = build_user_item_matrix(ratings_proc)
    sparsity_metrics = calculate_sparsity(ratings_proc, movies_proc)

    # 5. Save Processed Datasets
    print("[5/6] Saving processed datasets to ml/data/processed/...")
    processed_dir = config.PROCESSED_DATA_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Convert list column for clean CSV saving
    movies_save = movies_proc.copy()
    movies_save["genres_list"] = movies_save["genres_list"].apply(json.dumps)

    movies_save.to_csv(processed_dir / "movies_processed.csv", index=False)
    ratings_proc.to_csv(processed_dir / "ratings_processed.csv", index=False)
    user_stats.to_csv(processed_dir / "user_statistics.csv", index=False)
    movie_stats.to_csv(processed_dir / "movie_statistics.csv", index=False)
    genre_stats.to_csv(processed_dir / "genre_statistics.csv", index=False)

    print("      Saved processed CSV files.")

    # 6. Generate EDA Plots
    print("[6/6] Generating EDA visualizations...")
    eda_plots_dir = processed_dir / "eda"
    generated_plots = generate_all_eda_plots(
        movies_proc,
        ratings_proc,
        user_stats,
        movie_stats,
        genre_stats,
        user_item_matrix,
        output_dir=eda_plots_dir
    )
    print(f"      Generated {len(generated_plots)} EDA plots in {eda_plots_dir}")

    # Calculate final dataset metrics
    mean_rating = float(ratings_proc["rating"].mean())
    median_rating = float(ratings_proc["rating"].median())
    std_rating = float(ratings_proc["rating"].std())

    avg_ratings_per_user = float(user_stats["rating_count"].mean())
    avg_ratings_per_movie = float(movie_stats["rating_count"].mean())

    zero_rating_movies_count = int((movie_stats["rating_count"] == 0).sum())

    print()
    print("=" * 60)
    print("PHASE 2 SUMMARY & DATASET METRICS")
    print("=" * 60)
    print(f"  Users:                   {sparsity_metrics['num_users']:,}")
    print(f"  Movies (Catalog):        {sparsity_metrics['num_movies']:,}")
    print(f"  Ratings (Observed):      {sparsity_metrics['observed_ratings']:,}")
    print(f"  Zero-Rating Movies:      {zero_rating_movies_count}")
    print(f"  Mean Rating:             {mean_rating:.2f}")
    print(f"  Median Rating:           {median_rating:.2f}")
    print(f"  Rating Std Dev:          {std_rating:.2f}")
    print(f"  Avg Ratings / User:      {avg_ratings_per_user:.2f}")
    print(f"  Avg Ratings / Movie:     {avg_ratings_per_movie:.2f}")
    print(f"  Unique Genres:           {len(genre_stats)}")
    print(f"  Possible Interactions:   {sparsity_metrics['possible_interactions']:,}")
    print(f"  Sparsity Percentage:     {sparsity_metrics['sparsity_percent']:.2f}%")
    print(f"  Density Percentage:      {sparsity_metrics['density_percent']:.2f}%")
    print("=" * 60)
    print("Pipeline Execution Status: SUCCESS")
    print("=" * 60)


if __name__ == "__main__":
    main()

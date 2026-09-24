"""Exploratory Data Analysis (EDA) module for MovieLens dataset."""

import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from ml.src import config

logger = logging.getLogger(__name__)

# Apply clean plot style
sns.set_theme(style="whitegrid")


def compute_user_statistics(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate rating activity and score statistics for each user.

    Args:
        ratings_df: Preprocessed ratings DataFrame.

    Returns:
        DataFrame with columns: userId, rating_count, average_rating, median_rating, min_rating, max_rating.
    """
    user_stats = (
        ratings_df.groupby("userId")["rating"]
        .agg(
            rating_count="count",
            average_rating="mean",
            median_rating="median",
            min_rating="min",
            max_rating="max",
        )
        .reset_index()
    )
    return user_stats


def compute_movie_statistics(
    movies_df: pd.DataFrame,
    ratings_df: pd.DataFrame
) -> pd.DataFrame:
    """Calculate rating activity and score statistics for each movie in the catalog.

    CRITICAL REQUIREMENT: Performs a LEFT JOIN from the movie catalog to aggregated ratings,
    preserving movies with zero ratings (rating_count = 0, average_rating = NaN).

    Args:
        movies_df: Preprocessed movies DataFrame.
        ratings_df: Preprocessed ratings DataFrame.

    Returns:
        DataFrame containing movie details along with rating statistics.
    """
    aggregated_ratings = (
        ratings_df.groupby("movieId")["rating"]
        .agg(
            rating_count="count",
            average_rating="mean",
            median_rating="median",
        )
        .reset_index()
    )

    # Left join catalog to rating stats to keep 0-rating movies
    movie_stats = movies_df.merge(aggregated_ratings, on="movieId", how="left")

    # Fill missing rating counts with 0 (average_rating remains NaN)
    movie_stats["rating_count"] = movie_stats["rating_count"].fillna(0).astype(int)

    return movie_stats


def get_popular_movies(movie_stats_df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Extract top N popular movies based primarily on rating count.

    Args:
        movie_stats_df: Movie statistics DataFrame.
        n: Number of top movies to return.

    Returns:
        DataFrame containing top N movies sorted by rating count.
    """
    popular = movie_stats_df.sort_values(
        by=["rating_count", "average_rating"],
        ascending=[False, False]
    ).head(n)

    return popular[["movieId", "title", "rating_count", "average_rating"]]


def compute_genre_statistics(
    movies_df: pd.DataFrame,
    ratings_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Compute per-genre movie counts, rating counts, and average ratings.

    Explodes multi-genre movies so each genre is counted individually.

    Args:
        movies_df: Preprocessed movies DataFrame.
        ratings_df: Optional preprocessed ratings DataFrame.

    Returns:
        DataFrame with columns: genre, movie_count, rating_count, average_rating.
    """
    # Ensure genres_list column exists
    if "genres_list" not in movies_df.columns:
        movies_df = movies_df.copy()
        movies_df["genres_list"] = movies_df["genres"].apply(
            lambda g: [x.strip() for x in str(g).split("|")] if pd.notnull(g) else ["(no genres listed)"]
        )

    exploded_movies = movies_df.explode("genres_list").rename(columns={"genres_list": "genre"})

    if ratings_df is None:
        genre_stats = (
            exploded_movies.groupby("genre")
            .agg(movie_count=("movieId", "nunique"))
            .reset_index()
            .sort_values(by="movie_count", ascending=False)
        )
        genre_stats["rating_count"] = 0
        genre_stats["average_rating"] = np.nan
        return genre_stats

    # Merge exploded movies with ratings
    merged = exploded_movies.merge(ratings_df, on="movieId", how="inner")

    # Movie count per genre (from catalog)
    movie_counts = exploded_movies.groupby("genre")["movieId"].nunique().rename("movie_count")

    # Rating statistics per genre (from ratings)
    genre_ratings = (
        merged.groupby("genre")["rating"]
        .agg(rating_count="count", average_rating="mean")
    )

    # Combine metrics
    genre_stats = pd.concat([movie_counts, genre_ratings], axis=1).reset_index()
    genre_stats["rating_count"] = genre_stats["rating_count"].fillna(0).astype(int)
    genre_stats = genre_stats.sort_values(by="movie_count", ascending=False)

    return genre_stats


def build_user_item_matrix(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Construct a user-item interaction matrix from ratings.

    Rows: users (userId)
    Columns: movies (movieId)
    Values: rating score

    Unobserved interactions remain NaN (not 0).

    Args:
        ratings_df: Preprocessed ratings DataFrame.

    Returns:
        Pandas DataFrame representing user-item matrix.
    """
    user_item_matrix = ratings_df.pivot(
        index="userId",
        columns="movieId",
        values="rating"
    )
    return user_item_matrix


def calculate_sparsity(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame | None = None
) -> dict:
    """Calculate dataset matrix sparsity and interaction density metrics.

    Formula:
        sparsity = 1.0 - (observed_ratings / (number_of_users * number_of_movies))

    Args:
        ratings_df: Preprocessed ratings DataFrame.
        movies_df: Optional preprocessed movies DataFrame.

    Returns:
        Dictionary containing sparsity percentage, density percentage, and counts.
    """
    num_users = int(ratings_df["userId"].nunique())
    num_movies = int(movies_df["movieId"].nunique()) if movies_df is not None else int(ratings_df["movieId"].nunique())

    possible_interactions = num_users * num_movies
    observed_ratings = len(ratings_df)

    if possible_interactions == 0:
        sparsity = 0.0
    else:
        sparsity = 1.0 - (observed_ratings / possible_interactions)

    sparsity_percent = float(sparsity * 100.0)
    density_percent = float((1.0 - sparsity) * 100.0)

    return {
        "num_users": num_users,
        "num_movies": num_movies,
        "possible_interactions": possible_interactions,
        "observed_ratings": observed_ratings,
        "sparsity": float(sparsity),
        "sparsity_percent": sparsity_percent,
        "density_percent": density_percent,
    }


def generate_all_eda_plots(
    movies_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    user_stats: pd.DataFrame,
    movie_stats: pd.DataFrame,
    genre_stats: pd.DataFrame,
    user_item_matrix: pd.DataFrame,
    output_dir: Path | str | None = None
) -> list[Path]:
    """Generate and save the 8 required EDA visualizations.

    Args:
        movies_df: Processed movies.
        ratings_df: Processed ratings.
        user_stats: Computed user statistics.
        movie_stats: Computed movie statistics.
        genre_stats: Computed genre statistics.
        user_item_matrix: User-item interaction matrix.
        output_dir: Directory to save PNG plots.

    Returns:
        List of generated plot file paths.
    """
    out_path = Path(output_dir) if output_dir is not None else config.PROCESSED_DATA_DIR / "eda"
    out_path.mkdir(parents=True, exist_ok=True)
    generated_files: list[Path] = []

    # 1. Rating Distribution
    plt.figure(figsize=(8, 5))
    rating_counts = ratings_df["rating"].value_counts().sort_index()
    sns.barplot(x=rating_counts.index, y=rating_counts.values, color="royalblue")
    plt.title("MovieLens Rating Score Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Rating Score", fontsize=12)
    plt.ylabel("Frequency Count", fontsize=12)
    plt.tight_layout()
    p1 = out_path / "rating_distribution.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    generated_files.append(p1)

    # 2. Ratings per User Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(user_stats["rating_count"], bins=50, kde=True, color="teal")
    plt.title("User Rating Activity Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Number of Ratings per User", fontsize=12)
    plt.ylabel("Number of Users", fontsize=12)
    plt.tight_layout()
    p2 = out_path / "ratings_per_user_distribution.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    generated_files.append(p2)

    # 3. Ratings per Movie Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(movie_stats["rating_count"], bins=50, kde=True, color="coral")
    plt.title("Movie Rating Popularity Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Number of Ratings per Movie", fontsize=12)
    plt.ylabel("Number of Movies", fontsize=12)
    plt.tight_layout()
    p3 = out_path / "ratings_per_movie_distribution.png"
    plt.savefig(p3, dpi=300)
    plt.close()
    generated_files.append(p3)

    # 4. Top Popular Movies
    plt.figure(figsize=(10, 6))
    top_20 = get_popular_movies(movie_stats, n=20)
    sns.barplot(data=top_20, y="title", x="rating_count", hue="title", palette="viridis", legend=False)
    plt.title("Top 20 Most Popular Movies (by Rating Count)", fontsize=14, fontweight="bold")
    plt.xlabel("Total Rating Count", fontsize=12)
    plt.ylabel("Movie Title", fontsize=12)
    plt.tight_layout()
    p4 = out_path / "top_popular_movies.png"
    plt.savefig(p4, dpi=300)
    plt.close()
    generated_files.append(p4)

    # 5. Genre Distribution
    plt.figure(figsize=(10, 6))
    top_genres = genre_stats.head(20)
    sns.barplot(data=top_genres, y="genre", x="movie_count", hue="genre", palette="mako", legend=False)
    plt.title("Movie Count per Genre Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Number of Unique Movies", fontsize=12)
    plt.ylabel("Genre", fontsize=12)
    plt.tight_layout()
    p5 = out_path / "genre_distribution.png"
    plt.savefig(p5, dpi=300)
    plt.close()
    generated_files.append(p5)

    # 6. Rating vs Rating Count Scatter Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=movie_stats,
        x="rating_count",
        y="average_rating",
        alpha=0.5,
        color="darkpurple" if "darkpurple" in sns.xkcd_rgb else "purple"
    )
    plt.title("Average Rating vs. Rating Count per Movie", fontsize=14, fontweight="bold")
    plt.xlabel("Rating Count (Popularity)", fontsize=12)
    plt.ylabel("Average Rating Score", fontsize=12)
    plt.tight_layout()
    p6 = out_path / "rating_vs_rating_count_scatter.png"
    plt.savefig(p6, dpi=300)
    plt.close()
    generated_files.append(p6)

    # 7. Rating Activity Over Time
    plt.figure(figsize=(10, 5))
    if "datetime" in ratings_df.columns:
        ratings_df["year"] = ratings_df["datetime"].dt.year
        yearly_counts = ratings_df.groupby("year")["rating"].count().reset_index()
        sns.lineplot(data=yearly_counts, x="year", y="rating", marker="o", color="crimson")
        plt.title("MovieLens Rating Activity Over Time (Yearly)", fontsize=14, fontweight="bold")
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("Total Ratings Received", fontsize=12)
    plt.tight_layout()
    p7 = out_path / "rating_activity_over_time.png"
    plt.savefig(p7, dpi=300)
    plt.close()
    generated_files.append(p7)

    # 8. User-Item Matrix Sparsity Visualization (Sampled 50x50)
    plt.figure(figsize=(8, 7))
    sample_users = user_item_matrix.index[:50]
    sample_movies = user_item_matrix.columns[:50]
    sample_matrix = user_item_matrix.loc[sample_users, sample_movies]
    binary_matrix = sample_matrix.notnull().astype(int)

    sns.heatmap(binary_matrix, cbar=False, cmap="YlGnBu", linewidths=0.1)
    plt.title("User-Item Interaction Matrix Sparsity (50x50 Sample)", fontsize=14, fontweight="bold")
    plt.xlabel("Movie ID (Sample)", fontsize=12)
    plt.ylabel("User ID (Sample)", fontsize=12)
    plt.tight_layout()
    p8 = out_path / "user_item_matrix_sparsity.png"
    plt.savefig(p8, dpi=300)
    plt.close()
    generated_files.append(p8)

    logger.info("Generated %d EDA plots in %s.", len(generated_files), out_path)
    return generated_files

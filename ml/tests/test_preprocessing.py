"""Unit tests for preprocessing and EDA modules."""

import pytest
import numpy as np
import pandas as pd

from ml.src.preprocessing import (
    extract_year,
    split_genres,
    preprocess_movies,
    preprocess_ratings,
    preprocess_dataset,
)
from ml.src.eda import (
    compute_user_statistics,
    compute_movie_statistics,
    compute_genre_statistics,
    build_user_item_matrix,
    calculate_sparsity,
)


def test_extract_year():
    """Test four-digit year extraction from title strings."""
    assert extract_year("Toy Story (1995)") == 1995.0
    assert extract_year("Matrix, The (1999)") == 1999.0
    assert extract_year("Movie Without Year") is np.nan or pd.isna(extract_year("Movie Without Year"))
    assert extract_year(12345) is np.nan or pd.isna(extract_year(12345))


def test_split_genres():
    """Test splitting pipe-separated genre strings and handling no genres listed."""
    assert split_genres("Action|Adventure|Sci-Fi") == ["Action", "Adventure", "Sci-Fi"]
    assert split_genres("(no genres listed)") == ["(no genres listed)"]
    assert split_genres("") == ["(no genres listed)"]


def test_preprocess_movies():
    """Test movie preprocessing pipeline."""
    raw_movies = pd.DataFrame({
        "movieId": [1, 2],
        "title": ["Movie A (2020)", "Movie B (2021)"],
        "genres": ["Action|Comedy", "(no genres listed)"]
    })
    proc_movies = preprocess_movies(raw_movies)

    assert "year" in proc_movies.columns
    assert "genres_list" in proc_movies.columns
    assert proc_movies["year"].tolist() == [2020.0, 2021.0]
    assert proc_movies["genres_list"].iloc[0] == ["Action", "Comedy"]
    assert proc_movies["genres_list"].iloc[1] == ["(no genres listed)"]


def test_preprocess_ratings():
    """Test rating preprocessing pipeline and timestamp conversion."""
    raw_ratings = pd.DataFrame({
        "userId": [1, 1],
        "movieId": [10, 20],
        "rating": [4.0, 3.5],
        "timestamp": [1600000000, 1600000001]
    })
    proc_ratings = preprocess_ratings(raw_ratings)

    assert "datetime" in proc_ratings.columns
    assert pd.api.types.is_datetime64_any_dtype(proc_ratings["datetime"])
    assert proc_ratings["rating"].tolist() == [4.0, 3.5]


def test_movie_statistics_preserves_zero_ratings():
    """CRITICAL TEST: Ensure movies with 0 ratings are preserved in movie_statistics."""
    movies_df = pd.DataFrame({
        "movieId": [1, 2, 3],
        "title": ["Rated Movie 1", "Rated Movie 2", "Unrated Movie 3"],
        "genres": ["Action", "Drama", "Comedy"]
    })
    ratings_df = pd.DataFrame({
        "userId": [101, 102],
        "movieId": [1, 2],
        "rating": [4.0, 5.0],
        "timestamp": [1600000000, 1600000001]
    })

    movie_stats = compute_movie_statistics(movies_df, ratings_df)

    assert len(movie_stats) == 3  # All 3 catalog movies retained
    unrated_row = movie_stats[movie_stats["movieId"] == 3].iloc[0]
    assert unrated_row["rating_count"] == 0
    assert pd.isna(unrated_row["average_rating"])


def test_user_statistics():
    """Test user rating statistics aggregation."""
    ratings_df = pd.DataFrame({
        "userId": [1, 1, 2],
        "movieId": [10, 20, 10],
        "rating": [4.0, 2.0, 5.0],
        "timestamp": [1600000000, 1600000001, 1600000002]
    })
    user_stats = compute_user_statistics(ratings_df)

    user1 = user_stats[user_stats["userId"] == 1].iloc[0]
    assert user1["rating_count"] == 2
    assert user1["average_rating"] == 3.0
    assert user1["median_rating"] == 3.0
    assert user1["min_rating"] == 2.0
    assert user1["max_rating"] == 4.0


def test_genre_statistics():
    """Test per-genre exploding and movie_count vs rating_count distinction."""
    movies_df = pd.DataFrame({
        "movieId": [1, 2],
        "title": ["Multi-Genre", "Single-Genre"],
        "genres": ["Action|Comedy", "Action"],
        "genres_list": [["Action", "Comedy"], ["Action"]]
    })
    ratings_df = pd.DataFrame({
        "userId": [10, 20, 30],
        "movieId": [1, 1, 2],
        "rating": [5.0, 4.0, 3.0],
        "timestamp": [1, 2, 3]
    })

    genre_stats = compute_genre_statistics(movies_df, ratings_df)

    action_row = genre_stats[genre_stats["genre"] == "Action"].iloc[0]
    assert action_row["movie_count"] == 2  # Movie 1 and Movie 2 are both Action
    assert action_row["rating_count"] == 3  # 2 ratings for Movie 1 + 1 rating for Movie 2 = 3

    comedy_row = genre_stats[genre_stats["genre"] == "Comedy"].iloc[0]
    assert comedy_row["movie_count"] == 1  # Only Movie 1
    assert comedy_row["rating_count"] == 2  # 2 ratings for Movie 1


def test_user_item_matrix():
    """Test pivoting ratings into user-item interaction matrix preserving NaNs."""
    ratings_df = pd.DataFrame({
        "userId": [1, 1, 2],
        "movieId": [100, 200, 100],
        "rating": [4.5, 3.0, 5.0],
        "timestamp": [1, 2, 3]
    })
    matrix = build_user_item_matrix(ratings_df)

    assert matrix.shape == (2, 2)  # 2 users, 2 movies
    assert matrix.loc[1, 100] == 4.5
    assert matrix.loc[1, 200] == 3.0
    assert matrix.loc[2, 100] == 5.0
    assert pd.isna(matrix.loc[2, 200])  # Missing rating remains NaN


def test_calculate_sparsity():
    """Test exact sparsity calculation on small synthetic dataset.

    3 users, 4 movies -> 12 possible interactions.
    5 observed ratings -> sparsity = 1 - 5/12 = 7/12 = 58.333%
    """
    ratings_df = pd.DataFrame({
        "userId": [1, 1, 2, 2, 3],
        "movieId": [10, 20, 20, 30, 40],
        "rating": [4.0, 3.0, 5.0, 2.0, 1.0],
        "timestamp": [1, 2, 3, 4, 5]
    })
    metrics = calculate_sparsity(ratings_df)

    assert metrics["num_users"] == 3
    assert metrics["num_movies"] == 4
    assert metrics["possible_interactions"] == 12
    assert metrics["observed_ratings"] == 5
    assert pytest.approx(metrics["sparsity"], 0.001) == 0.58333
    assert pytest.approx(metrics["sparsity_percent"], 0.01) == 58.33

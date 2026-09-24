"""Unit tests for data_loader module."""

import pytest
import pandas as pd
from pathlib import Path

from ml.src.data_loader import (
    load_movies,
    load_ratings,
    load_dataset,
    find_dataset_file,
)


@pytest.fixture
def sample_data_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with valid sample movies and ratings CSVs."""
    movies_df = pd.DataFrame({
        "movieId": [1, 2, 3],
        "title": ["Movie A (2020)", "Movie B (2021)", "Movie C (2022)"],
        "genres": ["Action|Adventure", "Comedy", "Drama"]
    })
    ratings_df = pd.DataFrame({
        "userId": [101, 101, 102],
        "movieId": [1, 2, 1],
        "rating": [4.0, 3.5, 5.0],
        "timestamp": [1600000000, 1600000001, 1600000002]
    })

    movies_df.to_csv(tmp_path / "movies.csv", index=False)
    ratings_df.to_csv(tmp_path / "ratings.csv", index=False)
    return tmp_path


def test_load_movies_success(sample_data_dir: Path):
    """Test loading valid movies CSV."""
    df = load_movies(sample_data_dir)
    assert len(df) == 3
    assert list(df.columns) == ["movieId", "title", "genres"]
    assert df["movieId"].tolist() == [1, 2, 3]


def test_load_ratings_success(sample_data_dir: Path):
    """Test loading valid ratings CSV."""
    df = load_ratings(sample_data_dir)
    assert len(df) == 3
    assert list(df.columns) == ["userId", "movieId", "rating", "timestamp"]
    assert df["userId"].tolist() == [101, 101, 102]


def test_load_dataset_success(sample_data_dir: Path):
    """Test loading both movies and ratings together without auto-download."""
    movies_df, ratings_df = load_dataset(data_dir=sample_data_dir, auto_download=False)
    assert len(movies_df) == 3
    assert len(ratings_df) == 3


def test_missing_file_raises_error(tmp_path: Path):
    """Test missing dataset file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="movies.csv"):
        load_movies(tmp_path)

    with pytest.raises(FileNotFoundError, match="ratings.csv"):
        load_ratings(tmp_path)


def test_missing_required_columns_raises_error(tmp_path: Path):
    """Test loading file missing required columns raises ValueError."""
    invalid_movies = pd.DataFrame({"movieId": [1, 2], "title": ["A", "B"]})  # missing genres
    invalid_movies.to_csv(tmp_path / "movies.csv", index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_movies(tmp_path)


def test_find_dataset_file_in_subdirectory(tmp_path: Path):
    """Test finding dataset file in a nested subdirectory."""
    nested_dir = tmp_path / "ml-latest-small"
    nested_dir.mkdir()
    (nested_dir / "movies.csv").write_text("movieId,title,genres\n1,Test,Action\n")

    found_path = find_dataset_file("movies.csv", tmp_path)
    assert found_path == nested_dir / "movies.csv"

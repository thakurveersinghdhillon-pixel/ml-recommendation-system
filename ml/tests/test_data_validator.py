"""Unit tests for data_validator module."""

import pytest
import pandas as pd

from ml.src.data_validator import (
    validate_movies,
    validate_ratings,
    validate_dataset,
)


@pytest.fixture
def valid_movies_df() -> pd.DataFrame:
    return pd.DataFrame({
        "movieId": [1, 2, 3],
        "title": ["Toy Story (1995)", "Jumanji (1995)", "Grumpier Old Men (1995)"],
        "genres": ["Adventure|Animation", "Adventure|Children", "Comedy|Romance"]
    })


@pytest.fixture
def valid_ratings_df() -> pd.DataFrame:
    return pd.DataFrame({
        "userId": [1, 1, 2],
        "movieId": [1, 2, 1],
        "rating": [4.0, 5.0, 3.5],
        "timestamp": [964982703, 964981247, 964982224]
    })


def test_validate_valid_dataset(valid_movies_df, valid_ratings_df):
    """Test valid datasets pass validation."""
    summary = validate_dataset(valid_movies_df, valid_ratings_df)
    assert summary["passed"] is True
    assert len(summary["errors"]) == 0


def test_movies_missing_columns_fails():
    """Test missing required columns in movies DataFrame."""
    invalid_df = pd.DataFrame({"movieId": [1], "title": ["Toy Story"]})
    res = validate_movies(invalid_df)
    assert res["passed"] is False
    assert any("missing required columns" in err for err in res["errors"])


def test_movies_null_ids_detected(valid_movies_df):
    """Test null movieId is detected as error."""
    valid_movies_df.loc[0, "movieId"] = None
    res = validate_movies(valid_movies_df)
    assert res["passed"] is False
    assert any("null value(s) in 'movieId'" in err for err in res["errors"])


def test_movies_duplicate_ids_detected(valid_movies_df):
    """Test duplicate movie IDs are detected as error."""
    valid_movies_df.loc[2, "movieId"] = 1  # Duplicate of index 0
    res = validate_movies(valid_movies_df)
    assert res["passed"] is False
    assert any("duplicate 'movieId'" in err for err in res["errors"])


def test_ratings_null_ids_and_values_detected(valid_ratings_df):
    """Test null userId and rating values are detected."""
    valid_ratings_df.loc[0, "userId"] = None
    valid_ratings_df.loc[1, "rating"] = None
    res = validate_ratings(valid_ratings_df)
    assert res["passed"] is False
    assert any("null value(s) in 'userId'" in err for err in res["errors"])
    assert any("null value(s) in 'rating'" in err for err in res["errors"])


def test_ratings_invalid_range_detected(valid_ratings_df):
    """Test out-of-range rating values are detected."""
    valid_ratings_df.loc[0, "rating"] = 6.0  # Invalid (> 5.0)
    valid_ratings_df.loc[1, "rating"] = -1.0 # Invalid (< 0.5)
    res = validate_ratings(valid_ratings_df)
    assert res["passed"] is False
    assert any("outside allowed range" in err for err in res["errors"])


def test_ratings_foreign_key_mismatch_detected(valid_movies_df, valid_ratings_df):
    """Test rating referring to a non-existent movieId fails validation."""
    valid_ratings_df.loc[0, "movieId"] = 9999  # Movie 9999 does not exist in valid_movies_df
    res = validate_ratings(valid_ratings_df, movies_df=valid_movies_df)
    assert res["passed"] is False
    assert any("do not exist in the movies catalog" in err for err in res["errors"])
